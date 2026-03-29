# train.py
import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import cohen_kappa_score

from utils.data import IDRiDDataset
from utils.transforms import build_idrid_transform
from models.res_unet_pp import ResUNetPPMultiTask
from loss import IDRiDMultiTaskLoss


# Repro
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Device helpers
def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


# CORAL helpers
@torch.no_grad()
def coral_predict(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    # logits: [B, K-1]
    probs = torch.sigmoid(logits)
    return (probs > threshold).sum(dim=1)


def compute_coral_pos_weight(labels: np.ndarray, num_classes: int) -> Optional[torch.Tensor]:
    """
    For CORAL, each threshold j defines a binary task: label > j.
    pos_weight_j = negatives / positives
    """
    if labels.size == 0:
        return None

    weights = []
    for j in range(num_classes - 1):
        pos = np.sum(labels > j)
        neg = np.sum(labels <= j)

        if pos == 0 or neg == 0:
            weights.append(1.0)
        else:
            weights.append(float(neg / pos))

    return torch.tensor(weights, dtype=torch.float32)


# Segmentation class weighting
def compute_seg_class_weights(train_ds: IDRiDDataset, eps: float = 1e-8) -> Optional[torch.Tensor]:
    """
    Computes per-channel lesion weights from raw train-set masks only.
    Weight formula:
        w_c ∝ 1 / sqrt(p_c + eps)
    then normalized to mean 1 and clipped to [1, 5].
    """
    seg_df = train_ds.df[train_ds.df["has_seg"]].copy()
    if len(seg_df) == 0:
        return None

    pos_counts = np.zeros(4, dtype=np.float64)
    total_pixels = 0.0

    for _, row in seg_df.iterrows():
        image = train_ds._read_rgb(row["image_path"])
        h, w = image.shape[:2]
        seg = train_ds._load_seg_mask(row, (h, w))  # [H, W, 4], binary np.uint8
        pos_counts += seg.sum(axis=(0, 1)).astype(np.float64)
        total_pixels += float(h * w)

    prevalence = pos_counts / max(total_pixels, 1.0)
    weights = 1.0 / np.sqrt(prevalence + eps)

    # normalize mean weight to 1
    weights = weights / weights.mean()

    # clip to avoid crazy values
    weights = np.clip(weights, 1.0, 5.0)

    return torch.tensor(weights, dtype=torch.float32)


# Sampler
def build_train_sampler(train_ds: IDRiDDataset, task_mode: str, seg_oversample_factor: float):
    """
    Oversample segmentation-labeled rows in multitask mode because
    segmentation supervision is much rarer than grading supervision.
    """
    if task_mode != "multitask":
        return None

    weights = np.ones(len(train_ds), dtype=np.float64)

    has_seg = train_ds.df["has_seg"].to_numpy(dtype=bool)
    weights[has_seg] = float(seg_oversample_factor)

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(train_ds),
        replacement=True,
    )
    return sampler


# Optimizer / Scheduler
def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    backbone_lr_mult: float = 0.3,
) -> torch.optim.Optimizer:
    backbone_params = []
    head_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(p)
        else:
            head_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr * backbone_lr_mult},
            {"params": head_params, "lr": lr},
        ],
        weight_decay=weight_decay,
    )
    return optimizer


def build_scheduler(optimizer: torch.optim.Optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        threshold=1e-4,
        min_lr=1e-6,
    )


# Metrics
def init_epoch_stats(device: torch.device):
    return {
        "loss_total_sum": 0.0,
        "loss_seg_sum": 0.0,
        "loss_dr_sum": 0.0,
        "loss_dme_sum": 0.0,
        "num_batches": 0,

        "dr_correct": 0,
        "dr_count": 0,
        "dr_preds": [],
        "dr_targets": [],

        "dme_correct": 0,
        "dme_count": 0,
        "dme_preds": [],
        "dme_targets": [],

        "seg_inter": torch.zeros(4, dtype=torch.float64, device=device),
        "seg_union": torch.zeros(4, dtype=torch.float64, device=device),
        "seg_count": 0,
    }


@torch.no_grad()
def update_epoch_stats(stats, outputs, batch, loss_dict):
    stats["loss_total_sum"] += float(loss_dict["loss_total"].item())
    stats["loss_seg_sum"] += float(loss_dict["loss_seg"].item())
    stats["loss_dr_sum"] += float(loss_dict["loss_dr"].item())
    stats["loss_dme_sum"] += float(loss_dict["loss_dme"].item())
    stats["num_batches"] += 1

    grade_idx = batch["has_grade"]
    if grade_idx.any():
        dr_logits = outputs["dr_logits"][grade_idx]
        dme_logits = outputs["dme_logits"][grade_idx]
        dr_target = batch["dr_grade"][grade_idx]
        dme_target = batch["dme_grade"][grade_idx]

        dr_pred = coral_predict(dr_logits)
        dme_pred = coral_predict(dme_logits)

        stats["dr_correct"] += int((dr_pred == dr_target).sum().item())
        stats["dr_count"] += int(dr_target.numel())
        stats["dr_preds"].extend(dr_pred.detach().cpu().tolist())
        stats["dr_targets"].extend(dr_target.detach().cpu().tolist())

        stats["dme_correct"] += int((dme_pred == dme_target).sum().item())
        stats["dme_count"] += int(dme_target.numel())
        stats["dme_preds"].extend(dme_pred.detach().cpu().tolist())
        stats["dme_targets"].extend(dme_target.detach().cpu().tolist())

    seg_idx = batch["has_seg"]
    if seg_idx.any():
        seg_logits = outputs["seg_logits"][seg_idx]
        seg_target = batch["seg_mask"][seg_idx].float()

        seg_pred = (torch.sigmoid(seg_logits) > 0.5).float()

        inter = (seg_pred * seg_target).sum(dim=(0, 2, 3)).double()
        union = (seg_pred.sum(dim=(0, 2, 3)) + seg_target.sum(dim=(0, 2, 3))).double()

        stats["seg_inter"] += inter
        stats["seg_union"] += union
        stats["seg_count"] += 1


def finalize_epoch_stats(stats):
    eps = 1e-8
    out = {}

    n_batches = max(stats["num_batches"], 1)
    out["loss_total"] = stats["loss_total_sum"] / n_batches
    out["loss_seg"] = stats["loss_seg_sum"] / n_batches
    out["loss_dr"] = stats["loss_dr_sum"] / n_batches
    out["loss_dme"] = stats["loss_dme_sum"] / n_batches

    out["dr_acc"] = (
        float(stats["dr_correct"]) / max(stats["dr_count"], 1)
        if stats["dr_count"] > 0 else float("nan")
    )
    out["dme_acc"] = (
        float(stats["dme_correct"]) / max(stats["dme_count"], 1)
        if stats["dme_count"] > 0 else float("nan")
    )
    # QWK
    try:
        out["dr_qwk"] = cohen_kappa_score(
            stats["dr_targets"],
            stats["dr_preds"],
            labels=[0, 1, 2, 3, 4],
            weights="quadratic",
        ) if len(stats["dr_targets"]) > 0 else float("nan")
    except Exception:
        out["dr_qwk"] = float("nan")

    try:
        out["dme_qwk"] = cohen_kappa_score(
            stats["dme_targets"],
            stats["dme_preds"],
            labels=[0, 1, 2],
            weights="quadratic",
        ) if len(stats["dme_targets"]) > 0 else float("nan")
    except Exception:
        out["dme_qwk"] = float("nan")

    dice = (2.0 * stats["seg_inter"] + eps) / (stats["seg_union"] + eps)
    out["seg_dice_ma"] = float(dice[0].item())
    out["seg_dice_he"] = float(dice[1].item())
    out["seg_dice_ex"] = float(dice[2].item())
    out["seg_dice_se"] = float(dice[3].item())
    out["seg_dice_macro"] = float(dice.mean().item())

    return out


# Train / Validate
def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    train: bool,
):
    if train:
        model.train()
    else:
        model.eval()

    stats = init_epoch_stats(device)

    autocast_enabled = (device.type == "cuda")

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                outputs = model(batch["image"])
                loss_dict = criterion(outputs, batch)
                loss = loss_dict["loss_total"]

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        update_epoch_stats(stats, outputs, batch, loss_dict)

    return finalize_epoch_stats(stats)


# Checkpointing
def save_checkpoint(
    save_path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    best_val_loss: float,
    args: argparse.Namespace,
):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_val_loss": best_val_loss,
            "args": vars(args),
        },
        save_path,
    )


def load_checkpoint(
    ckpt_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    scaler=None,
    map_location="cpu",
):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    model.load_state_dict(ckpt["model"])

    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    start_epoch = ckpt.get("epoch", 0) + 1
    best_val_loss = ckpt.get("best_val_loss", math.inf)
    return start_epoch, best_val_loss


# Main
def parse_args():
    parser = argparse.ArgumentParser("IDRiD multitask train")

    # paths
    parser.add_argument("--data-root", type=str, default="data/archive")
    parser.add_argument("--save-dir", type=str, default="runs/idrid_mt")

    # data
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--task-mode", type=str, default="multitask",
                        choices=["classification", "segmentation", "multitask"])
    parser.add_argument("--require-both", action="store_true")
    parser.add_argument("--use-clahe", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seg-oversample-factor", type=float, default=4.0)

    # model
    parser.add_argument("--backbone", type=str, default="r50",
                        choices=["r18", "r34", "r50", "r101", "r152"])
    parser.add_argument("--cls-dropout", type=float, default=0.3)

    # optimization
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr-mult", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)

    # loss weights
    parser.add_argument("--lambda-seg", type=float, default=1.0)
    parser.add_argument("--lambda-dr", type=float, default=1.0)
    parser.add_argument("--lambda-dme", type=float, default=1.0)

    # resume
    parser.add_argument("--resume", type=str, default="")

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # save config
    with open(save_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # transforms
    train_tf = build_idrid_transform(
        train=True,
        image_size=args.image_size,
        use_clahe=args.use_clahe,
    )
    val_tf = build_idrid_transform(
        train=False,
        image_size=args.image_size,
        use_clahe=False,
    )

    # datasets
    train_ds = IDRiDDataset(
        root=args.data_root,
        split="train",
        task_mode=args.task_mode,
        require_both=args.require_both,
        transform=train_tf,
    )
    val_ds = IDRiDDataset(
        root=args.data_root,
        split="test",
        task_mode=args.task_mode,
        require_both=args.require_both,
        transform=val_tf,
    )

    print(f"Train size: {len(train_ds)}")
    print(f"Val size:   {len(val_ds)}")

    if hasattr(train_ds, "df"):
        n_grade = int(train_ds.df["has_grade"].sum())
        n_seg = int(train_ds.df["has_seg"].sum())
        n_both = int((train_ds.df["has_grade"] & train_ds.df["has_seg"]).sum())
        print(f"Train labeled: grade={n_grade}, seg={n_seg}, both={n_both}")

    # sampler
    train_sampler = build_train_sampler(
        train_ds=train_ds,
        task_mode=args.task_mode,
        seg_oversample_factor=args.seg_oversample_factor,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    # class / threshold weights
    dr_pos_weight = None
    dme_pos_weight = None
    seg_class_weights = None

    if hasattr(train_ds, "df"):
        grade_df = train_ds.df[train_ds.df["has_grade"]].copy()
        if len(grade_df) > 0:
            dr_labels = grade_df["dr_grade"].to_numpy(dtype=np.int64)
            dme_labels = grade_df["dme_grade"].to_numpy(dtype=np.int64)
            dr_pos_weight = compute_coral_pos_weight(dr_labels, num_classes=5)
            dme_pos_weight = compute_coral_pos_weight(dme_labels, num_classes=3)

        seg_class_weights = compute_seg_class_weights(train_ds)

    print("dr_pos_weight:", None if dr_pos_weight is None else dr_pos_weight.tolist())
    print("dme_pos_weight:", None if dme_pos_weight is None else dme_pos_weight.tolist())
    print("seg_class_weights:", None if seg_class_weights is None else seg_class_weights.tolist())

    # model
    model = ResUNetPPMultiTask(
        backbone_variant=args.backbone,
        seg_classes=4,
        dr_classes=5,
        dme_classes=3,
        cls_dropout=args.cls_dropout,
    )

    # Hard guard: model MUST already be changed for CORAL
    if getattr(model.dr_head, "out_features", None) != 4:
        raise ValueError(
            "dr_head.out_features must be 4 for CORAL on DR (5 classes -> 4 logits). "
            "Update models/res_unet_pp.py."
        )
    if getattr(model.dme_head, "out_features", None) != 2:
        raise ValueError(
            "dme_head.out_features must be 2 for CORAL on DME (3 classes -> 2 logits). "
            "Update models/res_unet_pp.py."
        )

    model = model.to(device)

    # loss
    criterion = IDRiDMultiTaskLoss(
        dr_num_classes=5,
        dme_num_classes=3,
        seg_class_weights=seg_class_weights.to(device) if seg_class_weights is not None else None,
        dr_pos_weight=dr_pos_weight.to(device) if dr_pos_weight is not None else None,
        dme_pos_weight=dme_pos_weight.to(device) if dme_pos_weight is not None else None,
        lambda_seg=args.lambda_seg,
        lambda_dr=args.lambda_dr,
        lambda_dme=args.lambda_dme,
    )

    optimizer = build_optimizer(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        backbone_lr_mult=args.backbone_lr_mult,
    )
    scheduler = build_scheduler(optimizer)
    scaler = torch.amp.GradScaler(device=device, enabled=(device.type == "cuda"))

    start_epoch = 1
    best_val_loss = math.inf

    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        print(f"Resumed from {args.resume} at epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")

    history_path = save_dir / "history.jsonl"

    # training
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_stats = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            train=True,
        )

        val_stats = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            scaler=None,
            device=device,
            train=False,
        )

        scheduler.step(val_stats["loss_total"])

        lr_group0 = optimizer.param_groups[0]["lr"]
        lr_group1 = optimizer.param_groups[1]["lr"]
        elapsed = time.time() - t0

        if device.type == "cuda":
            torch.cuda.empty_cache()

        log_row = {
            "epoch": epoch,
            "time_sec": round(elapsed, 2),
            "lr_backbone": lr_group0,
            "lr_heads": lr_group1,
            "train": train_stats,
            "val": val_stats,
        }

        with open(history_path, "a") as f:
            f.write(json.dumps(log_row) + "\n")

        print(
            f"[{epoch:03d}/{args.epochs:03d}] "
            f"time={elapsed:.1f}s | "
            f"train_loss={train_stats['loss_total']:.4f} | "
            f"val_loss={val_stats['loss_total']:.4f} | "
            f"val_dr_acc={val_stats['dr_acc']:.4f} | "
            f"val_dr_qwk={val_stats['dr_qwk']:.4f} | "
            f"val_dme_acc={val_stats['dme_acc']:.4f} | "
            f"val_dme_qwk={val_stats['dme_qwk']:.4f} | "
            f"val_seg_dice={val_stats['seg_dice_macro']:.4f}"
        )

        # save latest
        save_checkpoint(
            save_path=save_dir / "last.pt",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_val_loss=best_val_loss,
            args=args,
        )

        # save best by val total loss
        if val_stats["loss_total"] < best_val_loss:
            best_val_loss = val_stats["loss_total"]
            save_checkpoint(
                save_path=save_dir / "best.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_val_loss=best_val_loss,
                args=args,
            )
            print(f"  -> saved new best to {save_dir / 'best.pt'}")

    print("Training complete.")
    print(f"Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
