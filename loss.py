import torch
import torch.nn as nn
import torch.nn.functional as F


def coral_levels_from_labels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    labels: [B] int64 in {0, ..., K-1}
    returns: [B, K-1], where level[j] = 1 iff label > j
    """
    if labels.ndim != 1:
        raise ValueError(f"labels must be [B], got {labels.shape}")
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")

    thresholds = torch.arange(num_classes - 1, device=labels.device).unsqueeze(0)  # [1, K-1]
    levels = (labels.unsqueeze(1) > thresholds).float()
    return levels


class CoralLoss(nn.Module):
    """
    CORAL loss: BCE over cumulative ordinal levels.
    logits shape must be [B, K-1].
    """
    def __init__(self, num_classes: int, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.num_classes = num_classes
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """forward method"""
        if logits.ndim != 2:
            raise ValueError(f"logits must be [B, K-1], got {logits.shape}")
        if logits.size(1) != self.num_classes - 1:
            raise ValueError(
                f"Expected logits second dim {self.num_classes - 1}, got {logits.size(1)}"
            )

        levels = coral_levels_from_labels(labels, self.num_classes)
        return F.binary_cross_entropy_with_logits(
            logits, levels, pos_weight=self.pos_weight, reduction="mean"
        )


@torch.no_grad()
def coral_predict(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Hard ordinal prediction from CORAL logits.
    logits: [B, K-1]
    returns: [B] predicted class in {0, ..., K-1}
    """
    probs = torch.sigmoid(logits)
    return (probs > threshold).sum(dim=1)


class MultiLabelFocalTverskyLoss(nn.Module):
    """
    Multi-label focal Tversky for lesion segmentation.
    Expects logits [B, C, H, W] and target [B, C, H, W] with target in {0,1}.
    """
    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        focal_power: float = 0.75,   # paper's gamma=4/3 => exponent 1/gamma = 0.75
        smooth: float = 1.0,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.focal_power = focal_power
        self.smooth = smooth

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """forward method"""
        if logits.shape != target.shape:
            raise ValueError(f"logits and target must match, got {logits.shape} vs {target.shape}")

        probs = torch.sigmoid(logits)
        target = target.float()

        # per-sample, per-class
        dims = (2, 3)
        tp = (probs * target).sum(dim=dims)
        fn = ((1.0 - probs) * target).sum(dim=dims)
        fp = (probs * (1.0 - target)).sum(dim=dims)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fn + self.beta * fp + self.smooth
        )
        loss = (1.0 - tversky).pow(self.focal_power)  # [B, C]

        if self.class_weights is not None:
            w = self.class_weights.view(1, -1)
            loss = loss * w
            loss = loss.sum(dim=1) / w.sum()
            return loss.mean()
        return loss.mean()

class SegmentationLoss(nn.Module):
    """
    Stable practical loss for 4-channel lesion head:
        0.5 * BCEWithLogits + 0.5 * FocalTversky
    """
    def __init__(
        self,
        ft_alpha: float = 0.7,
        ft_beta: float = 0.3,
        ft_power: float = 0.75,
        class_weights: torch.Tensor | None = None,
        bce_weight: float = 0.5,
        ft_weight: float = 0.5,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.ft_weight = ft_weight

        self.ft = MultiLabelFocalTverskyLoss(
            alpha=ft_alpha,
            beta=ft_beta,
            focal_power=ft_power,
            class_weights=class_weights,
        )

        if class_weights is not None:
            self.register_buffer("bce_pos_weight", class_weights.float())
        else:
            self.bce_pos_weight = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """forward method"""
        bce = F.binary_cross_entropy_with_logits(
            logits, target.float(),
            reduction="mean",
            pos_weight=self.bce_pos_weight.view(1, -1, 1, 1) if self.bce_pos_weight is not None else None
        )
        ft = self.ft(logits, target)
        return self.bce_weight * bce + self.ft_weight * ft


class IDRiDMultiTaskLoss(nn.Module):
    """Multi-task loss"""
    def __init__(
        self,
        dr_num_classes: int = 5,
        dme_num_classes: int = 3,
        seg_class_weights: torch.Tensor | None = None,
        dr_pos_weight: torch.Tensor | None = None,   # shape [4]
        dme_pos_weight: torch.Tensor | None = None,  # shape [2]
        lambda_seg: float = 1.0,
        lambda_dr: float = 1.0,
        lambda_dme: float = 1.0,
    ):
        super().__init__()
        self.lambda_seg = lambda_seg
        self.lambda_dr = lambda_dr
        self.lambda_dme = lambda_dme

        self.seg_loss = SegmentationLoss(class_weights=seg_class_weights)
        self.dr_loss = CoralLoss(num_classes=dr_num_classes, pos_weight=dr_pos_weight)
        self.dme_loss = CoralLoss(num_classes=dme_num_classes, pos_weight=dme_pos_weight)

    def forward(self, outputs: dict, batch: dict) -> dict:
        """forward method"""
        total = outputs["seg_logits"].new_tensor(0.0)
        loss_dict = {}

        # segmentation
        seg_idx = batch["has_seg"]
        if seg_idx.any():
            seg_logits = outputs["seg_logits"][seg_idx]
            seg_target = batch["seg_mask"][seg_idx]
            seg_loss = self.seg_loss(seg_logits, seg_target)
            total = total + self.lambda_seg * seg_loss
            loss_dict["loss_seg"] = seg_loss
        else:
            loss_dict["loss_seg"] = total.new_tensor(0.0)

        # DR ordinal
        grade_idx = batch["has_grade"]
        if grade_idx.any():
            dr_logits = outputs["dr_logits"][grade_idx]
            dr_target = batch["dr_grade"][grade_idx]
            dr_loss = self.dr_loss(dr_logits, dr_target)
            total = total + self.lambda_dr * dr_loss
            loss_dict["loss_dr"] = dr_loss
        else:
            loss_dict["loss_dr"] = total.new_tensor(0.0)

        # DME ordinal
        if grade_idx.any():
            dme_logits = outputs["dme_logits"][grade_idx]
            dme_target = batch["dme_grade"][grade_idx]
            dme_loss = self.dme_loss(dme_logits, dme_target)
            total = total + self.lambda_dme * dme_loss
            loss_dict["loss_dme"] = dme_loss
        else:
            loss_dict["loss_dme"] = total.new_tensor(0.0)

        loss_dict["loss_total"] = total
        return loss_dict
