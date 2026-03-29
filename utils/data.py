import re
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class IDRiDDataset(Dataset):
    """
    IDRiD dataset for classification + lesion segmentation.

    Supports:
        - task_mode="classification"  -> all grading-labeled samples
        - task_mode="segmentation"    -> all segmentation-labeled samples
        - task_mode="multitask"       -> union of both (partial labels)
        - task_mode="multitask", require_both=True -> overlap only

    Returns a dict:
        {
            "image":      FloatTensor [3, H, W] in [0,1],
            "dr_grade":   LongTensor scalar, -1 if unavailable,
            "dme_grade":  LongTensor scalar, -1 if unavailable,
            "seg_mask":   FloatTensor [4, H, W], channels = [MA, HE, EX, SE],
            "has_grade":  BoolTensor scalar,
            "has_seg":    BoolTensor scalar,
            "id_int":     LongTensor scalar,
            "id_str":     str,
            "image_path": str,
        }

    Notes:
        - For segmentation images, missing lesion mask files are treated as all-zero masks.
        - For overlap samples, the segmentation image path is preferred so masks align naturally.
        - For batching, use a resize transform so all samples have the same shape.
        - For segmentation/multitask, prefer Albumentations-style transforms that accept
        image=..., mask=... and use nearest interpolation for masks.
    """

    LESION_ORDER = ("MA", "HE", "EX", "SE")

    def __init__(
        self,
        root: str,
        split: str = "train",                 # "train" | "test" | "all"
        task_mode: str = "multitask",         # "classification" | "segmentation" | "multitask"
        require_both: bool = False,
        transform=None,
    ):
        super().__init__()

        if split not in {"train", "test", "all"}:
            raise ValueError(f"split must be one of ['train', 'test', 'all'], got {split}")
        if task_mode not in {"classification", "segmentation", "multitask"}:
            raise ValueError(
                f"task_mode must be one of ['classification', 'segmentation', 'multitask'], got {task_mode}"
            )

        self.root = Path(root)
        self.split = split
        self.task_mode = task_mode
        self.require_both = require_both
        self.transform = transform

        self.seg_root = self.root / "A20Segmentation" / "A. Segmentation"
        self.grade_root = self.root / "B20Disease%20Grading" / "B. Disease Grading"

        self.df = self._build_master_index()
        self.df = self._filter_index(self.df).reset_index(drop=True)

        if len(self.df) == 0:
            raise RuntimeError("No samples found after filtering. Check root/split/task_mode settings.")

    # ------------------------------------------------------------------
    # ID helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_id_int(name: str) -> int:
        m = re.search(r"IDRiD_(\d+)", str(name))
        if m is None:
            raise ValueError(f"Could not parse ID from: {name}")
        return int(m.group(1))

    @staticmethod
    def _canonical_id(id_int: int) -> str:
        return f"IDRiD_{id_int:03d}"

    # ------------------------------------------------------------------
    # CSV loading
    # ------------------------------------------------------------------
    def _load_grading_csv(self, csv_path: Path, raw_split: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df.columns = [str(c).strip() for c in df.columns]

        # Keep only the first 3 meaningful columns; trailing commas create junk columns
        df = df.iloc[:, :3].copy()
        df.columns = ["image_name", "dr_grade", "dme_grade"]

        df["id_int"] = df["image_name"].apply(self._extract_id_int)
        df["id_str"] = df["id_int"].apply(self._canonical_id)

        df["dr_grade"] = df["dr_grade"].astype(int)
        df["dme_grade"] = df["dme_grade"].astype(int)

        split_dir = "a. Training Set" if raw_split == "grade_train" else "b. Testing Set"
        df["grade_image_path"] = df["id_str"].apply(
            lambda s: str(self.grade_root / "1. Original Images" / split_dir / f"{s}.jpg")
        )
        df["grade_split_raw"] = raw_split
        df["has_grade"] = True

        return df[[
            "id_int", "id_str",
            "dr_grade", "dme_grade",
            "grade_image_path", "grade_split_raw", "has_grade"
        ]]

    # ------------------------------------------------------------------
    # Segmentation indexing
    # ------------------------------------------------------------------
    def _build_seg_index(self) -> pd.DataFrame:
        lesion_dirs = {
            "MA": "1. Microaneurysms",
            "HE": "2. Haemorrhages",
            "EX": "3. Hard Exudates",
            "SE": "4. Soft Exudates",
        }

        rows = []

        def process_split(img_dir: Path, mask_root: Path, raw_split: str):
            for img_path in sorted(img_dir.glob("*.jpg")):
                id_int = self._extract_id_int(img_path.stem)
                id_str = self._canonical_id(id_int)

                row = {
                    "id_int": id_int,
                    "id_str": id_str,
                    "seg_image_path": str(img_path),
                    "seg_split_raw": raw_split,
                    "has_seg": True,
                }

                # Important:
                # segmentation filenames may use shorter zero-padding, so use img_path.stem
                # to construct mask filenames, not canonical 3-digit names.
                stem = img_path.stem

                for lesion, folder in lesion_dirs.items():
                    mask_path = mask_root / folder / f"{stem}_{lesion}.tif"
                    row[f"mask_{lesion.lower()}_path"] = str(mask_path) if mask_path.exists() else None

                rows.append(row)

        train_img_dir = self.seg_root / "1. Original Images" / "a. Training Set"
        test_img_dir  = self.seg_root / "1. Original Images" / "b. Testing Set"

        train_mask_root = self.seg_root / "2. All Segmentation Groundtruths" / "a. Training Set"
        test_mask_root  = self.seg_root / "2. All Segmentation Groundtruths" / "b. Testing Set"

        process_split(train_img_dir, train_mask_root, "seg_train")
        process_split(test_img_dir,  test_mask_root,  "seg_test")

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Build + filter master table
    # ------------------------------------------------------------------
    def _build_master_index(self) -> pd.DataFrame:
        grade_train_csv = self.grade_root / "2. Groundtruths" / "a. IDRiD_Disease Grading_Training Labels.csv"
        grade_test_csv  = self.grade_root / "2. Groundtruths" / "b. IDRiD_Disease Grading_Testing Labels.csv"

        # load grade splits separately
        df_grade_train = self._load_grading_csv(grade_train_csv, "grade_train")
        df_grade_test  = self._load_grading_csv(grade_test_csv,  "grade_test")

        # load seg splits separately
        df_seg_all = self._build_seg_index()
        df_seg_train = df_seg_all[df_seg_all["seg_split_raw"] == "seg_train"].copy()
        df_seg_test  = df_seg_all[df_seg_all["seg_split_raw"] == "seg_test"].copy()

        # merge train with train, test with test
        df_train = pd.merge(df_grade_train, df_seg_train, on=["id_int", "id_str"], how="outer")
        df_test  = pd.merge(df_grade_test,  df_seg_test,  on=["id_int", "id_str"], how="outer")

        # fill bools split-wise
        for df_ in (df_train, df_test):
            if "has_grade" not in df_:
                df_["has_grade"] = False
            if "has_seg" not in df_:
                df_["has_seg"] = False

            df_["has_grade"] = df_["has_grade"].astype("boolean").fillna(False).astype(bool)
            df_["has_seg"]   = df_["has_seg"].astype("boolean").fillna(False).astype(bool)

            df_["image_path"] = np.where(
                df_["has_seg"],
                df_["seg_image_path"],
                df_["grade_image_path"]
            )

        if self.split == "train":
            df = df_train
        elif self.split == "test":
            df = df_test
        else:
            df = pd.concat([df_train, df_test], ignore_index=True)

        df = df[df["image_path"].notna()].copy()
        return df.sort_values(["id_int"]).reset_index(drop=True)

    def _filter_index(self, df: pd.DataFrame) -> pd.DataFrame:
        # split filter
        if self.split == "train":
            keep = (
                (df["grade_split_raw"] == "grade_train") |
                (df["seg_split_raw"] == "seg_train")
            )
            df = df[keep].copy()
        elif self.split == "test":
            keep = (
                (df["grade_split_raw"] == "grade_test") |
                (df["seg_split_raw"] == "seg_test")
            )
            df = df[keep].copy()
        # else: all

        # task filter
        if self.task_mode == "classification":
            df = df[df["has_grade"]].copy()

        elif self.task_mode == "segmentation":
            df = df[df["has_seg"]].copy()

        elif self.task_mode == "multitask":
            if self.require_both:
                df = df[df["has_grade"] & df["has_seg"]].copy()
            else:
                df = df[df["has_grade"] | df["has_seg"]].copy()

        return df.sort_values("id_int").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Image/mask loading
    # ------------------------------------------------------------------
    @staticmethod
    def _read_rgb(image_path: str) -> np.ndarray:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    @staticmethod
    def _read_binary_mask(mask_path: Optional[str], shape_hw: Tuple[int, int]) -> np.ndarray:
        """
        Returns binary mask of shape [H, W], dtype uint8, values {0,1}.
        If mask_path is None/missing, returns all-zero mask.

        This is correct for IDRiD segmentation:
        if a lesion file is absent for a segmentation-labeled image,
        treat that lesion type as absent (all zeros).
        """
        h, w = shape_hw
        if mask_path is None:
            return np.zeros((h, w), dtype=np.uint8)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")

        # robust binarization: handles 0/1, 0/255, or slight artifacts
        mask = (mask > 0).astype(np.uint8)
        return mask

    def _load_seg_mask(self, row: pd.Series, image_hw: Tuple[int, int]) -> np.ndarray:
        """
        Returns multi-channel lesion mask [H, W, 4], channel order = [MA, HE, EX, SE].
        For non-seg samples, returns all-zero tensor of correct spatial size.
        """
        h, w = image_hw
        channels = []

        if bool(row["has_seg"]):
            channels.append(self._read_binary_mask(row.get("mask_ma_path", None), (h, w)))
            channels.append(self._read_binary_mask(row.get("mask_he_path", None), (h, w)))
            channels.append(self._read_binary_mask(row.get("mask_ex_path", None), (h, w)))
            channels.append(self._read_binary_mask(row.get("mask_se_path", None), (h, w)))
        else:
            for _ in range(4):
                channels.append(np.zeros((h, w), dtype=np.uint8))

        seg = np.stack(channels, axis=-1)  # [H, W, 4]
        return seg

    # ------------------------------------------------------------------
    # Tensor conversion
    # ------------------------------------------------------------------
    @staticmethod
    def _image_to_tensor(image: Any) -> torch.Tensor:
        """
        Converts image to float tensor [C, H, W].
        Behavior:
        - uint8 / integer images in [0,255] -> scaled to [0,1]
        - float images already normalized (e.g. Albumentations Normalize) -> left unchanged
        - float images in [0,255] -> scaled to [0,1]
        """
        if isinstance(image, torch.Tensor):
            x = image
            if x.ndim == 3 and x.shape[0] in (1, 3):   # already CHW
                x = x.float()
            elif x.ndim == 3 and x.shape[-1] in (1, 3):  # HWC -> CHW
                x = x.permute(2, 0, 1).float()
            else:
                raise ValueError(f"Unexpected image tensor shape: {tuple(x.shape)}")

            # only scale if it still looks like raw image data
            if not torch.is_floating_point(image):
                x = x / 255.0
            else:
                # float tensor: scale only if clearly raw [0,255]-style data
                if x.min() >= 0.0 and x.max() > 1.0:
                    x = x / 255.0
            return x

        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected image shape [H, W, 3], got {image.shape}")

        x = torch.from_numpy(image).permute(2, 0, 1).float()
        # Integer images are raw pixels -> scale
        if np.issubdtype(image.dtype, np.integer):
            x = x / 255.0
        else:
            # Float image:
            # If already normalized, it likely contains negatives or centered values -> leave it
            # If still raw float in [0,255], scale it
            if image.min() >= 0.0 and image.max() > 1.0:
                x = x / 255.0
        return x

    @staticmethod
    def _mask_to_tensor(mask: Any) -> torch.Tensor:
        if isinstance(mask, torch.Tensor):
            x = mask
            if x.ndim == 3 and x.shape[0] == 4:        # CHW already
                return (x > 0.5).float()
            elif x.ndim == 3 and x.shape[-1] == 4:     # HWC tensor
                return (x.permute(2, 0, 1) > 0.5).float()
            else:
                raise ValueError(f"Unexpected mask tensor shape: {tuple(x.shape)}")

        mask = np.asarray(mask)
        if mask.ndim != 3 or mask.shape[2] != 4:
            raise ValueError(f"Expected mask shape [H, W, 4], got {mask.shape}")

        x = torch.from_numpy(mask).permute(2, 0, 1).float()
        return (x > 0.5).float()

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        image_path = row["image_path"]
        image = self._read_rgb(image_path)
        h, w = image.shape[:2]

        seg_mask = self._load_seg_mask(row, (h, w))  # [H, W, 4]

        # Albumentations-style transform preferred
        if self.transform is not None:
            transformed = self.transform(image=image, mask=seg_mask)
            image = transformed["image"]
            seg_mask = transformed["mask"]

        image = self._image_to_tensor(image)
        seg_mask = self._mask_to_tensor(seg_mask)

        has_grade = bool(row["has_grade"])
        has_seg = bool(row["has_seg"])

        dr_grade = int(row["dr_grade"]) if has_grade else -1
        dme_grade = int(row["dme_grade"]) if has_grade else -1

        sample = {
            "image": image,  # [3, H, W]
            "dr_grade": torch.tensor(dr_grade, dtype=torch.long),
            "dme_grade": torch.tensor(dme_grade, dtype=torch.long),
            "seg_mask": seg_mask,  # [4, H, W] -> [MA, HE, EX, SE]
            "has_grade": torch.tensor(has_grade, dtype=torch.bool),
            "has_seg": torch.tensor(has_seg, dtype=torch.bool),
            "id_int": torch.tensor(int(row["id_int"]), dtype=torch.long),
            "id_str": row["id_str"],
            "image_path": image_path,
        }
        return sample
