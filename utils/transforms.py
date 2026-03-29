import cv2
import numpy as np
import albumentations as A


class CropBlackBorders:
    """
    Deterministic retinal ROI crop.

    Strategy:
    - grayscale
    - light blur for threshold stability
    - Otsu binarization
    - keep largest connected component only
    - crop with margin
    - optional sanity fallback if crop is suspiciously small
    """
    def __init__(self, margin: int = 15, min_area_ratio: float = 0.15):
        self.margin = margin
        self.min_area_ratio = min_area_ratio

    def __call__(self, image: np.ndarray, mask: np.ndarray | None = None):
        if image.ndim != 3 or image.shape[2] != 3:
            return image, mask

        H, W = image.shape[:2]

        # 1) grayscale + slight blur for more stable thresholding
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # 2) Otsu binarization
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # 3) Largest connected component only
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)

        if num_labels <= 1:
            return image, mask  # only background found

        # skip label 0 = background
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = 1 + np.argmax(areas)

        # sanity check: if component is implausibly tiny, keep original
        largest_area = stats[largest_idx, cv2.CC_STAT_AREA]
        if largest_area < self.min_area_ratio * (H * W):
            return image, mask

        x = stats[largest_idx, cv2.CC_STAT_LEFT]
        y = stats[largest_idx, cv2.CC_STAT_TOP]
        w = stats[largest_idx, cv2.CC_STAT_WIDTH]
        h = stats[largest_idx, cv2.CC_STAT_HEIGHT]

        # 4) margin with boundary checks
        x1 = max(0, x - self.margin)
        y1 = max(0, y - self.margin)
        x2 = min(W, x + w + self.margin)
        y2 = min(H, y + h + self.margin)

        image_cropped = image[y1:y2, x1:x2]
        mask_cropped = mask[y1:y2, x1:x2] if mask is not None else None

        return image_cropped, mask_cropped


class IDRiDTransform:
    """
    Full-frame transform for joint classification + segmentation.

    Design choices:
    - no random crops: shared transform must preserve global grading context
    - no channel shuffle: unrealistic for fundus imaging
    - no vertical flip / 360 rotation: too aggressive for retinal anatomy
    - mild CLAHE only as augmentation, not mandatory preprocessing
    - no ToTensorV2 here because dataset class already tensorizes
    """
    def __init__(
        self,
        train: bool = True,
        image_size: int = 1024,
        use_clahe: bool = True,
    ):
        self.train = train
        self.image_size = image_size
        self.border_crop = CropBlackBorders(margin=6)

        imagenet_mean = (0.485, 0.456, 0.406)
        imagenet_std = (0.229, 0.224, 0.225)

        if train:
            self.aug = A.Compose([
                # resize full frame while preserving aspect ratio
                A.LongestMaxSize(
                    max_size=image_size,
                    interpolation=cv2.INTER_AREA,
                ),
                A.PadIfNeeded(
                    min_height=image_size,
                    min_width=image_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=0,
                    fill_mask=0,
                ),

                # modest, anatomy-preserving geometry
                A.HorizontalFlip(p=0.5),

                A.ShiftScaleRotate(
                    shift_limit=0.02,
                    scale_limit=0.08,
                    rotate_limit=25,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=0,
                    fill_mask=0,
                    p=0.6,
                ),

                # mild photometric robustness
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.12,
                        contrast_limit=0.12,
                        p=1.0,
                    ),
                    A.RandomGamma(
                        gamma_limit=(90, 110),
                        p=1.0,
                    ),
                ], p=0.45),

                A.OneOf([
                    A.GaussNoise(
                        std_range=(0.01, 0.03),
                        mean_range=(0.0, 0.0),
                        p=1.0,
                    ),
                    A.GaussianBlur(
                        blur_limit=(3, 5),
                        p=1.0,
                    ),
                ], p=0.15),

                # optional, low-probability contrast enhancement
                A.CLAHE(
                    clip_limit=(1.0, 2.0),
                    tile_grid_size=(8, 8),
                    p=0.12 if use_clahe else 0.0,
                ),

                A.Normalize(
                    mean=imagenet_mean,
                    std=imagenet_std,
                    max_pixel_value=255.0,
                ),
            ])
        else:
            self.aug = A.Compose([
                A.LongestMaxSize(
                    max_size=image_size,
                    interpolation=cv2.INTER_AREA,
                ),
                A.PadIfNeeded(
                    min_height=image_size,
                    min_width=image_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=0,
                    fill_mask=0,
                ),
                # optional, low-probability contrast enhancement
                A.CLAHE(
                    clip_limit=(1.0, 2.0),
                    tile_grid_size=(8, 8),
                    p=1.0 if use_clahe else 0.0,
                ),

                A.Normalize(
                    mean=imagenet_mean,
                    std=imagenet_std,
                    max_pixel_value=255.0,
                ),
            ])

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        # deterministic crop first
        image, mask = self.border_crop(image, mask)

        # stochastic / deterministic Albumentations pipeline
        out = self.aug(image=image, mask=mask)

        # keep mask strictly binary after interpolation/ops
        out["mask"] = (out["mask"] > 0.5).astype(np.float32)

        return out


def build_idrid_transform(train: bool, image_size: int = 1024, use_clahe: bool = True):
    """
    Factory function so usage stays simple.
    """
    return IDRiDTransform(train=train, image_size=image_size, use_clahe=use_clahe)
