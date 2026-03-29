from typing import Literal
import torch
import torch.nn as nn
from torchvision.models.segmentation.deeplabv3 import ASPP as TorchVisionASPP
from models.blocks import DecoderBlock, make_gn
from models.resnet_fe import BackboneFeatureExtractor


class CoralHead(nn.Module):
    """CORAL Head"""
    def __init__(self, dim: int, num_classes: int) -> None:
        super().__init__()
        self.out_features = num_classes
        self.weight = nn.Parameter(
            torch.empty((dim, 1), dtype=torch.float32),
            requires_grad=True)
        self.bias = nn.Parameter(
            torch.zeros((self.out_features), dtype=torch.float32),
            requires_grad=True)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward method
        logit_j = x @ w + b_j
        """
        x = x @ self.weight # [B, D] @ [D, 1] -> [B, 1]
        # [B, 1] + [1, K-1] -> [B, K-1]
        x = x + self.bias.unsqueeze(0).expand(x.size(0), -1)
        return x

class ResUNetPPMultiTask(nn.Module):
    """segmentation(multi-label) + classification(ordinal)"""
    def __init__(
        self,
        backbone_variant: Literal['r18', 'r34', 'r50', 'r101', 'r152'] = 'r50',
        seg_classes: int = 4,   # MA, HE, EX, SE
        dr_classes: int = 5,    # 0..4
        dme_classes: int = 3,   # 0..2
        cls_dropout: float = 0.3,
    ):
        super().__init__()

        self.backbone = BackboneFeatureExtractor(backbone_variant)
        ch = self.backbone.channels

        # deep context for segmentation
        self.b1 = TorchVisionASPP(
            in_channels=ch[4],
            atrous_rates=[6, 12, 18],
            out_channels=512,
        )

        self.d1 = DecoderBlock([ch[3], 512], 256)
        self.d2 = DecoderBlock([ch[2], 256], 128)
        self.d3 = DecoderBlock([ch[1], 128], 64)
        self.d4 = DecoderBlock([ch[0], 64], 32)

        # light segmentation head (replace second ASPP)
        self.seg_head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            make_gn(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, seg_classes, kernel_size=1)
        )

        # classification heads from backbone
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cls_drop = nn.Dropout(cls_dropout)
        # since these two are ordinal and not independent classes
        self.dr_head = CoralHead(ch[4], dr_classes - 1)   # 4 logits for DR 0..4
        self.dme_head = CoralHead(ch[4], dme_classes - 1) # 2 logits for DME 0..2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1, c2, c3, c4, c5 = self.backbone(x)

        # classification branch
        cls_feat = self.pool(c5).flatten(1)
        cls_feat = self.cls_drop(cls_feat)
        dr_logits = self.dr_head(cls_feat)
        dme_logits = self.dme_head(cls_feat)

        # segmentation branch
        b1 = self.b1(c5)
        d1 = self.d1(c4, b1)
        d2 = self.d2(c3, d1)
        d3 = self.d3(c2, d2)
        d4 = self.d4(c1, d3)

        seg_logits = self.seg_head(d4)
        seg_logits = nn.functional.interpolate(
            seg_logits, size=x.shape[-2:],
            mode="bilinear", align_corners=False)

        return {
            "seg_logits": seg_logits,
            "dr_logits": dr_logits,
            "dme_logits": dme_logits,
        }
