import torch
import torch.nn as nn
from torchvision.ops import SqueezeExcitation


# small helper
def make_gn(c: int, max_groups: int = 8):
    """make gn helper fn"""
    g = min(max_groups, c)
    while c % g != 0:
        g -= 1
    return nn.GroupNorm(g, c)

class ConvGNAct(nn.Module):
    """using BN is not optimal for small batches since we use large imgs"""
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False),
            make_gn(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """forward method"""
        return self.block(x)

class ResNetBlock(nn.Module):
    """Residual conv block for decoder. GN is safer than BN for micro-batches."""
    def __init__(self, in_c: int, out_c: int, stride: int):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride, bias=False),
            make_gn(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            make_gn(out_c),
        )

        if in_c != out_c or stride != 1:
            self.c2 = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                make_gn(out_c),
            )
        else:
            self.c2 = nn.Identity()

        self.attn = SqueezeExcitation(
            input_channels=out_c,
            squeeze_channels=max(1, out_c // 8)
        )
        self.out_act = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """forward method"""
        x = self.c1(inputs)
        s = self.c2(inputs)
        x = self.attn(x)
        return self.out_act(x + s)

class AttentionBlock(nn.Module):
    """
    Deep feature gates the skip connection.
    Preserves current call style: forward(g, x) where
        g = skip feature
        x = deep feature
    """
    def __init__(self, in_c):
        super().__init__()
        skip_c, deep_c = in_c
        inter_c = max(1, min(skip_c, deep_c) // 2)

        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_c, inter_c, kernel_size=1, bias=False),
            make_gn(inter_c),
        )
        self.deep_proj = nn.Sequential(
            nn.Conv2d(deep_c, inter_c, kernel_size=1, bias=False),
            make_gn(inter_c),
        )
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_c, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        g: skip feature  [B, skip_c, H, W]
        x: deep feature  [B, deep_c, h, w]
        returns: gated skip [B, skip_c, H, W]
        """
        skip = g
        deep = x
        deep_up = nn.functional.interpolate(
            deep,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        skip_proj = self.skip_proj(skip)
        deep_proj = self.deep_proj(deep_up)
        attn = self.psi(skip_proj + deep_proj)   # [B, 1, H, W]
        return skip * attn

class DecoderBlock(nn.Module):
    """Decoder block with deep->skip attention gating."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.a1 = AttentionBlock(in_c)
        self.r1 = ResNetBlock(in_c[0] + in_c[1], out_c, stride=1)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        g: skip feature
        x: deep feature
        """
        gated_skip = self.a1(g, x)
        deep_up = nn.functional.interpolate(
            x,
            size=gated_skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        d = torch.cat([deep_up, gated_skip], dim=1)
        return self.r1(d)
