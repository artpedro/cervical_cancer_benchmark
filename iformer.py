from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.layers import DropPath, trunc_normal_
except ImportError:
    from timm.models.layers import DropPath, trunc_normal_


@dataclass
class IFormerMConfig:
    in_chans: int = 3
    num_classes: int = 2

    # Exact iFormer-M macro-architecture
    depths: Sequence[int] = (2, 2, 22, 6)
    dims: Sequence[int] = (48, 96, 192, 384)
    downsample_kernels: Sequence[int] = (5, 3, 3, 3)

    # Keep these explicit so you can tune them later if needed
    drop_path_rate: float = 0.0
    layer_scale_init_value: float = 0.0


class Conv2dBN(nn.Sequential):
    """
    Conv + BN projection used throughout the network.

    This is one of the fundamental building blocks in the original code.
    """
    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bn_weight_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_chs,
                out_chs,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
        )
        self.add_module("bn", nn.BatchNorm2d(out_chs))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0.0)


class BNLinear(nn.Sequential):
    """
    BN + Linear classifier head used by the original iFormer implementation
    when use_bn=True.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.add_module("bn", nn.BatchNorm1d(in_features))
        self.add_module("fc", nn.Linear(in_features, out_features, bias=bias))
        trunc_normal_(self.fc.weight, std=0.02)
        if bias:
            nn.init.constant_(self.fc.bias, 0.0)


class Residual(nn.Module):
    """
    Residual wrapper with optional DropPath and optional LayerScale.
    """
    def __init__(
        self,
        block: nn.Module,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 0.0,
        dim: int | None = None,
    ) -> None:
        super().__init__()
        self.block = block
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        if layer_scale_init_value > 0.0:
            if dim is None:
                raise ValueError("dim must be provided when layer_scale_init_value > 0.")
            self.gamma = nn.Parameter(
                layer_scale_init_value * torch.ones((1, dim, 1, 1)),
                requires_grad=True,
            )
        else:
            self.gamma = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        y = self.drop_path(y)
        if self.gamma is not None:
            y = self.gamma * y
        return x + y


class EdgeResidual(nn.Module):
    """
    Fused inverted bottleneck used in the stem.
    This is the 'FusedIB' stem choice from the original code.
    """
    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        exp_kernel_size: int = 3,
        stride: int = 1,
        exp_ratio: float = 4.0,
        act_layer: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        mid_chs = int(in_chs * exp_ratio)
        self.conv_exp = Conv2dBN(
            in_chs,
            mid_chs,
            kernel_size=exp_kernel_size,
            stride=stride,
            padding=exp_kernel_size // 2,
        )
        self.act = act_layer()
        self.conv_pwl = Conv2dBN(mid_chs, out_chs, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_exp(x)
        x = self.act(x)
        x = self.conv_pwl(x)
        return x


class ConvBlock(nn.Module):
    """
    Pure convolutional token mixer block.

    In iFormer-M this is the local modeling block used heavily in the early
    stages and also at the beginning/end of stage 3.
    """
    def __init__(
        self,
        dim: int,
        ratio: int = 4,
        kernel: int = 7,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        mid_chs = dim * ratio
        self.block = Residual(
            nn.Sequential(
                Conv2dBN(
                    dim,
                    dim,
                    kernel_size=kernel,
                    stride=1,
                    padding=kernel // 2,
                    groups=dim,  # depthwise
                ),
                Conv2dBN(dim, mid_chs, kernel_size=1),
                act_layer(),
                Conv2dBN(mid_chs, dim, kernel_size=1),
            ),
            drop_path=drop_path,
            layer_scale_init_value=layer_scale_init_value,
            dim=dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RepCPE(nn.Module):
    """
    Conditional positional encoding via residual depthwise convolution.

    This is the 'RepCPE_k3' block in the original model definition.
    """
    def __init__(self, dim: int, kernel: int = 3) -> None:
        super().__init__()
        self.block = Residual(
            Conv2dBN(
                dim,
                dim,
                kernel_size=kernel,
                stride=1,
                padding=kernel // 2,
                groups=dim,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SHMA(nn.Module):
    """
    Spatial attention block used by iFormer-M.

    This is the important hybrid-attention module in the original architecture.
    For the M variant, num_heads=1 everywhere, so this implementation matches
    the exact M usage without extra unused branches.
    """
    def __init__(
        self,
        dim: int,
        ratio: int = 1,
        head_dim_reduce_ratio: int = 2,
        num_heads: int = 1,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()

        if num_heads != 1:
            raise ValueError("This cleaned iFormer-M implementation expects num_heads=1.")

        mid_dim = int(dim * ratio)
        dim_attn = dim // head_dim_reduce_ratio

        self.scale = dim_attn ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)

        self.q = Conv2dBN(dim, dim_attn, kernel_size=1)
        self.k = Conv2dBN(dim, dim_attn, kernel_size=1)
        self.v_gate = Conv2dBN(dim, 2 * mid_dim, kernel_size=1)
        self.proj = Conv2dBN(mid_dim, dim, kernel_size=1)

        self.gate_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape

        v, gate = self.gate_act(self.v_gate(x)).chunk(2, dim=1)

        q = self.q(x).flatten(2)  # (B, Cq, HW)
        k = self.k(x).flatten(2)  # (B, Cq, HW)
        v = v.flatten(2)          # (B, Cv, HW)

        q = q * self.scale
        attn = q.transpose(-2, -1) @ k          # (B, HW, HW)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (v @ attn.transpose(-2, -1)).view(b, -1, h, w)
        x = x * gate
        x = self.proj(x)
        return x


class SHMABlock(nn.Module):
    """
    Residual wrapper around SHMA.
    """
    def __init__(
        self,
        dim: int,
        ratio: int = 1,
        head_dim_reduce_ratio: int = 2,
        num_heads: int = 1,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 0.0,
    ) -> None:
        super().__init__()
        self.block = Residual(
            SHMA(
                dim=dim,
                ratio=ratio,
                head_dim_reduce_ratio=head_dim_reduce_ratio,
                num_heads=num_heads,
            ),
            drop_path=drop_path,
            layer_scale_init_value=layer_scale_init_value,
            dim=dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FFN2d(nn.Module):
    """
    Feed-forward network in 2D feature-map form.

    This is the MLP-equivalent block after attention in the hybrid stages.
    """
    def __init__(
        self,
        dim: int,
        ratio: int = 3,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        mid_chs = dim * ratio
        self.block = Residual(
            nn.Sequential(
                Conv2dBN(dim, mid_chs, kernel_size=1),
                act_layer(),
                Conv2dBN(mid_chs, dim, kernel_size=1),
            ),
            drop_path=drop_path,
            layer_scale_init_value=layer_scale_init_value,
            dim=dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class IFormerM(nn.Module):
    """
    Clean iFormer-M for image classification.

    This keeps the actual M architecture:
      depths = [2, 2, 22, 6]
      dims   = [48, 96, 192, 384]

    and exposes a pipeline-friendly API:
      - forward_features
      - forward_head
      - get_classifier
      - reset_classifier
    """
    def __init__(self, cfg: IFormerMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_classes = cfg.num_classes
        self.num_features = cfg.dims[-1]

        depths = list(cfg.depths)
        dims = list(cfg.dims)
        downsample_kernels = list(cfg.downsample_kernels)

        if depths != [2, 2, 22, 6]:
            raise ValueError("This cleaned definition is specific to iFormer-M depths=[2,2,22,6].")
        if dims != [48, 96, 192, 384]:
            raise ValueError("This cleaned definition is specific to iFormer-M dims=[48,96,192,384].")

        # Stochastic depth schedule across all 32 blocks
        dp_rates = torch.linspace(0, cfg.drop_path_rate, sum(depths)).tolist()
        dp_iter = iter(dp_rates)

        # Stem: exact original FusedIB stem
        stem_kernel = downsample_kernels[0]
        self.stem = nn.Sequential(
            Conv2dBN(cfg.in_chans, dims[0] // 2, kernel_size=stem_kernel, stride=2, padding=stem_kernel // 2),
            nn.GELU(),
            EdgeResidual(
                dims[0] // 2,
                dims[0],
                exp_kernel_size=stem_kernel,
                stride=2,
                exp_ratio=4.0,
                act_layer=nn.GELU,
            ),
        )

        # Downsample layers between stages
        self.downsample1 = Conv2dBN(dims[0], dims[1], kernel_size=downsample_kernels[1], stride=2, padding=downsample_kernels[1] // 2)
        self.downsample2 = Conv2dBN(dims[1], dims[2], kernel_size=downsample_kernels[2], stride=2, padding=downsample_kernels[2] // 2)
        self.downsample3 = Conv2dBN(dims[2], dims[3], kernel_size=downsample_kernels[3], stride=2, padding=downsample_kernels[3] // 2)

        # Stage 1: 2 x ConvBlock_k7_r4
        self.stage1 = nn.Sequential(*[
            ConvBlock(
                dim=dims[0],
                ratio=4,
                kernel=7,
                drop_path=next(dp_iter),
                layer_scale_init_value=cfg.layer_scale_init_value,
            )
            for _ in range(2)
        ])

        # Stage 2: 2 x ConvBlock_k7_r4
        self.stage2 = nn.Sequential(*[
            ConvBlock(
                dim=dims[1],
                ratio=4,
                kernel=7,
                drop_path=next(dp_iter),
                layer_scale_init_value=cfg.layer_scale_init_value,
            )
            for _ in range(2)
        ])

        # Stage 3:
        #   9 x ConvBlock_k7_r4
        #   4 x [RepCPE_k3 + SHMABlock_r1_hdrr2_nh1 + FFN2d_r3]
        #   1 x ConvBlock_k7_r4
        stage3_blocks: list[nn.Module] = []

        for _ in range(9):
            stage3_blocks.append(
                ConvBlock(
                    dim=dims[2],
                    ratio=4,
                    kernel=7,
                    drop_path=next(dp_iter),
                    layer_scale_init_value=cfg.layer_scale_init_value,
                )
            )

        for _ in range(4):
            stage3_blocks.append(RepCPE(dim=dims[2], kernel=3))
            stage3_blocks.append(
                SHMABlock(
                    dim=dims[2],
                    ratio=1,
                    head_dim_reduce_ratio=2,
                    num_heads=1,
                    drop_path=next(dp_iter),
                    layer_scale_init_value=cfg.layer_scale_init_value,
                )
            )
            stage3_blocks.append(
                FFN2d(
                    dim=dims[2],
                    ratio=3,
                    drop_path=next(dp_iter),
                    layer_scale_init_value=cfg.layer_scale_init_value,
                )
            )

        stage3_blocks.append(
            ConvBlock(
                dim=dims[2],
                ratio=4,
                kernel=7,
                drop_path=next(dp_iter),
                layer_scale_init_value=cfg.layer_scale_init_value,
            )
        )
        self.stage3 = nn.Sequential(*stage3_blocks)

        # Stage 4:
        #   2 x [RepCPE_k3 + SHMABlock_r1_hdrr4_nh1 + FFN2d_r3]
        stage4_blocks: list[nn.Module] = []
        for _ in range(2):
            stage4_blocks.append(RepCPE(dim=dims[3], kernel=3))
            stage4_blocks.append(
                SHMABlock(
                    dim=dims[3],
                    ratio=1,
                    head_dim_reduce_ratio=4,
                    num_heads=1,
                    drop_path=next(dp_iter),
                    layer_scale_init_value=cfg.layer_scale_init_value,
                )
            )
            stage4_blocks.append(
                FFN2d(
                    dim=dims[3],
                    ratio=3,
                    drop_path=next(dp_iter),
                    layer_scale_init_value=cfg.layer_scale_init_value,
                )
            )
        self.stage4 = nn.Sequential(*stage4_blocks)

        # Original iFormer code with use_bn=True ends with GAP + BN_Linear
        if cfg.num_classes > 0:
            self.head = BNLinear(self.num_features, cfg.num_classes)
        else:
            self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def get_classifier(self) -> nn.Module:
        if isinstance(self.head, BNLinear):
            return self.head.fc
        return self.head

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.head = BNLinear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)         # /4
        x = self.stage1(x)

        x = self.downsample1(x)  # /8
        x = self.stage2(x)

        x = self.downsample2(x)  # /16
        x = self.stage3(x)

        x = self.downsample3(x)  # /32
        x = self.stage4(x)

        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return x

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def build_iformer_m(
    num_classes: int = 2,
    in_chans: int = 3,
    drop_path_rate: float = 0.0,
    layer_scale_init_value: float = 0.0,
) -> IFormerM:
    cfg = IFormerMConfig(
        in_chans=in_chans,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        layer_scale_init_value=layer_scale_init_value,
    )
    return IFormerM(cfg)


if __name__ == "__main__":
    model = build_iformer_m(num_classes=4)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print("logits:", y.shape)         # (2, 4)

    feats = model.forward_features(x)
    print("features:", feats.shape)   # (2, 384)