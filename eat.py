from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_2tuple(x: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(x, tuple):
        if len(x) != 2:
            raise ValueError("Expected a tuple of length 2.")
        return x
    return (x, x)


@dataclass
class EATConfig:
    # This is now the BASE size used to initialize positional embeddings.
    # It is NOT a hard constraint on the forward input size anymore.
    img_size: int | tuple[int, int] = 144

    in_channels: int = 3
    patch_size: int | tuple[int, int] = 9
    embed_dim: int = 64
    depth: int = 8

    # Table-2-faithful EA setup
    num_heads: int = 16
    expand_ratio: int = 4
    memory_dim: int = 16

    num_classes: int = 2

    attn_dropout: float = 0.2
    proj_dropout: float = 0.2
    ffn_dropout: float = 0.2

    layer_norm_eps: float = 1e-5
    l1_norm_eps: float = 1e-9

    use_pos_embed: bool = True
    use_cls_token: bool = False
    pool: str = "gap"  # "gap" or "cls"
    use_ffn_activation: bool = False
    patch_proj_bias: bool = False

    # New dynamic-shape options
    pad_if_needed: bool = True
    patch_pad_value: float = 0.0
    pos_embed_interp_mode: str = "bicubic"

    @property
    def patch_hw(self) -> tuple[int, int]:
        return _to_2tuple(self.patch_size)

    @property
    def patch_dim(self) -> int:
        ph, pw = self.patch_hw
        return ph * pw * self.in_channels

    @property
    def base_grid_size(self) -> tuple[int, int]:
        h, w = _to_2tuple(self.img_size)
        ph, pw = self.patch_hw
        gh = math.ceil(h / ph)
        gw = math.ceil(w / pw)
        return gh, gw

    @property
    def base_num_patches(self) -> int:
        gh, gw = self.base_grid_size
        return gh * gw

    @property
    def expanded_dim(self) -> int:
        return self.embed_dim * self.expand_ratio

    @property
    def head_dim(self) -> int:
        if self.expanded_dim % self.num_heads != 0:
            raise ValueError("expanded_dim must be divisible by num_heads.")
        return self.expanded_dim // self.num_heads


class PatchExtract(nn.Module):
    def __init__(
        self,
        patch_size: int | tuple[int, int],
        pad_if_needed: bool = True,
        pad_value: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_hw = _to_2tuple(patch_size)
        self.pad_if_needed = pad_if_needed
        self.pad_value = pad_value
        self.unfold = nn.Unfold(kernel_size=self.patch_hw, stride=self.patch_hw)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        # x: (B, C, H, W)
        _, _, h, w = x.shape
        ph, pw = self.patch_hw

        pad_h = (ph - (h % ph)) % ph
        pad_w = (pw - (w % pw)) % pw

        if (pad_h > 0 or pad_w > 0) and not self.pad_if_needed:
            raise ValueError(
                f"Input size {(h, w)} is not divisible by patch size {(ph, pw)}. "
                "Enable pad_if_needed=True or choose a compatible patch size."
            )

        if pad_h > 0 or pad_w > 0:
            # Pad on bottom and right only
            x = F.pad(x, (0, pad_w, 0, pad_h), value=self.pad_value)

        gh = (h + pad_h) // ph
        gw = (w + pad_w) // pw

        patches = self.unfold(x)  # (B, patch_dim, N)
        patches = patches.transpose(1, 2)  # (B, N, patch_dim)
        return patches, (gh, gw)


class PatchEmbedding(nn.Module):
    def __init__(self, cfg: EATConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.extract = PatchExtract(
            patch_size=cfg.patch_size,
            pad_if_needed=cfg.pad_if_needed,
            pad_value=cfg.patch_pad_value,
        )

        self.proj = nn.Linear(
            cfg.patch_dim,
            cfg.embed_dim,
            bias=cfg.patch_proj_bias,
        )

        seq_len = cfg.base_num_patches + (1 if cfg.use_cls_token else 0)

        if cfg.use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, cfg.embed_dim))
        else:
            self.register_parameter("pos_embed", None)

        if cfg.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        else:
            self.register_parameter("cls_token", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _interpolate_pos_embed(self, grid_size: tuple[int, int]) -> torch.Tensor | None:
        if self.pos_embed is None:
            return None

        target_gh, target_gw = grid_size
        base_gh, base_gw = self.cfg.base_grid_size
        embed_dim = self.cfg.embed_dim

        if self.cfg.use_cls_token:
            cls_pos = self.pos_embed[:, :1, :]
            patch_pos = self.pos_embed[:, 1:, :]
        else:
            cls_pos = None
            patch_pos = self.pos_embed

        if patch_pos.shape[1] != base_gh * base_gw:
            raise RuntimeError(
                "Stored positional embedding does not match base_grid_size."
            )

        if (target_gh, target_gw) == (base_gh, base_gw):
            resized_patch_pos = patch_pos
        else:
            resized_patch_pos = patch_pos.reshape(1, base_gh, base_gw, embed_dim)
            resized_patch_pos = resized_patch_pos.permute(0, 3, 1, 2)

            mode = self.cfg.pos_embed_interp_mode
            if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
                resized_patch_pos = F.interpolate(
                    resized_patch_pos,
                    size=(target_gh, target_gw),
                    mode=mode,
                    align_corners=False,
                )
            else:
                resized_patch_pos = F.interpolate(
                    resized_patch_pos,
                    size=(target_gh, target_gw),
                    mode=mode,
                )

            resized_patch_pos = resized_patch_pos.permute(0, 2, 3, 1)
            resized_patch_pos = resized_patch_pos.reshape(
                1, target_gh * target_gw, embed_dim
            )

        if cls_pos is not None:
            return torch.cat([cls_pos, resized_patch_pos], dim=1)

        return resized_patch_pos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, grid_size = self.extract(x)  # (B, N, patch_dim), dynamic grid
        x = self.proj(x)  # (B, N, embed_dim)

        if self.cfg.use_cls_token:
            cls = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat([cls, x], dim=1)

        pos_embed = self._interpolate_pos_embed(grid_size)
        if pos_embed is not None:
            x = x + pos_embed.to(dtype=x.dtype, device=x.device)

        return x


class MultiHeadExternalAttention(nn.Module):
    def __init__(self, cfg: EATConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.in_proj = nn.Linear(cfg.embed_dim, cfg.expanded_dim, bias=True)
        self.mk = nn.Linear(cfg.head_dim, cfg.memory_dim, bias=True)
        self.mv = nn.Linear(cfg.memory_dim, cfg.head_dim, bias=True)
        self.out_proj = nn.Linear(cfg.expanded_dim, cfg.embed_dim, bias=True)

        self.attn_drop = nn.Dropout(cfg.attn_dropout)
        self.proj_drop = nn.Dropout(cfg.proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape

        x = self.in_proj(x)  # (B, N, expanded_dim)

        x = x.view(b, n, self.cfg.num_heads, self.cfg.head_dim)
        x = x.permute(0, 2, 1, 3)  # (B, H, N, Dh)

        attn = self.mk(x)  # (B, H, N, M)
        attn = torch.softmax(attn, dim=2)
        attn = attn / (attn.sum(dim=3, keepdim=True) + self.cfg.l1_norm_eps)
        attn = self.attn_drop(attn)

        x = self.mv(attn)  # (B, H, N, Dh)

        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(b, n, self.cfg.expanded_dim)

        x = self.out_proj(x)
        x = self.proj_drop(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, cfg: EATConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.fc2 = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.drop1 = nn.Dropout(cfg.ffn_dropout)
        self.drop2 = nn.Dropout(cfg.ffn_dropout)
        self.act = nn.GELU() if cfg.use_ffn_activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class EATBlock(nn.Module):
    def __init__(self, cfg: EATConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.embed_dim, eps=cfg.layer_norm_eps)
        self.attn = MultiHeadExternalAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.embed_dim, eps=cfg.layer_norm_eps)
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class EATClassifier(nn.Module):
    def __init__(self, cfg: EATConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbedding(cfg)
        self.blocks = nn.ModuleList([EATBlock(cfg) for _ in range(cfg.depth)])
        self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)

        for block in self.blocks:
            x = block(x)

        if self.cfg.pool == "cls":
            if not self.cfg.use_cls_token:
                raise ValueError("pool='cls' requires use_cls_token=True")
            return x[:, 0]

        if self.cfg.pool == "gap":
            if self.cfg.use_cls_token:
                return x[:, 1:].mean(dim=1)
            return x.mean(dim=1)

        raise ValueError("pool must be 'gap' or 'cls'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        logits = self.head(feats)
        return logits


def build_eat(
    num_classes: int = 4,
    img_size: int | tuple[int, int] = 144,
) -> EATClassifier:
    cfg = EATConfig(
        img_size=img_size,
        num_classes=num_classes,
        num_heads=16,
        memory_dim=16,
        use_cls_token=False,
        pool="gap",
        use_ffn_activation=False,
        pad_if_needed=True,
    )
    return EATClassifier(cfg)


if __name__ == "__main__":
    model = build_eat(num_classes=4, img_size=144)

    for size in [144, 244, 64, 96]:
        x = torch.randn(2, 3, size, size)
        y = model(x)
        print(f"input={size}x{size} -> output={tuple(y.shape)}")
