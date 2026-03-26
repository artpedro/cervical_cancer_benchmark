from __future__ import annotations

from pathlib import Path
from typing import Any

import timm
import torch
import torchvision.models as tvm
from eat import EATClassifier, EATConfig, build_eat
from torch import nn


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def _adapt_head(model: nn.Module, num_classes: int = 2) -> int:
    """
    Replace the model's final classification layer(s) with a fresh head that
    outputs `num_classes` logits.

    Returns the input feature dimension of the new head.
    """
    if hasattr(model, "reset_classifier"):
        old_head = model.get_classifier()
        in_feats = getattr(
            old_head,
            "in_features",
            getattr(old_head, "in_channels", None),
        )
        model.reset_classifier(num_classes)
        return in_feats

    for attr in ("head", "classifier", "fc", "_fc"):
        if not hasattr(model, attr):
            continue

        head = getattr(model, attr)

        if isinstance(head, nn.Linear):
            in_feats = head.in_features
            setattr(model, attr, nn.Linear(in_feats, num_classes))
            return in_feats

        if isinstance(head, nn.Sequential):
            layers = list(head.children())

            for idx in range(len(layers) - 1, -1, -1):
                layer = layers[idx]

                if isinstance(layer, nn.Linear):
                    in_feats = layer.in_features
                    layers[idx] = nn.Linear(in_feats, num_classes)
                    setattr(model, attr, nn.Sequential(*layers))
                    return in_feats

                if isinstance(layer, nn.Conv2d):
                    in_ch = layer.in_channels
                    layers[idx] = nn.Conv2d(in_ch, num_classes, kernel_size=1)
                    setattr(model, attr, nn.Sequential(*layers))
                    return in_ch

    raise RuntimeError("Could not find a Linear/Conv2d classification head.")


def _load_checkpoint_if_needed(
    model: nn.Module,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> nn.Module:
    if checkpoint_path is None:
        return model

    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            state_dict = ckpt["model_state"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint format at: {checkpoint_path}")

    model.load_state_dict(state_dict, strict=strict)
    return model


def _build_eat_from_kwargs(
    num_classes: int,
    img_size: int | tuple[int, int] = 144,
    **kwargs: Any,
) -> EATClassifier:
    if not kwargs:
        return build_eat(num_classes=num_classes, img_size=img_size)

    cfg = EATConfig(
        img_size=img_size,
        num_classes=num_classes,
        **kwargs,
    )
    return EATClassifier(cfg)


MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "eat": {
        "source": "custom",
        "canonical_name": "eat",
        "max_params_m": 0.40,
    },
    "external_attention_transformer": {
        "source": "custom",
        "canonical_name": "eat",
        "max_params_m": 0.40,
    },
    "mobilevitv2_100": {
        "source": "timm",
        "canonical_name": "mobilevitv2_100.cvnets_in1k",
        "max_params_m": 5.0,
    },
    "mobilevitv2_100.cvnets_in1k": {
        "source": "timm",
        "canonical_name": "mobilevitv2_100.cvnets_in1k",
        "max_params_m": 5.0,
    },
    "mobilenet_v4": {
        "source": "timm",
        "canonical_name": "mobilenetv4_conv_small.e2400_r224_in1k",
        "max_params_m": 6.0,
    },
    "mobilenetv4_conv_small.e2400_r224_in1k": {
        "source": "timm",
        "canonical_name": "mobilenetv4_conv_small.e2400_r224_in1k",
        "max_params_m": 6.0,
    },
    "efficientformerv2_s0": {
        "source": "timm",
        "canonical_name": "efficientformerv2_s0.snap_dist_in1k",
        "max_params_m": 4.0,
    },
    "efficientformerv2_s0.snap_dist_in1k": {
        "source": "timm",
        "canonical_name": "efficientformerv2_s0.snap_dist_in1k",
        "max_params_m": 4.0,
    },
    "efficientformerv2_s1": {
        "source": "timm",
        "canonical_name": "efficientformerv2_s1.snap_dist_in1k",
        "max_params_m": 7.0,
    },
    "efficientformerv2_s1.snap_dist_in1k": {
        "source": "timm",
        "canonical_name": "efficientformerv2_s1.snap_dist_in1k",
        "max_params_m": 7.0,
    },
    "efficientformerv2_s2": {
        "source": "timm",
        "canonical_name": "efficientformerv2_s2.snap_dist_in1k",
        "max_params_m": 13.0,
    },
    "efficientformerv2_s2.snap_dist_in1k": {
        "source": "timm",
        "canonical_name": "efficientformerv2_s2.snap_dist_in1k",
        "max_params_m": 13.0,
    },
    "fastvit_t8": {
        "source": "timm",
        "canonical_name": "fastvit_t8.apple_in1k",
        "max_params_m": 5.0,
    },
    "fastvit_t8.apple_in1k": {
        "source": "timm",
        "canonical_name": "fastvit_t8.apple_in1k",
        "max_params_m": 5.0,
    },
    "levit_128s": {
        "source": "timm",
        "canonical_name": "levit_128s.fb_dist_in1k",
        "max_params_m": 8.5,
    },
    "levit_128s.fb_dist_in1k": {
        "source": "timm",
        "canonical_name": "levit_128s.fb_dist_in1k",
        "max_params_m": 8.5,
    },
    "tv_squeezenet1_1": {"source": "torchvision", "canonical_name": "tv_squeezenet1_1"},
    "tv_shufflenet_v2_x1_0": {"source": "torchvision", "canonical_name": "tv_shufflenet_v2_x1_0"},
    "tv_mobilenet_v2": {"source": "torchvision", "canonical_name": "tv_mobilenet_v2"},
    "mobilenetv2_100": {"source": "torchvision", "canonical_name": "mobilenetv2_100"},
    "efficientnet_b0": {"source": "torchvision", "canonical_name": "efficientnet_b0"},
    "efficientnet_b1": {"source": "torchvision", "canonical_name": "efficientnet_b1"},
    "efficientnet_b2": {"source": "torchvision", "canonical_name": "efficientnet_b2"},
    "efficientnet_b3": {"source": "torchvision", "canonical_name": "efficientnet_b3"},
    "efficientnet_b4": {"source": "torchvision", "canonical_name": "efficientnet_b4"},
    "efficientnet_b5": {"source": "torchvision", "canonical_name": "efficientnet_b5"},
    "efficientnet_b6": {"source": "torchvision", "canonical_name": "efficientnet_b6"},
    "efficientnet_b7": {"source": "torchvision", "canonical_name": "efficientnet_b7"},
    "ghostnet": {"source": "hub", "canonical_name": "ghostnet"},
}


TV_REGISTRY = {
    "tv_squeezenet1_1": tvm.squeezenet1_1,
    "tv_shufflenet_v2_x1_0": tvm.shufflenet_v2_x1_0,
    "tv_mobilenet_v2": tvm.mobilenet_v2,
    "mobilenetv2_100": tvm.mobilenet_v2,
    "efficientnet_b0": tvm.efficientnet_b0,
    "efficientnet_b1": tvm.efficientnet_b1,
    "efficientnet_b2": tvm.efficientnet_b2,
    "efficientnet_b3": tvm.efficientnet_b3,
    "efficientnet_b4": tvm.efficientnet_b4,
    "efficientnet_b5": tvm.efficientnet_b5,
    "efficientnet_b6": tvm.efficientnet_b6,
    "efficientnet_b7": tvm.efficientnet_b7,
}


def load_any(
    name: str,
    num_classes: int = 2,
    pretrained: bool = True,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    max_params_m: float = 20.0,
    strict_checkpoint: bool = True,
    **model_kwargs: Any,
):
    key = name.lower()

    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model alias: {name}\n"
            f"Available keys: {sorted(MODEL_REGISTRY.keys())}"
        )

    spec = MODEL_REGISTRY[key]
    source = spec["source"]
    canonical_name = spec["canonical_name"]

    model: nn.Module
    origin: str

    if source == "custom":
        img_size = model_kwargs.pop("img_size", 144)
        model = _build_eat_from_kwargs(
            num_classes=num_classes,
            img_size=img_size,
            **model_kwargs,
        )
        in_features = model.head.in_features
        origin = f"custom:{canonical_name}"
    elif source == "timm":
        try:
            model = timm.create_model(
                canonical_name,
                pretrained=pretrained,
                num_classes=num_classes,
                **model_kwargs,
            )
        except TypeError:
            model = timm.create_model(
                canonical_name,
                pretrained=pretrained,
                **model_kwargs,
            )
            _adapt_head(model, num_classes)

        origin = f"timm:{canonical_name}"

        if hasattr(model, "get_classifier"):
            cls = model.get_classifier()
            in_features = getattr(
                cls,
                "in_features",
                getattr(cls, "in_channels", None),
            )
        else:
            in_features = _adapt_head(model, num_classes)
    elif source == "torchvision":
        tv_ctor = TV_REGISTRY[canonical_name]

        weights = None
        if pretrained:
            weight_enum = getattr(tv_ctor, "Weights", None)
            if weight_enum is not None:
                weights = weight_enum.DEFAULT

        try:
            model = tv_ctor(weights=weights)
        except TypeError:
            model = tv_ctor(pretrained=pretrained)

        in_features = _adapt_head(model, num_classes)
        origin = f"torchvision:{canonical_name}"
    elif source == "hub":
        if canonical_name == "ghostnet":
            model = torch.hub.load("pytorch/vision", "ghostnet_1x", pretrained=pretrained)
            in_features = _adapt_head(model, num_classes)
            origin = "hub:ghostnet"
        else:
            raise ValueError(f"Unsupported hub model: {canonical_name}")
    else:
        raise ValueError(f"Unsupported source type: {source}")

    if checkpoint_path is not None:
        model = _load_checkpoint_if_needed(
            model=model,
            checkpoint_path=checkpoint_path,
            device=device,
            strict=strict_checkpoint,
        )

    n_params = count_parameters(model, trainable_only=False)
    n_params_m = n_params / 1_000_000
    if n_params_m > max_params_m:
        raise ValueError(
            f"Model '{name}' has {n_params_m:.3f}M parameters, "
            f"which exceeds max_params_m={max_params_m:.1f}M."
        )

    return model, in_features, origin, n_params


__all__ = ["load_any", "MODEL_REGISTRY", "count_parameters"]

