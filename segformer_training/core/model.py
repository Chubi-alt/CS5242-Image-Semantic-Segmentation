from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import load_config_snapshot
from core.constants import BACKBONE_CHANNELS, ID2LABEL, LABEL2ID, SEGFORMER_BACKBONE_DEPTHS
from core.utils import find_snapshot_for_export_path, has_processor_artifacts


@dataclass
class SegFormerModelOutput:
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    aux_logits: Optional[List[torch.Tensor]] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    adapted_stage_feats: Optional[List[torch.Tensor]] = None  # populated by SegFormerWithAdaptors


class SegFormerWithAuxHeads(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        aux_stages: List[int],
        aux_channels: int,
        num_labels: int,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.aux_stages = list(aux_stages)
        self.aux_channels = aux_channels
        self.num_labels = num_labels
        hidden_sizes = list(getattr(base_model.config, "hidden_sizes", []))
        self.aux_heads = nn.ModuleDict()
        for stage in self.aux_stages:
            stage_channels = hidden_sizes[stage]
            self.aux_heads[str(stage)] = nn.Sequential(
                nn.Conv2d(stage_channels, aux_channels, kernel_size=1),
                nn.BatchNorm2d(aux_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(aux_channels, num_labels, kernel_size=1),
            )

    def forward(self, pixel_values: torch.Tensor, labels: Optional[torch.Tensor] = None, **kwargs) -> SegFormerModelOutput:
        del labels
        outputs = self.base_model(pixel_values=pixel_values, output_hidden_states=True, **kwargs)
        logits = outputs.logits
        hidden_states = tuple(outputs.hidden_states) if outputs.hidden_states is not None else None
        aux_logits: Optional[List[torch.Tensor]] = None
        if hidden_states is not None and self.training and self.aux_heads:
            target_size = logits.shape[-2:]
            aux_logits = []
            for stage in self.aux_stages:
                aux_head = self.aux_heads[str(stage)]
                aux_logit = aux_head(hidden_states[stage])
                if aux_logit.shape[-2:] != target_size:
                    aux_logit = F.interpolate(aux_logit, size=target_size, mode="bilinear", align_corners=False)
                aux_logits.append(aux_logit)
        return SegFormerModelOutput(
            logits=logits,
            aux_logits=aux_logits,
            hidden_states=hidden_states,
        )

    def get_main_model(self):
        return self.base_model


def build_model(config) -> Tuple[nn.Module, object]:
    try:
        from transformers import SegformerConfig, SegformerForSemanticSegmentation, SegformerImageProcessor
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("transformers is required to build the SegFormer model.") from exc

    if config.resume_from is not None:
        model, processor = _build_model_from_resume_export(
            config=config,
            segformer_config_cls=SegformerConfig,
            segformer_model_cls=SegformerForSemanticSegmentation,
            processor_cls=SegformerImageProcessor,
        )
        return model, processor

    cache_dir = os.environ.get("HF_HOME")
    local_files_only = bool(os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE"))
    if config.pretrained:
        base_model = SegformerForSemanticSegmentation.from_pretrained(
            config.backbone,
            num_labels=config.num_labels,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        processor = SegformerImageProcessor.from_pretrained(
            config.backbone,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    else:
        base_model = SegformerForSemanticSegmentation(_build_offline_segformer_config(SegformerConfig, config.backbone, config.num_labels))
        processor = _build_offline_processor(SegformerImageProcessor)

    model = _maybe_wrap_with_aux(base_model, config)
    processor = _finalize_processor(processor)
    return model, processor


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
    }


def export_inference_bundle(model: nn.Module, processor: object, output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    base_model = get_base_model(model)
    if hasattr(base_model, "save_pretrained"):
        base_model.save_pretrained(output_path)
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(output_path)


def get_base_model(model: nn.Module) -> nn.Module:
    # Import here to avoid circular import (distillation imports from model)
    from core.distillation import SegFormerWithAdaptors
    if isinstance(model, (SegFormerWithAuxHeads, SegFormerWithAdaptors)):
        return model.get_main_model()
    return model


def load_full_model_state(model: nn.Module, checkpoint_path: str | Path) -> Tuple[List[str], List[str]]:
    checkpoint = Path(checkpoint_path)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise RuntimeError(f"Expected state_dict at {checkpoint}, got {type(state).__name__}")

    from core.distillation import SegFormerWithAdaptors
    is_wrapper = isinstance(model, (SegFormerWithAuxHeads, SegFormerWithAdaptors))
    has_base_prefix = any(key.startswith("base_model.") for key in state)

    if is_wrapper and has_base_prefix:
        result = model.load_state_dict(state, strict=False)
    elif is_wrapper:
        wrapper_state = {f"base_model.{key}": value for key, value in state.items()}
        result = model.load_state_dict(wrapper_state, strict=False)
    elif has_base_prefix:
        base_state = {
            key.removeprefix("base_model."): value
            for key, value in state.items()
            if key.startswith("base_model.")
        }
        result = model.load_state_dict(base_state, strict=False)
    else:
        result = model.load_state_dict(state, strict=False)
    return list(result.missing_keys), list(result.unexpected_keys)


def _build_model_from_resume_export(config, segformer_config_cls, segformer_model_cls, processor_cls):
    resume_path = Path(config.resume_from).expanduser().resolve()

    if resume_path.is_file():
        # Prefer HF config.json next to the checkpoint (exact match).
        # Falls back to offline-built config (DECODER_HIDDEN_SIZES lookup) if absent.
        sibling_config = resume_path.parent / "config.json"
        if sibling_config.is_file():
            hf_config = segformer_config_cls.from_pretrained(str(resume_path.parent))
            # Override num_labels / id2label / label2id in case the saved ones differ
            hf_config.num_labels = config.num_labels
            hf_config.id2label = ID2LABEL
            hf_config.label2id = LABEL2ID
            base_model = segformer_model_cls(hf_config)
        else:
            base_model = segformer_model_cls(_build_offline_segformer_config(segformer_config_cls, config.backbone, config.num_labels))
        model = _maybe_wrap_with_aux(base_model, config)
        processor = _load_processor_from_export(resume_path.parent / "segformer", processor_cls)
        missing_keys, unexpected_keys = load_full_model_state(model, resume_path)
        _warn_on_state_mismatch(resume_path, missing_keys, unexpected_keys)
        return model, processor

    if resume_path.is_dir():
        base_model = segformer_model_cls.from_pretrained(
            str(resume_path),
            local_files_only=True,
            num_labels=config.num_labels,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )
        model = _maybe_wrap_with_aux(base_model, config)
        processor = _load_processor_from_export(resume_path, processor_cls)
        return model, processor

    raise FileNotFoundError(f"Unsupported resume export path: {resume_path}")


def _load_processor_from_export(segformer_dir: Path, processor_cls):
    if has_processor_artifacts(segformer_dir):
        processor = processor_cls.from_pretrained(str(segformer_dir), local_files_only=True)
        return _finalize_processor(processor)

    snapshot_path = find_snapshot_for_export_path(segformer_dir)
    if snapshot_path is None:
        raise FileNotFoundError(f"Processor artifacts not found under {segformer_dir}")
    warnings.warn(f"Processor artifacts missing under {segformer_dir}; rebuilding from {snapshot_path}.")
    _ = load_config_snapshot(snapshot_path, validate=False)
    return _finalize_processor(_build_offline_processor(processor_cls))


def _maybe_wrap_with_aux(base_model: nn.Module, config) -> nn.Module:
    if not config.auxiliary_heads.enabled:
        return base_model
    return SegFormerWithAuxHeads(
        base_model=base_model,
        aux_stages=list(config.auxiliary_heads.stages),
        aux_channels=int(config.auxiliary_heads.channels),
        num_labels=int(config.num_labels),
    )


def _warn_on_state_mismatch(checkpoint_path: Path, missing_keys: List[str], unexpected_keys: List[str]) -> None:
    filtered_missing = [key for key in missing_keys if not key.endswith("num_batches_tracked")]
    if filtered_missing or unexpected_keys:
        warnings.warn(
            "Loaded checkpoint with non-strict state_dict compatibility. "
            f"path={checkpoint_path} missing={filtered_missing} unexpected={unexpected_keys}"
        )


def _build_offline_segformer_config(segformer_config_cls, backbone: str, num_labels: int):
    from core.constants import DECODER_HIDDEN_SIZES
    return segformer_config_cls(
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        depths=SEGFORMER_BACKBONE_DEPTHS[backbone],
        hidden_sizes=BACKBONE_CHANNELS[backbone],
        decoder_hidden_size=DECODER_HIDDEN_SIZES.get(backbone, 256),
        num_attention_heads=(1, 2, 5, 8),
        sr_ratios=(8, 4, 2, 1),
        patch_sizes=(7, 3, 3, 3),
        strides=(4, 2, 2, 2),
        mlp_ratios=(4, 4, 4, 4),
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        classifier_dropout_prob=0.1,
        initializer_range=0.02,
        drop_path_rate=0.1,
        semantic_loss_ignore_index=255,
        reshape_last_stage=True,
    )


def _build_offline_processor(processor_cls):
    return processor_cls(
        do_resize=False,
        do_reduce_labels=False,
        do_rescale=True,
        rescale_factor=1.0 / 255.0,
        do_normalize=True,
        image_mean=(0.485, 0.456, 0.406),
        image_std=(0.229, 0.224, 0.225),
        size={"height": 512, "width": 512},
    )


def _finalize_processor(processor):
    if hasattr(processor, "do_resize"):
        processor.do_resize = False
    if hasattr(processor, "do_reduce_labels"):
        processor.do_reduce_labels = False
    return processor
