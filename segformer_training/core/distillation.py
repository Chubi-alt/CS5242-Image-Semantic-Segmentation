"""core/distillation.py — Knowledge Distillation components for SegFormer B2→B0."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple  # noqa: F401 (Any, Tuple used in later tasks)

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401 (used in loss functions added in Tasks 4-6)

from core.constants import BACKBONE_CHANNELS

# Channel dims for backbone feature adaptor projections
_B0_CHANNELS: List[int] = list(BACKBONE_CHANNELS["nvidia/mit-b0"])   # [32, 64, 160, 256]
_B2_CHANNELS: List[int] = list(BACKBONE_CHANNELS["nvidia/mit-b2"])   # [64, 128, 320, 512]


class SegFormerFeatureExtractor:
    """Captures the 256-dim pre-classification decoder feature via a persistent forward pre-hook.

    Attach to a model at initialisation; the hook fires on every forward pass through
    ``model.decode_head.classifier``.  Call ``remove()`` when done to detach the hook.
    """

    def __init__(self, model: nn.Module) -> None:
        self._feats: Dict[str, Optional[torch.Tensor]] = {"decoder": None}
        self._hook = model.decode_head.classifier.register_forward_pre_hook(
            self._capture
        )

    def _capture(self, module: nn.Module, inputs: Tuple) -> None:
        # inputs[0] is the (B, 256, H, W) tensor fed into the 1×1 classifier conv
        self._feats["decoder"] = inputs[0]

    def get_decoder_feat(self) -> Optional[torch.Tensor]:
        return self._feats["decoder"]

    def remove(self) -> None:
        self._hook.remove()
        self._feats["decoder"] = None


class FeatureAdaptorBank(nn.Module):
    """Four 1×1 Conv layers projecting B0 backbone channel dims to B2 dims.

    B0: [32, 64, 160, 256] → B2: [64, 128, 320, 512]
    Used only for Multi-level KD (Method 3).  Parameters are trainable and
    included in the student model's optimizer group via SegFormerWithAdaptors.
    """

    def __init__(self) -> None:
        super().__init__()
        self.adaptors = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
            for in_ch, out_ch in zip(_B0_CHANNELS, _B2_CHANNELS)
        ])

    def forward(self, stage_feats: List[torch.Tensor]) -> List[torch.Tensor]:
        return [conv(feat) for conv, feat in zip(self.adaptors, stage_feats)]


class SegFormerWithAdaptors(nn.Module):
    """Wraps a SegFormer B0 student with a trainable FeatureAdaptorBank.

    Always requests ``output_hidden_states=True`` from the inner model so that
    backbone stage features are available for Multi-level KD.  Returns a
    ``SegFormerModelOutput`` with ``adapted_stage_feats`` populated.

    ``get_main_model()`` returns the unwrapped base model, enabling
    ``export_inference_bundle`` to save a standard B0 without adaptor weights.
    """

    def __init__(self, base_model: nn.Module, adaptor_bank: FeatureAdaptorBank) -> None:
        super().__init__()
        self.base_model = base_model
        self.adaptor_bank = adaptor_bank

    def forward(self, pixel_values: torch.Tensor, **kwargs) -> Any:
        from core.model import SegFormerModelOutput
        kwargs["output_hidden_states"] = True
        base_out = self.base_model(pixel_values=pixel_values, **kwargs)
        logits = base_out.logits if hasattr(base_out, "logits") else base_out["logits"]
        hidden_states = tuple(base_out.hidden_states) if base_out.hidden_states is not None else None
        if hidden_states is None:
            raise RuntimeError(
                "SegFormerWithAdaptors: base_model returned hidden_states=None "
                "despite output_hidden_states=True. Ensure the inner model honours this kwarg."
            )
        adapted = self.adaptor_bank(list(hidden_states))
        return SegFormerModelOutput(
            logits=logits,
            hidden_states=hidden_states,
            adapted_stage_feats=adapted,
        )

    def get_main_model(self) -> nn.Module:
        return self.base_model


class LogitKDLoss:
    """Method 1: Pixel-wise KL-divergence knowledge distillation on output logits.

    Returns the KD loss term (already scaled by alpha and T²).  Add this to the
    task loss computed by the main ``loss_fn`` in the trainer:

        total_loss = task_loss + kd_loss_fn(student_logits, teacher_logits, labels)

    Valid pixels are determined by ``labels != ignore_index``.
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        ignore_index: int = 255,
    ) -> None:
        self.T = temperature
        self.alpha = alpha
        self.ignore_index = ignore_index

    def __call__(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        student_outputs: Any = None,
        teacher_outputs: Any = None,
    ) -> torch.Tensor:
        valid_mask = (labels != self.ignore_index).unsqueeze(1).float()  # (B,1,H,W)

        s_log = F.log_softmax(student_logits / self.T, dim=1)
        t_prob = F.softmax(teacher_logits / self.T, dim=1)

        # Per-pixel KL: sum over classes, then mean over valid pixels
        kl_per_pixel = F.kl_div(s_log, t_prob, reduction="none").sum(dim=1, keepdim=True)
        n_valid = valid_mask.sum().clamp(min=1.0)
        kl = (kl_per_pixel * valid_mask).sum() / n_valid

        return self.alpha * (self.T ** 2) * kl


def _normalize_feat(feat: torch.Tensor) -> torch.Tensor:
    """L2-normalise along the channel dimension for stable feature MSE."""
    return F.normalize(feat, p=2, dim=1)


class DecoderKDLoss:
    """Method 2: Logit KD + decoder feature alignment.

    Both B0 and B2 use an All-MLP decoder with ``decoder_hidden_size=256``,
    so the 256-dim feature before the final classifier conv can be matched
    directly without a channel adaptor.  Features are L2-normalised before MSE.

    Requires ``SegFormerFeatureExtractor`` instances attached to both the
    student and teacher models (done in ``build_kd_loss``).
    """

    def __init__(
        self,
        student_extractor: SegFormerFeatureExtractor,
        teacher_extractor: SegFormerFeatureExtractor,
        temperature: float = 4.0,
        alpha: float = 0.5,
        feature_weight: float = 0.5,
        ignore_index: int = 255,
    ) -> None:
        self._logit_kd = LogitKDLoss(temperature, alpha, ignore_index)
        self.student_extractor = student_extractor
        self.teacher_extractor = teacher_extractor
        self.feature_weight = feature_weight

    def __call__(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        student_outputs: Any = None,
        teacher_outputs: Any = None,
    ) -> torch.Tensor:
        logit_loss = self._logit_kd(student_logits, teacher_logits, labels)

        dec_s = self.student_extractor.get_decoder_feat()
        dec_t = self.teacher_extractor.get_decoder_feat()
        if dec_s is None or dec_t is None or self.feature_weight == 0.0:
            return logit_loss

        feat_loss = F.mse_loss(_normalize_feat(dec_s), _normalize_feat(dec_t.detach()))
        return logit_loss + self.feature_weight * feat_loss


class MultiLevelKDLoss:
    """Method 3: Logit KD + decoder feature + backbone stage feature alignment.

    Student backbone features are projected to B2 channel dims by the
    ``FeatureAdaptorBank`` inside ``SegFormerWithAdaptors`` (available via
    ``student_outputs.adapted_stage_feats``).  Teacher backbone features come
    from ``teacher_outputs.hidden_states``.  All features are L2-normalised.
    """

    def __init__(
        self,
        student_extractor: SegFormerFeatureExtractor,
        teacher_extractor: SegFormerFeatureExtractor,
        temperature: float = 4.0,
        alpha: float = 0.5,
        feature_weight: float = 0.5,
        stage_weight: float = 0.3,
        ignore_index: int = 255,
    ) -> None:
        self._decoder_kd = DecoderKDLoss(
            student_extractor, teacher_extractor,
            temperature, alpha, feature_weight, ignore_index,
        )
        self.stage_weight = stage_weight

    def __call__(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        student_outputs: Any = None,
        teacher_outputs: Any = None,
    ) -> torch.Tensor:
        loss = self._decoder_kd(student_logits, teacher_logits, labels,
                                student_outputs, teacher_outputs)

        if self.stage_weight == 0.0:
            return loss

        adapted_feats = getattr(student_outputs, "adapted_stage_feats", None)
        teacher_hidden = getattr(teacher_outputs, "hidden_states", None)
        if adapted_feats is None or teacher_hidden is None:
            return loss

        stage_losses = [
            F.mse_loss(_normalize_feat(s_feat), _normalize_feat(t_feat.detach()))
            for s_feat, t_feat in zip(adapted_feats, teacher_hidden)
        ]
        return loss + self.stage_weight * torch.stack(stage_losses).mean()

    @property
    def student_extractor(self):
        return self._decoder_kd.student_extractor

    @property
    def teacher_extractor(self):
        return self._decoder_kd.teacher_extractor


def build_kd_loss(
    kd_config,
    student_model: nn.Module,
    teacher_model: nn.Module,
    ignore_index: int = 255,
) -> Tuple[nn.Module, Optional[Any]]:
    """Factory: create the appropriate KD loss and optionally wrap the student model.

    Returns ``(model, kd_loss_fn)`` where ``model`` may be the original student
    or a ``SegFormerWithAdaptors`` wrapper (for method='multilevel').
    Returns ``(student_model, None)`` when ``kd_config.enabled`` is False.
    """
    if not kd_config.enabled:
        return student_model, None

    method = kd_config.method

    if method == "logit":
        return student_model, LogitKDLoss(
            temperature=kd_config.temperature,
            alpha=kd_config.alpha,
            ignore_index=ignore_index,
        )

    # Methods 2 & 3 need decoder feature extractors
    if method == "multilevel":
        adaptor_bank = FeatureAdaptorBank()
        student_model = SegFormerWithAdaptors(student_model, adaptor_bank)

    # Hook on the base model's decode_head (unwrap if necessary)
    student_base = (
        student_model.get_main_model()
        if hasattr(student_model, "get_main_model")
        else student_model
    )
    student_extractor = SegFormerFeatureExtractor(student_base)
    teacher_extractor = SegFormerFeatureExtractor(teacher_model)

    if method == "decoder":
        return student_model, DecoderKDLoss(
            student_extractor, teacher_extractor,
            temperature=kd_config.temperature,
            alpha=kd_config.alpha,
            feature_weight=kd_config.feature_weight,
            ignore_index=ignore_index,
        )

    if method == "multilevel":
        return student_model, MultiLevelKDLoss(
            student_extractor, teacher_extractor,
            temperature=kd_config.temperature,
            alpha=kd_config.alpha,
            feature_weight=kd_config.feature_weight,
            stage_weight=kd_config.stage_weight,
            ignore_index=ignore_index,
        )

    raise ValueError(f"Unknown KD method: {method!r}. Expected one of: logit, decoder, multilevel")
