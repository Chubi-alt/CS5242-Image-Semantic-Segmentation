from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def build_loss_fn(
    config,
    num_classes: int = 11,
    ignore_index: int = 255,
    class_frequencies: Optional[np.ndarray] = None,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    ce_loss = CrossEntropyLoss(
        ignore_index=ignore_index,
        class_frequencies=class_frequencies if config.class_weights else None,
    )
    dice_loss = DiceLoss(ignore_index=ignore_index, num_classes=num_classes)
    focal_loss = FocalLoss(ignore_index=ignore_index, gamma=config.focal_gamma)
    boundary_loss = BoundaryLoss(ignore_index=ignore_index, sigma=config.boundary.sigma) if config.boundary.enabled else None

    component_builders = {
        "ce": ce_loss,
        "dice": dice_loss,
        "focal": focal_loss,
    }

    if config.type == "combined":
        components: List[Tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], float]] = []
        for component in config.components:
            loss_fn = component_builders[component.type]
            if config.ohem.enabled and component.type == "ce":
                loss_fn = OHEMWrapper(loss_fn=ce_loss, ratio=config.ohem.ratio, ignore_index=ignore_index)
            components.append((loss_fn, component.weight))
        return CombinedLoss(
            components=components,
            boundary_loss=boundary_loss,
            boundary_weight=config.boundary.weight,
        )

    if config.type == "ce":
        main_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = ce_loss
        if config.ohem.enabled:
            main_loss = OHEMWrapper(loss_fn=ce_loss, ratio=config.ohem.ratio, ignore_index=ignore_index)
    elif config.type == "dice":
        main_loss = dice_loss
    elif config.type == "focal":
        main_loss = focal_loss
    else:  # pragma: no cover - validate_config prevents this branch
        raise ValueError(f"Unsupported loss.type: {config.type}")

    if boundary_loss is not None:
        return CombinedLoss(
            components=[(main_loss, 1.0)],
            boundary_loss=boundary_loss,
            boundary_weight=config.boundary.weight,
        )
    return main_loss


class CrossEntropyLoss:
    def __init__(self, ignore_index: int = 255, class_frequencies: Optional[np.ndarray] = None) -> None:
        self.ignore_index = ignore_index
        self.weight = _build_class_weight_tensor(class_frequencies) if class_frequencies is not None else None

    def per_pixel(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(logits.device) if self.weight is not None else None
        return F.cross_entropy(logits, labels, weight=weight, ignore_index=self.ignore_index, reduction="none")

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        per_pixel = self.per_pixel(logits, labels)
        valid = labels != self.ignore_index
        if not torch.any(valid):
            return logits.new_zeros(())
        return per_pixel[valid].mean()


class DiceLoss:
    def __init__(self, ignore_index: int = 255, num_classes: int = 11) -> None:
        self.ignore_index = ignore_index
        self.num_classes = num_classes

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(logits, dim=1)
        valid_mask = labels != self.ignore_index
        safe_labels = labels.clone()
        safe_labels[~valid_mask] = 0
        one_hot = F.one_hot(safe_labels, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        valid_mask_f = valid_mask.unsqueeze(1).float()
        probabilities = probabilities * valid_mask_f
        one_hot = one_hot * valid_mask_f
        dims = (0, 2, 3)
        intersection = (probabilities * one_hot).sum(dim=dims)
        denominator = probabilities.sum(dim=dims) + one_hot.sum(dim=dims)
        dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
        return 1.0 - dice.mean()


class FocalLoss:
    def __init__(self, ignore_index: int = 255, gamma: float = 2.0) -> None:
        self.ignore_index = ignore_index
        self.gamma = gamma

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, labels, ignore_index=self.ignore_index, reduction="none")
        valid = labels != self.ignore_index
        if not torch.any(valid):
            return logits.new_zeros(())
        pt = torch.exp(-ce[valid])
        loss = ((1.0 - pt) ** self.gamma) * ce[valid]
        return loss.mean()


class BoundaryLoss:
    def __init__(self, ignore_index: int = 255, sigma: float = 5.0) -> None:
        self.ignore_index = ignore_index
        self.sigma = sigma

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        try:
            from scipy.ndimage import distance_transform_edt
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("scipy is required for BoundaryLoss.") from exc

        per_pixel_ce = F.cross_entropy(logits, labels, ignore_index=self.ignore_index, reduction="none")
        weight_maps = []
        labels_np = labels.detach().cpu().numpy()
        for sample in labels_np:
            valid = sample != self.ignore_index
            boundary = _label_to_boundary_map(sample, ignore_index=self.ignore_index)
            distance = distance_transform_edt(~boundary)
            weights = 1.0 + np.exp(-distance / self.sigma)
            weights[~valid] = 0.0
            weight_maps.append(torch.from_numpy(weights).float())
        weight_tensor = torch.stack(weight_maps, dim=0).to(logits.device)
        valid = labels != self.ignore_index
        if not torch.any(valid):
            return logits.new_zeros(())
        weighted = per_pixel_ce * weight_tensor
        return weighted[valid].mean()


class OHEMWrapper:
    def __init__(self, loss_fn: CrossEntropyLoss, ratio: float = 0.7, ignore_index: int = 255) -> None:
        self.loss_fn = loss_fn
        self.ratio = ratio
        self.ignore_index = ignore_index

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        per_pixel = self.loss_fn.per_pixel(logits, labels)
        valid = labels != self.ignore_index
        if not torch.any(valid):
            return logits.new_zeros(())
        values = per_pixel[valid].reshape(-1)
        keep_count = max(1, int(torch.ceil(torch.tensor(values.numel() * self.ratio)).item()))
        hardest = torch.topk(values, k=keep_count, largest=True).values
        return hardest.mean()


class CombinedLoss:
    def __init__(
        self,
        components: List[Tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], float]],
        boundary_loss: Optional[BoundaryLoss] = None,
        boundary_weight: float = 0.0,
    ) -> None:
        self.components = components
        self.boundary_loss = boundary_loss
        self.boundary_weight = boundary_weight

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        total = logits.new_zeros(())
        for loss_fn, weight in self.components:
            total = total + float(weight) * loss_fn(logits, labels)
        if self.boundary_loss is not None and self.boundary_weight > 0:
            total = total + float(self.boundary_weight) * self.boundary_loss(logits, labels)
        return total


def _build_class_weight_tensor(class_frequencies: Optional[np.ndarray]) -> Optional[torch.Tensor]:
    if class_frequencies is None:
        return None
    frequencies = np.asarray(class_frequencies, dtype=np.float64)
    frequencies = np.clip(frequencies, 1e-12, None)
    inverse = 1.0 / frequencies
    normalized = inverse / inverse.mean()
    return torch.tensor(normalized, dtype=torch.float32)


def _label_to_boundary_map(label: np.ndarray, ignore_index: int) -> np.ndarray:
    boundary = np.zeros_like(label, dtype=bool)
    valid = label != ignore_index
    if label.shape[0] > 1:
        vertical = (label[1:, :] != label[:-1, :]) & valid[1:, :] & valid[:-1, :]
        boundary[1:, :] |= vertical
        boundary[:-1, :] |= vertical
    if label.shape[1] > 1:
        horizontal = (label[:, 1:] != label[:, :-1]) & valid[:, 1:] & valid[:, :-1]
        boundary[:, 1:] |= horizontal
        boundary[:, :-1] |= horizontal
    return boundary
