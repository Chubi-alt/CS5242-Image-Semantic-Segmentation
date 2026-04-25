from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from core.constants import IGNORE_INDEX, NUM_CLASSES


def build_train_transform(config: Any) -> Any:
    albumentations = _import_albumentations()
    if config.strategy == "basic":
        transforms = [albumentations.Resize(config.image_size, config.image_size)]
        if config.horizontal_flip:
            transforms.append(albumentations.HorizontalFlip(p=0.5))
        return albumentations.Compose(transforms)

    if config.strategy not in {"strong", "classmix"}:
        raise ValueError(f"Unsupported augmentation.strategy: {config.strategy}")

    transforms = []
    if config.random_scale.enabled:
        scale_min, scale_max = config.random_scale.range
        transforms.append(albumentations.RandomScale(scale_limit=(scale_min - 1.0, scale_max - 1.0), p=1.0))
        transforms.append(
            albumentations.PadIfNeeded(
                min_height=config.image_size,
                min_width=config.image_size,
                border_mode=0,
                fill=0,
                fill_mask=IGNORE_INDEX,
            )
        )
        transforms.append(albumentations.RandomCrop(config.image_size, config.image_size))
    else:
        transforms.append(albumentations.Resize(config.image_size, config.image_size))

    if config.horizontal_flip:
        transforms.append(albumentations.HorizontalFlip(p=0.5))
    if config.color_jitter.enabled:
        transforms.append(
            albumentations.ColorJitter(
                brightness=config.color_jitter.brightness,
                contrast=config.color_jitter.contrast,
                saturation=config.color_jitter.saturation,
                hue=config.color_jitter.hue,
                p=0.8,
            )
        )
    if config.gaussian_blur.enabled:
        kernel_min, kernel_max = config.gaussian_blur.kernel_range
        transforms.append(albumentations.GaussianBlur(blur_limit=(kernel_min, kernel_max), p=0.2))
    return albumentations.Compose(transforms)


def build_val_transform(config: Any) -> Any:
    albumentations = _import_albumentations()
    return albumentations.Compose([albumentations.Resize(config.image_size, config.image_size)])


def apply_classmix(
    image_a: np.ndarray,
    label_a: np.ndarray,
    image_b: np.ndarray,
    label_b: np.ndarray,
    num_classes: Optional[int],
    total_classes: int = NUM_CLASSES,
) -> Tuple[np.ndarray, np.ndarray]:
    valid_classes = sorted(
        {
            int(value)
            for value in np.unique(label_b)
            if 0 <= int(value) < total_classes and int(value) != IGNORE_INDEX
        }
    )
    if not valid_classes:
        return np.array(image_a, copy=True), np.array(label_a, copy=True)

    generator = np.random.default_rng()
    if num_classes is None:
        upper = max(1, min(len(valid_classes), total_classes // 2))
        num_classes_to_mix = int(generator.integers(1, upper + 1))
    else:
        num_classes_to_mix = max(1, min(len(valid_classes), int(num_classes)))
    selected = generator.choice(valid_classes, size=num_classes_to_mix, replace=False)
    mask = np.isin(label_b, selected)

    mixed_image = np.array(image_a, copy=True)
    mixed_label = np.array(label_a, copy=True)
    mixed_image[mask] = image_b[mask]
    mixed_label[mask] = label_b[mask]
    mixed_label[mixed_label == 11] = IGNORE_INDEX
    return mixed_image.astype(np.uint8), mixed_label.astype(np.uint8)


def _import_albumentations() -> Any:
    try:
        import albumentations as albumentations  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("albumentations is required for SegExpress dataset transforms.") from exc
    return albumentations
