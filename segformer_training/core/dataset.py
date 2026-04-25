from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import random

import numpy as np
import torch
from PIL import Image

from core.augmentations import apply_classmix, build_train_transform, build_val_transform
from core.constants import IGNORE_INDEX, NUM_CLASSES


@dataclass
class DatasetBundle:
    train_dataset: torch.utils.data.Dataset
    val_dataset: torch.utils.data.Dataset
    test_dataset: Optional[torch.utils.data.Dataset]
    class_frequencies: Optional[np.ndarray] = None


@dataclass
class RawSample:
    image: Image.Image
    label: np.ndarray
    image_path: str
    label_path: str


class CamVidDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_paths: List[Path],
        label_paths: List[Path],
        transform,
        processor,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform
        self.processor = processor
        self.ignore_index = ignore_index

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_array, label_array = self.get_augmented_arrays(index)
        encoded = self.processor(
            images=image_array,
            segmentation_maps=label_array,
            return_tensors="pt",
        )
        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels": encoded["labels"].squeeze(0),
        }

    def get_raw_sample(self, index: int) -> RawSample:
        image_path = self.image_paths[index]
        label_path = self.label_paths[index]
        image = Image.open(image_path).convert("RGB")
        label = np.asarray(Image.open(label_path), dtype=np.uint8)
        label = _canonicalize_label(label, ignore_index=self.ignore_index)
        return RawSample(
            image=image,
            label=label,
            image_path=str(image_path),
            label_path=str(label_path),
        )

    def get_augmented_arrays(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        sample = self.get_raw_sample(index)
        image = np.asarray(sample.image, dtype=np.uint8)
        label = np.asarray(sample.label, dtype=np.uint8)
        if self.transform is None:
            return image, label
        transformed = self.transform(image=image, mask=label)
        transformed_image = np.asarray(transformed["image"], dtype=np.uint8)
        transformed_label = np.asarray(transformed["mask"], dtype=np.uint8)
        transformed_label = _canonicalize_label(transformed_label, ignore_index=self.ignore_index)
        return transformed_image, transformed_label


class ClassMixDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dataset: CamVidDataset,
        classmix_prob: float,
        num_classes_to_mix: Optional[int] = None,
        total_classes: int = NUM_CLASSES,
    ) -> None:
        self.base_dataset = base_dataset
        self.classmix_prob = classmix_prob
        self.num_classes_to_mix = num_classes_to_mix
        self.total_classes = total_classes
        self.processor = base_dataset.processor

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        image_a, label_a = self.base_dataset.get_augmented_arrays(index)
        if len(self.base_dataset) > 1 and random.random() < self.classmix_prob:
            candidate = random.randrange(len(self.base_dataset) - 1)
            mix_index = candidate if candidate < index else candidate + 1
            image_b, label_b = self.base_dataset.get_augmented_arrays(mix_index)
            mixed_image, mixed_label = apply_classmix(
                image_a=image_a,
                label_a=label_a,
                image_b=image_b,
                label_b=label_b,
                num_classes=self.num_classes_to_mix,
                total_classes=self.total_classes,
            )
        else:
            mixed_image, mixed_label = image_a, label_a
        encoded = self.processor(images=mixed_image, segmentation_maps=mixed_label, return_tensors="pt")
        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels": encoded["labels"].squeeze(0),
        }


def build_datasets(config, processor, data_dir: "str | None" = None) -> DatasetBundle:
    root = Path(data_dir) if data_dir is not None else Path(config.data.dir)
    train_pairs = _resolve_split_pairs(root, "train")
    val_pairs = _resolve_split_pairs(root, "val")
    test_pairs = _resolve_optional_split_pairs(root, "test")

    train_base = CamVidDataset(
        image_paths=train_pairs[0],
        label_paths=train_pairs[1],
        transform=build_train_transform(config.augmentation),
        processor=processor,
        ignore_index=config.data.ignore_index,
    )
    train_dataset: torch.utils.data.Dataset = train_base
    if config.augmentation.strategy == "classmix":
        train_dataset = ClassMixDataset(
            base_dataset=train_base,
            classmix_prob=config.augmentation.classmix.prob,
            num_classes_to_mix=config.augmentation.classmix.num_classes,
            total_classes=config.model.num_labels,
        )

    val_dataset = CamVidDataset(
        image_paths=val_pairs[0],
        label_paths=val_pairs[1],
        transform=build_val_transform(config.augmentation),
        processor=processor,
        ignore_index=config.data.ignore_index,
    )
    test_dataset = None
    if test_pairs is not None:
        test_dataset = CamVidDataset(
            image_paths=test_pairs[0],
            label_paths=test_pairs[1],
            transform=build_val_transform(config.augmentation),
            processor=processor,
            ignore_index=config.data.ignore_index,
        )

    class_frequencies = None
    if config.loss.class_weights:
        class_frequencies = _compute_class_frequencies(train_pairs[1], ignore_index=config.data.ignore_index, num_classes=config.model.num_labels)

    return DatasetBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        class_frequencies=class_frequencies,
    )


def _resolve_split_pairs(root: Path, split: str) -> Tuple[List[Path], List[Path]]:
    image_dir = root / split
    label_dir = root / f"{split}annot"
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Missing annotation directory: {label_dir}")
    image_paths = sorted(image_dir.glob("*.png"))
    label_paths = sorted(label_dir.glob("*.png"))
    image_names = [path.name for path in image_paths]
    label_names = [path.name for path in label_paths]
    if image_names != label_names:
        raise RuntimeError(f"Image/label mismatch under split '{split}'.")
    return image_paths, label_paths


def _resolve_optional_split_pairs(root: Path, split: str) -> Optional[Tuple[List[Path], List[Path]]]:
    image_dir = root / split
    label_dir = root / f"{split}annot"
    if not image_dir.exists() and not label_dir.exists():
        return None
    return _resolve_split_pairs(root, split)


def _canonicalize_label(label: np.ndarray, ignore_index: int) -> np.ndarray:
    canonical = np.array(label, copy=True, dtype=np.uint8)
    canonical[canonical == 11] = ignore_index
    return canonical


def _compute_class_frequencies(
    label_paths: List[Path],
    ignore_index: int,
    num_classes: int,
) -> np.ndarray:
    counts = np.zeros((num_classes,), dtype=np.float64)
    for label_path in label_paths:
        label = _canonicalize_label(np.asarray(Image.open(label_path), dtype=np.uint8), ignore_index=ignore_index)
        valid = label != ignore_index
        values, value_counts = np.unique(label[valid], return_counts=True)
        for value, value_count in zip(values.tolist(), value_counts.tolist()):
            if 0 <= int(value) < num_classes:
                counts[int(value)] += float(value_count)
    total = counts.sum()
    if total <= 0:
        return np.ones((num_classes,), dtype=np.float64) / float(num_classes)
    return counts / total
