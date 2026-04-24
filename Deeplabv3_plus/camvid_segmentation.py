#!/usr/bin/env python3
"""CamVid semantic segmentation — training, evaluation, comparison & plotting.

Supports two DeepLabV3 backends via ``--model``:
  * ``custom``      — hand-implemented DeepLabV3 from ``model.py``
  * ``torchvision`` — ``torchvision.models.segmentation.deeplabv3_resnet50``

Sub-commands
------------
  train    Train a model on CamVid.
  eval     Evaluate a saved checkpoint on the test split.
  predict  Predict a segmentation mask for a single image.
  compare  Compare two checkpoints side-by-side (metrics + visualisations).
  plot     Plot training curves from one or more history JSON files.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    import torch
    from torch import nn
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch.utils.data import DataLoader, Dataset
    from torchvision.transforms.functional import InterpolationMode
    from torchvision.transforms.functional import normalize, pil_to_tensor, resize
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install the training stack first, for example:\n"
        "python -m pip install torch torchvision pillow numpy"
    ) from exc


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

MEAN = (0.390, 0.405, 0.414)
STD = (0.274, 0.285, 0.297)
AVAILABLE_MODELS = ("custom", "torchvision", "improved")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
#  Dataset helpers
# ---------------------------------------------------------------------------

def load_class_info(csv_path: Path) -> Tuple[List[str], np.ndarray, Dict[int, int]]:
    class_names: List[str] = []
    palette: List[Tuple[int, int, int]] = []
    color_to_index: Dict[int, int] = {}

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            r = int(row["r"].strip())
            g = int(row["g"].strip())
            b = int(row["b"].strip())
            class_names.append(row["name"].strip())
            palette.append((r, g, b))
            color_to_index[(r << 16) | (g << 8) | b] = idx

    return class_names, np.asarray(palette, dtype=np.uint8), color_to_index


def rgb_mask_to_class(mask: np.ndarray, color_to_index: Dict[int, int]) -> np.ndarray:
    flat_codes = (
        (mask[..., 0].astype(np.int32) << 16)
        | (mask[..., 1].astype(np.int32) << 8)
        | mask[..., 2].astype(np.int32)
    )
    class_mask = np.zeros(mask.shape[:2], dtype=np.int64)
    for color_code, class_idx in color_to_index.items():
        class_mask[flat_codes == color_code] = class_idx
    return class_mask


def class_mask_to_rgb(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    return palette[mask]


class CamVidDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split: str,
        color_to_index: Dict[int, int],
        image_size: Tuple[int, int],
        augment: bool = False,
    ) -> None:
        self.image_dir = data_root / split
        self.mask_dir = data_root / f"{split}_labels"
        self.color_to_index = color_to_index
        self.image_size = image_size
        self.augment = augment
        self.samples = self._collect_samples()

    def _collect_samples(self) -> List[Tuple[Path, Path]]:
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        samples: List[Tuple[Path, Path]] = []
        for image_path in sorted(self.image_dir.glob("*.png")):
            mask_path = self.mask_dir / f"{image_path.stem}_L.png"
            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Missing mask for {image_path.name}: {mask_path}"
                )
            samples.append((image_path, mask_path))

        if not samples:
            raise RuntimeError(f"No PNG images found in {self.image_dir}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _sync_hflip(image: Image.Image, mask: Image.Image):
        """Horizontal flip applied identically to image and mask."""
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return image, mask

    @staticmethod
    def _sync_random_crop(
        image: Image.Image, mask: Image.Image,
        crop_h: int, crop_w: int,
    ):
        """Random crop applied identically to image and mask."""
        w, h = image.size
        if h <= crop_h or w <= crop_w:
            return image, mask
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)
        image = image.crop((left, top, left + crop_w, top + crop_h))
        mask = mask.crop((left, top, left + crop_w, top + crop_h))
        return image, mask

    @staticmethod
    def _color_jitter(image_tensor: torch.Tensor) -> torch.Tensor:
        """Random brightness / contrast / saturation on the image tensor."""
        from torchvision.transforms import ColorJitter
        jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        return jitter(image_tensor)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.samples[index]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        if self.augment:
            # Resize to slightly larger than target, then random-crop
            up_h = int(self.image_size[0] * 1.15)
            up_w = int(self.image_size[1] * 1.15)
            image = resize(image, (up_h, up_w), interpolation=InterpolationMode.BILINEAR)
            mask = resize(mask, (up_h, up_w), interpolation=InterpolationMode.NEAREST)
            image, mask = self._sync_random_crop(
                image, mask, self.image_size[0], self.image_size[1],
            )
            image, mask = self._sync_hflip(image, mask)
        else:
            image = resize(image, self.image_size, interpolation=InterpolationMode.BILINEAR)
            mask = resize(mask, self.image_size, interpolation=InterpolationMode.NEAREST)

        image_tensor = pil_to_tensor(image).float() / 255.0

        if self.augment:
            image_tensor = self._color_jitter(image_tensor)

        image_tensor = normalize(image_tensor, mean=MEAN, std=STD)

        mask_array = np.array(mask, dtype=np.uint8)
        class_mask = rgb_mask_to_class(mask_array, self.color_to_index)
        mask_tensor = torch.from_numpy(class_mask).long()

        return image_tensor, mask_tensor


# ---------------------------------------------------------------------------
#  Model factory
# ---------------------------------------------------------------------------

def build_model(
    model_name: str, num_classes: int, pretrained_backbone: bool,
) -> nn.Module:
    """Instantiate a segmentation model by name."""
    if model_name == "custom":
        try:
            from model import CustomDeepLabV3
        except ImportError:
            raise ImportError(
                "Cannot import CustomDeepLabV3 from model.py. "
                "Make sure model.py is in the same directory or on PYTHONPATH."
            )
        return CustomDeepLabV3(
            num_classes=num_classes,
            pretrained_backbone=pretrained_backbone,
        )

    if model_name == "improved":
        try:
            from model_modified import ImprovedDeepLabV3Plus
        except ImportError:
            raise ImportError(
                "Cannot import ImprovedDeepLabV3Plus from model_modified.py. "
                "Make sure model_modified.py is in the same directory or on PYTHONPATH."
            )
        return ImprovedDeepLabV3Plus(
            num_classes=num_classes,
            pretrained_backbone=pretrained_backbone,
        )

    if model_name == "torchvision":
        return _torchvision_deeplabv3(num_classes, pretrained_backbone)

    raise ValueError(
        f"Unknown model '{model_name}'. Choose from {AVAILABLE_MODELS}"
    )


def _torchvision_deeplabv3(num_classes: int, pretrained_backbone: bool) -> nn.Module:
    from torchvision.models.segmentation import deeplabv3_resnet50

    # New API (torchvision >= 0.13): weights / weights_backbone
    try:
        from torchvision.models import ResNet50_Weights
        bb = ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        model = deeplabv3_resnet50(weights=None, weights_backbone=bb)
    except (ImportError, TypeError, ValueError):
        # Old API
        try:
            model = deeplabv3_resnet50(
                pretrained=False, pretrained_backbone=pretrained_backbone,
            )
        except TypeError:
            model = deeplabv3_resnet50(pretrained=False)

    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model


def forward_logits(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    outputs = model(images)
    if isinstance(outputs, dict):
        return outputs["out"]
    return outputs


# ---------------------------------------------------------------------------
#  Metrics
# ---------------------------------------------------------------------------

def update_confusion_matrix(
    confusion: np.ndarray,
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> None:
    preds_np = preds.detach().cpu().numpy().astype(np.int64)
    targets_np = targets.detach().cpu().numpy().astype(np.int64)
    valid = (targets_np >= 0) & (targets_np < num_classes)
    labels = num_classes * targets_np[valid] + preds_np[valid]
    confusion += np.bincount(
        labels, minlength=num_classes**2
    ).reshape(num_classes, num_classes)


def compute_metrics(confusion: np.ndarray) -> Dict[str, Any]:
    intersection = np.diag(confusion)
    union = confusion.sum(axis=0) + confusion.sum(axis=1) - intersection
    iou = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float64),
        where=union > 0,
    )
    pixel_accuracy = intersection.sum() / max(confusion.sum(), 1)
    return {
        "pixel_accuracy": float(pixel_accuracy),
        "mIoU": float(iou.mean()),
        "per_class_iou": iou.tolist(),
    }


# ---------------------------------------------------------------------------
#  Training / evaluation loop
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Adam | None,
    num_classes: int,
    max_batches: int | None,
) -> Dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    batches = 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for batch_idx, (images, masks) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device)
        masks = masks.to(device)

        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            if isinstance(outputs, dict):
                logits = outputs["out"]
                aux_logits = outputs.get("aux")
            else:
                logits = outputs
                aux_logits = None

            loss = criterion(logits, masks)
            if aux_logits is not None and is_train:
                loss = loss + 0.4 * criterion(aux_logits, masks)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        preds = logits.argmax(dim=1)
        update_confusion_matrix(confusion, preds, masks, num_classes)
        total_loss += loss.item()
        batches += 1

    if batches == 0:
        raise RuntimeError(
            "No batches were processed. Check dataset paths and batch settings."
        )

    metrics = compute_metrics(confusion)
    metrics["loss"] = total_loss / batches
    return metrics


# ---------------------------------------------------------------------------
#  Visualisation helpers
# ---------------------------------------------------------------------------

def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image = image * np.asarray(STD) + np.asarray(MEAN)
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def save_visualizations(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    palette: np.ndarray,
    output_dir: Path,
    limit: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    model.eval()

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            logits = forward_logits(model, images)
            preds = logits.argmax(dim=1).cpu()

            for idx in range(images.shape[0]):
                image_rgb = denormalize_image(images[idx])
                gt_rgb = class_mask_to_rgb(masks[idx].cpu().numpy(), palette)
                pred_rgb = class_mask_to_rgb(preds[idx].numpy(), palette)
                canvas = np.concatenate([image_rgb, gt_rgb, pred_rgb], axis=1)
                Image.fromarray(canvas).save(
                    output_dir / f"sample_{saved:03d}.png"
                )
                saved += 1
                if saved >= limit:
                    return


def save_comparison_samples(
    model_a: nn.Module,
    model_b: nn.Module,
    loader: DataLoader,
    device: torch.device,
    palette: np.ndarray,
    label_a: str,
    label_b: str,
    output_dir: Path,
    limit: int,
) -> None:
    """Save [Original | GT | Pred-A | Pred-B] side-by-side images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    model_a.eval()
    model_b.eval()

    with torch.no_grad():
        for images, masks in loader:
            images_dev = images.to(device)
            preds_a = forward_logits(model_a, images_dev).argmax(dim=1).cpu()
            preds_b = forward_logits(model_b, images_dev).argmax(dim=1).cpu()

            for idx in range(images.shape[0]):
                img = denormalize_image(images[idx])
                gt = class_mask_to_rgb(masks[idx].numpy(), palette)
                pa = class_mask_to_rgb(preds_a[idx].numpy(), palette)
                pb = class_mask_to_rgb(preds_b[idx].numpy(), palette)
                canvas = np.concatenate([img, gt, pa, pb], axis=1)
                Image.fromarray(canvas).save(
                    output_dir / f"compare_{saved:03d}.png"
                )
                saved += 1
                if saved >= limit:
                    print(
                        f"Saved {saved} comparison samples to {output_dir}\n"
                        f"  Columns: Original | GT | {label_a} | {label_b}"
                    )
                    return

    print(
        f"Saved {saved} comparison samples to {output_dir}\n"
        f"  Columns: Original | GT | {label_a} | {label_b}"
    )


# ---------------------------------------------------------------------------
#  DataLoader builder
# ---------------------------------------------------------------------------

def build_loaders(
    data_root: Path,
    color_to_index: Dict[int, int],
    image_size: Tuple[int, int],
    batch_size: int,
    num_workers: int,
    augment: bool = True,
) -> Tuple[CamVidDataset, CamVidDataset, CamVidDataset,
           DataLoader, DataLoader, DataLoader]:
    train_dataset = CamVidDataset(
        data_root, "train", color_to_index, image_size, augment=augment,
    )
    val_dataset = CamVidDataset(data_root, "val", color_to_index, image_size)
    test_dataset = CamVidDataset(data_root, "test", color_to_index, image_size)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    return (train_dataset, val_dataset, test_dataset,
            train_loader, val_loader, test_loader)


# ---------------------------------------------------------------------------
#  Checkpoint I/O
# ---------------------------------------------------------------------------

def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Adam,
    epoch: int,
    metrics: Dict[str, Any],
    class_names: Sequence[str],
    image_size: Tuple[int, int],
    model_name: str,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "class_names": list(class_names),
            "image_size": list(image_size),
            "model_name": model_name,
        },
        checkpoint_path,
    )


# ---------------------------------------------------------------------------
#  Sub-command: train
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    class_names, palette, color_to_index = load_class_info(data_root / "class_dict.csv")
    num_classes = len(class_names)
    image_size = (args.height, args.width)
    model_name = args.model

    if args.batch_size < 2:
        raise ValueError(
            "batch_size >= 2 required (BatchNorm fails on single-sample batches)."
        )

    use_augment = not getattr(args, "no_augment", False)
    (_, _, _, train_loader, val_loader, test_loader) = build_loaders(
        data_root, color_to_index, image_size, args.batch_size, args.num_workers,
        augment=use_augment,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    pretrained = not args.no_pretrained
    model = build_model(model_name, num_classes, pretrained).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Model: {model_name} | pretrained_backbone: {pretrained} | device: {device}\n"
        f"  Total params: {total_params:,} | Trainable: {train_params:,} "
        f"({100 * train_params / max(total_params, 1):.1f}%)"
    )

    criterion = nn.CrossEntropyLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_miou = -1.0
    patience_counter = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(
            model, train_loader, criterion, device,
            optimizer, num_classes, args.max_train_batches,
        )
        val_m = run_epoch(
            model, val_loader, criterion, device,
            None, num_classes, args.max_val_batches,
        )

        scheduler.step(val_m["loss"])
        lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": train_m["loss"],
            "train_pixel_accuracy": train_m["pixel_accuracy"],
            "train_mIoU": train_m["mIoU"],
            "val_loss": val_m["loss"],
            "val_pixel_accuracy": val_m["pixel_accuracy"],
            "val_mIoU": val_m["mIoU"],
            "lr": lr,
        })

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_m['loss']:.4f} "
            f"train_mIoU={train_m['mIoU']:.4f} "
            f"val_loss={val_m['loss']:.4f} "
            f"val_mIoU={val_m['mIoU']:.4f} "
            f"val_pixel_acc={val_m['pixel_accuracy']:.4f} "
            f"lr={lr:.2e}"
        )

        if val_m["mIoU"] > best_miou:
            best_miou = val_m["mIoU"]
            patience_counter = 0
            save_checkpoint(
                Path(args.checkpoint), model, optimizer, epoch,
                val_m, class_names, image_size, model_name,
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    # Save training history
    history_path = Path(args.history)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Training history saved to {history_path}")

    # Evaluate best checkpoint on test set
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_m = run_epoch(
        model, test_loader, criterion, device,
        None, num_classes, args.max_test_batches,
    )
    print(
        f"Test | loss={test_m['loss']:.4f} "
        f"mIoU={test_m['mIoU']:.4f} "
        f"pixel_acc={test_m['pixel_accuracy']:.4f}"
    )

    # Per-class IoU
    _print_per_class_iou(class_names, test_m["per_class_iou"])

    if args.save_samples > 0:
        save_visualizations(
            model, test_loader, device, palette,
            Path(args.samples_dir), args.save_samples,
        )


# ---------------------------------------------------------------------------
#  Sub-command: eval
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    class_names, palette, color_to_index = load_class_info(data_root / "class_dict.csv")
    num_classes = len(class_names)
    image_size = (args.height, args.width)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if "image_size" in checkpoint:
        image_size = tuple(checkpoint["image_size"])
    model_name = checkpoint.get("model_name", "torchvision")

    (_, _, _, _, _, test_loader) = build_loaders(
        data_root, color_to_index, image_size, args.batch_size, args.num_workers,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    model = build_model(model_name, num_classes, pretrained_backbone=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss()

    test_m = run_epoch(
        model, test_loader, criterion, device,
        None, num_classes, args.max_test_batches,
    )

    print(f"\nModel : {model_name}")
    print(f"mIoU  : {test_m['mIoU']:.4f}")
    print(f"Pixel : {test_m['pixel_accuracy']:.4f}")
    print(f"Loss  : {test_m['loss']:.4f}")
    _print_per_class_iou(class_names, test_m["per_class_iou"])

    if args.save_samples > 0:
        save_visualizations(
            model, test_loader, device, palette,
            Path(args.samples_dir), args.save_samples,
        )


# ---------------------------------------------------------------------------
#  Sub-command: predict
# ---------------------------------------------------------------------------

def predict(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    class_names, palette, _ = load_class_info(data_root / "class_dict.csv")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    image_size = tuple(checkpoint.get("image_size", [args.height, args.width]))
    model_name = checkpoint.get("model_name", "torchvision")

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    model = build_model(
        model_name, len(class_names), pretrained_backbone=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image = Image.open(args.image).convert("RGB")
    original_size = image.size
    resized = resize(image, image_size, interpolation=InterpolationMode.BILINEAR)
    image_tensor = pil_to_tensor(resized).float() / 255.0
    image_tensor = normalize(image_tensor, mean=MEAN, std=STD).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = forward_logits(model, image_tensor)
        pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)

    pred_rgb = class_mask_to_rgb(pred_mask, palette)
    pred_image = Image.fromarray(pred_rgb).resize(
        original_size, resample=Image.Resampling.NEAREST,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_image.save(output_path)
    print(f"Saved prediction to {output_path}")


# ---------------------------------------------------------------------------
#  Sub-command: compare
# ---------------------------------------------------------------------------

def compare(args: argparse.Namespace) -> None:
    """Compare two checkpoints on the test set."""
    data_root = Path(args.data_root)
    class_names, palette, color_to_index = load_class_info(data_root / "class_dict.csv")
    num_classes = len(class_names)
    image_size = (args.height, args.width)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    ckpt_a = torch.load(args.checkpoint_a, map_location=device)
    ckpt_b = torch.load(args.checkpoint_b, map_location=device)
    name_a = ckpt_a.get("model_name", "torchvision")
    name_b = ckpt_b.get("model_name", "torchvision")
    label_a = args.label_a or name_a
    label_b = args.label_b or name_b

    model_a = build_model(name_a, num_classes, pretrained_backbone=False).to(device)
    model_a.load_state_dict(ckpt_a["model_state_dict"])

    model_b = build_model(name_b, num_classes, pretrained_backbone=False).to(device)
    model_b.load_state_dict(ckpt_b["model_state_dict"])

    (_, _, _, _, _, test_loader) = build_loaders(
        data_root, color_to_index, image_size, args.batch_size, args.num_workers,
    )
    criterion = nn.CrossEntropyLoss()

    print(f"Evaluating [{label_a}] ...")
    m_a = run_epoch(model_a, test_loader, criterion, device, None, num_classes, None)
    print(f"Evaluating [{label_b}] ...")
    m_b = run_epoch(model_b, test_loader, criterion, device, None, num_classes, None)

    # ---- Metrics summary table ----
    print(f"\n{'Metric':<20} {label_a:<16} {label_b:<16} {'Diff':>8}")
    print("-" * 62)
    for key in ("mIoU", "pixel_accuracy", "loss"):
        va, vb = m_a[key], m_b[key]
        diff = vb - va
        sign = "+" if diff >= 0 else ""
        print(f"{key:<20} {va:<16.4f} {vb:<16.4f} {sign}{diff:>7.4f}")

    # ---- Per-class IoU table ----
    iou_a, iou_b = m_a["per_class_iou"], m_b["per_class_iou"]
    print(f"\n{'Class':<22} {label_a:<12} {label_b:<12} {'Diff':>8}")
    print("-" * 56)
    for name, ia, ib in zip(class_names, iou_a, iou_b):
        diff = ib - ia
        sign = "+" if diff >= 0 else ""
        print(f"{name:<22} {ia:<12.4f} {ib:<12.4f} {sign}{diff:>7.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Side-by-side sample images ----
    if args.save_samples > 0:
        save_comparison_samples(
            model_a, model_b, test_loader, device, palette,
            label_a, label_b, output_dir / "samples", args.save_samples,
        )

    # ---- Per-class IoU bar chart ----
    _plot_iou_comparison(
        iou_a, iou_b, class_names, label_a, label_b,
        output_dir / "per_class_iou.png",
    )


# ---------------------------------------------------------------------------
#  Sub-command: plot
# ---------------------------------------------------------------------------

def plot_curves(args: argparse.Namespace) -> None:
    """Plot training curves from one or more history JSON files."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit(
            "Install matplotlib first: pip install matplotlib"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    histories = []
    for hp in args.history_files:
        histories.append(json.loads(Path(hp).read_text(encoding="utf-8")))

    labels = (
        args.labels if args.labels
        else [Path(p).stem for p in args.history_files]
    )
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, (history, label) in enumerate(zip(histories, labels)):
        epochs = [h["epoch"] for h in history]
        c = colors[i % len(colors)]
        axes[0].plot(
            epochs, [h["train_loss"] for h in history],
            "--", color=c, alpha=0.6, label=f"{label} train",
        )
        axes[0].plot(
            epochs, [h["val_loss"] for h in history],
            "-", color=c, label=f"{label} val",
        )
        axes[1].plot(
            epochs, [h["train_mIoU"] for h in history],
            "--", color=c, alpha=0.6, label=f"{label} train",
        )
        axes[1].plot(
            epochs, [h["val_mIoU"] for h in history],
            "-", color=c, label=f"{label} val",
        )

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mIoU")
    axes[1].set_title("Training & Validation mIoU")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    save_path = output_dir / "training_curves.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved training curves to {save_path}")


# ---------------------------------------------------------------------------
#  Printing & plotting helpers
# ---------------------------------------------------------------------------

def _print_per_class_iou(class_names: List[str], per_class_iou: list) -> None:
    print(f"\n{'Class':<22} {'IoU':>8}")
    print("-" * 32)
    for name, iou in zip(class_names, per_class_iou):
        print(f"{name:<22} {iou:>8.4f}")


def _plot_iou_comparison(
    iou_a: list,
    iou_b: list,
    class_names: List[str],
    label_a: str,
    label_b: str,
    output_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib for per-class IoU charts: pip install matplotlib")
        return

    x = np.arange(len(class_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - width / 2, iou_a, width, label=label_a, alpha=0.85)
    ax.bar(x + width / 2, iou_b, width, label=label_b, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("IoU")
    ax.set_title("Per-class IoU Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved per-class IoU chart to {output_path}")


# ---------------------------------------------------------------------------
#  Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CamVid semantic segmentation with DeepLabV3.",
    )
    parser.add_argument("--data-root", default="CamVid",
                        help="CamVid dataset root directory.")
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU even if CUDA is available.")
    parser.add_argument("--seed", type=int, default=42)

    sub = parser.add_subparsers(dest="command", required=True)

    # ---- train ----
    tp = sub.add_parser("train", help="Train on CamVid.")
    tp.add_argument("--checkpoint", default="checkpoints/camvid_deeplabv3.pt")
    tp.add_argument("--model", choices=AVAILABLE_MODELS, default="custom",
                    help="Which DeepLabV3 implementation to use.")
    tp.add_argument("--epochs", type=int, default=40)
    tp.add_argument("--lr", type=float, default=1e-4)
    tp.add_argument("--weight-decay", type=float, default=1e-4)
    tp.add_argument("--patience", type=int, default=5)
    tp.add_argument("--no-pretrained", action="store_true",
                    help="Do NOT load ImageNet backbone weights.")
    tp.add_argument("--no-augment", action="store_true",
                    help="Disable training data augmentation.")
    tp.add_argument("--history", default="outputs/train_history.json")
    tp.add_argument("--samples-dir", default="outputs/test_samples")
    tp.add_argument("--save-samples", type=int, default=4)
    tp.add_argument("--max-train-batches", type=int, default=None)
    tp.add_argument("--max-val-batches", type=int, default=None)
    tp.add_argument("--max-test-batches", type=int, default=None)
    tp.set_defaults(func=train)

    # ---- eval ----
    ep = sub.add_parser("eval",
                        help="Evaluate a saved checkpoint on the test split.")
    ep.add_argument("--checkpoint", default="checkpoints/camvid_deeplabv3.pt")
    ep.add_argument("--samples-dir", default="outputs/eval_samples")
    ep.add_argument("--save-samples", type=int, default=4)
    ep.add_argument("--max-test-batches", type=int, default=None)
    ep.set_defaults(func=evaluate)

    # ---- predict ----
    pp = sub.add_parser("predict", help="Predict a mask for one image.")
    pp.add_argument("--checkpoint", default="checkpoints/camvid_deeplabv3.pt")
    pp.add_argument("--image", required=True, help="Path to the input image.")
    pp.add_argument("--output", default="outputs/prediction.png")
    pp.set_defaults(func=predict)

    # ---- compare ----
    cp = sub.add_parser("compare",
                        help="Compare two checkpoints on the test set.")
    cp.add_argument("--checkpoint-a", required=True, help="First checkpoint.")
    cp.add_argument("--checkpoint-b", required=True, help="Second checkpoint.")
    cp.add_argument("--label-a", default=None, help="Display label for model A.")
    cp.add_argument("--label-b", default=None, help="Display label for model B.")
    cp.add_argument("--output-dir", default="outputs/comparison")
    cp.add_argument("--save-samples", type=int, default=6)
    cp.set_defaults(func=compare)

    # ---- plot ----
    plp = sub.add_parser("plot",
                         help="Plot training curves from history JSON files.")
    plp.add_argument("--history-files", nargs="+", required=True,
                     help="One or more training history JSON files.")
    plp.add_argument("--labels", nargs="+", default=None,
                     help="Legend labels (same order as --history-files).")
    plp.add_argument("--output-dir", default="outputs/plots")
    plp.set_defaults(func=plot_curves)

    return parser


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    args.func(args)


if __name__ == "__main__":
    main()
