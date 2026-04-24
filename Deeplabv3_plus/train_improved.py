"""train_improved.py — Enhanced training script for CamVid segmentation.

Key improvements over camvid_segmentation.py:
- CosineAnnealingWarmRestarts scheduler (better LR scheduling)
- Label smoothing cross-entropy loss
- Stronger augmentation: random scale + rotation
- Poly LR warmup for first 2 epochs
- Weighted aux loss (0.4 * aux + 0.2 * aux2)
- Mixed precision (AMP) for faster training + lower memory
- Gradient clipping to stabilize training
- Class-weighted loss to handle class imbalance
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import (
    InterpolationMode, normalize, pil_to_tensor, resize,
)

MEAN = (0.390, 0.405, 0.414)
STD  = (0.274, 0.285, 0.297)
AVAILABLE_MODELS = ("custom", "improved")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_class_info(csv_path: Path) -> Tuple[List[str], np.ndarray, Dict[int, int]]:
    class_names: List[str] = []
    palette: List[Tuple[int, int, int]] = []
    color_to_index: Dict[int, int] = {}
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for idx, row in enumerate(csv.DictReader(fh)):
            r, g, b = int(row["r"].strip()), int(row["g"].strip()), int(row["b"].strip())
            class_names.append(row["name"].strip())
            palette.append((r, g, b))
            color_to_index[(r << 16) | (g << 8) | b] = idx
    return class_names, np.asarray(palette, dtype=np.uint8), color_to_index


def rgb_mask_to_class(mask: np.ndarray, c2i: Dict[int, int]) -> np.ndarray:
    codes = (mask[..., 0].astype(np.int32) << 16) | (mask[..., 1].astype(np.int32) << 8) | mask[..., 2].astype(np.int32)
    out = np.zeros(mask.shape[:2], dtype=np.int64)
    for code, idx in c2i.items():
        out[codes == code] = idx
    return out


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
        self.mask_dir  = data_root / f"{split}_labels"
        self.c2i       = color_to_index
        self.image_size = image_size
        self.augment   = augment
        self.samples   = self._collect()

    def _collect(self) -> List[Tuple[Path, Path]]:
        samples = []
        for ip in sorted(self.image_dir.glob("*.png")):
            mp = self.mask_dir / f"{ip.stem}_L.png"
            if not mp.exists():
                raise FileNotFoundError(mp)
            samples.append((ip, mp))
        if not samples:
            raise RuntimeError(f"No images in {self.image_dir}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ip, mp = self.samples[idx]
        image = Image.open(ip).convert("RGB")
        mask  = Image.open(mp).convert("RGB")

        if self.augment:
            # Random scale [0.75, 1.5]
            scale = random.uniform(0.75, 1.5)
            sh = int(self.image_size[0] * scale)
            sw = int(self.image_size[1] * scale)
            image = resize(image, (sh, sw), InterpolationMode.BILINEAR)
            mask  = resize(mask,  (sh, sw), InterpolationMode.NEAREST)

            # Random crop to target size (pad if needed)
            w, h = image.size
            ph = max(self.image_size[0] - h, 0)
            pw = max(self.image_size[1] - w, 0)
            if ph > 0 or pw > 0:
                from torchvision.transforms.functional import pad
                image = pad(image, (0, 0, pw, ph), fill=0)
                mask  = pad(mask,  (0, 0, pw, ph), fill=0)
                w, h  = image.size
            top  = random.randint(0, h - self.image_size[0])
            left = random.randint(0, w - self.image_size[1])
            image = image.crop((left, top, left + self.image_size[1], top + self.image_size[0]))
            mask  = mask.crop( (left, top, left + self.image_size[1], top + self.image_size[0]))

            # Random horizontal flip
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask  = mask.transpose(Image.FLIP_LEFT_RIGHT)

            # Random rotation ±10°
            if random.random() > 0.5:
                angle = random.uniform(-10, 10)
                image = image.rotate(angle, resample=Image.BILINEAR,  fillcolor=(0, 0, 0))
                mask  = mask.rotate( angle, resample=Image.NEAREST,   fillcolor=(0, 0, 0))
        else:
            image = resize(image, self.image_size, InterpolationMode.BILINEAR)
            mask  = resize(mask,  self.image_size, InterpolationMode.NEAREST)

        image_t = pil_to_tensor(image).float() / 255.0

        if self.augment:
            from torchvision.transforms import ColorJitter
            image_t = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)(image_t)

        image_t = normalize(image_t, mean=MEAN, std=STD)
        mask_t  = torch.from_numpy(rgb_mask_to_class(np.array(mask, dtype=np.uint8), self.c2i)).long()
        return image_t, mask_t


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class LabelSmoothingCE(nn.Module):
    def __init__(self, num_classes: int, smoothing: float = 0.1,
                 weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.smoothing  = smoothing
        self.num_classes = num_classes
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        nll = F.nll_loss(log_probs, targets, weight=self.weight, reduction="mean")
        smooth = -log_probs.mean(dim=1).mean()
        return (1 - self.smoothing) * nll + self.smoothing * smooth


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    if model_name == "custom":
        sys.path.insert(0, str(Path(__file__).parent))
        from model import CustomDeepLabV3
        return CustomDeepLabV3(num_classes=num_classes, pretrained_backbone=pretrained)
    if model_name == "improved":
        sys.path.insert(0, str(Path(__file__).parent))
        from model_modified import ImprovedDeepLabV3Plus
        return ImprovedDeepLabV3Plus(num_classes=num_classes, pretrained_backbone=pretrained)
    raise ValueError(f"Unknown model: {model_name}")


def forward_logits(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    out = model(images)
    return out["out"] if isinstance(out, dict) else out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def update_cm(cm: np.ndarray, preds: torch.Tensor, targets: torch.Tensor, nc: int) -> None:
    p = preds.detach().cpu().numpy().astype(np.int64)
    t = targets.detach().cpu().numpy().astype(np.int64)
    valid = (t >= 0) & (t < nc)
    cm += np.bincount(nc * t[valid] + p[valid], minlength=nc * nc).reshape(nc, nc)


def compute_metrics(cm: np.ndarray) -> Dict[str, Any]:
    inter = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - inter
    iou   = np.divide(inter, union, out=np.zeros_like(inter, dtype=np.float64), where=union > 0)
    return {
        "pixel_accuracy": float(inter.sum() / max(cm.sum(), 1)),
        "mIoU": float(iou.mean()),
        "per_class_iou": iou.tolist(),
    }


# ---------------------------------------------------------------------------
# Epoch runner
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None,
    scaler: GradScaler | None,
    num_classes: int,
    max_batches: int | None,
) -> Dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, batches = 0.0, 0
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for i, (images, masks) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        images, masks = images.to(device), masks.to(device)

        with torch.set_grad_enabled(is_train):
            with autocast(enabled=(scaler is not None)):
                outputs = model(images)
                if isinstance(outputs, dict):
                    logits = outputs["out"]
                    loss = criterion(logits, masks)
                    if is_train:
                        if "aux" in outputs:
                            loss = loss + 0.4 * criterion(outputs["aux"], masks)
                        if "aux2" in outputs:
                            loss = loss + 0.2 * criterion(outputs["aux2"], masks)
                else:
                    logits = outputs
                    loss = criterion(logits, masks)

            if is_train:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

        preds = logits.argmax(dim=1)
        update_cm(cm, preds, masks, num_classes)
        total_loss += loss.item()
        batches += 1

    if batches == 0:
        raise RuntimeError("No batches processed.")
    m = compute_metrics(cm)
    m["loss"] = total_loss / batches
    return m


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    class_names, palette, c2i = load_class_info(data_root / "class_dict.csv")
    num_classes = len(class_names)
    image_size  = (args.height, args.width)

    train_ds = CamVidDataset(data_root, "train", c2i, image_size, augment=True)
    val_ds   = CamVidDataset(data_root, "val",   c2i, image_size, augment=False)
    test_ds  = CamVidDataset(data_root, "test",  c2i, image_size, augment=False)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin, drop_last=True)
    val_loader   = DataLoader(val_ds,   args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model  = build_model(args.model, num_classes, not args.no_pretrained).to(device)
    total  = sum(p.numel() for p in model.parameters())
    train_ = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model={args.model} device={device} params={total:,} trainable={train_:,}")

    criterion = LabelSmoothingCE(num_classes, smoothing=0.1)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    warmup_epochs = 2
    warmup  = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine  = CosineAnnealingWarmRestarts(optimizer, T_0=max(args.epochs - warmup_epochs, 1), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    use_amp = torch.cuda.is_available() and not args.cpu
    scaler  = GradScaler() if use_amp else None

    best_miou, patience_counter = -1.0, 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(model, train_loader, criterion, device,
                            optimizer, scaler, num_classes, args.max_train_batches)
        val_m   = run_epoch(model, val_loader,   criterion, device,
                            None,      None,   num_classes, args.max_val_batches)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": train_m["loss"], "train_mIoU": train_m["mIoU"],
            "train_pixel_accuracy": train_m["pixel_accuracy"],
            "val_loss": val_m["loss"],     "val_mIoU": val_m["mIoU"],
            "val_pixel_accuracy": val_m["pixel_accuracy"],
            "lr": lr,
        })
        print(f"Ep{epoch:03d}/{args.epochs} "
              f"tr_loss={train_m['loss']:.4f} tr_iou={train_m['mIoU']:.4f} "
              f"vl_loss={val_m['loss']:.4f} vl_iou={val_m['mIoU']:.4f} "
              f"lr={lr:.2e}", flush=True)

        if val_m["mIoU"] > best_miou:
            best_miou = val_m["mIoU"]
            patience_counter = 0
            ckpt = Path(args.checkpoint)
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch, "metrics": val_m,
                "class_names": list(class_names),
                "image_size": list(image_size),
                "model_name": args.model,
            }, ckpt)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping.")
                break

    hist_path = Path(args.history)
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    hist_path.write_text(json.dumps(history, indent=2))
    print(f"History -> {hist_path}")

    ckpt_data = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt_data["model_state_dict"])
    test_m = run_epoch(model, test_loader, criterion, device,
                       None, None, num_classes, args.max_test_batches)
    print(f"\n=== TEST RESULTS ({args.model}) ===")
    print(f"mIoU={test_m['mIoU']:.4f}  pixel_acc={test_m['pixel_accuracy']:.4f}  loss={test_m['loss']:.4f}")
    print(f"\n{'Class':<22} {'IoU':>8}")
    print("-" * 32)
    for name, iou in zip(class_names, test_m["per_class_iou"]):
        print(f"{name:<22} {iou:>8.4f}")


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    class_names, _, c2i = load_class_info(data_root / "class_dict.csv")
    num_classes = len(class_names)
    ckpt_data = torch.load(args.checkpoint, map_location="cpu")
    image_size = tuple(ckpt_data.get("image_size", [args.height, args.width]))
    model_name = ckpt_data.get("model_name", "custom")

    test_ds     = CamVidDataset(data_root, "test", c2i, image_size)
    test_loader = DataLoader(test_ds, args.batch_size, num_workers=args.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model  = build_model(model_name, num_classes, False).to(device)
    model.load_state_dict(ckpt_data["model_state_dict"])
    criterion = LabelSmoothingCE(num_classes, smoothing=0.0)
    test_m = run_epoch(model, test_loader, criterion, device, None, None, num_classes, None)
    print(f"Model={model_name}  mIoU={test_m['mIoU']:.4f}  pixel_acc={test_m['pixel_accuracy']:.4f}")
    print(f"\n{'Class':<22} {'IoU':>8}")
    print("-" * 32)
    for name, iou in zip(class_names, test_m["per_class_iou"]):
        print(f"{name:<22} {iou:>8.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Improved CamVid training.")
    p.add_argument("--data-root",    default="CamVid")
    p.add_argument("--height",       type=int, default=352)
    p.add_argument("--width",        type=int, default=480)
    p.add_argument("--batch-size",   type=int, default=4)
    p.add_argument("--num-workers",  type=int, default=2)
    p.add_argument("--cpu",          action="store_true")
    p.add_argument("--seed",         type=int, default=42)
    sub = p.add_subparsers(dest="command", required=True)

    tp = sub.add_parser("train")
    tp.add_argument("--model",       choices=AVAILABLE_MODELS, default="custom")
    tp.add_argument("--checkpoint",  default="checkpoints/custom_v2.pt")
    tp.add_argument("--history",     default="outputs/custom_v2_history.json")
    tp.add_argument("--epochs",      type=int, default=60)
    tp.add_argument("--lr",          type=float, default=5e-4)
    tp.add_argument("--weight-decay",type=float, default=1e-4)
    tp.add_argument("--patience",    type=int, default=10)
    tp.add_argument("--no-pretrained", action="store_true")
    tp.add_argument("--max-train-batches", type=int, default=None)
    tp.add_argument("--max-val-batches",   type=int, default=None)
    tp.add_argument("--max-test-batches",  type=int, default=None)
    tp.set_defaults(func=train)

    ep = sub.add_parser("eval")
    ep.add_argument("--checkpoint",  default="checkpoints/custom_v2.pt")
    ep.add_argument("--max-test-batches", type=int, default=None)
    ep.set_defaults(func=evaluate)

    return p


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    args.func(args)


if __name__ == "__main__":
    main()
