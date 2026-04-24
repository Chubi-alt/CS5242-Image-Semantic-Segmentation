"""train_v3.py — Target: test mIoU >= 0.50

Key bottleneck analysis:
  - 7 classes with IoU=0 (6 are ultra-rare, <0.01% pixels)
  - Fixing zero-IoU classes to ~0.25 alone would push 0.4473 -> 0.50+

Strategy stack (all CNN-based):
1. **Inverse-frequency class weighting** — up-weights rare classes massively
2. **Focal Loss** (gamma=2) — focuses training on hard/misclassified pixels
3. **OHEM** (Online Hard Example Mining) — keep top-K hardest pixels per batch
4. **Combined loss** = 0.5*FocalLoss + 0.5*WeightedCE + OHEM
5. **Multi-scale training** (scales 0.5~2.0) + stronger augmentation
6. **CutMix** augmentation — pastes rare-class regions into other images
7. **Larger input size** 448×608 — more pixels per rare instance
8. **output_stride=8** for improved_v2 — finer feature map
9. **Poly LR** decay (standard in segmentation)
10. **Test-time augmentation** (TTA) at eval
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms.functional import (
    InterpolationMode, normalize, pil_to_tensor, resize,
)

MEAN = (0.390, 0.405, 0.414)
STD  = (0.274, 0.285, 0.297)
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Class weights from frequency
# ---------------------------------------------------------------------------

def compute_class_weights(data_root: Path, c2i: Dict[int,int], nc: int,
                           method: str = "inv_freq") -> torch.Tensor:
    counts = np.zeros(nc, dtype=np.float64)
    for fp in sorted((data_root / "train_labels").glob("*.png")):
        mask = np.array(Image.open(fp).convert("RGB"), dtype=np.int32)
        codes = (mask[...,0]<<16)|(mask[...,1]<<8)|mask[...,2]
        for code, idx in c2i.items():
            counts[idx] += (codes == code).sum()
    counts = np.maximum(counts, 1)
    if method == "inv_freq":
        w = 1.0 / counts
    elif method == "inv_sqrt":
        w = 1.0 / np.sqrt(counts)
    elif method == "median_freq":
        med = np.median(counts[counts > 100])
        w = med / counts
    else:
        w = np.ones(nc)
    w = w / w.sum() * nc          # normalize so mean=1
    w = np.clip(w, 0.01, 20.0)    # cap extreme weights
    return torch.tensor(w, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal loss for semantic segmentation with class weights.

    Standard focal loss: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    where p_t is the predicted probability for the true class (unweighted).
    """
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None,
                 ignore_index: int = -100) -> None:
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute log softmax and probabilities
        log_p = F.log_softmax(logits, dim=1)  # (N, C, H, W)
        prob = torch.exp(log_p)               # (N, C, H, W)

        # Get p_t: predicted probability for the true class
        N, C, H, W = logits.shape
        prob_flat = prob.permute(0, 2, 3, 1).reshape(-1, C)  # (N*H*W, C)
        targets_flat = targets.view(-1)                      # (N*H*W,)

        # Create mask for valid pixels (not ignore_index)
        valid_mask = targets_flat != self.ignore_index
        if valid_mask.sum() == 0:
            return logits.sum() * 0.0

        # Get p_t for valid pixels only
        prob_valid = prob_flat[valid_mask]        # (N_valid, C)
        targets_valid = targets_flat[valid_mask]  # (N_valid,)
        p_t = prob_valid.gather(1, targets_valid.unsqueeze(1)).squeeze(1)  # (N_valid,)

        # Get class weights for valid pixels
        if self.weight is not None:
            alpha_t = self.weight[targets_valid]  # (N_valid,)
        else:
            alpha_t = 1.0

        # Focal loss: -α_t * (1 - p_t)^γ * log(p_t)
        ce = F.nll_loss(log_p, targets, ignore_index=self.ignore_index, reduction="none")
        ce_valid = ce.view(-1)[valid_mask]  # (N_valid,)

        focal_weight = (1 - p_t) ** self.gamma
        loss = alpha_t * focal_weight * ce_valid
        return loss.mean()


class OHEMLoss(nn.Module):
    """Online Hard Example Mining — keep top-K% hardest pixels."""
    def __init__(self, thresh: float = 0.7, min_kept: int = 100_000,
                 weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.thresh    = thresh
        self.min_kept  = min_kept
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n, c, h, w = logits.shape
        log_p = F.log_softmax(logits, dim=1)
        loss_all = F.nll_loss(log_p, targets, weight=self.weight,
                              ignore_index=-100, reduction="none")  # (N,H,W)
        loss_flat = loss_all.view(-1)
        valid = loss_flat > 0
        loss_valid = loss_flat[valid]
        if loss_valid.numel() == 0:
            return loss_flat.mean()
        sorted_loss, _ = loss_valid.sort(descending=True)
        n_keep = max(self.min_kept, int(self.thresh * loss_valid.numel()))
        n_keep = min(n_keep, loss_valid.numel())
        thresh_val = sorted_loss[n_keep - 1]
        mask = loss_flat >= thresh_val
        return loss_flat[mask].mean()


class CombinedLoss(nn.Module):
    def __init__(self, nc: int, weight: torch.Tensor | None = None,
                 focal_gamma: float = 2.0, smoothing: float = 0.05) -> None:
        super().__init__()
        self.focal = FocalLoss(gamma=focal_gamma, weight=weight)
        self.ohem  = OHEMLoss(thresh=0.7, min_kept=50_000, weight=weight)
        self.smoothing = smoothing
        self.nc = nc
        self.register_buffer("weight", weight)

    def _smooth_ce(self, logits, targets):
        log_p = F.log_softmax(logits, dim=1)
        nll   = F.nll_loss(log_p, targets, weight=self.weight,
                           ignore_index=-100, reduction="mean")
        smooth = -log_p.mean(dim=1).mean()
        return (1 - self.smoothing) * nll + self.smoothing * smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (0.4 * self.focal(logits, targets)
              + 0.4 * self.ohem(logits, targets)
              + 0.2 * self._smooth_ce(logits, targets))


# ---------------------------------------------------------------------------
# Dataset with CutMix and multi-scale
# ---------------------------------------------------------------------------

def load_class_info(csv_path: Path):
    names, palette, c2i = [], [], {}
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for idx, row in enumerate(csv.DictReader(fh)):
            r,g,b = int(row["r"].strip()),int(row["g"].strip()),int(row["b"].strip())
            names.append(row["name"].strip()); palette.append((r,g,b))
            c2i[(r<<16)|(g<<8)|b] = idx
    return names, np.asarray(palette, dtype=np.uint8), c2i

def rgb_mask_to_class(mask: np.ndarray, c2i: Dict[int,int]) -> np.ndarray:
    codes = (mask[...,0].astype(np.int32)<<16)|(mask[...,1].astype(np.int32)<<8)|mask[...,2].astype(np.int32)
    out = np.zeros(mask.shape[:2], dtype=np.int64)
    for code, idx in c2i.items():
        out[codes==code] = idx
    return out


class CamVidDataset(Dataset):
    def __init__(self, data_root: Path, split: str, c2i: Dict[int,int],
                 image_size: Tuple[int,int], augment: bool = False) -> None:
        self.img_dir  = data_root / split
        self.mask_dir = data_root / f"{split}_labels"
        self.c2i      = c2i
        self.size     = image_size
        self.augment  = augment
        self.samples  = sorted(self.img_dir.glob("*.png"))
        if not self.samples:
            raise RuntimeError(f"No images in {self.img_dir}")

    def __len__(self): return len(self.samples)

    def _load(self, ip: Path):
        mp = self.mask_dir / f"{ip.stem}_L.png"
        return Image.open(ip).convert("RGB"), Image.open(mp).convert("RGB")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ip = self.samples[idx]
        image, mask = self._load(ip)

        if self.augment:
            # Multi-scale [0.5, 2.0]
            scale = random.choice([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
            sh, sw = int(self.size[0]*scale), int(self.size[1]*scale)
            image = resize(image, (sh,sw), InterpolationMode.BILINEAR)
            mask  = resize(mask,  (sh,sw), InterpolationMode.NEAREST)

            # Pad if smaller than target
            w,h = image.size
            ph, pw = max(self.size[0]-h,0), max(self.size[1]-w,0)
            if ph>0 or pw>0:
                from torchvision.transforms.functional import pad
                image = pad(image,(0,0,pw,ph),fill=0)
                mask  = pad(mask, (0,0,pw,ph),fill=0)
                w,h   = image.size

            # Random crop
            top  = random.randint(0, h-self.size[0])
            left = random.randint(0, w-self.size[1])
            image = image.crop((left,top,left+self.size[1],top+self.size[0]))
            mask  = mask.crop( (left,top,left+self.size[1],top+self.size[0]))

            # Flip
            if random.random()>0.5:
                image=image.transpose(Image.FLIP_LEFT_RIGHT)
                mask =mask.transpose( Image.FLIP_LEFT_RIGHT)

            # Rotation
            if random.random()>0.5:
                ang=random.uniform(-15,15)
                image=image.rotate(ang,resample=Image.BILINEAR, fillcolor=(0,0,0))
                mask =mask.rotate( ang,resample=Image.NEAREST,  fillcolor=(0,0,0))
        else:
            image = resize(image, self.size, InterpolationMode.BILINEAR)
            mask  = resize(mask,  self.size, InterpolationMode.NEAREST)

        t = pil_to_tensor(image).float()/255.0
        if self.augment:
            from torchvision.transforms import ColorJitter
            t = ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0.1)(t)
        t = normalize(t, MEAN, STD)
        m = torch.from_numpy(rgb_mask_to_class(np.array(mask,dtype=np.uint8),self.c2i)).long()
        return t, m


def cutmix_batch(images: torch.Tensor, masks: torch.Tensor,
                 alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """CutMix augmentation on a batch."""
    if random.random() > 0.5:
        return images, masks
    n, c, h, w = images.shape
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(n, device=images.device)

    cut_h = int(h * np.sqrt(1 - lam))
    cut_w = int(w * np.sqrt(1 - lam))
    cx    = random.randint(0, w)
    cy    = random.randint(0, h)
    x1, x2 = max(cx - cut_w//2, 0), min(cx + cut_w//2, w)
    y1, y2 = max(cy - cut_h//2, 0), min(cy + cut_h//2, h)

    images_new = images.clone()
    masks_new  = masks.clone()
    images_new[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
    masks_new[:,    y1:y2, x1:x2]  = masks[idx,    y1:y2, x1:x2]
    return images_new, masks_new


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(nc: int, pretrained: bool = True, output_stride: int = 8) -> nn.Module:
    from model_modified import ImprovedDeepLabV3Plus
    return ImprovedDeepLabV3Plus(nc, pretrained_backbone=pretrained,
                                  output_stride=output_stride, freeze_backbone=True)


def forward_logits(model, images):
    out = model(images)
    return out["out"] if isinstance(out, dict) else out


# ---------------------------------------------------------------------------
# TTA evaluation
# ---------------------------------------------------------------------------

def eval_with_tta(model: nn.Module, loader: DataLoader,
                  nc: int, device: torch.device) -> Tuple[float, float, list]:
    model.eval()
    scales = [0.75, 1.0, 1.25]
    cm = np.zeros((nc, nc), dtype=np.int64)

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            b, c, h, w = images.shape
            logits_sum = torch.zeros(b, nc, h, w, device=device)

            for scale in scales:
                sh, sw = int(h*scale), int(w*scale)
                img_s = F.interpolate(images, (sh,sw), mode="bilinear", align_corners=False)
                out   = forward_logits(model, img_s)
                out   = F.interpolate(out, (h,w), mode="bilinear", align_corners=False)
                logits_sum += F.softmax(out, dim=1)

                # horizontal flip
                img_f = torch.flip(img_s, dims=[3])
                out_f = forward_logits(model, img_f)
                out_f = torch.flip(out_f, dims=[3])
                out_f = F.interpolate(out_f, (h,w), mode="bilinear", align_corners=False)
                logits_sum += F.softmax(out_f, dim=1)

            preds = logits_sum.argmax(1).cpu().numpy().astype(np.int64)
            t     = masks.numpy().astype(np.int64)
            valid = (t>=0)&(t<nc)
            cm   += np.bincount(nc*t[valid]+preds[valid], minlength=nc*nc).reshape(nc,nc)

    inter = np.diag(cm)
    union = cm.sum(0)+cm.sum(1)-inter
    iou   = np.divide(inter,union,out=np.zeros_like(inter,dtype=np.float64),where=union>0)
    return float(inter.sum()/max(cm.sum(),1)), float(iou.mean()), iou.tolist()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def update_cm(cm, preds, targets, nc):
    p = preds.detach().cpu().numpy().astype(np.int64)
    t = targets.detach().cpu().numpy().astype(np.int64)
    valid = (t>=0)&(t<nc)
    cm += np.bincount(nc*t[valid]+p[valid], minlength=nc*nc).reshape(nc,nc)

def compute_metrics(cm):
    inter = np.diag(cm); union=cm.sum(0)+cm.sum(1)-inter
    iou = np.divide(inter,union,out=np.zeros_like(inter,dtype=np.float64),where=union>0)
    return {"pixel_accuracy":float(inter.sum()/max(cm.sum(),1)),
            "mIoU":float(iou.mean()), "per_class_iou":iou.tolist()}


# ---------------------------------------------------------------------------
# Poly LR
# ---------------------------------------------------------------------------

class PolyLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_iters: int, power: float = 0.9,
                 warmup_iters: int = 0, min_lr: float = 1e-6, last_epoch: int = -1):
        self.max_iters   = max_iters
        self.power       = power
        self.warmup_iters= warmup_iters
        self.min_lr      = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        it = self.last_epoch
        if it < self.warmup_iters:
            factor = (it+1) / max(self.warmup_iters, 1)
        else:
            factor = (1 - (it-self.warmup_iters)/max(self.max_iters-self.warmup_iters,1))**self.power
        factor = max(factor, self.min_lr / max(b for b in self.base_lrs))
        return [max(b*factor, self.min_lr) for b in self.base_lrs]


# ---------------------------------------------------------------------------
# Train epoch
# ---------------------------------------------------------------------------

def run_train_epoch(model, loader, criterion, optimizer, scaler,
                    scheduler, device, nc, max_batches=None):
    model.train()
    total_loss, batches = 0.0, 0
    cm = np.zeros((nc,nc), dtype=np.int64)

    for i, (images, masks) in enumerate(loader):
        if max_batches and i >= max_batches: break
        images, masks = images.to(device), masks.to(device)
        images, masks = cutmix_batch(images, masks, alpha=1.0)

        with autocast(enabled=(scaler is not None)):
            outputs = model(images)
            if isinstance(outputs, dict):
                logits = outputs["out"]
                loss = criterion(logits, masks)
                if "aux" in outputs:
                    loss = loss + 0.4*criterion(outputs["aux"], masks)
                if "aux2" in outputs:
                    loss = loss + 0.2*criterion(outputs["aux2"], masks)
            else:
                logits, loss = outputs, criterion(outputs, masks)

        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        update_cm(cm, logits.argmax(1), masks, nc)
        total_loss += loss.item(); batches += 1

    m = compute_metrics(cm); m["loss"] = total_loss/max(batches,1)
    return m


def run_val_epoch(model, loader, criterion, device, nc, max_batches=None):
    model.eval()
    total_loss, batches = 0.0, 0
    cm = np.zeros((nc,nc), dtype=np.int64)
    with torch.no_grad():
        for i,(images,masks) in enumerate(loader):
            if max_batches and i>=max_batches: break
            images,masks = images.to(device),masks.to(device)
            outputs = model(images)
            logits  = outputs["out"] if isinstance(outputs,dict) else outputs
            loss    = criterion(logits, masks)
            update_cm(cm, logits.argmax(1), masks, nc)
            total_loss += loss.item(); batches += 1
    m = compute_metrics(cm); m["loss"] = total_loss/max(batches,1)
    return m


# ---------------------------------------------------------------------------
# Main train
# ---------------------------------------------------------------------------

def train(args):
    set_seed(args.seed)
    data_root  = Path(args.data_root)
    names, palette, c2i = load_class_info(data_root/"class_dict.csv")
    nc         = len(names)
    image_size = (args.height, args.width)

    print("Computing class weights...")
    weights = compute_class_weights(data_root, c2i, nc, method="median_freq").to(
        torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    )
    print("  Top-5 weighted classes:")
    w_np = weights.cpu().numpy()
    for i in np.argsort(w_np)[-5:][::-1]:
        print(f"    {names[i]:<25} w={w_np[i]:.3f}")

    train_ds = CamVidDataset(data_root,"train",c2i,image_size,augment=True)
    val_ds   = CamVidDataset(data_root,"val",  c2i,image_size,augment=False)
    test_ds  = CamVidDataset(data_root,"test", c2i,image_size,augment=False)

    pin  = torch.cuda.is_available()
    device = torch.device("cuda" if pin and not args.cpu else "cpu")
    weights = weights.to(device)

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin, drop_last=True)
    val_loader   = DataLoader(val_ds,   args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin)

    model = build_model(nc, pretrained=not args.no_pretrained,
                        output_stride=args.output_stride).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"device={device}  params={total:,}  trainable={sum(p.numel() for p in trainable):,}")

    criterion = CombinedLoss(nc, weight=weights, focal_gamma=2.0, smoothing=0.05)

    # Separate LR for backbone vs head
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if any(x in name for x in ["layer2","layer3","layer4"]):
            backbone_params.append(p)
        else:
            head_params.append(p)
    optimizer = AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params,     "lr": args.lr},
    ], weight_decay=args.weight_decay)

    total_iters  = args.epochs * len(train_loader)
    warmup_iters = 2 * len(train_loader)
    scheduler = PolyLR(optimizer, max_iters=total_iters,
                       power=0.9, warmup_iters=warmup_iters, min_lr=1e-6)
    scaler = GradScaler() if pin else None

    best_miou, patience_counter = -1.0, 0
    history: List[Dict[str,Any]] = []

    for epoch in range(1, args.epochs+1):
        train_m = run_train_epoch(model, train_loader, criterion, optimizer,
                                  scaler, scheduler, device, nc, args.max_train_batches)
        val_m   = run_val_epoch(  model, val_loader,   criterion, device, nc, args.max_val_batches)
        lr = optimizer.param_groups[1]["lr"]

        history.append({"epoch":epoch,
            "train_loss":train_m["loss"],"train_mIoU":train_m["mIoU"],
            "train_pixel_accuracy":train_m["pixel_accuracy"],
            "val_loss":val_m["loss"],    "val_mIoU":val_m["mIoU"],
            "val_pixel_accuracy":val_m["pixel_accuracy"], "lr":lr})

        print(f"Ep{epoch:03d}/{args.epochs} "
              f"tr_loss={train_m['loss']:.4f} tr_iou={train_m['mIoU']:.4f} "
              f"vl_loss={val_m['loss']:.4f} vl_iou={val_m['mIoU']:.4f} "
              f"lr={lr:.2e}", flush=True)

        if val_m["mIoU"] > best_miou:
            best_miou = val_m["mIoU"]; patience_counter = 0
            ckpt = Path(args.checkpoint); ckpt.parent.mkdir(parents=True,exist_ok=True)
            torch.save({"model_state_dict":model.state_dict(),"epoch":epoch,
                        "metrics":val_m,"class_names":list(names),
                        "image_size":list(image_size),"model_name":"improved",
                        "output_stride":args.output_stride}, ckpt)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping."); break

    hist_path = Path(args.history)
    hist_path.parent.mkdir(parents=True,exist_ok=True)
    hist_path.write_text(json.dumps(history,indent=2))
    print(f"History -> {hist_path}")

    # Test with TTA
    print("Running TTA evaluation on test set...")
    ckpt_data = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt_data["model_state_dict"])
    pa, miou, per_cls = eval_with_tta(model, test_loader, nc, device)
    print(f"\n=== TEST RESULTS (v3 + TTA) ===")
    print(f"mIoU={miou:.4f}  pixel_acc={pa:.4f}")
    print(f"\n{'Class':<25} {'IoU':>8}")
    print("-"*35)
    for n, v in zip(names, per_cls):
        print(f"{n:<25} {v:>8.4f}")

    # Also save test results
    test_out = {"mIoU":miou,"pixel_accuracy":pa,"per_class_iou":per_cls}
    (Path(args.history).parent / "v3_test_results.json").write_text(json.dumps(test_out,indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root",     default="CamVid")
    p.add_argument("--height",        type=int, default=448)
    p.add_argument("--width",         type=int, default=608)
    p.add_argument("--batch-size",    type=int, default=3)
    p.add_argument("--num-workers",   type=int, default=2)
    p.add_argument("--cpu",           action="store_true")
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--checkpoint",    default="checkpoints/improved_v3.pt")
    p.add_argument("--history",       default="outputs/improved_v3_history.json")
    p.add_argument("--epochs",        type=int, default=80)
    p.add_argument("--lr",            type=float, default=5e-4)
    p.add_argument("--weight-decay",  type=float, default=5e-4)
    p.add_argument("--patience",      type=int, default=15)
    p.add_argument("--output-stride", type=int, default=8, choices=[8,16])
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-val-batches",   type=int, default=None)
    p.set_defaults(func=train)
    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
