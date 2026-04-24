"""analyze.py — Generate all tables, plots, and visualizations for the report.

Outputs (all in outputs/report/):
  tables/
    model_comparison.csv        — overall metrics for all 5 runs
    per_class_iou.csv           — per-class IoU for all models
    overfitting_summary.csv     — train/val gap analysis
  plots/
    training_curves_loss.png    — loss curves all models
    training_curves_miou.png    — mIoU curves all models
    overfitting_gap.png         — train-val gap over epochs
    per_class_iou_bar.png       — per-class IoU comparison (v2 models)
    lr_schedule.png             — LR schedule comparison
  samples/                      — prediction visualizations
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms.functional import InterpolationMode, normalize, pil_to_tensor, resize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT   = Path(__file__).parent
OUT    = ROOT / "outputs" / "report"
CKPT   = ROOT / "checkpoints"
DATA   = ROOT / "CamVid"
MEAN   = (0.390, 0.405, 0.414)
STD    = (0.274, 0.285, 0.297)

(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "plots").mkdir(parents=True, exist_ok=True)
(OUT / "samples").mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_class_info(csv_path):
    names, palette, c2i = [], [], {}
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for idx, row in enumerate(csv.DictReader(fh)):
            r,g,b = int(row["r"].strip()), int(row["g"].strip()), int(row["b"].strip())
            names.append(row["name"].strip())
            palette.append((r,g,b))
            c2i[(r<<16)|(g<<8)|b] = idx
    return names, np.asarray(palette, dtype=np.uint8), c2i

def rgb_mask_to_class(mask, c2i):
    codes = (mask[...,0].astype(np.int32)<<16)|(mask[...,1].astype(np.int32)<<8)|mask[...,2].astype(np.int32)
    out = np.zeros(mask.shape[:2], dtype=np.int64)
    for code,idx in c2i.items():
        out[codes==code] = idx
    return out

def denorm(t):
    img = t.cpu().numpy().transpose(1,2,0)
    img = img * np.array(STD) + np.array(MEAN)
    return np.clip(img*255, 0, 255).astype(np.uint8)

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, c2i, image_size):
        self.img_dir  = data_root / "test"
        self.mask_dir = data_root / "test_labels"
        self.c2i      = c2i
        self.size     = image_size
        self.samples  = sorted(self.img_dir.glob("*.png"))
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        ip = self.samples[i]
        mp = self.mask_dir / f"{ip.stem}_L.png"
        image = Image.open(ip).convert("RGB")
        mask  = Image.open(mp).convert("RGB")
        image = resize(image, self.size, InterpolationMode.BILINEAR)
        mask  = resize(mask,  self.size, InterpolationMode.NEAREST)
        t = pil_to_tensor(image).float()/255.0
        t = normalize(t, MEAN, STD)
        m = torch.from_numpy(rgb_mask_to_class(np.array(mask,dtype=np.uint8), self.c2i)).long()
        return t, m, ip.name

def forward_logits(model, images):
    out = model(images)
    return out["out"] if isinstance(out, dict) else out

def compute_metrics_from_cm(cm):
    inter = np.diag(cm)
    union = cm.sum(0)+cm.sum(1)-inter
    iou   = np.divide(inter, union, out=np.zeros_like(inter,dtype=np.float64), where=union>0)
    return float(inter.sum()/max(cm.sum(),1)), float(iou.mean()), iou.tolist()

def eval_model(model, loader, nc, device):
    model.eval()
    cm = np.zeros((nc,nc), dtype=np.int64)
    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device)
            preds  = forward_logits(model, images).argmax(1).cpu().numpy().astype(np.int64)
            t      = masks.numpy().astype(np.int64)
            valid  = (t>=0)&(t<nc)
            cm    += np.bincount(nc*t[valid]+preds[valid], minlength=nc*nc).reshape(nc,nc)
    return compute_metrics_from_cm(cm)

def build_model_v1_custom(nc):
    """ResNet-50 based CustomDeepLabV3 matching old checkpoint structure."""
    from torchvision.models import resnet50
    class _ASPPConvV1(nn.Sequential):
        def __init__(self, in_ch, out_ch, dilation):
            super().__init__(nn.Conv2d(in_ch,out_ch,3,padding=dilation,dilation=dilation,bias=False),
                             nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    class _ASPPPoolingV1(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.pool=nn.AdaptiveAvgPool2d(1)
            self.conv=nn.Sequential(nn.Conv2d(in_ch,out_ch,1,bias=False),nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True))
        def forward(self,x):
            sz=x.shape[2:]; x=self.conv(self.pool(x))
            return F.interpolate(x,size=sz,mode="bilinear",align_corners=False)
    class ASPPV1(nn.Module):
        def __init__(self,in_ch,rates=(6,12,18),out_ch=256):
            super().__init__()
            branches=[nn.Sequential(nn.Conv2d(in_ch,out_ch,1,bias=False),nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True))]
            for r in rates: branches.append(_ASPPConvV1(in_ch,out_ch,r))
            branches.append(_ASPPPoolingV1(in_ch,out_ch))
            self.branches=nn.ModuleList(branches)
            self.project=nn.Sequential(nn.Conv2d(out_ch*len(self.branches),out_ch,1,bias=False),
                                       nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True),nn.Dropout(0.5))
        def forward(self,x): return self.project(torch.cat([b(x) for b in self.branches],dim=1))
    class CustomV1(nn.Module):
        def __init__(self,nc):
            super().__init__()
            bb=resnet50(weights=None,replace_stride_with_dilation=[False,False,True])
            self.stem=nn.Sequential(bb.conv1,bb.bn1,bb.relu,bb.maxpool)
            self.layer1=bb.layer1; self.layer2=bb.layer2; self.layer3=bb.layer3; self.layer4=bb.layer4
            self.aspp=ASPPV1(2048)
            self.classifier=nn.Sequential(nn.Conv2d(256,256,3,padding=1,bias=False),nn.BatchNorm2d(256),
                                          nn.ReLU(inplace=True),nn.Conv2d(256,nc,1))
        def forward(self,x):
            sz=x.shape[2:]; x=self.stem(x); x=self.layer1(x); x=self.layer2(x)
            x=self.layer3(x); x=self.layer4(x); x=self.aspp(x); x=self.classifier(x)
            return F.interpolate(x,size=sz,mode="bilinear",align_corners=False)
    return CustomV1(nc)

def build_model_v1_improved(nc):
    """ResNet-50 based ImprovedDeepLabV3Plus matching old checkpoint structure."""
    from torchvision.models import resnet50
    # reuse current model_modified but with resnet50 backbone swap
    # The old improved.pt was trained with the old model_modified (resnet50)
    # We rebuild old structure inline
    class DSConv(nn.Module):
        def __init__(self,ic,oc,k=3,p=1,d=1):
            super().__init__()
            self.depthwise=nn.Conv2d(ic,ic,k,padding=p,dilation=d,groups=ic,bias=False)
            self.bn1=nn.BatchNorm2d(ic); self.pointwise=nn.Conv2d(ic,oc,1,bias=False)
            self.bn2=nn.BatchNorm2d(oc); self.relu=nn.ReLU(inplace=True)
        def forward(self,x): return self.relu(self.bn2(self.pointwise(self.relu(self.bn1(self.depthwise(x))))))
    class SEBlock(nn.Module):
        def __init__(self,c,r=16):
            super().__init__()
            mid=max(c//r,8)
            self.pool=nn.AdaptiveAvgPool2d(1)
            self.fc=nn.Sequential(nn.Linear(c,mid,bias=False),nn.ReLU(inplace=True),nn.Linear(mid,c,bias=False),nn.Sigmoid())
        def forward(self,x):
            b,c,_,_=x.shape; w=self.pool(x).view(b,c); w=self.fc(w).view(b,c,1,1); return x*w
    class ASPPSep(nn.Module):
        def __init__(self,ic,oc,d): super().__init__(); self.conv=DSConv(ic,oc,p=d,d=d)
        def forward(self,x): return self.conv(x)
    class ASPPPool(nn.Module):
        def __init__(self,ic,oc):
            super().__init__()
            self.pool=nn.AdaptiveAvgPool2d(1)
            self.conv=nn.Sequential(nn.Conv2d(ic,oc,1,bias=False),nn.BatchNorm2d(oc),nn.ReLU(inplace=True))
        def forward(self,x):
            sz=x.shape[2:]; x=self.conv(self.pool(x))
            return F.interpolate(x,size=sz,mode="bilinear",align_corners=False)
    class ASPPV1I(nn.Module):
        def __init__(self,ic,rates=(6,12,18),oc=256):
            super().__init__()
            b=[nn.Sequential(nn.Conv2d(ic,oc,1,bias=False),nn.BatchNorm2d(oc),nn.ReLU(inplace=True))]
            for r in rates: b.append(ASPPSep(ic,oc,r))
            b.append(ASPPPool(ic,oc)); self.branches=nn.ModuleList(b)
            self.project=nn.Sequential(nn.Conv2d(oc*len(self.branches),oc,1,bias=False),
                                       nn.BatchNorm2d(oc),nn.ReLU(inplace=True),nn.Dropout(0.5))
        def forward(self,x): return self.project(torch.cat([b(x) for b in self.branches],dim=1))
    class Decoder(nn.Module):
        def __init__(self,lc,ac,nc):
            super().__init__()
            self.reduce_low=nn.Sequential(nn.Conv2d(lc,48,1,bias=False),nn.BatchNorm2d(48),nn.ReLU(inplace=True))
            self.refine=nn.Sequential(DSConv(ac+48,256),DSConv(256,256),nn.Conv2d(256,nc,1))
        def forward(self,a,l):
            low=self.reduce_low(l); high=F.interpolate(a,size=low.shape[2:],mode="bilinear",align_corners=False)
            return self.refine(torch.cat([high,low],dim=1))
    class AuxH(nn.Sequential):
        def __init__(self,ic,nc):
            super().__init__(nn.Conv2d(ic,256,3,padding=1,bias=False),nn.BatchNorm2d(256),
                             nn.ReLU(inplace=True),nn.Dropout(0.1),nn.Conv2d(256,nc,1))
    class ImprovedV1(nn.Module):
        def __init__(self,nc):
            super().__init__()
            bb=resnet50(weights=None,replace_stride_with_dilation=[False,False,True])
            self.stem=nn.Sequential(bb.conv1,bb.bn1,bb.relu,bb.maxpool)
            self.layer1=bb.layer1; self.layer2=bb.layer2; self.layer3=bb.layer3; self.layer4=bb.layer4
            self.aspp=ASPPV1I(2048); self.se=SEBlock(256)
            self.decoder=Decoder(256,256,nc); self.aux_head=AuxH(1024,nc)
        def forward(self,x):
            sz=x.shape[2:]; x=self.stem(x); ll=self.layer1(x)
            x=self.layer2(ll); x=self.layer3(x); x=self.layer4(x)
            x=self.se(self.aspp(x)); x=self.decoder(x,ll)
            return F.interpolate(x,size=sz,mode="bilinear",align_corners=False)
    return ImprovedV1(nc)

def build_model(name, nc, pretrained=False, version="v2"):
    if version == "v1":
        if name == "custom":   return build_model_v1_custom(nc)
        if name == "improved": return build_model_v1_improved(nc)
    if name == "custom":
        from model import CustomDeepLabV3
        return CustomDeepLabV3(nc, pretrained_backbone=pretrained)
    if name == "improved":
        from model_modified import ImprovedDeepLabV3Plus
        return ImprovedDeepLabV3Plus(nc, pretrained_backbone=pretrained)
    # torchvision
    from torchvision.models.segmentation import deeplabv3_resnet50
    try:
        m = deeplabv3_resnet50(weights=None, weights_backbone=None)
    except Exception:
        m = deeplabv3_resnet50(pretrained=False)
    m.classifier[4] = nn.Conv2d(256, nc, 1)
    return m


# ---------------------------------------------------------------------------
# 1. Load histories
# ---------------------------------------------------------------------------
print("Loading histories...")
histories = {
    "custom_v1":   json.loads((ROOT/"outputs"/"custom_history.json").read_text()),
    "torchvision": json.loads((ROOT/"outputs"/"torchvision_history.json").read_text()),
    "improved_v1": json.loads((ROOT/"outputs"/"improved_history.json").read_text()),
    "custom_v2":   json.loads((ROOT/"outputs"/"custom_v2_history.json").read_text()),
    "improved_v2": json.loads((ROOT/"outputs"/"improved_v2_history.json").read_text()),
}
LABELS = {
    "custom_v1":   "Custom-v1 (ResNet50)",
    "torchvision": "Torchvision (ResNet50)",
    "improved_v1": "Improved-v1 (ResNet50+SE)",
    "custom_v2":   "Custom-v2 (ResNet101+V3+)",
    "improved_v2": "Improved-v2 (ResNet101+CBAM)",
}
COLORS = {
    "custom_v1":   "#1f77b4",
    "torchvision": "#ff7f0e",
    "improved_v1": "#2ca02c",
    "custom_v2":   "#d62728",
    "improved_v2": "#9467bd",
}


# ---------------------------------------------------------------------------
# 2. Evaluate all models on test set
# ---------------------------------------------------------------------------
print("Evaluating models on test set...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
names, palette, c2i = load_class_info(DATA/"class_dict.csv")
nc = len(names)
image_size = (352, 480)
ds      = TestDataset(DATA, c2i, image_size)
loader  = DataLoader(ds, batch_size=4, num_workers=2, pin_memory=True)

ckpt_map = {
    "custom_v1":   ("custom.pt",       "custom",      "v1"),
    "torchvision": ("torchvision.pt",  "torchvision", "v2"),
    "improved_v1": ("improved.pt",     "improved",    "v1"),
    "custom_v2":   ("custom_v2.pt",    "custom",      "v2"),
    "improved_v2": ("improved_v2.pt",  "improved",    "v2"),
}

test_results = {}
for key, (ckpt_file, mname, ver) in ckpt_map.items():
    ckpt_path = CKPT / ckpt_file
    if not ckpt_path.exists():
        print(f"  SKIP {key} (no checkpoint)")
        continue
    print(f"  Evaluating {key}...")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(mname, nc, version=ver).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    pa, miou, per_cls = eval_model(model, loader, nc, device)
    test_results[key] = {"pixel_accuracy": pa, "mIoU": miou, "per_class_iou": per_cls}
    print(f"    mIoU={miou:.4f}  pixel_acc={pa:.4f}")

    # best val mIoU from history
    h = histories[key]
    best_val = max(e["val_mIoU"] for e in h)
    best_train = max(e["train_mIoU"] for e in h)
    last_train = h[-1]["train_mIoU"]
    last_val   = h[-1]["val_mIoU"]
    test_results[key]["best_val_mIoU"]   = best_val
    test_results[key]["best_train_mIoU"] = best_train
    test_results[key]["overfit_gap"]     = last_train - last_val
    test_results[key]["epochs"]          = len(h)
    del model


# ---------------------------------------------------------------------------
# 3. Tables
# ---------------------------------------------------------------------------
print("Writing tables...")

# 3a. Model comparison table
rows = []
for key in ["custom_v1","torchvision","improved_v1","custom_v2","improved_v2"]:
    if key not in test_results: continue
    r = test_results[key]
    rows.append([
        LABELS[key],
        f"{r['mIoU']:.4f}",
        f"{r['pixel_accuracy']:.4f}",
        f"{r['best_val_mIoU']:.4f}",
        f"{r['overfit_gap']:.4f}",
        str(r["epochs"]),
    ])
with (OUT/"tables"/"model_comparison.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Model","Test mIoU","Test PixAcc","Best Val mIoU","Train-Val Gap (last)","Epochs"])
    w.writerows(rows)

# 3b. Per-class IoU
with (OUT/"tables"/"per_class_iou.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Class"] + [LABELS[k] for k in ckpt_map if k in test_results])
    for i, cname in enumerate(names):
        row = [cname]
        for k in ckpt_map:
            if k in test_results:
                row.append(f"{test_results[k]['per_class_iou'][i]:.4f}")
        w.writerow(row)

# 3c. Overfitting summary
with (OUT/"tables"/"overfitting_summary.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Model","Final Train mIoU","Final Val mIoU","Gap","Best Val mIoU"])
    for key in ["custom_v1","improved_v1","custom_v2","improved_v2"]:
        if key not in test_results: continue
        h  = histories[key]
        ft = h[-1]["train_mIoU"]
        fv = h[-1]["val_mIoU"]
        w.writerow([LABELS[key], f"{ft:.4f}", f"{fv:.4f}",
                    f"{ft-fv:.4f}", f"{test_results[key]['best_val_mIoU']:.4f}"])

print("  Tables written.")


# ---------------------------------------------------------------------------
# 4. Plots
# ---------------------------------------------------------------------------
print("Generating plots...")

# 4a. Training curves — Loss
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle("Training & Validation Curves — All Models", fontsize=13)
for key, h in histories.items():
    c = COLORS[key]; lbl = LABELS[key]
    ep = [e["epoch"] for e in h]
    axes[0].plot(ep, [e["train_loss"] for e in h], "--", color=c, alpha=0.5, lw=1.2)
    axes[0].plot(ep, [e["val_loss"]   for e in h], "-",  color=c, lw=1.8, label=lbl)
    axes[1].plot(ep, [e["train_mIoU"] for e in h], "--", color=c, alpha=0.5, lw=1.2)
    axes[1].plot(ep, [e["val_mIoU"]   for e in h], "-",  color=c, lw=1.8, label=lbl)
for ax, ylabel, title in zip(axes, ["Loss","mIoU"],["Validation Loss","Validation mIoU"]):
    ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    solid = plt.Line2D([],[],color="gray",lw=1.8,label="— val")
    dash  = plt.Line2D([],[],color="gray",lw=1.2,ls="--",alpha=0.5,label="-- train")
    ax.legend(handles=ax.get_legend_handles_labels()[0]+[solid,dash], fontsize=6.5)
fig.tight_layout()
fig.savefig(OUT/"plots"/"training_curves.png", dpi=150)
plt.close(fig)

# 4b. Overfitting gap over epochs
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Overfitting Analysis: Train–Val mIoU Gap", fontsize=13)
keys_v1 = ["custom_v1","improved_v1"]
keys_v2 = ["custom_v2","improved_v2"]
for ax, keys, title in zip(axes, [keys_v1, keys_v2], ["v1 Models (ResNet-50)","v2 Models (ResNet-101)"]):
    for key in keys:
        h = histories[key]
        ep  = [e["epoch"] for e in h]
        gap = [e["train_mIoU"]-e["val_mIoU"] for e in h]
        ax.plot(ep, gap, "-", color=COLORS[key], lw=2, label=LABELS[key])
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Train mIoU − Val mIoU")
    ax.set_title(title); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_ylim(-0.02, 0.18)
fig.tight_layout()
fig.savefig(OUT/"plots"/"overfitting_gap.png", dpi=150)
plt.close(fig)

# 4c. Per-class IoU bar chart (v2 models + torchvision baseline)
show_keys = ["custom_v2","improved_v2","torchvision"]
show_keys = [k for k in show_keys if k in test_results]
x = np.arange(nc)
width = 0.25
fig, ax = plt.subplots(figsize=(18, 6))
for i, key in enumerate(show_keys):
    iou = test_results[key]["per_class_iou"]
    ax.bar(x + i*width - width, iou, width, label=LABELS[key], color=COLORS[key], alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=55, ha="right", fontsize=8)
ax.set_ylabel("IoU"); ax.set_title("Per-class IoU — Test Set")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT/"plots"/"per_class_iou_bar.png", dpi=150)
plt.close(fig)

# 4d. Summary bar chart — mIoU comparison
fig, ax = plt.subplots(figsize=(10, 5))
keys_ordered = ["custom_v1","torchvision","improved_v1","custom_v2","improved_v2"]
keys_ordered = [k for k in keys_ordered if k in test_results]
mious = [test_results[k]["mIoU"] for k in keys_ordered]
lbls  = [LABELS[k] for k in keys_ordered]
colors= [COLORS[k] for k in keys_ordered]
bars  = ax.barh(lbls, mious, color=colors, alpha=0.85)
for bar, v in zip(bars, mious):
    ax.text(v+0.002, bar.get_y()+bar.get_height()/2, f"{v:.4f}", va="center", fontsize=9)
ax.axvline(0.428, color="gray", ls="--", lw=1, label="custom_v1 baseline")
ax.set_xlabel("Test mIoU"); ax.set_title("Model mIoU Comparison (Test Set)")
ax.set_xlim(0, 0.58); ax.legend(fontsize=8); ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT/"plots"/"miou_comparison.png", dpi=150)
plt.close(fig)

# 4e. LR schedule
fig, ax = plt.subplots(figsize=(10, 4))
for key in ["custom_v2","improved_v2"]:
    h = histories[key]
    ax.plot([e["epoch"] for e in h], [e["lr"] for e in h],
            color=COLORS[key], lw=2, label=LABELS[key])
ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate")
ax.set_title("Learning Rate Schedule (CosineAnnealingWarmRestarts)")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT/"plots"/"lr_schedule.png", dpi=150)
plt.close(fig)

print("  Plots written.")


# ---------------------------------------------------------------------------
# 5. Prediction Visualizations
# ---------------------------------------------------------------------------
print("Generating prediction visualizations...")

def save_vis(models_dict, loader, device, palette, out_dir, n=6):
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    model_list = list(models_dict.items())
    for m in models_dict.values():
        m.eval()
    with torch.no_grad():
        for images, masks, fnames in loader:
            images_dev = images.to(device)
            preds = {}
            for k, m in model_list:
                preds[k] = forward_logits(m, images_dev).argmax(1).cpu().numpy()
            for bi in range(images.shape[0]):
                if saved >= n: break
                panels = [denorm(images[bi]),
                          palette[masks[bi].numpy()]]
                for k, _ in model_list:
                    panels.append(palette[preds[k][bi]])
                canvas = np.concatenate(panels, axis=1)
                Image.fromarray(canvas).save(out_dir / f"vis_{saved:03d}.png")
                saved += 1
            if saved >= n: break

# Load v2 models for vis
vis_models = {}
for key, (ckpt_file, mname, ver) in [("custom_v2",("custom_v2.pt","custom","v2")),
                                  ("improved_v2",("improved_v2.pt","improved","v2"))]:
    ckpt = torch.load(CKPT/ckpt_file, map_location=device)
    m = build_model(mname, nc, version=ver).to(device)
    m.load_state_dict(ckpt["model_state_dict"])
    vis_models[key] = m

save_vis(vis_models, loader, device, palette, OUT/"samples", n=8)
print("  Visualizations saved.")

# 5b. Failure case analysis — find worst predictions
print("  Finding failure cases...")
improved_v2_m = vis_models["improved_v2"]
improved_v2_m.eval()
per_img_iou = []
with torch.no_grad():
    for images, masks, fnames in loader:
        images_dev = images.to(device)
        preds = forward_logits(improved_v2_m, images_dev).argmax(1).cpu().numpy()
        for bi in range(images.shape[0]):
            t = masks[bi].numpy().astype(np.int64)
            p = preds[bi]
            valid = (t>=0)&(t<nc)
            cm_i  = np.bincount(nc*t[valid]+p[valid], minlength=nc*nc).reshape(nc,nc)
            inter = np.diag(cm_i); union = cm_i.sum(0)+cm_i.sum(1)-inter
            iou_i = np.divide(inter, union, out=np.zeros_like(inter,dtype=np.float64), where=union>0)
            per_img_iou.append((iou_i.mean(), bi, images[bi], masks[bi], p, fnames[bi]))

per_img_iou.sort(key=lambda x: x[0])
fail_dir = OUT/"samples"/"failure_cases"
fail_dir.mkdir(parents=True, exist_ok=True)
for rank, (iou_v, _, img_t, mask_t, pred_np, fname) in enumerate(per_img_iou[:6]):
    panels = [denorm(img_t), palette[mask_t.numpy()], palette[pred_np]]
    canvas = np.concatenate(panels, axis=1)
    Image.fromarray(canvas).save(fail_dir / f"fail_{rank:02d}_iou{iou_v:.3f}_{fname}")

print("  Failure cases saved.")

# cleanup
for m in vis_models.values():
    del m

print("\nDone! All outputs in:", OUT)

# Print summary table to stdout
print("\n" + "="*70)
print(f"{'Model':<35} {'Test mIoU':>10} {'PixAcc':>8} {'Gap':>8} {'Epochs':>7}")
print("-"*70)
for key in ["custom_v1","torchvision","improved_v1","custom_v2","improved_v2"]:
    if key not in test_results: continue
    r = test_results[key]
    print(f"{LABELS[key]:<35} {r['mIoU']:>10.4f} {r['pixel_accuracy']:>8.4f} {r['overfit_gap']:>8.4f} {r['epochs']:>7}")
print("="*70)
