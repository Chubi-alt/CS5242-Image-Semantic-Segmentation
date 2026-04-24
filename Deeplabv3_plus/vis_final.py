"""vis_final.py
生成三方法对比可视化 + 更新完整实验表格。

输出:
  outputs/report/
    samples/
      grid_NNN.png          — [原图 | GT | Custom-v2 | Improved-v2 | v3(TTA)] 5列对比
      scene_variety/        — 按场景类型精选 (道路/夜/行人/复杂交叉口)
      diff_maps/            — v2 vs v3 差异热力图
    tables/
      model_comparison_full.csv   — 含v3的完整对比
      per_class_iou_full.csv      — 含v3的逐类IoU
      overfitting_summary_full.csv
    plots/
      training_curves_all.png     — 含v3曲线
      per_class_iou_full.png      — 含v3的柱状图
      miou_comparison_full.png    — 横向柱状图
"""
from __future__ import annotations
import csv, json, random, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import InterpolationMode, normalize, pil_to_tensor, resize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).parent
OUT  = ROOT / "outputs" / "report"
CKPT = ROOT / "checkpoints"
DATA = ROOT / "CamVid"
MEAN = (0.390, 0.405, 0.414)
STD  = (0.274, 0.285, 0.297)

for d in ["samples/grid","samples/scene_variety","samples/diff_maps","tables","plots"]:
    (OUT/d).mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))


# ─── helpers ────────────────────────────────────────────────────────────────

def load_class_info(csv_path):
    names, palette, c2i = [], [], {}
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for idx, row in enumerate(csv.DictReader(fh)):
            r,g,b = int(row["r"].strip()),int(row["g"].strip()),int(row["b"].strip())
            names.append(row["name"].strip()); palette.append((r,g,b))
            c2i[(r<<16)|(g<<8)|b] = idx
    return names, np.asarray(palette,dtype=np.uint8), c2i

def rgb_mask_to_class(mask, c2i):
    codes=(mask[...,0].astype(np.int32)<<16)|(mask[...,1].astype(np.int32)<<8)|mask[...,2].astype(np.int32)
    out=np.zeros(mask.shape[:2],dtype=np.int64)
    for code,idx in c2i.items(): out[codes==code]=idx
    return out

def denorm(t):
    img=t.cpu().numpy().transpose(1,2,0)
    img=img*np.array(STD)+np.array(MEAN)
    return np.clip(img*255,0,255).astype(np.uint8)

def add_label_bar(img_array, text, height=22, bg=(40,40,40), fg=(255,255,255)):
    h, w = img_array.shape[:2]
    bar = np.full((height, w, 3), bg, dtype=np.uint8)
    pil = Image.fromarray(bar)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0,0), text, font=font)
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    draw.text(((w-tw)//2, (height-th)//2), text, fill=fg, font=font)
    bar = np.array(pil)
    return np.vstack([bar, img_array])

def forward_tta(model, images, nc):
    """Multi-scale + flip TTA."""
    b,c,h,w = images.shape
    logits_sum = torch.zeros(b,nc,h,w,device=images.device)
    for scale in [0.75, 1.0, 1.25]:
        sh,sw = int(h*scale),int(w*scale)
        img_s = F.interpolate(images,(sh,sw),mode="bilinear",align_corners=False)
        out   = model(img_s)
        out   = out["out"] if isinstance(out,dict) else out
        out   = F.interpolate(out,(h,w),mode="bilinear",align_corners=False)
        logits_sum += F.softmax(out,dim=1)
        img_f = torch.flip(img_s,dims=[3])
        out_f = model(img_f)
        out_f = out_f["out"] if isinstance(out_f,dict) else out_f
        out_f = torch.flip(out_f,dims=[3])
        out_f = F.interpolate(out_f,(h,w),mode="bilinear",align_corners=False)
        logits_sum += F.softmax(out_f,dim=1)
    return logits_sum


def forward_logits(model, images):
    out = model(images)
    return out["out"] if isinstance(out,dict) else out


# ─── Dataset ────────────────────────────────────────────────────────────────

class TestDataset(Dataset):
    def __init__(self, data_root, c2i, image_size):
        self.img_dir  = data_root/"test"
        self.mask_dir = data_root/"test_labels"
        self.c2i      = c2i
        self.size     = image_size
        self.samples  = sorted(self.img_dir.glob("*.png"))
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        ip = self.samples[i]
        mp = self.mask_dir/f"{ip.stem}_L.png"
        img  = Image.open(ip).convert("RGB")
        mask = Image.open(mp).convert("RGB")
        img  = resize(img,  self.size, InterpolationMode.BILINEAR)
        mask = resize(mask, self.size, InterpolationMode.NEAREST)
        t = normalize(pil_to_tensor(img).float()/255.0, MEAN, STD)
        m = torch.from_numpy(rgb_mask_to_class(np.array(mask,dtype=np.uint8),self.c2i)).long()
        return t, m, ip.name


# ─── Model builders (version-aware) ─────────────────────────────────────────

def build_v1_custom(nc):
    from torchvision.models import resnet50
    class _AC(nn.Sequential):
        def __init__(self,ic,oc,d):
            super().__init__(nn.Conv2d(ic,oc,3,padding=d,dilation=d,bias=False),nn.BatchNorm2d(oc),nn.ReLU(inplace=True))
    class _AP(nn.Module):
        def __init__(self,ic,oc):
            super().__init__(); self.pool=nn.AdaptiveAvgPool2d(1)
            self.conv=nn.Sequential(nn.Conv2d(ic,oc,1,bias=False),nn.BatchNorm2d(oc),nn.ReLU(inplace=True))
        def forward(self,x):
            sz=x.shape[2:]; return F.interpolate(self.conv(self.pool(x)),size=sz,mode="bilinear",align_corners=False)
    class ASPPV1(nn.Module):
        def __init__(self,ic,rates=(6,12,18),oc=256):
            super().__init__()
            self.branches=nn.ModuleList([
                nn.Sequential(nn.Conv2d(ic,oc,1,bias=False),nn.BatchNorm2d(oc),nn.ReLU(inplace=True)),
                _AC(ic,oc,rates[0]),_AC(ic,oc,rates[1]),_AC(ic,oc,rates[2]),_AP(ic,oc)])
            self.project=nn.Sequential(nn.Conv2d(oc*5,oc,1,bias=False),nn.BatchNorm2d(oc),nn.ReLU(inplace=True),nn.Dropout(0.5))
        def forward(self,x): return self.project(torch.cat([b(x) for b in self.branches],dim=1))
    class V1(nn.Module):
        def __init__(self,nc):
            super().__init__()
            bb=resnet50(weights=None,replace_stride_with_dilation=[False,False,True])
            self.stem=nn.Sequential(bb.conv1,bb.bn1,bb.relu,bb.maxpool)
            self.layer1=bb.layer1;self.layer2=bb.layer2;self.layer3=bb.layer3;self.layer4=bb.layer4
            self.aspp=ASPPV1(2048)
            self.classifier=nn.Sequential(nn.Conv2d(256,256,3,padding=1,bias=False),nn.BatchNorm2d(256),nn.ReLU(inplace=True),nn.Conv2d(256,nc,1))
        def forward(self,x):
            sz=x.shape[2:]; x=self.stem(x); x=self.layer1(x); x=self.layer2(x); x=self.layer3(x); x=self.layer4(x)
            return F.interpolate(self.classifier(self.aspp(x)),size=sz,mode="bilinear",align_corners=False)
    return V1(nc)

def build_model(key, nc):
    if key == "custom_v1":   return build_v1_custom(nc)
    if key == "torchvision":
        from torchvision.models.segmentation import deeplabv3_resnet50
        try:    m = deeplabv3_resnet50(weights=None,weights_backbone=None)
        except: m = deeplabv3_resnet50(pretrained=False)
        m.classifier[4] = nn.Conv2d(256,nc,1); return m
    if key == "custom_v2":
        from model import CustomDeepLabV3
        return CustomDeepLabV3(nc, pretrained_backbone=False)
    if key == "improved_v2":
        from model_modified import ImprovedDeepLabV3Plus
        return ImprovedDeepLabV3Plus(nc, pretrained_backbone=False, output_stride=16)
    if key == "improved_v3":
        from model_modified import ImprovedDeepLabV3Plus
        return ImprovedDeepLabV3Plus(nc, pretrained_backbone=False, output_stride=8)
    raise ValueError(key)


# ─── Load everything ─────────────────────────────────────────────────────────

print("Loading class info...")
names, palette, c2i = load_class_info(DATA/"class_dict.csv")
nc = len(names)

# All 5 configs
CONFIGS = [
    ("custom_v1",   "custom.pt",    (352,480), "Custom-v1\n(ResNet50)"),
    ("torchvision", "torchvision.pt",(352,480), "Torchvision\n(ResNet50)"),
    ("custom_v2",   "custom_v2.pt", (352,480), "Custom-v2\n(ResNet101+V3+)"),
    ("improved_v2", "improved_v2.pt",(352,480), "Improved-v2\n(ResNet101+CBAM)"),
    ("improved_v3", "improved_v3.pt",(448,608), "V3+TTA\n(ResNet101+Focal+OHEM)"),
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = {}
for key, ckpt_file, img_size, label in CONFIGS:
    ckpt_path = CKPT/ckpt_file
    if not ckpt_path.exists(): print(f"  SKIP {key}"); continue
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = build_model(key, nc).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    models[key] = (model, img_size, label)
    print(f"  Loaded {key}")


# ─── Per-model metrics ───────────────────────────────────────────────────────

def eval_model(key, model, img_size, use_tta=False):
    ds     = TestDataset(DATA, c2i, img_size)
    loader = DataLoader(ds, batch_size=3, num_workers=2, pin_memory=True)
    cm = np.zeros((nc,nc), dtype=np.int64)
    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device)
            if use_tta:
                probs = forward_tta(model, images, nc)
                preds = probs.argmax(1).cpu().numpy().astype(np.int64)
            else:
                preds = forward_logits(model, images).argmax(1).cpu().numpy().astype(np.int64)
            t     = masks.numpy().astype(np.int64)
            valid = (t>=0)&(t<nc)
            cm   += np.bincount(nc*t[valid]+preds[valid], minlength=nc*nc).reshape(nc,nc)
    inter = np.diag(cm); union=cm.sum(0)+cm.sum(1)-inter
    iou   = np.divide(inter,union,out=np.zeros_like(inter,dtype=np.float64),where=union>0)
    pa    = float(inter.sum()/max(cm.sum(),1))
    return pa, float(iou.mean()), iou.tolist()

print("\nEvaluating all models on test set...")
results = {}
for key,(model,img_size,label) in models.items():
    use_tta = (key=="improved_v3")
    pa, miou, per_cls = eval_model(key, model, img_size, use_tta)
    results[key] = {"pa":pa,"miou":miou,"per_cls":per_cls,"label":label.replace("\n"," ")}
    print(f"  {key:<15} mIoU={miou:.4f}  PA={pa:.4f}")


# ─── Tables ──────────────────────────────────────────────────────────────────

print("\nWriting tables...")

HIST_MAP = {
    "custom_v1":   "custom_history.json",
    "torchvision": "torchvision_history.json",
    "improved_v1": "improved_history.json",
    "custom_v2":   "custom_v2_history.json",
    "improved_v2": "improved_v2_history.json",
    "improved_v3": "improved_v3_history.json",
}
histories = {}
for k,f in HIST_MAP.items():
    p = ROOT/"outputs"/f
    if p.exists(): histories[k]=json.loads(p.read_text())

# Full comparison
all_keys_ordered = ["custom_v1","torchvision","improved_v1","custom_v2","improved_v2","improved_v3"]
with (OUT/"tables"/"model_comparison_full.csv").open("w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["Model","Backbone","Key Improvements","Test mIoU","Test PixAcc",
                "Best Val mIoU","Train-Val Gap","Epochs","TTA"])
    meta = {
        "custom_v1":   ("ResNet-50",  "ASPP baseline"),
        "torchvision": ("ResNet-50",  "Official DeepLabV3"),
        "improved_v1": ("ResNet-50",  "V3+ decoder + SE"),
        "custom_v2":   ("ResNet-101", "V3+ decoder + aux head"),
        "improved_v2": ("ResNet-101", "CBAM + dual aux + CosLR"),
        "improved_v3": ("ResNet-101", "Focal+OHEM+CutMix+os=8+TTA"),
    }
    for key in all_keys_ordered:
        if key not in results and key not in ["improved_v1"]:
            continue
        r = results.get(key, {})
        h = histories.get(key, [])
        bb, impr = meta.get(key,("?","?"))
        miou  = f"{r.get('miou',0):.4f}" if r else "N/A"
        pa    = f"{r.get('pa',0):.4f}"   if r else "N/A"
        best_val = f"{max((e['val_mIoU'] for e in h),default=0):.4f}" if h else "N/A"
        gap   = f"{h[-1]['train_mIoU']-h[-1]['val_mIoU']:.4f}" if h else "N/A"
        epochs= str(len(h)) if h else "N/A"
        tta   = "Yes" if key=="improved_v3" else "No"
        w.writerow([key,bb,impr,miou,pa,best_val,gap,epochs,tta])

# Per-class IoU full
eval_keys = [k for k in all_keys_ordered if k in results]
with (OUT/"tables"/"per_class_iou_full.csv").open("w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["Class"]+[results[k]["label"] for k in eval_keys])
    for i,cname in enumerate(names):
        w.writerow([cname]+[f"{results[k]['per_cls'][i]:.4f}" for k in eval_keys])

print("  Tables written.")


# ─── Plots ───────────────────────────────────────────────────────────────────

print("Generating plots...")

COLORS = {"custom_v1":"#1f77b4","torchvision":"#ff7f0e","improved_v1":"#2ca02c",
          "custom_v2":"#d62728","improved_v2":"#9467bd","improved_v3":"#e377c2"}
PLOT_LABELS = {
    "custom_v1":   "Custom-v1 (ResNet50)",
    "torchvision": "Torchvision (ResNet50)",
    "improved_v1": "Improved-v1 (ResNet50+SE)",
    "custom_v2":   "Custom-v2 (ResNet101+V3+)",
    "improved_v2": "Improved-v2 (ResNet101+CBAM)",
    "improved_v3": "V3 (ResNet101+Focal+OHEM+TTA)",
}

# Training curves (all including v3)
fig, axes = plt.subplots(1,2,figsize=(16,5))
fig.suptitle("Training & Validation Curves — All Methods", fontsize=13, fontweight="bold")
for key, h in histories.items():
    c=COLORS.get(key,"#333"); lbl=PLOT_LABELS.get(key,key)
    ep=[e["epoch"] for e in h]
    axes[0].plot(ep,[e["val_loss"]   for e in h],"-", color=c,lw=2,  label=lbl)
    axes[0].plot(ep,[e["train_loss"] for e in h],"--",color=c,lw=1,alpha=0.4)
    axes[1].plot(ep,[e["val_mIoU"]   for e in h],"-", color=c,lw=2,  label=lbl)
    axes[1].plot(ep,[e["train_mIoU"] for e in h],"--",color=c,lw=1,alpha=0.4)
for ax,yl,tl in zip(axes,["Loss","mIoU"],["Loss","mIoU"]):
    ax.set_xlabel("Epoch",fontsize=11); ax.set_ylabel(yl,fontsize=11)
    ax.set_title(f"Validation {tl}  (— val  -- train)",fontsize=11)
    ax.legend(fontsize=7.5); ax.grid(alpha=0.3)
axes[1].axhline(0.50,color="red",ls=":",lw=1.5,label="0.50 target")
axes[1].legend(fontsize=7.5)
fig.tight_layout(); fig.savefig(OUT/"plots"/"training_curves_all.png",dpi=150); plt.close(fig)

# mIoU bar
fig, ax = plt.subplots(figsize=(11,5.5))
plot_keys = [k for k in all_keys_ordered if k in results]
mious  = [results[k]["miou"] for k in plot_keys]
lbls   = [PLOT_LABELS[k]     for k in plot_keys]
colors = [COLORS[k]          for k in plot_keys]
bars   = ax.barh(lbls, mious, color=colors, alpha=0.88, height=0.55)
for bar,v in zip(bars,mious):
    ax.text(v+0.003, bar.get_y()+bar.get_height()/2, f"{v:.4f}", va="center", fontsize=10, fontweight="bold")
ax.axvline(0.50, color="red", ls="--", lw=1.5, label="0.50 target")
ax.set_xlabel("Test mIoU", fontsize=12); ax.set_title("Model Comparison — Test mIoU", fontsize=13, fontweight="bold")
ax.set_xlim(0.30, 0.56); ax.legend(fontsize=10); ax.grid(axis="x", alpha=0.3)
fig.tight_layout(); fig.savefig(OUT/"plots"/"miou_comparison_full.png",dpi=150); plt.close(fig)

# Per-class IoU bar (v2, v3, torchvision baseline)
show = [k for k in ["torchvision","improved_v2","improved_v3"] if k in results]
x=np.arange(nc); width=0.27
fig, ax = plt.subplots(figsize=(20,6))
for i,k in enumerate(show):
    ax.bar(x+i*width-width, results[k]["per_cls"], width,
           label=PLOT_LABELS[k], color=COLORS[k], alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(names,rotation=55,ha="right",fontsize=7.5)
ax.axhline(0.5,color="red",ls=":",lw=1,alpha=0.6)
ax.set_ylabel("IoU",fontsize=11); ax.set_title("Per-class IoU — Test Set",fontsize=13,fontweight="bold")
ax.legend(fontsize=9); ax.grid(axis="y",alpha=0.3)
fig.tight_layout(); fig.savefig(OUT/"plots"/"per_class_iou_full.png",dpi=150); plt.close(fig)

print("  Plots written.")


# ─── Visualizations ──────────────────────────────────────────────────────────

print("Generating visualizations...")

# We need predictions from 3 methods side by side: custom_v2, improved_v2, improved_v3
VIS_KEYS  = ["custom_v2", "improved_v2", "improved_v3"]
VIS_SIZE  = (352, 480)   # common size for display
V3_SIZE   = (448, 608)

# Collect all test images with predictions
ds_std = TestDataset(DATA, c2i, VIS_SIZE)
ds_v3  = TestDataset(DATA, c2i, V3_SIZE)

# Pre-compute all predictions for each model
print("  Computing predictions...")
all_preds = {k: [] for k in VIS_KEYS}
all_images, all_masks, all_fnames = [], [], []

loader_std = DataLoader(ds_std, batch_size=4, num_workers=2)
loader_v3  = DataLoader(ds_v3,  batch_size=3, num_workers=2)

# std models
std_models = {k: models[k][0] for k in ["custom_v2","improved_v2"] if k in models}
with torch.no_grad():
    for images, masks, fnames in loader_std:
        images_dev = images.to(device)
        all_images.append(images); all_masks.append(masks)
        for fname in fnames: all_fnames.append(fname)
        for k, m in std_models.items():
            preds = forward_logits(m, images_dev).argmax(1).cpu()
            all_preds[k].append(preds)

# v3 model with TTA
if "improved_v3" in models:
    v3_model = models["improved_v3"][0]
    with torch.no_grad():
        for images_v3, _, _ in loader_v3:
            images_v3 = images_v3.to(device)
            probs = forward_tta(v3_model, images_v3, nc)
            preds = probs.argmax(1).cpu()
            # resize to VIS_SIZE for display
            preds_rs = F.interpolate(preds.unsqueeze(1).float(), VIS_SIZE, mode="nearest").squeeze(1).long()
            all_preds["improved_v3"].append(preds_rs)

# Flatten
all_images = torch.cat(all_images, 0)
all_masks  = torch.cat(all_masks,  0)
for k in VIS_KEYS:
    if all_preds[k]: all_preds[k] = torch.cat(all_preds[k], 0)

n_samples = len(all_fnames)
print(f"  Total test samples: {n_samples}")

col_labels = ["Input", "Ground Truth", "Custom-v2\n(ResNet101+V3+)",
              "Improved-v2\n(ResNet101+CBAM)", "V3\n(Focal+OHEM+TTA)"]

def make_grid_row(idx):
    """Return (H, 5W, 3) numpy image for one test sample."""
    img  = denorm(all_images[idx])
    gt   = palette[all_masks[idx].numpy()]
    cols = [img, gt]
    for k in VIS_KEYS:
        if all_preds[k] is not None and len(all_preds[k]) > idx:
            cols.append(palette[all_preds[k][idx].numpy()])
        else:
            cols.append(np.zeros_like(img))
    return cols

def save_grid(indices, path, title=""):
    H, W = VIS_SIZE
    n_cols = 5; n_rows = len(indices)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3.2, n_rows*2.4))
    if n_rows == 1: axes = axes[np.newaxis]
    for row, idx in enumerate(indices):
        cols = make_grid_row(idx)
        for col, (panel, clabel) in enumerate(zip(cols, col_labels)):
            axes[row][col].imshow(panel)
            axes[row][col].axis("off")
            if row == 0:
                axes[row][col].set_title(clabel, fontsize=9, fontweight="bold", pad=4)
        # filename label on left
        axes[row][0].set_ylabel(all_fnames[idx][:16], fontsize=6.5, rotation=0,
                                 labelpad=55, va="center")
    if title: fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.97] if title else None)
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)

# 1. Main grid: 12 diverse samples (evenly spaced)
step = max(n_samples//12, 1)
diverse_idx = list(range(0, min(n_samples, 12*step), step))[:12]
save_grid(diverse_idx, OUT/"samples"/"grid"/"main_grid.png", "三方法分割对比 — 测试集多样样本")
print("  main_grid.png saved")

# 2. Per-sample IoU computation for smart selection
def img_miou(idx):
    scores = []
    for k in VIS_KEYS:
        if not all_preds[k] is not None or len(all_preds[k]) <= idx: continue
        t = all_masks[idx].numpy().astype(np.int64)
        p = all_preds[k][idx].numpy().astype(np.int64)
        v = (t>=0)&(t<nc)
        cm = np.bincount(nc*t[v]+p[v],minlength=nc*nc).reshape(nc,nc)
        inter=np.diag(cm); union=cm.sum(0)+cm.sum(1)-inter
        iou=np.divide(inter,union,out=np.zeros_like(inter,dtype=np.float64),where=union>0)
        scores.append(iou.mean())
    return np.mean(scores) if scores else 0

print("  Computing per-image IoU for smart selection...")
img_ious = [img_miou(i) for i in range(n_samples)]
img_ious = np.array(img_ious)

# Sort and select
sorted_idx = np.argsort(img_ious)
worst_idx  = sorted_idx[:6].tolist()
best_idx   = sorted_idx[-6:].tolist()
mid_idx    = sorted_idx[len(sorted_idx)//2-3:len(sorted_idx)//2+3].tolist()

save_grid(best_idx,  OUT/"samples"/"scene_variety"/"best_cases.png",  "最佳预测样本 (mIoU最高)")
save_grid(worst_idx, OUT/"samples"/"scene_variety"/"worst_cases.png", "最难样本 (mIoU最低)")
save_grid(mid_idx,   OUT/"samples"/"scene_variety"/"medium_cases.png","中等难度样本")
print("  Scene variety grids saved.")

# 3. Difference maps: v2 vs v3
print("  Generating difference maps...")
diff_idx = list(range(0, min(n_samples, 9)))   # first 9 samples
H, W = VIS_SIZE
fig, axes = plt.subplots(3, 9, figsize=(27, 9))
diff_cmap = LinearSegmentedColormap.from_list("diff", ["#2166ac","#f7f7f7","#d6604d"])

for col, idx in enumerate(diff_idx):
    ax_img  = axes[0][col]
    ax_v2   = axes[1][col]
    ax_diff = axes[2][col]

    img = denorm(all_images[idx])
    gt  = all_masks[idx].numpy().astype(np.int64)
    ax_img.imshow(img); ax_img.axis("off")
    if col==0: ax_img.set_ylabel("Input", fontsize=9, fontweight="bold")

    # v2 correct pixels
    if len(all_preds["improved_v2"]) > idx:
        p_v2 = all_preds["improved_v2"][idx].numpy().astype(np.int64)
        correct_v2 = (p_v2 == gt).astype(np.float32)
        ax_v2.imshow(correct_v2, cmap="RdYlGn", vmin=0, vmax=1)
        ax_v2.axis("off")
        if col==0: ax_v2.set_ylabel("Improved-v2\ncorrect px", fontsize=9, fontweight="bold")

    # improvement: v3 correct where v2 wrong
    if len(all_preds["improved_v3"]) > idx and len(all_preds["improved_v2"]) > idx:
        p_v3 = all_preds["improved_v3"][idx].numpy().astype(np.int64)
        p_v2 = all_preds["improved_v2"][idx].numpy().astype(np.int64)
        diff = p_v3.astype(np.float32) - p_v2.astype(np.float32)  # just show change
        v3_better  = ((p_v3==gt)&(p_v2!=gt)).astype(np.float32)
        v2_better  = ((p_v2==gt)&(p_v3!=gt)).astype(np.float32)
        diff_map   = v3_better - v2_better   # +1=v3 better, -1=v2 better, 0=same
        ax_diff.imshow(diff_map, cmap=diff_cmap, vmin=-1, vmax=1)
        ax_diff.axis("off")
        if col==0: ax_diff.set_ylabel("V3 vs V2\n(red=v3better)", fontsize=9, fontweight="bold")
    ax_img.set_title(f"#{idx}", fontsize=8)

fig.suptitle("差异图：V3 vs Improved-v2  (绿=v3更好, 蓝=v2更好)", fontsize=12, fontweight="bold")
plt.tight_layout(); fig.savefig(OUT/"samples"/"diff_maps"/"v3_vs_v2_diff.png",dpi=130,bbox_inches="tight")
plt.close(fig)
print("  Difference maps saved.")

# 4. Legend panel for color palette
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis("off")
n_cols_leg = 4
patches = [mpatches.Patch(color=np.array(palette[i])/255, label=names[i]) for i in range(nc)]
ax.legend(handles=patches, loc="center", ncol=n_cols_leg, fontsize=9,
          frameon=True, title="CamVid Class Palette", title_fontsize=11)
fig.tight_layout(); fig.savefig(OUT/"samples"/"class_legend.png",dpi=130); plt.close(fig)
print("  Class legend saved.")

print(f"\n{'='*65}")
print(f"{'Model':<35} {'Test mIoU':>10} {'PixAcc':>9}")
print(f"{'-'*65}")
for k in all_keys_ordered:
    if k in results:
        r=results[k]
        print(f"{PLOT_LABELS[k]:<35} {r['miou']:>10.4f} {r['pa']:>9.4f}")
print(f"{'='*65}")
print(f"\nAll outputs -> {OUT}")
