"""补充实验：TTA消融、参数量统计、Confusion Matrix (无需重新训练)

不需要训练的实验：
1. TTA消融: {single-scale, 3-scale, 3-scale+flip}
2. 模型统计: #params, #trainable, GFLOPs, inference latency
3. Confusion Matrix heatmap (揭示类别混淆模式)
4. 边界F-score (trimap评估)
"""
from __future__ import annotations
import csv, json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms.functional import InterpolationMode, normalize, pil_to_tensor, resize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT   = Path(__file__).parent
OUT    = ROOT / "outputs" / "supplement"
CKPT   = ROOT / "checkpoints"
DATA   = ROOT / "CamVid"
MEAN   = (0.390, 0.405, 0.414)
STD    = (0.274, 0.285, 0.297)

(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "plots").mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from train_v3 import build_model, CamVidDataset, forward_logits


def load_class_info(csv_path):
    names, palette, c2i = [], [], {}
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for idx, row in enumerate(csv.DictReader(fh)):
            r,g,b = int(row["r"].strip()), int(row["g"].strip()), int(row["b"].strip())
            names.append(row["name"].strip())
            palette.append((r,g,b))
            c2i[(r<<16)|(g<<8)|b] = idx
    return names, np.asarray(palette, dtype=np.uint8), c2i


# ---------------------------------------------------------------------------
# 1. TTA 消融实验
# ---------------------------------------------------------------------------

def eval_single_scale(model, loader, nc, device):
    """单尺度推理（无TTA）"""
    model.eval()
    cm = np.zeros((nc, nc), dtype=np.int64)
    
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            logits = forward_logits(model, images)
            preds = logits.argmax(1).cpu().numpy()
            t = masks.numpy()
            valid = (t >= 0) & (t < nc)
            cm += np.bincount(nc*t[valid]+preds[valid], minlength=nc*nc).reshape(nc,nc)
    
    inter = np.diag(cm).astype(np.float64)
    union = cm.sum(0) + cm.sum(1) - np.diag(cm)
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union>0)
    return float(iou.mean()), cm


def eval_multi_scale(model, loader, nc, device, scales=[0.75, 1.0, 1.25], use_flip=False):
    """多尺度TTA推理"""
    model.eval()
    cm = np.zeros((nc, nc), dtype=np.int64)
    
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            b, c, h, w = images.shape
            logits_sum = torch.zeros(b, nc, h, w, device=device)
            n_aug = 0
            
            for scale in scales:
                sh, sw = int(h*scale), int(w*scale)
                img_s = F.interpolate(images, (sh, sw), mode="bilinear", align_corners=False)
                out = forward_logits(model, img_s)
                out = F.interpolate(out, (h, w), mode="bilinear", align_corners=False)
                logits_sum += F.softmax(out, dim=1)
                n_aug += 1
                
                if use_flip:
                    img_f = torch.flip(img_s, dims=[3])
                    out_f = forward_logits(model, img_f)
                    out_f = torch.flip(out_f, dims=[3])
                    out_f = F.interpolate(out_f, (h, w), mode="bilinear", align_corners=False)
                    logits_sum += F.softmax(out_f, dim=1)
                    n_aug += 1
            
            preds = logits_sum.argmax(1).cpu().numpy()
            t = masks.numpy()
            valid = (t >= 0) & (t < nc)
            cm += np.bincount(nc*t[valid]+preds[valid], minlength=nc*nc).reshape(nc,nc)
    
    inter = np.diag(cm).astype(np.float64)
    union = cm.sum(0) + cm.sum(1) - np.diag(cm)
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union>0)
    return float(iou.mean()), cm


def run_tta_ablation(model, loader, nc, device):
    """运行TTA消融实验"""
    print("\n=== TTA Ablation Study ===")
    results = {}
    
    # Single-scale (baseline)
    t0 = time.time()
    miou, _ = eval_single_scale(model, loader, nc, device)
    t1 = time.time()
    results["single_scale"] = {"mIoU": miou, "time": t1-t0, "augs": 1}
    print(f"Single-scale: mIoU={miou:.4f}, time={t1-t0:.1f}s")
    
    # 3-scale (no flip)
    t0 = time.time()
    miou, _ = eval_multi_scale(model, loader, nc, device, scales=[0.75, 1.0, 1.25], use_flip=False)
    t1 = time.time()
    results["3_scale"] = {"mIoU": miou, "time": t1-t0, "augs": 3}
    print(f"3-scale:      mIoU={miou:.4f}, time={t1-t0:.1f}s")
    
    # 3-scale + flip (6 augmentations)
    t0 = time.time()
    miou, _ = eval_multi_scale(model, loader, nc, device, scales=[0.75, 1.0, 1.25], use_flip=True)
    t1 = time.time()
    results["3_scale_flip"] = {"mIoU": miou, "time": t1-t0, "augs": 6}
    print(f"3-scale+flip: mIoU={miou:.4f}, time={t1-t0:.1f}s")
    
    return results


# ---------------------------------------------------------------------------
# 2. 模型统计 (Params, FLOPs, Latency)
# ---------------------------------------------------------------------------

def count_params(model):
    """统计参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_flops(model, input_size=(1, 3, 352, 480)):
    """估算FLOPs (使用thop或简单估算)"""
    try:
        from thop import profile
        dummy_input = torch.randn(input_size)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        return flops / 1e9  # GFLOPs
    except ImportError:
        # 简化估算：假设主要计算在backbone
        h, w = input_size[2], input_size[3]
        # ResNet-101 约 7.8e9 FLOPs for 224x224
        # 按比例估算
        scale = (h * w) / (224 * 224)
        return 7.8 * scale  # 约 20 GFLOPs for 352x480


def measure_latency(model, device, input_size=(1, 3, 352, 480), n_iters=100):
    """测量推理延迟"""
    model.eval()
    dummy_input = torch.randn(input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = forward_logits(model, dummy_input)
    
    # Measure
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_iters):
            _ = forward_logits(model, dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
    t1 = time.time()
    
    return (t1 - t0) / n_iters * 1000  # ms per image


def run_model_analysis(model, device):
    """运行模型统计"""
    print("\n=== Model Statistics ===")
    
    total, trainable = count_params(model)
    print(f"Total params:    {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Frozen params:   {total-trainable:,} ({(total-trainable)/total*100:.1f}%)")
    
    flops = count_flops(model)
    print(f"GFLOPs (est.):   {flops:.2f}")
    
    latency = measure_latency(model, device)
    print(f"Latency (352x480): {latency:.2f} ms")
    
    # 不同输入尺寸的延迟
    for size in [(352, 480), (448, 608)]:
        lat = measure_latency(model, device, input_size=(1, 3, size[0], size[1]), n_iters=50)
        print(f"Latency ({size[0]}x{size[1]}): {lat:.2f} ms")
    
    return {
        "total_params": total,
        "trainable_params": trainable,
        "gflops": flops,
        "latency_ms": latency
    }


# ---------------------------------------------------------------------------
# 3. Confusion Matrix Heatmap
# ---------------------------------------------------------------------------

def compute_confusion_matrix(model, loader, nc, device):
    """计算完整混淆矩阵"""
    model.eval()
    cm = np.zeros((nc, nc), dtype=np.int64)
    
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            logits = forward_logits(model, images)
            preds = logits.argmax(1).cpu().numpy()
            t = masks.numpy()
            valid = (t >= 0) & (t < nc)
            cm += np.bincount(nc*t[valid]+preds[valid], minlength=nc*nc).reshape(nc,nc)
    
    return cm


def plot_confusion_matrix(cm, class_names, save_path, top_k=15):
    """绘制混淆矩阵热力图（只显示最频繁的k个类别）"""
    # 选择样本最多的top_k类别
    class_counts = cm.sum(axis=1)
    top_indices = np.argsort(class_counts)[-top_k:]
    
    cm_subset = cm[np.ix_(top_indices, top_indices)]
    
    # 归一化为每行（真实类别）
    cm_norm = np.divide(cm_subset, cm_subset.sum(axis=1, keepdims=True), 
                        out=np.zeros_like(cm_subset, dtype=float), 
                        where=cm_subset.sum(axis=1, keepdims=True)>0)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_norm, cmap="YlOrRd", vmin=0, vmax=1)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Normalized Count", rotation=-90, va="bottom")
    
    # Set ticks
    tick_labels = [class_names[i] for i in top_indices]
    ax.set_xticks(np.arange(len(tick_labels)))
    ax.set_yticks(np.arange(len(tick_labels)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticklabels(tick_labels)
    
    # Add text annotations
    for i in range(len(tick_labels)):
        for j in range(len(tick_labels)):
            text = ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                         ha="center", va="center", color="black" if cm_norm[i, j] < 0.5 else "white",
                         fontsize=8)
    
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (Top-{top_k} Classes)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix to {save_path}")


def analyze_common_confusions(cm, class_names, top_k=10):
    """分析最常见的混淆对"""
    # 排除对角线
    cm_off_diag = cm.copy()
    np.fill_diagonal(cm_off_diag, 0)
    
    # 找到最大混淆对
    confusions = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confusions.append((cm[i, j], class_names[i], class_names[j]))
    
    confusions.sort(reverse=True)
    
    print("\n=== Top Confusion Pairs ===")
    for count, true_cls, pred_cls in confusions[:top_k]:
        print(f"{true_cls} → {pred_cls}: {count} pixels")
    
    return confusions[:top_k]


# ---------------------------------------------------------------------------
# 4. 边界F-score (Trimap)
# ---------------------------------------------------------------------------

def compute_boundary_mask(mask, kernel_size=3):
    """计算边界区域（形态学膨胀减原图）"""
    from scipy import ndimage
    mask_bool = mask > 0
    dilated = ndimage.binary_dilation(mask_bool, iterations=kernel_size//2)
    eroded = ndimage.binary_erosion(mask_bool, iterations=kernel_size//2)
    boundary = dilated ^ eroded
    return boundary


def eval_boundary_fscore(model, loader, nc, device, trimap_width=5):
    """评估边界F-score"""
    from scipy import ndimage
    
    model.eval()
    f_scores = []
    
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            logits = forward_logits(model, images)
            preds = logits.argmax(1).cpu().numpy()
            masks_np = masks.numpy()
            
            for b in range(images.shape[0]):
                pred, mask = preds[b], masks_np[b]
                
                # 计算每个类别的边界
                for cls in range(1, nc):  # 跳过背景类0
                    mask_cls = (mask == cls).astype(np.uint8)
                    pred_cls = (pred == cls).astype(np.uint8)
                    
                    if mask_cls.sum() == 0:
                        continue
                    
                    # 计算边界
                    mask_boundary = compute_boundary_mask(mask_cls, trimap_width*2+1)
                    pred_boundary = compute_boundary_mask(pred_cls, trimap_width*2+1)
                    
                    # 在真实边界周围创建trimap
                    trimap = ndimage.binary_dilation(mask_boundary, iterations=trimap_width)
                    
                    # 只在trimap区域计算
                    intersection = (pred_boundary & mask_boundary & trimap).sum()
                    precision = intersection / (pred_boundary & trimap).sum() if (pred_boundary & trimap).sum() > 0 else 0
                    recall = intersection / (mask_boundary & trimap).sum() if (mask_boundary & trimap).sum() > 0 else 0
                    
                    if precision + recall > 0:
                        f1 = 2 * precision * recall / (precision + recall)
                        f_scores.append(f1)
    
    mean_fscore = np.mean(f_scores) if f_scores else 0
    print(f"\n=== Boundary F-score (trimap width={trimap_width}) ===")
    print(f"Mean F-score: {mean_fscore:.4f}")
    return mean_fscore


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load class info
    names, palette, c2i = load_class_info(DATA / "class_dict.csv")
    nc = len(names)
    
    # Test dataset (352x480)
    test_ds = CamVidDataset(DATA, "test", c2i, (352, 480), augment=False)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4)
    
    # Load v3 model
    print("\nLoading improved_v3 model...")
    sys.path.insert(0, str(ROOT))
    from model_modified import ImprovedDeepLabV3Plus
    
    model = ImprovedDeepLabV3Plus(num_classes=nc, output_stride=16, pretrained_backbone=False, freeze_backbone=False).to(device)
    ckpt = torch.load(CKPT / "improved_v3.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    
    # 1. TTA Ablation
    tta_results = run_tta_ablation(model, test_loader, nc, device)
    
    # 2. Model Statistics
    stats = run_model_analysis(model, device)
    
    # 3. Confusion Matrix
    print("\nComputing confusion matrix...")
    cm = compute_confusion_matrix(model, test_loader, nc, device)
    plot_confusion_matrix(cm, names, OUT / "plots" / "confusion_matrix.png", top_k=15)
    confusions = analyze_common_confusions(cm, names, top_k=10)
    
    # 4. Boundary F-score
    try:
        from scipy import ndimage
        fscore = eval_boundary_fscore(model, test_loader, nc, device, trimap_width=5)
    except ImportError:
        print("scipy not available, skipping boundary F-score")
        fscore = None
    
    # Save results
    results = {
        "tta_ablation": tta_results,
        "model_stats": stats,
        "top_confusions": [(int(c), t, p) for c, t, p in confusions],
        "boundary_fscore": fscore
    }
    
    with open(OUT / "tables" / "supplement_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {OUT}/tables/supplement_results.json")


if __name__ == "__main__":
    main()
