"""Improved DeepLabV3+ for CamVid semantic segmentation.

Improvements over previous version
------------------------------------
1. **ResNet-101 backbone** — deeper, richer features vs ResNet-50.
2. **CBAM attention** (channel + spatial) after ASPP — better feature
   recalibration than SE-only.
3. **Larger low-level feature channel** (64 instead of 48) in decoder.
4. **Deeper decoder** with three separable conv blocks.
5. **Aux head on layer2** in addition to layer3 — more gradient signal.
6. **Backbone freezing** (stem + layer1) to reduce overfitting.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101


class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int,
        kernel_size: int = 3, padding: int = 1, dilation: int = 1,
    ) -> None:
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding,
                            dilation=dilation, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn2(self.pw(self.relu(self.bn1(self.dw(x))))))


class CBAMChannel(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg = self.mlp(x.mean(dim=[2, 3])).view(b, c, 1, 1)
        mx = self.mlp(x.amax(dim=[2, 3])).view(b, c, 1, 1)
        return x * torch.sigmoid(avg + mx)


class CBAMSpatial(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        w = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * w


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.channel = CBAMChannel(channels, reduction)
        self.spatial = CBAMSpatial()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


class _ASPPSepConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int) -> None:
        super().__init__()
        self.conv = DepthwiseSeparableConv(in_ch, out_ch, padding=dilation, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _ASPPPooling(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        x = self.conv(self.pool(x))
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(
        self, in_ch: int,
        atrous_rates: tuple[int, ...] = (6, 12, 18),
        out_ch: int = 256,
    ) -> None:
        super().__init__()
        branches: list[nn.Module] = [
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        ]
        for rate in atrous_rates:
            branches.append(_ASPPSepConv(in_ch, out_ch, rate))
        branches.append(_ASPPPooling(in_ch, out_ch))
        self.branches = nn.ModuleList(branches)
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * len(self.branches), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([b(x) for b in self.branches], dim=1))


class DeepLabV3PlusDecoder(nn.Module):
    def __init__(self, low_ch: int, aspp_ch: int, num_classes: int) -> None:
        super().__init__()
        self.reduce_low = nn.Sequential(
            nn.Conv2d(low_ch, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.refine = nn.Sequential(
            DepthwiseSeparableConv(aspp_ch + 64, 256),
            DepthwiseSeparableConv(256, 256),
            DepthwiseSeparableConv(256, 256),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, aspp_out: torch.Tensor, low_level: torch.Tensor) -> torch.Tensor:
        low = self.reduce_low(low_level)
        high = F.interpolate(aspp_out, size=low.shape[2:], mode="bilinear", align_corners=False)
        return self.refine(torch.cat([high, low], dim=1))


class AuxHead(nn.Sequential):
    def __init__(self, in_ch: int, num_classes: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1),
        )


class ImprovedDeepLabV3Plus(nn.Module):
    """DeepLabV3+ with ResNet-101, CBAM attention, deeper decoder,
    dual auxiliary heads, and backbone freezing.

    During training: returns {"out": logits, "aux": aux_logits, "aux2": aux2_logits}
    During eval: returns plain logits tensor.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained_backbone: bool = True,
        output_stride: int = 16,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self._freeze_bn_stages = freeze_backbone

        if output_stride == 16:
            replace_stride = [False, False, True]
            atrous_rates = (6, 12, 18)
        elif output_stride == 8:
            replace_stride = [False, True, True]
            atrous_rates = (12, 24, 36)
        else:
            raise ValueError(f"output_stride must be 8 or 16, got {output_stride}")

        backbone = self._make_backbone(pretrained_backbone, replace_stride)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.aspp = ASPP(2048, atrous_rates)
        self.cbam = CBAM(256)

        self.decoder = DeepLabV3PlusDecoder(low_ch=256, aspp_ch=256, num_classes=num_classes)
        self.aux_head = AuxHead(1024, num_classes)
        self.aux_head2 = AuxHead(512, num_classes)

        if freeze_backbone:
            self._freeze_early_layers()

    @staticmethod
    def _make_backbone(pretrained: bool, replace_stride: list[bool]):
        try:
            from torchvision.models import ResNet101_Weights
            weights = ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
            return resnet101(weights=weights, replace_stride_with_dilation=replace_stride)
        except (ImportError, TypeError):
            return resnet101(pretrained=pretrained, replace_stride_with_dilation=replace_stride)

    def _freeze_early_layers(self) -> None:
        for module in [self.stem, self.layer1]:
            for param in module.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self._freeze_bn_stages:
            self.stem.eval()
            self.layer1.eval()
        return self

    def forward(self, x: torch.Tensor):
        input_size = x.shape[2:]
        x = self.stem(x)
        low_level = self.layer1(x)
        x = self.layer2(low_level)
        aux2_feat = x
        x = self.layer3(x)
        aux_feat = x
        x = self.layer4(x)

        x = self.cbam(self.aspp(x))
        x = self.decoder(x, low_level)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)

        if self.training:
            aux = self.aux_head(aux_feat)
            aux = F.interpolate(aux, size=input_size, mode="bilinear", align_corners=False)
            aux2 = self.aux_head2(aux2_feat)
            aux2 = F.interpolate(aux2, size=input_size, mode="bilinear", align_corners=False)
            return {"out": x, "aux": aux, "aux2": aux2}

        return x
