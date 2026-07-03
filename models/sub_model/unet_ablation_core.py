import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pytorch_wavelets import DWTForward
except ImportError:
    DWTForward = None


def _num_groups(channels: int, max_groups: int = 8) -> int:
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return max(groups, 1)


class RMSNorm2d(nn.Module):

    def __init__(self, channels: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.scale


class ConvUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvUnit(in_channels, out_channels),
            ConvUnit(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttnResEncoderStage(nn.Module):
    def __init__(self, stage_index: int, stage_channels: List[int], stem_channels: int):
        super().__init__()
        self.stage_index = stage_index
        self.out_channels = stage_channels[stage_index]

        source_channels = [stem_channels] + stage_channels[:stage_index] + [self.out_channels]
        self.adapters = nn.ModuleList()
        for in_channels in source_channels:
            if in_channels == self.out_channels:
                self.adapters.append(nn.Identity())
            else:
                self.adapters.append(nn.Conv2d(in_channels, self.out_channels, kernel_size=1, bias=False))

        self.query1 = nn.Parameter(torch.zeros(self.out_channels))
        self.query2 = nn.Parameter(torch.zeros(self.out_channels))
        self.norm1 = RMSNorm2d(self.out_channels)
        self.norm2 = RMSNorm2d(self.out_channels)

        self.conv1 = ConvUnit(self.out_channels, self.out_channels)
        self.conv2 = ConvUnit(self.out_channels, self.out_channels)

    def _target_size(self, stem: torch.Tensor, prev_blocks: List[torch.Tensor]) -> Tuple[int, int]:
        if self.stage_index == 0:
            return stem.shape[-2:]
        height, width = prev_blocks[-1].shape[-2:]
        return height // 2, width // 2

    def _project_and_resize(
            self,
            x: torch.Tensor,
            adapter_idx: int,
            target_size: Tuple[int, int],
    ) -> torch.Tensor:
        x = self.adapters[adapter_idx](x)
        if x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x

    def _attn_res(
            self,
            sources: List[torch.Tensor],
            adapter_indices: List[int],
            query: torch.Tensor,
            norm: RMSNorm2d,
            target_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        values = [
            self._project_and_resize(src, idx, target_size)
            for src, idx in zip(sources, adapter_indices)
        ]
        value_tensor = torch.stack(values, dim=1)
        key_tensor = torch.stack([norm(value) for value in values], dim=1)

        q = query.view(1, 1, -1, 1, 1)
        logits = (key_tensor * q).sum(dim=2) / math.sqrt(query.numel())
        alpha = torch.softmax(logits, dim=1)
        mixed = (alpha.unsqueeze(2) * value_tensor).sum(dim=1)
        return mixed, alpha

    def forward(
            self,
            stem: torch.Tensor,
            prev_blocks: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        target_size = self._target_size(stem, prev_blocks)
        history = [stem] + list(prev_blocks)

        adapter_indices_1 = list(range(len(history)))
        h1, alpha1 = self._attn_res(history, adapter_indices_1, self.query1, self.norm1, target_size)
        partial = self.conv1(h1)

        adapter_indices_2 = list(range(len(history))) + [len(history)]
        h2, alpha2 = self._attn_res(history + [partial], adapter_indices_2, self.query2, self.norm2, target_size)
        block_representation = partial + self.conv2(h2)

        attn_info = {
            "layer1": alpha1,
            "layer2": alpha2,
        }
        return block_representation, attn_info


class WaveletSkipBlock(nn.Module):
    def __init__(self, in_channels: int, previous_wavelet_channels: Optional[int] = None):
        super().__init__()
        if DWTForward is None:
            raise ImportError(
                "WT variants require pytorch_wavelets. Install it with "
                "`pip install pytorch_wavelets PyWavelets`."
            )

        self.dwt = DWTForward(J=1, mode="zero", wave="haar")

        if previous_wavelet_channels is None:
            self.previous_adapter = None
            self.input_fuse = None
        else:
            self.previous_adapter = ConvNormAct(previous_wavelet_channels, in_channels, kernel_size=1)
            self.input_fuse = ConvNormAct(in_channels * 2, in_channels, kernel_size=3)

        self.down_fusion = nn.Sequential(
            ConvNormAct(in_channels * 4, in_channels * 4, kernel_size=3),
            ConvNormAct(in_channels * 4, in_channels * 4, kernel_size=3),
        )

        self.skip_restore = nn.Sequential(
            ConvNormAct(in_channels * 4, in_channels * 2, kernel_size=3),
            ConvNormAct(in_channels * 2, in_channels, kernel_size=1),
        )
        self.residual_refine = ConvNormAct(in_channels, in_channels, kernel_size=3)

    def _resize_to(self, x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        if x.shape[-2:] != size:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return x

    def forward(
            self,
            encoder_feature: torch.Tensor,
            previous_wavelet_feature: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_dtype = encoder_feature.dtype
        device_type = encoder_feature.device.type

        with torch.amp.autocast(device_type=device_type, enabled=False):
            encoder_feature_fp32 = encoder_feature.float()
            fused_input = encoder_feature_fp32

            if previous_wavelet_feature is not None:
                if self.previous_adapter is None or self.input_fuse is None:
                    raise ValueError("Current wavelet block does not expect previous wavelet features.")
                previous_wavelet_feature = self._resize_to(previous_wavelet_feature, encoder_feature.shape[-2:]).float()
                previous_wavelet_feature = self.previous_adapter(previous_wavelet_feature)
                fused_input = self.input_fuse(torch.cat([encoder_feature_fp32, previous_wavelet_feature], dim=1))
                fused_input = fused_input + encoder_feature_fp32

            low_frequency, high_frequency = self.dwt(fused_input)
            high_frequency = high_frequency[0]

            hl = high_frequency[:, :, 0, :, :]
            lh = high_frequency[:, :, 1, :, :]
            hh = high_frequency[:, :, 2, :, :]

            stacked_frequency = torch.cat([low_frequency, hl, lh, hh], dim=1)
            propagated_feature = self.down_fusion(stacked_frequency)

            skip_feature = self.skip_restore(propagated_feature)
            skip_feature = self._resize_to(skip_feature, encoder_feature.shape[-2:])
            skip_feature = self.residual_refine(skip_feature + encoder_feature_fp32)

        if skip_feature.dtype != input_dtype:
            skip_feature = skip_feature.to(dtype=input_dtype)
        if propagated_feature.dtype != input_dtype:
            propagated_feature = propagated_feature.to(dtype=input_dtype)

        return skip_feature, propagated_feature


class WaveletSkipPyramid(nn.Module):
    def __init__(self, encoder_channels: Sequence[int]):
        super().__init__()
        if len(encoder_channels) != 4:
            raise ValueError("WaveletSkipPyramid expects 4 encoder stages.")

        blocks: List[WaveletSkipBlock] = []
        for idx, channels in enumerate(encoder_channels):
            previous_wavelet_channels = None if idx == 0 else encoder_channels[idx - 1] * 4
            blocks.append(
                WaveletSkipBlock(
                    in_channels=channels,
                    previous_wavelet_channels=previous_wavelet_channels,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(
            self,
            encoder_features: Sequence[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if len(encoder_features) != len(self.blocks):
            raise ValueError(f"Expected {len(self.blocks)} encoder features, got {len(encoder_features)}.")

        decoder_skips: List[torch.Tensor] = []
        propagated_features: List[torch.Tensor] = []
        previous_wavelet_feature: Optional[torch.Tensor] = None

        for block, encoder_feature in zip(self.blocks, encoder_features):
            skip_feature, previous_wavelet_feature = block(encoder_feature, previous_wavelet_feature)
            decoder_skips.append(skip_feature)
            propagated_features.append(previous_wavelet_feature)

        return decoder_skips, propagated_features


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.fuse = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class DecoderBridgeFusion(nn.Module):
    def __init__(self, deep_channels: int, shallow_channels: int, out_channels: int):
        super().__init__()
        self.deep_proj = ConvNormAct(deep_channels, out_channels, kernel_size=1)
        self.shallow_proj = ConvNormAct(shallow_channels, out_channels, kernel_size=1)
        self.gate = nn.Sequential(
            ConvNormAct(out_channels * 2, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, 2, kernel_size=1, bias=True),
        )
        self.refine = ConvBlock(out_channels * 2, out_channels)

    def forward(self, deep_feature: torch.Tensor, shallow_feature: torch.Tensor) -> torch.Tensor:
        deep_feature = F.interpolate(
            deep_feature,
            size=shallow_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        deep_feature = self.deep_proj(deep_feature)
        shallow_feature = self.shallow_proj(shallow_feature)

        pair_feature = torch.cat([deep_feature, shallow_feature], dim=1)
        gate = torch.softmax(self.gate(pair_feature), dim=1)
        bridged_feature = deep_feature * gate[:, 0:1] + shallow_feature * gate[:, 1:2]

        fused_feature = self.refine(pair_feature)
        return fused_feature + bridged_feature


class DecoderBridgeHead(nn.Module):
    def __init__(self, branch_channels: int, num_classes: int):
        super().__init__()
        self.weight_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvNormAct(branch_channels * 3, branch_channels, kernel_size=1),
            nn.Conv2d(branch_channels, 3, kernel_size=1, bias=True),
        )
        self.fuse = ConvBlock(branch_channels * 3, branch_channels)
        self.head = nn.Conv2d(branch_channels, num_classes, kernel_size=1)

    def forward(
            self,
            bridge_43: torch.Tensor,
            bridge_32: torch.Tensor,
            bridge_21: torch.Tensor,
    ) -> torch.Tensor:
        target_size = bridge_21.shape[-2:]
        bridge_43 = F.interpolate(bridge_43, size=target_size, mode="bilinear", align_corners=False)
        bridge_32 = F.interpolate(bridge_32, size=target_size, mode="bilinear", align_corners=False)

        stacked = torch.cat([bridge_43, bridge_32, bridge_21], dim=1)
        weights = torch.softmax(self.weight_predictor(stacked), dim=1)

        consensus = (
                bridge_43 * weights[:, 0:1]
                + bridge_32 * weights[:, 1:2]
                + bridge_21 * weights[:, 2:3]
        )
        fused_feature = self.fuse(stacked) + consensus
        return self.head(fused_feature)


class AblationUNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            num_classes: int = 2,
            base_channels: int = 32,
            use_ar: bool = False,
            use_wt: bool = False,
            use_db: bool = False,
    ):
        super().__init__()
        self.use_ar = use_ar
        self.use_wt = use_wt
        self.use_db = use_db

        stage_channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]

        if self.use_ar:
            self.stem = ConvBlock(in_channels, stage_channels[0])
            self.encoder_stages = nn.ModuleList(
                [
                    AttnResEncoderStage(i, stage_channels=stage_channels, stem_channels=stage_channels[0])
                    for i in range(4)
                ]
            )
        else:
            self.enc1 = ConvBlock(in_channels, stage_channels[0])
            self.enc2 = DownBlock(stage_channels[0], stage_channels[1])
            self.enc3 = DownBlock(stage_channels[1], stage_channels[2])
            self.enc4 = DownBlock(stage_channels[2], stage_channels[3])

        if self.use_wt:
            self.wavelet_skip = WaveletSkipPyramid(stage_channels)

        self.dec3 = UpBlock(stage_channels[3], stage_channels[2], stage_channels[2])
        self.dec2 = UpBlock(stage_channels[2], stage_channels[1], stage_channels[1])
        self.dec1 = UpBlock(stage_channels[1], stage_channels[0], stage_channels[0])

        if self.use_db:
            bridge_channels = stage_channels[0]
            self.bridge_fusion_43 = DecoderBridgeFusion(stage_channels[3], stage_channels[2], bridge_channels)
            self.bridge_fusion_32 = DecoderBridgeFusion(stage_channels[2], stage_channels[1], bridge_channels)
            self.bridge_fusion_21 = DecoderBridgeFusion(stage_channels[1], stage_channels[0], bridge_channels)
            self.bridge_head = DecoderBridgeHead(bridge_channels, num_classes)
        else:
            self.head = nn.Conv2d(stage_channels[0], num_classes, kernel_size=1)

    def _encode(
            self,
            x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]:
        if self.use_ar:
            stem = self.stem(x)
            x1, a1 = self.encoder_stages[0](stem, [])
            x2, a2 = self.encoder_stages[1](stem, [x1])
            x3, a3 = self.encoder_stages[2](stem, [x1, x2])
            x4, a4 = self.encoder_stages[3](stem, [x1, x2, x3])
            attn_maps = {
                "enc1": a1,
                "enc2": a2,
                "enc3": a3,
                "enc4": a4,
            }
            return x1, x2, x3, x4, attn_maps

        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        return x1, x2, x3, x4, {}

    def forward(
            self,
            x: torch.Tensor,
            return_attn: bool = False,
    ):
        x1, x2, x3, x4, attn_maps = self._encode(x)

        if self.use_wt:
            wavelet_skips, _ = self.wavelet_skip([x1, x2, x3, x4])
            skip1, skip2, skip3, skip4 = wavelet_skips
        else:
            skip1, skip2, skip3, skip4 = x1, x2, x3, x4

        d4 = skip4
        d3 = self.dec3(d4, skip3)
        d2 = self.dec2(d3, skip2)
        d1 = self.dec1(d2, skip1)

        if self.use_db:
            bridge_43 = self.bridge_fusion_43(d4, d3)
            bridge_32 = self.bridge_fusion_32(d3, d2)
            bridge_21 = self.bridge_fusion_21(d2, d1)
            logits = self.bridge_head(bridge_43, bridge_32, bridge_21)
        else:
            logits = self.head(d1)

        if not return_attn:
            return {'out': logits}

        return {'out': logits, 'attn_maps': attn_maps}
