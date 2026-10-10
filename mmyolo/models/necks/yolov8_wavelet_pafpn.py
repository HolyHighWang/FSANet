# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.utils import ConfigType, OptMultiConfig
from mmengine import MessageHub
from torch import Tensor

from mmyolo.registry import MODELS
from ..utils import make_divisible
from .yolov8_pafpn import YOLOv8PAFPN


class HaarDWTMixin:
    """Utility mixin for fixed Haar wavelet decomposition/reconstruction."""

    @staticmethod
    def _pad_to_even(x: Tensor) -> Tensor:
        pad_h = x.shape[-2] % 2
        pad_w = x.shape[-1] % 2
        if pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

    def haar_dwt(self, x: Tensor):
        x = self._pad_to_even(x)
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (x00 - x01 + x10 - x11) * 0.5
        hl = (x00 + x01 - x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return ll, lh, hl, hh

    @staticmethod
    def haar_high_reconstruct(lh: Tensor, hl: Tensor, hh: Tensor,
                              out_size: Sequence[int]) -> Tensor:
        y00 = (lh + hl + hh) * 0.5
        y01 = (-lh + hl - hh) * 0.5
        y10 = (lh - hl - hh) * 0.5
        y11 = (-lh - hl + hh) * 0.5

        out = y00.new_zeros(y00.shape[0], y00.shape[1], y00.shape[2] * 2,
                            y00.shape[3] * 2)
        out[:, :, 0::2, 0::2] = y00
        out[:, :, 0::2, 1::2] = y01
        out[:, :, 1::2, 0::2] = y10
        out[:, :, 1::2, 1::2] = y11
        return out[:, :, :out_size[0], :out_size[1]]


class WaveletDownsample(nn.Module, HaarDWTMixin):
    """Haar DWT downsampling that keeps low- and high-frequency bands."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_cfg: ConfigType,
                 act_cfg: ConfigType):
        super().__init__()
        self.fuse = ConvModule(
            in_channels * 4,
            out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x: Tensor) -> Tensor:
        bands = self.haar_dwt(x)
        return self.fuse(torch.cat(bands, dim=1))


class WaveletFeatureEnhance(nn.Module, HaarDWTMixin):
    """Residual high-frequency enhancement for FPN outputs."""

    def __init__(self,
                 channels: int,
                 norm_cfg: ConfigType,
                 act_cfg: ConfigType,
                 init_scale: float = 0.0,
                 max_scale: float = -1.0):
        super().__init__()
        self.high_fuse = ConvModule(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.max_scale = float(max_scale)
        if self.max_scale > 0:
            init_scale = max(min(float(init_scale), self.max_scale),
                             -self.max_scale)
            init_raw = torch.atanh(
                torch.tensor(init_scale / self.max_scale, dtype=torch.float32))
            self.scale = nn.Parameter(init_raw)
        else:
            self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.runtime_scale = 1.0

    def forward(self, x: Tensor) -> Tensor:
        _, lh, hl, hh = self.haar_dwt(x)
        high = self.haar_high_reconstruct(lh, hl, hh, x.shape[-2:])
        if self.max_scale > 0:
            scale = self.max_scale * self.scale.tanh()
        else:
            scale = self.scale
        return x + self.runtime_scale * scale * self.high_fuse(high)


@MODELS.register_module()
class YOLOv8WaveletPAFPN(YOLOv8PAFPN):
    """YOLOv8 PAFPN with Haar-wavelet frequency enhancement.

    The bottom-up path can replace stride-2 convolution with fixed Haar DWT
    downsampling, and selected output levels can receive residual high-frequency
    enhancement. The default setting focuses on P3 for small objects.
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 freeze_all: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 use_wavelet_downsample: bool = True,
                 wavelet_enhance_levels: Sequence[int] = (0,),
                 wavelet_enhance_init_scale: float = 0.0,
                 wavelet_enhance_max_scale: float = -1.0,
                 wavelet_enhance_warmup_start_epoch: int = -1,
                 wavelet_enhance_warmup_end_epoch: int = -1,
                 init_cfg: OptMultiConfig = None):
        self.use_wavelet_downsample = use_wavelet_downsample
        self.wavelet_enhance_levels = tuple(wavelet_enhance_levels)
        self.wavelet_enhance_init_scale = wavelet_enhance_init_scale
        self.wavelet_enhance_max_scale = wavelet_enhance_max_scale
        self.wavelet_enhance_warmup_start_epoch = (
            wavelet_enhance_warmup_start_epoch)
        self.wavelet_enhance_warmup_end_epoch = (
            wavelet_enhance_warmup_end_epoch)
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            num_csp_blocks=num_csp_blocks,
            freeze_all=freeze_all,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)

    def _level_channels(self, idx: int) -> int:
        if isinstance(self.out_channels, int):
            channels = self.out_channels
        else:
            channels = self.out_channels[idx]
        return make_divisible(channels, self.widen_factor)

    def build_downsample_layer(self, idx: int) -> nn.Module:
        if not self.use_wavelet_downsample:
            return super().build_downsample_layer(idx)

        channels = self._level_channels(idx)
        return WaveletDownsample(
            channels,
            channels,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_out_layer(self, idx: int) -> nn.Module:
        if idx not in self.wavelet_enhance_levels:
            return nn.Identity()

        return WaveletFeatureEnhance(
            self._level_channels(idx),
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            init_scale=self.wavelet_enhance_init_scale,
            max_scale=self.wavelet_enhance_max_scale)

    def _current_wavelet_enhance_scale(self) -> float:
        if (self.wavelet_enhance_warmup_start_epoch < 0
                or self.wavelet_enhance_warmup_end_epoch <=
                self.wavelet_enhance_warmup_start_epoch):
            return 1.0

        message_hub = MessageHub.get_current_instance()
        current_epoch = message_hub.get_info('epoch')
        if current_epoch is None:
            return 1.0
        current_epoch = int(current_epoch)

        if current_epoch < self.wavelet_enhance_warmup_start_epoch:
            return 0.0
        if current_epoch >= self.wavelet_enhance_warmup_end_epoch:
            return 1.0

        progress = (
            (current_epoch - self.wavelet_enhance_warmup_start_epoch) /
            (self.wavelet_enhance_warmup_end_epoch -
             self.wavelet_enhance_warmup_start_epoch))
        return float(progress)

    def forward(self, inputs: List[Tensor]) -> tuple:
        warmup_scale = self._current_wavelet_enhance_scale()
        for layer in self.out_layers:
            if isinstance(layer, WaveletFeatureEnhance):
                layer.runtime_scale = warmup_scale
        return super().forward(inputs)
