# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import math
import torch.nn as nn

from torch import Tensor
from typing import (Union, List, Tuple)

from aLab.register import NECK


__all__ = ['CenterNetUpsampleNeck']


@NECK.register_module()
class CenterNetUpsampleNeck(nn.Module):
    # deconv配置, {deconv_kernel: (padding, output_padding)
    deconv_cfg = {
        4: (1, 0),
        3: (1, 1),
        2: (0, 0)
    }

    def __init__(self,
                 in_channels: int,
                 num_deconv_layers: int = 3,
                 upsample_mode: str = 'nearest',
                 num_deconv_kernels: Union[List[int], Tuple[int]] = [4, 4, 4],
                 num_deconv_filters: Union[List[int], Tuple[int]] = [256, 128, 64]) -> None:
        super(CenterNetUpsampleNeck, self).__init__()

        assert num_deconv_layers == len(num_deconv_filters), 'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_deconv_layers == len(num_deconv_kernels), 'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        self.deconv_layers = nn.ModuleList()

        # conv + bn + relu + up + (conv) + bn + relu
        for i in range(num_deconv_layers):
            layers = []
            deconv_kernel = num_deconv_kernels[i]
            feat_channels = num_deconv_filters[i]

            layers.append(nn.Conv2d(in_channels, feat_channels, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(feat_channels))
            layers.append(nn.ReLU(inplace=True))

            if upsample_mode == 'deconv':
                padding, output_padding = self.deconv_cfg[deconv_kernel]

                up = nn.ConvTranspose2d(
                    in_channels=feat_channels,
                    out_channels=feat_channels,
                    kernel_size=deconv_kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False)
                self._fill_up_weights(up)

                layers.append(up)
            else:
                assert deconv_kernel % 2 == 0, ValueError(f'When using nn.Upsample, deconv_kernel must be divisible by 2, but got {deconv_kernel}')

                up_cfg = dict(scale_factor=deconv_kernel // 2, mode=upsample_mode)
                if upsample_mode == 'bilinear':
                    up_cfg['align_corners'] = False

                layers.append(nn.Upsample(**up_cfg))
                layers.append(nn.Conv2d(feat_channels, feat_channels, kernel_size=3, stride=1, padding=1, bias=False))

            layers.append(nn.BatchNorm2d(feat_channels))
            layers.append(nn.ReLU(inplace=True))
            
            in_channels = feat_channels

            self.deconv_layers.append(nn.Sequential(*layers))

        self.init_weights()

    def init_weights(self) -> None:
        """
        Initialize the parameters.
        """
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    @staticmethod
    def _fill_up_weights(up: nn.Module) -> None:
        w = up.weight.data
        f = math.ceil(w.size(2) / 2)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(w.size(2)):
            for j in range(w.size(3)):
                w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))

        for c in range(1, w.size(0)):
            w[c, 0, :, :] = w[0, 0, :, :]

    def forward(self, img_feats: Union[Tuple[Tensor], Tensor]) -> Tuple[Tensor]:
        if isinstance(img_feats, Tensor):
            img_feats = [img_feats]

        img_feat = img_feats[-1]

        for layer in self.deconv_layers:
            img_feat = layer(img_feat)
        
        return (img_feat, )
