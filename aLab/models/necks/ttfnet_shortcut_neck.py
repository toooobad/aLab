# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import torch.nn as nn

from torch import Tensor
from typing import Union, List, Tuple

from aLab.register import NECK


__all__ = ['TTFNetShortcutNeck']


@NECK.register_module()
class TTFNetShortcutNeck(nn.Module):
    def __init__(self,
                 kernel_size: int = 3,
                 upsample_mode: str = 'nearest',
                 stage_convs: Union[List[int], Tuple[int]] = (1, 2, 3),
                 in_channels: Union[List[int], Tuple[int]] = (64, 128, 256, 512),
                 out_channels: Union[List[int], Tuple[int]] = (256, 128, 64)) -> None:
        super(TTFNetShortcutNeck, self).__init__()
        assert len(out_channels) in [1, 2, 3, 4]

        shortcut_num = min(len(in_channels) - 1, len(out_channels))
        assert shortcut_num == len(stage_convs)

        self.out_channels = out_channels
        self.upsample_mode = upsample_mode

        # repeat upsampling n times.
        self.deconv_layers = nn.ModuleList([self.build_upsample(in_channels[-1], out_channels[0])])

        for i in range(1, len(out_channels)):
            self.deconv_layers.append(self.build_upsample(out_channels[i - 1], out_channels[i]))
            
        # shortcut
        padding = (kernel_size - 1) // 2
        self.shortcut_layers = self.build_shortcut(in_channels[:-1][::-1][:shortcut_num], 
                                                   out_channels[:shortcut_num], 
                                                   stage_convs,
                                                   kernel_size=kernel_size, 
                                                   padding=padding)
    
    def build_upsample(self, in_channels: int, out_channels: int) -> nn.Sequential:
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),  # TODO: replace by dcn
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]

        if self.upsample_mode == 'nearest':
            up = nn.Upsample(scale_factor=2, mode='nearest')
        elif self.upsample_mode == 'bilinear': 
            up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            up = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        
        layers.append(up)

        return nn.Sequential(*layers)

    def build_shortcut(self,
                       in_channels: Union[List[int], Tuple[int]],
                       out_channels: Union[List[int], Tuple[int]],
                       stage_convs: Union[List[int], Tuple[int]],
                       kernel_size: int = 3,
                       padding: int = 1) -> nn.ModuleList:
        assert len(in_channels) == len(out_channels) == len(stage_convs)

        shortcut_layers = nn.ModuleList()
        for (inp, outp, num_conv) in zip(in_channels, out_channels, stage_convs):
            assert num_conv > 0
            layer = ShortcutConv2d(inp, outp, [kernel_size] * num_conv, [padding] * num_conv)
            shortcut_layers.append(layer)

        return shortcut_layers
    
    def init_weights(self) -> None:
        """
        Initialize the parameters.
        """
        for _, m in self.shortcut_layers.named_modules():
            if isinstance(m, nn.Conv2d):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='relu')

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, feats: Tuple[Tensor]) -> Tuple[Tensor]:
        """model forward."""
        x = feats[-1]

        for i, upsample_layer in enumerate(self.deconv_layers):
            x = upsample_layer(x)
            if i < len(self.shortcut_layers):
                shortcut = self.shortcut_layers[i](feats[-i - 2])
                x = x + shortcut

        return x,
    

class ShortcutConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_sizes: Union[List[int], Tuple[int]],
                 paddings: Union[List[int], Tuple[int]],
                 activation_last: bool = False):
        super(ShortcutConv2d, self).__init__()
        assert len(kernel_sizes) == len(paddings)

        layers = []
        for i, (kernel_size, padding) in enumerate(zip(kernel_sizes, paddings)):
            inc = in_channels if i == 0 else out_channels
            layers.append(nn.Conv2d(inc, out_channels, kernel_size, padding=padding))
            if i < len(kernel_sizes) - 1 or activation_last:
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
