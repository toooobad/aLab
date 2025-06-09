# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import torch
import torch.nn as nn

from torch import Tensor
from loguru import logger
from torch.hub import load_state_dict_from_url
from typing import (Callable, List, Optional, Type, Union, Tuple)
from torchvision.models.resnet import (BasicBlock, Bottleneck, conv1x1)

from aLab.register import BACKBONE


__all__ = ['ResNet']


@BACKBONE.register_module()
class ResNet(nn.Module):
    configs = {
        18: dict(block=BasicBlock, layers=(2, 2, 2, 2), pretrained='https://download.pytorch.org/models/resnet18-f37072fd.pth'),
        34: dict(block=BasicBlock, layers=(3, 4, 6, 3), pretrained='https://download.pytorch.org/models/resnet34-b627a593.pth'),
        50: dict(block=Bottleneck, layers=(3, 4, 6, 3), pretrained='https://download.pytorch.org/models/resnet50-0676ba61.pth'),
        101: dict(block=Bottleneck, layers=(3, 4, 23, 3), pretrained='https://download.pytorch.org/models/resnet101-63fe2227.pth'),
        152: dict(block=Bottleneck, layers=(3, 8, 36, 3), pretrained='https://download.pytorch.org/models/resnet152-394f9c45.pth'),
        '50_32x4d': dict(block=Bottleneck, layers=(3, 4, 6, 3), groups=32, width_per_group=4, pretrained='https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth'),
        '101_32x8d': dict(block=Bottleneck, layers=(3, 4, 23, 3), groups=32, width_per_group=8, pretrained='https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth'),
        'wide_50_2': dict(block=Bottleneck, layers=(3, 4, 6, 3), width_per_group=64 * 2, pretrained='https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth'),
        'wide_101_2': dict(block=Bottleneck, layers=(3, 4, 23, 3), width_per_group=64 * 2, pretrained='https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth')
    }

    def __init__(self,
                 depth: Union[int, str],
                 deep_stem: bool = False,
                 num_classes: int = 1000,
                 return_classes: bool = False,
                 zero_init_residual: bool = False,
                 out_indices: Tuple[int, ...]=(0, 1, 2, 3),
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 replace_stride_with_dilation: Optional[List[bool]] = None) -> None:
        """
        Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>

        Args:
            depth (str): Depth of resnet.
            deep_stem (bool, optional): Replace 7x7 conv in input stem with 3 3x3 conv. Defaults to False.
            num_classes (int, optional): _description_. Defaults to 1000.
            return_classes (bool, optional): _description_. Defaults to False.
            zero_init_residual (bool, optional): _description_. Defaults to False.
            out_indices (Tuple[int, ...], optional): _description_. Defaults to (0, 1, 2, 3).
            norm_layer (Optional[Callable[..., nn.Module]], optional): _description_. Defaults to None.
            replace_stride_with_dilation (Optional[List[bool]], optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
        """
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        _config = self.configs.get(depth, None)
        assert _config is not None, ValueError(f'Expect {list(self.configs.keys())}, but got "{depth}"')

        self.dilation = 1
        self.inplanes = 64
        self.groups = _config.get('groups', 1)
        self._norm_layer = norm_layer
        self.out_indices = out_indices
        self.base_width = _config.get('width_per_group', 64)
        
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError(f"replace_stride_with_dilation should be None or a 3-element tuple, got {replace_stride_with_dilation}")

        # conv1
        if deep_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(self.inplanes),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(self.inplanes),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # stage1, 2, 3, 4
        block = self.configs[depth]['block']
        layers = self.configs[depth]['layers']
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        self.return_classes = return_classes
        if self.return_classes:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

        # load pretrained
        pretrained = _config.get('pretrained', None)
        if pretrained is not None:
            self.load_state_dict(load_state_dict_from_url(pretrained, progress=True), strict=False)
            logger.info(f'Load ResNet{depth} checkpoints from "{pretrained}".')
    
    def _make_layer(self,
                    block: Type[Union[BasicBlock, Bottleneck]],
                    planes: int,
                    blocks: int,
                    stride: int = 1,
                    dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)
    
    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outs = []
        for i in range(1, 5):
            x = getattr(self, f'layer{i}')(x)

            if (i - 1) in self.out_indices:
                outs.append(x)
        
        if self.return_classes:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x
        else:
            return tuple(outs)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        return self._forward_impl(x)
