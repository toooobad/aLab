# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

from torch.optim.lr_scheduler import StepLR as TorchStepLR
from torch.optim.lr_scheduler import MultiStepLR as TorchMultiStepLR
from torch.optim.lr_scheduler import CosineAnnealingLR as TorchCosineAnnealingLR

from aLab.register import LRSCHEDULER


__all__ = ['StepLR', 'MultiStepLR', 'CosineAnnealingLR']


@LRSCHEDULER.register_module()
class StepLR(TorchStepLR):
    def __init__(self, *args, **kwargs) -> None:
        super(StepLR, self).__init__(*args, **kwargs)


@LRSCHEDULER.register_module()
class MultiStepLR(TorchMultiStepLR):
    def __init__(self, *args, **kwargs) -> None:
        super(MultiStepLR, self).__init__(*args, **kwargs)


@LRSCHEDULER.register_module()
class CosineAnnealingLR(TorchCosineAnnealingLR):
    def __init__(self, *args, **kwargs) -> None:
        super(CosineAnnealingLR, self).__init__(*args, **kwargs)