# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

from typing import (Union, Callable, Sequence, Optional, List)

from aLab.register import PREPROCESS

from .flip import *
from .crop import *
from .color import *
from .resize import *
from .loading import *
from .normalize import *


class ComposePreprocess:
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict, callable], optional): Sequence of transform
            object or config dict to be composed.
    """

    def __init__(self, transforms: Optional[Sequence[Union[dict, Callable]]]):
        self.transforms: List[Callable] = []

        if transforms is None:
            transforms = []

        for transform in transforms:
            # `Compose` can be built with config dict with type and corresponding arguments.
            if isinstance(transform, dict):
                transform = PREPROCESS.build(transform)
                if not callable(transform):
                    raise TypeError(f'transform should be a callable object, but got {type(transform)}')
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(f'transform must be a callable object or dict, but got {type(transform)}')

    def __call__(self, data: dict) -> Optional[dict]:
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """
        for t in self.transforms:
            data = t(data)
            # The transform will return None when it failed to load images or
            # cannot find suitable augmentation parameters to augment the data.
            # Here we simply return None if the transform returns None and the
            # dataset will handle it by randomly selecting another data sample.
            if data is None:
                return None
            
        return data

    def __repr__(self):
        """Print ``self.transforms`` in sequence.

        Returns:
            str: Formatted string.
        """
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string