# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import torch
import numpy as np

from copy import deepcopy
from typing import (Optional, Any, Union, Iterator, Tuple, Type)


class DataSample:
    """
    A base data interface that supports Tensor-like and dict-like operations.
    copy from "mmengine.structures.base_data_element"
    """
    def __init__(self, *, metainfo: Optional[dict] = None, **kwargs) -> None:
        
        self._data_fields: set = set()
        self._metainfo_fields: set = set()

        if kwargs:
            self.set_data(kwargs)

        if metainfo is not None:
            self.set_metainfo(metainfo=metainfo)

    def set_data(self, data: dict) -> None:
        """
        Set or change key-value pairs in ``data_field`` by parameter ``data``.

        grammar: obj.key = value

        Args:
            data (dict): A dict contains annotations of image or model predictions.
        """
        assert isinstance(data, dict), f'data should be a `dict` but got {data}'
        for k, v in data.items():
            # Use `setattr()` rather than `self.set_field` to allow `set_data` to set property method.
            setattr(self, k, v)
    
    def set_metainfo(self, metainfo: dict) -> None:
        """
        Set or change key-value pairs in ``metainfo_field`` by parameter ``metainfo``.

        Args:
            metainfo (dict): A dict contains the meta information of image, such as ``img_shape``, ``scale_factor``, etc.
        """
        assert isinstance(metainfo, dict), f'metainfo should be a ``dict`` but got {type(metainfo)}'
        meta = deepcopy(metainfo)

        for k, v in meta.items():
            self.set_field(name=k, value=v, field_type='metainfo', dtype=None)

    def set_field(self, value: Any, name: str, dtype: Optional[Union[Type, Tuple[Type, ...]]] = None, field_type: str = 'data') -> None:
        """
        Special method for set union field, used as property.setter functions.
        """
        assert field_type in ['metainfo', 'data']
        if dtype is not None:
            assert isinstance(value, dtype), f'{value} should be a {dtype} but got {type(value)}'

        if field_type == 'metainfo':
            if name in self._data_fields:
                raise AttributeError(f'Cannot set {name} to be a field of metainfo because {name} is already a data field')
            self._metainfo_fields.add(name)
        else:
            if name in self._metainfo_fields:
                raise AttributeError(f'Cannot set {name} to be a field of data because {name} is already a metainfo field')
            self._data_fields.add(name)

        super().__setattr__(name, value)

    def metainfo_keys(self) -> list:
        """
        Returns:
            list: Contains all keys in metainfo_fields.
        """
        return list(self._metainfo_fields)
    
    def metainfo_values(self) -> list:
        """
        Returns:
            list: Contains all values in metainfo.
        """
        return [getattr(self, k) for k in self.metainfo_keys()]
    
    def metainfo_items(self) -> Iterator[Tuple[str, Any]]:
        """
        Returns:
            iterator: An iterator object whose element is (key, value) tuple pairs for ``metainfo``.
        """
        for k in self.metainfo_keys():
            yield (k, getattr(self, k))

    def keys(self) -> list:
        """
        Returns:
            list: Contains all keys in data_fields.
        """
        # We assume that the name of the attribute related to property is
        # '_' + the name of the property. We use this rule to filter out
        # private keys.
        # TODO: Use a more robust way to solve this problem
        private_keys = {'_' + key for key in self._data_fields if isinstance(getattr(type(self), key, None), property)}
        return list(self._data_fields - private_keys)

    def values(self) -> list:
        """
        Returns:
            list: Contains all values in data.
        """
        return [getattr(self, k) for k in self.keys()]
    
    def items(self) -> Iterator[Tuple[str, Any]]:
        """
        Returns:
            iterator: An iterator object whose element is (key, value) tuple
            pairs for ``data``.
        """
        for k in self.keys():
            yield (k, getattr(self, k))

    def all_keys(self) -> list:
        """
        Returns:
            list: Contains all keys in metainfo and data.
        """
        return self.metainfo_keys() + self.keys()

    def all_values(self) -> list:
        """
        Returns:
            list: Contains all values in metainfo and data.
        """
        return self.metainfo_values() + self.values()

    def all_items(self) -> Iterator[Tuple[str, Any]]:
        """
        Returns:
            iterator: An iterator object whose element is (key, value) tuple
            pairs for ``metainfo`` and ``data``.
        """
        for k in self.all_keys():
            yield (k, getattr(self, k))

    def get(self, key, default=None) -> Any:
        """Get property in data and metainfo as the same as python."""
        # Use `getattr()` rather than `self.__dict__.get()` to allow getting
        # properties.
        return getattr(self, key, default)
    
    def pop(self, *args) -> Any:
        """Pop property in data and metainfo as the same as python."""
        assert len(args) < 3, '``pop`` get more than 2 arguments'
        name = args[0]
        if name in self._metainfo_fields:
            self._metainfo_fields.remove(args[0])
            return self.__dict__.pop(*args)

        elif name in self._data_fields:
            self._data_fields.remove(args[0])
            return self.__dict__.pop(*args)

        # with default value
        elif len(args) == 2:
            return args[1]
        else:
            # don't just use 'self.__dict__.pop(*args)' for only popping key in
            # metainfo or data
            raise KeyError(f'{args[0]} is not contained in metainfo or data')
    
    def update(self, instance: 'DataSample') -> None:
        """
        The update() method updates the DataSample with the elements from another DataSample object.

        Args:
            instance (DataSample): Another DataSample object for update the current object.
        """
        assert isinstance(instance, DataSample), f'instance should be a `DataSample` but got {type(instance)}'
        self.set_metainfo(dict(instance.metainfo_items()))
        self.set_data(dict(instance.items()))

    def new(self, *, metainfo: Optional[dict] = None, **kwargs) -> 'DataSample':
        """
        Return a new data element with same type. If ``metainfo`` and
        ``data`` are None, the new data element will have same metainfo and
        data. If metainfo or data is not None, the new result will overwrite it
        with the input value.

        Args:
            metainfo (dict, optional): A dict contains the meta information
                of image, such as ``img_shape``, ``scale_factor``, etc.
                Defaults to None.
            kwargs (dict): A dict contains annotations of image or
                model predictions.

        Returns:
            DataSample: A new data element with same type.
        """
        new_data = self.__class__()

        if metainfo is not None:
            new_data.set_metainfo(metainfo)
        else:
            new_data.set_metainfo(dict(self.metainfo_items()))

        if kwargs:
            new_data.set_data(kwargs)
        else:
            new_data.set_data(dict(self.items()))
            
        return new_data

    def __contains__(self, item: str) -> bool:
        """Whether the item is in dataelement.

        Args:
            item (str): The key to inquire.
        """
        return item in self._data_fields or item in self._metainfo_fields
    
    def __setattr__(self, name: str, value: Any):
        """Setattr is only used to set data."""
        if name in ('_metainfo_fields', '_data_fields'):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(f'{name} has been used as a private attribute, which is immutable.')
        else:
            self.set_field(name=name, value=value, field_type='data', dtype=None)
    
    def __delattr__(self, item: str):
        """
        Delete the item in dataelement.

        Args:
            item (str): The key to delete.
        """
        if item in ('_metainfo_fields', '_data_fields'):
            raise AttributeError(f'{item} has been used as a private attribute, which is immutable.')
        super().__delattr__(item)
        if item in self._metainfo_fields:
            self._metainfo_fields.remove(item)
        elif item in self._data_fields:
            self._data_fields.remove(item)

    # dict-like methods
    __delitem__ = __delattr__

    # Tensor-like methods
    def to(self, *args, **kwargs) -> 'DataSample':
        """Apply same name function to all tensors in data_fields."""
        new_data = self.new()
        for k, v in self.items():
            if hasattr(v, 'to'):
                v = v.to(*args, **kwargs)
                data = {k: v}
                new_data.set_data(data)
        return new_data
    
    # Tensor-like methods
    def cpu(self) -> 'DataSample':
        """Convert all tensors to CPU in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, DataSample)):
                v = v.cpu()
                data = {k: v}
                new_data.set_data(data)
        return new_data
    
    # Tensor-like methods
    def cuda(self) -> 'DataSample':
        """Convert all tensors to GPU in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, DataSample)):
                v = v.cuda()
                data = {k: v}
                new_data.set_data(data)
        return new_data
    
    # Tensor-like methods
    def detach(self) -> 'DataSample':
        """Detach all tensors in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, DataSample)):
                v = v.detach()
                data = {k: v}
                new_data.set_data(data)
        return new_data
    
    def clone(self):
        """
        Deep copy the current data element.

        Returns:
            DataSample: The copy of current data element.
        """
        clone_data = self.__class__()
        clone_data.set_metainfo(dict(self.metainfo_items()))
        clone_data.set_data(dict(self.items()))
        return clone_data
    
    # Tensor-like methods
    def numpy(self) -> 'DataSample':
        """Convert all tensors to np.ndarray in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, DataSample)):
                v = v.detach().cpu().numpy()
                data = {k: v}
                new_data.set_data(data)
        return new_data
    
    def to_tensor(self) -> 'DataSample':
        """Convert all np.ndarray to tensor in data."""
        new_data = self.new()
        for k, v in self.items():
            data = {}
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
                data[k] = v
            elif isinstance(v, DataSample):
                v = v.to_tensor()
                data[k] = v
            new_data.set_data(data)
        return new_data
    
    def to_dict(self) -> dict:
        """Convert DataSample to dict."""
        return {
            k: v.to_dict() if isinstance(v, DataSample) else v
            for k, v in self.all_items()
        }
    
    @property
    def metainfo(self) -> dict:
        """dict: A dict contains metainfo of current data element."""
        return dict(self.metainfo_items())
    
    def __repr__(self) -> str:
        """Represent the object."""

        def _addindent(s_: str, num_spaces: int) -> str:
            """This func is modified from `pytorch` https://github.com/pytorch/
            pytorch/blob/b17b2b1cc7b017c3daaeff8cc7ec0f514d42ec37/torch/nn/modu
            les/module.py#L29.

            Args:
                s_ (str): The string to add spaces.
                num_spaces (int): The num of space to add.

            Returns:
                str: The string after add indent.
            """
            s = s_.split('\n')
            # don't do anything for single-line stuff
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * ' ') + line for line in s]
            s = '\n'.join(s)  # type: ignore
            s = first + '\n' + s  # type: ignore
            return s  # type: ignore

        def dump(obj: Any) -> str:
            """Represent the object.

            Args:
                obj (Any): The obj to represent.

            Returns:
                str: The represented str.
            """
            _repr = ''
            if isinstance(obj, dict):
                for k, v in obj.items():
                    _repr += f'\n{k}: {_addindent(dump(v), 4)}'
            elif isinstance(obj, DataSample):
                _repr += '\n\n    META INFORMATION'
                metainfo_items = dict(obj.metainfo_items())
                _repr += _addindent(dump(metainfo_items), 4)
                _repr += '\n\n    DATA FIELDS'
                items = dict(obj.items())
                _repr += _addindent(dump(items), 4)
                classname = obj.__class__.__name__
                _repr = f'<{classname}({_repr}\n) at {hex(id(obj))}>'
            else:
                _repr += repr(obj)
            return _repr

        return dump(self)
    

if __name__ == '__main__':
    obj = DataSample()
    obj.a = 123
    obj['b'] = 234