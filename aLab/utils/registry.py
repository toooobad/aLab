# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

from loguru import logger
from tabulate import tabulate
from typing import (Dict, Type, Optional, List, Union, Callable, Any)


__all__ = ['Registry']
    

class Registry:
    def __init__(self, name: str) -> None:
        self._name = name
        self._module_dict: Dict[str, Type] = dict()
    
    @property
    def name(self):
        return self._name
    
    @property
    def module_dict(self):
        return self._module_dict
    
    def __len__(self) -> int:
        return len(self._module_dict)
    
    def __repr__(self) -> str:
        data = []
        for name, obj in self.module_dict.items():
            data.append([name, str(obj)])
        table = tabulate(data, headers=[self.name, 'Objects'], tablefmt='grid')
        return table
    
    def get(self, name: str) -> Any:
        if name in self._module_dict:
            obj_cls = self._module_dict[name]
            return obj_cls
        else:
            logger.error(f'{name} not in {self.name} Registry!')

    def _register_module(self,
                         module: Type,
                         module_name: Optional[Union[str, List[str]]] = None,
                         force: bool = False) -> None:
        """Register a module.

        Args:
            module (type): Module to be registered. Typically a class or a
                function, but generally all ``Callable`` are acceptable.
            module_name (str or list of str, optional): The module name to be
                registered. If not specified, the class name will be used.
                Defaults to None.
            force (bool): Whether to override an existing class with the same
                name. Defaults to False.
        """
        if not callable(module):
            raise TypeError(f'module must be Callable, but got {type(module)}')

        if module_name is None:
            module_name = module.__name__

        if isinstance(module_name, str):
            module_name = [module_name]

        for name in module_name:
            if not force and name in self._module_dict:
                existed_module = self.module_dict[name]
                raise KeyError(f'{name} is already registered in {self.name} at {existed_module.__module__}')
            self._module_dict[name] = module

    def register_module(self,
                        name: Optional[Union[str, List[str]]] = None,
                        force: bool = False,
                        module: Optional[Type] = None) -> Union[type, Callable]:
        
        if not isinstance(force, bool):
            raise TypeError(f'force must be a boolean, but got {type(force)}')

        # raise the error ahead of time
        if not (name is None or isinstance(name, str)):
            raise TypeError(f'name must be None, an instance of str but got {type(name)}')

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(module=module, module_name=name, force=force)
            return module

        # use it as a decorator: @x.register_module()
        def _register(module):
            self._register_module(module=module, module_name=name, force=force)
            return module

        return _register
    
    def build(self, cfg: dict):
        return build_from_cfg(cfg, registry=self)
    

def build_from_cfg(cfg: dict, registry: Registry) -> Any:
    assert 'type' in cfg, ValueError(f"'type' must be ")
    obj_type = cfg.pop('type')
    obj_cls = registry.get(obj_type)
    return obj_cls(**cfg)
