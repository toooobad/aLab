# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import os
import onnx
import torch
import shutil
import onnxruntime
import torch.nn as nn
import os.path as osp

from loguru import logger
from argparse import Namespace

from aLab.register import DEPLOYER
from aLab.builder import build_model
from aLab.utils import (heading, load_checkpoint, parameter, cosine_similarity)


__all__ = ['Det2dDeployer']


@DEPLOYER.register_module(name='det2d')
class Det2dDeployer:
    def __init__(self, cfgs: Namespace) -> None:
        # init
        self.cfgs = cfgs

        # load from
        self.load_from = self.cfgs.load_from

        # model
        self.model = self._build_model()

        if self.cfgs.input_h > 0 and self.cfgs.input_w > 0:
            self.input_size = (self.cfgs.input_h, self.cfgs.input_w)
        else:
            preprocesses = self.cfgs.train_dataset['preprocesses']
            for p in preprocesses:
                if 'input_size' in p:
                    self.input_size = p['input_size']
                    break

    def _build_model(self) -> nn.Module:
        # init model
        model = build_model(self.cfgs)

        # load checkpoint
        model = self._load_checkpoints(model)

        return model

    def _load_checkpoints(self, model: nn.Module) -> nn.Module:
        parameter(f'load from', self.load_from)
        assert osp.exists(self.load_from), ValueError(f'"{self.load_from}" is not exists.')

        checkpoints = torch.load(self.load_from, map_location='cpu')
        weights = load_checkpoint(model, checkpoints)

        model.load_state_dict(weights)

        class_names = checkpoints.get('class_names', None)
        parameter('model class names', class_names)

        # 备份权重
        basename = osp.basename(self.load_from)
        shutil.copy(self.load_from, osp.join(self.cfgs.work_dir, f'load_from-{basename}'))
        
        return model
    
    def _prepare_input_data(self, input_names: list) -> tuple:
        input_data = []
        for input_name in input_names:
            inputs = torch.randn(1, 3, self.input_size[0], self.input_size[1])
            input_data.append(inputs)
            parameter(input_name, inputs.shape)
        
        return tuple(input_data)

    def _convert2onnx(self, input_data: tuple, input_names: list, output_names: list, onnx_file: str) -> None:
        torch.onnx.export(
            self.model,
            input_data,
            onnx_file,
            opset_version=self.cfgs.opset,
            do_constant_folding=False,
            verbose=self.cfgs.verbose,
            input_names=input_names, 
            output_names=output_names)

        logger.success(f'Saving ONNX model to {onnx_file}')

    def _simplified(self, onnx_file: str) -> str:
        heading('simplify')

        simplified_onnx_file = onnx_file.replace('_origin.onnx', '.onnx')
        cmd = 'python -m onnxsim {} {} 1'.format(onnx_file, simplified_onnx_file)
        os.system(cmd)
        logger.success(f'Saving simplified ONNX model to {simplified_onnx_file}')

        return simplified_onnx_file

    def _check(self, onnx_file: str, input_data: tuple, output_names: list) -> None:
        heading('check')
        onnx_model = onnx.load(onnx_file)
        onnx.checker.check_model(onnx_model)

        # cosine similarity
        pytorch_outputs = self.model(*input_data)
        onnxruntime_session = onnxruntime.InferenceSession(onnx_file, 
                                                           providers=['CPUExecutionProvider'])

        onnxruntime_inputs = {}
        onnx_inputs = onnxruntime_session.get_inputs()
        for idx, onnx_input in enumerate(onnx_inputs):
            onnxruntime_inputs[onnx_input.name] = input_data[idx].numpy()

        onnx_outputs = onnxruntime_session.run(None, onnxruntime_inputs)
        assert len(onnx_outputs) == len(pytorch_outputs)

        for idx, onnx_output in enumerate(onnx_outputs):
            pytorch_output = pytorch_outputs[idx].detach().clone().numpy()

            cosine_sim = cosine_similarity(onnx_output, pytorch_output)

            parameter(f'{output_names[idx]} (cs)', cosine_sim)
        
        logger.success('done')

    @torch.no_grad()
    def deploy(self) -> None:
        heading('start det2d deploying')
        parameter('opset', self.cfgs.opset)
        parameter('verbose', self.cfgs.verbose)
        parameter('input size', self.input_size)
        self.model.eval()

        # input & output names
        heading('inputs and outputs')
        input_names = self.model.input_names
        parameter('input names', input_names)
        input_data = self._prepare_input_data(input_names)
 
        output_names = self.model.output_names
        parameter('output names', output_names)

        # convert to onnx
        heading('convert')
        self.model.forward = self.model.forward_deploy

        onnx_file = osp.join(self.cfgs.work_dir, f'{self.cfgs.model_name}_origin.onnx')
        self._convert2onnx(input_data, input_names, output_names, onnx_file)

        # simplified
        simplified_onnx_file = self._simplified(onnx_file)

        # check
        self._check(simplified_onnx_file, input_data, output_names)
