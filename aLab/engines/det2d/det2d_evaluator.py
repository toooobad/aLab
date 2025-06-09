# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import time
import torch
import shutil
import datetime
import onnxruntime
import os.path as osp
import torch.nn as nn

from copy import deepcopy
from loguru import logger
from argparse import Namespace
from typing import (List, Tuple)
from torch.utils.data import (DataLoader, SequentialSampler)

from aLab.structures import DetDataSample
from aLab.register import (EVALUATOR, POSTPROCESS)
from aLab.builder import (build_model, build_dataset, build_metric)
from aLab.utils import (heading, load_checkpoint, parameter, StatusLogger)


__all__ = ['Det2dEvaluator']


@EVALUATOR.register_module(name='det2d')
class Det2dEvaluator:
    def __init__(self, cfgs: Namespace) -> None:
        # init
        self.cfgs = cfgs
        self.infer_model = cfgs.infer_model

        # device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # model
        self.torch_model, self.onnx_model = self._build_model()

        self.onnx_input_names = None
        self.onnx_output_names = None
        if self.onnx_model is not None:
            self.onnx_input_names = [inp.name for inp in self.onnx_model.get_inputs()]
            self.onnx_output_names = [out.name for out in self.onnx_model.get_outputs()]

            # postprocess
            postprocess_cfgs = deepcopy(self.cfgs.model['head'].get('initial_cfgs', dict()))
            postprocess_cfgs['type'] = f"{self.cfgs.model['type']}Postprocess"
            self.onnx_postprocesser = POSTPROCESS.build(postprocess_cfgs)

        # datasets
        self.test_loader = self._build_datasets()

        # metric
        self.torch_metric, self.onnx_metric = self._build_metric()

        # misc
        self.delimiter = ' | '
        self.metric_fmt = '{value:.6f}'
        self.window_size = self.cfgs.log_interval
    
    def _build_model(self) -> tuple:
        if 'torch' in self.infer_model:
            torch_model = build_model(self.cfgs)
            torch_model = torch_model.to(self.device)
            torch_model = self._load_checkpoints(torch_model)
        else:
            torch_model = None

        if 'onnx' in self.infer_model:
            onnx_model = onnxruntime.InferenceSession(self.cfgs.onnx_model, 
                                                      providers=['CPUExecutionProvider'])
        else:
            onnx_model = None

        return torch_model, onnx_model

    def _load_checkpoints(self, model: nn.Module) -> nn.Module:
        parameter(f'load from', self.cfgs.torch_model)
        assert osp.exists(self.cfgs.torch_model), ValueError(f'"{self.cfgs.torch_model}" is not exists.')

        checkpoints = torch.load(self.cfgs.torch_model, map_location='cpu')
        weights = load_checkpoint(model, checkpoints)

        model.load_state_dict(weights)

        class_names = checkpoints.get('class_names', None)
        parameter('model class names', class_names)

        # 备份权重
        basename = osp.basename(self.cfgs.torch_model)
        shutil.copy(self.cfgs.torch_model, osp.join(self.cfgs.work_dir, f'torch-model-{basename}'))
        
        return model

    def _build_datasets(self) -> DataLoader:
        num_workers = self.cfgs.num_workers

        # 制作dataset-cfg
        if hasattr(self.cfgs, 'val_dataset'):
            test_dataset_cfg = self.cfgs.val_dataset   # 使用config中的test_dataset
        else:
            raise ValueError('"test_dataset" is not defined.')  # test-dataset & demo都没定义
        preprocesses = test_dataset_cfg['preprocesses']
        
        # 替换input_size
        if (self.cfgs.input_w > 0) and (self.cfgs.input_h > 0):
            for pre in preprocesses:
                if 'input_size' in pre:
                    pre['input_size'] = [self.cfgs.input_h, self.cfgs.input_w]
         
        test_dataset_cfg['preprocesses'] = preprocesses
        self.cfgs.test_dataset = test_dataset_cfg

        test_dataset = build_dataset(self.cfgs, mode='test')
        test_loader = DataLoader(test_dataset,
                                 1,
                                 sampler=SequentialSampler(test_dataset),
                                 drop_last=False,
                                 num_workers=num_workers,
                                 collate_fn=test_dataset.collate_fn)

        return test_loader

    def _build_metric(self) -> Tuple[object, object]:
        if 'torch' in self.infer_model:
            torch_metric = build_metric(self.cfgs)
        else:
            torch_metric = None
        
        if 'onnx' in self.infer_model:
            onnx_metric = build_metric(self.cfgs)
        else:
            onnx_metric = None

        return torch_metric, onnx_metric
    
    def _torch_model_infer(self, inputs: dict, batch_data_samples: List[DetDataSample]) -> None:
        # to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # forward
        torch_predictions = self.torch_model(inputs, batch_data_samples=batch_data_samples, training=False)

        self.torch_metric.process_preds(torch_predictions)

    def _onnx_model_infer(self, inputs: dict, batch_data_samples: List[DetDataSample]) -> None:
        onnx_inputs = dict()

        for name in self.onnx_input_names:
            onnx_inputs[name] = inputs[name].numpy()
        
        onnx_preds = self.onnx_model.run(None, onnx_inputs)
        onnx_preds = {self.onnx_output_names[idx]: torch.from_numpy(pred) for idx, pred in enumerate(onnx_preds)}
        onnx_predictions = self.onnx_postprocesser.postprocess(onnx_preds, batch_data_samples=batch_data_samples, rescale=True, to_numpy=True)

        self.onnx_metric.process_preds(onnx_predictions)

    @torch.no_grad()
    def eval(self) -> None:
        heading('start det2d evaluating')
        parameter('infer model', self.infer_model)

        start_time = time.time()
        if self.torch_model is not None:
            self.torch_model.eval()

        eval_status = StatusLogger(delimiter=self.delimiter,
                                   metric_fmt=self.metric_fmt,
                                   window_size=self.window_size)

        for batch_samples in eval_status.log_every(self.test_loader, 'Eval'):
            inputs = batch_samples.pop('inputs')
            batch_data_samples = batch_samples.pop('data_samples')

            if 'torch' in self.infer_model:
                self._torch_model_infer(inputs, batch_data_samples)
            
            if 'onnx' in self.infer_model:
                self._onnx_model_infer(inputs, batch_data_samples)
            
        # metric
        if 'torch' in self.infer_model:
            heading('torch')
            self.torch_metric.compute_metric()
            self.torch_metric.save_metric(osp.join(self.cfgs.work_dir, 'metrics', 'torch'))
            self.torch_metric.reset()

        if 'onnx' in self.infer_model:
            heading('onnx')
            self.onnx_metric.compute_metric()
            self.onnx_metric.save_metric(osp.join(self.cfgs.work_dir, 'metrics', 'onnx'))
            self.onnx_metric.reset()
        
        # end evaluating
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))

        logger.success(f'Evaluating completed, total time spent {total_time_str}.')
