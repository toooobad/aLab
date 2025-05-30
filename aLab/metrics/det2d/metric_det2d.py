# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import os
import json

from typing import Tuple


class MetricDet2D:
    def __init__(self) -> None:
        self.epoch = -1
        self._best_metric = {}

        # 用于保存当前的评估指标
        self._metrics = {}
        self._classwise_metrics = {}

        # 用于保存预测结果
        self._predictions = []

        # 用于绘制pr曲线
        self._classwise_precisions = {}

    @property
    def metrics(self) -> dict:
        return self._metrics
    
    @property
    def classwise_metrics(self) -> dict:
        return self._classwise_metrics
    
    @property
    def best_metric(self) -> dict:
        return self._best_metric
    
    @property
    def predictions(self) -> list:
        return self._predictions
    
    @property
    def classwise_precisions(self) -> dict:
        return self._classwise_precisions
    
    @staticmethod
    def _round(value: float, num_digits: int = 3) -> float:
        return float(round(value, num_digits))
    
    @staticmethod
    def _get_f1_score(p: float, r: float, eps: float = 1e-4) -> float:
        return 2 * p * r / (p + r + eps)
    
    def process_preds(self) -> None:
        raise NotImplementedError
    
    def compute_metric(self) -> Tuple[dict]:
        raise NotImplementedError
    
    def reset(self):
        self._predictions = []
        self._classwise_precision = {}

        self._metrics = {}
        self._classwise_metrics = {}

    def save_metric(self, save_dir: str) -> None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        def _dump_metric(metric, filename):
            if len(metric):
                metric.update({'epoch': self.epoch})

                with open(os.path.join(save_dir, f'{filename}.json'), 'a') as f:
                    f.write(json.dumps(metric) + '\n')

        # dump
        _dump_metric(self.metrics, 'metrics')
        _dump_metric(self.classwise_metrics, 'classwise_metrics')
        _dump_metric(self.classwise_precisions, 'classwise_precisions')