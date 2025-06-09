# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import itertools
import numpy as np

from loguru import logger
from pycocotools.coco import COCO
from collections import OrderedDict
from terminaltables import AsciiTable
from pycocotools.cocoeval import COCOeval
from typing import (Sequence, List, Tuple, Optional)

from aLab.utils import heading
from aLab.register import METRIC
from aLab.structures import DetDataSample
from .metric_det2d import MetricDet2D


__all__ = ['CocoMetricDet2D']


@METRIC.register_module()
class CocoMetricDet2D(MetricDet2D):
    def __init__(self,
                 ann_file: str,
                 max_dets: Optional[Sequence[int]] = None,
                 object_sizes: Optional[Sequence[float]] = None) -> None:
        # max_dets: (1, 10, 100)
        # object_sizes: (32, 96, 1e5)
        self.ann_file = ann_file

        # max dets
        if max_dets is not None:
            self.max_dets = list(max_dets)
        else:
            self.max_dets = None

        if object_sizes is not None:
            self.area_ranges = (
                [0 ** 2, object_sizes[2] ** 2],     # all
                [0 ** 2, object_sizes[0] ** 2],     # small
                [object_sizes[0] ** 2, object_sizes[1] ** 2],   # medium
                [object_sizes[1] ** 2, object_sizes[2] ** 2])   # large
        else:
            self.area_ranges = None
        
        self.iou_thrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)

        super(CocoMetricDet2D, self).__init__()
    
    @property
    def classwise_metric_header(self) -> list:
        return ['category', 'images', 'instances', 
                'AP', 'AR', 'F1', 'AP@.5', 'AP@.75', 'AP@small', 'AP@medium', 'AP@large']

    @property
    def average_metric_header(self) -> list:
        return ['mAP', 'mAP@.5', 'mAP@.75', 'mAP@small', 'mAP@medium', 'mAP@large']

    def xyxy2xywh(self, bbox: np.ndarray) -> list:
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox: List = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]
    
    def process_preds(self, batch_predictions: List[DetDataSample]) -> None:
        # 将dts转换成coco的格式
        for prediction in batch_predictions:
            img_id = prediction.metainfo['img_id']

            pred_instances = prediction.pred_instances
            labels = pred_instances.labels
            bboxes = pred_instances.bboxes
            scores = pred_instances.scores

            for bbox, label, score in zip(bboxes, labels, scores):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(bbox)    # xyxy -> xywh list
                data['score'] = float(score)
                data['category_id'] = int(label)       # 未映射成coco的类别
                self._predictions.append(data)
    
    def compute_metric(self) -> Tuple[OrderedDict, OrderedDict]:
        heading('compute metric')

        if len(self._predictions) == 0:
            logger.info('No Predictions!')
            return self.metrics, self.classwise_metrics
        
        # 加载gt
        gts = COCO(self.ann_file)
        cat_mapping = {idx: cat_id for idx, cat_id in enumerate(gts.getCatIds())}

        # 加载dt
        for pred in self._predictions:
            category_id = pred['category_id']
            pred['category_id'] = cat_mapping[int(category_id)]
        dts = gts.loadRes(self._predictions)

        # 创建coco-evalutor
        evaluator = COCOeval(gts, dts, iouType='bbox')

        # gt cls ids
        gt_cls_ids = gts.getCatIds()

        # set params
        if self.max_dets is not None:
            evaluator.params.maxDets = self.max_dets

        if self.area_ranges is not None:
            evaluator.params.areaRng = self.area_ranges

        evaluator.params.iouThrs = self.iou_thrs

        # 计算平均指标
        evaluator.evaluate()
        evaluator.accumulate()
        logger.info('{:-^110}'.format(' COCO Metric '))
        evaluator.summarize()

        # 计算每个类别的指标
        recalls = evaluator.eval['recall']   # (iou, cls, area range, max dets)
        precisions = evaluator.eval['precision']    # (iou, recall, cls, area range, max dets)
        assert len(gt_cls_ids) == precisions.shape[2] == recalls.shape[1]

        classwise_metric_table = []   # 用于打印每个类别的指标
        for idx, cls_id in enumerate(gt_cls_ids):
            class_metric = dict()  # 记录当前类别的指标

            # 获取类别名称
            cls_name = gts.loadCats(cls_id)[0]['name']
            if cls_name not in self._classwise_precisions:
                self._classwise_precisions[cls_name] = dict()
            
            # 获取该类别的图片的id
            img_ids = gts.getImgIds(catIds=cls_id)

            # 包含该类别图片的数量
            num_cls_imgs = len(img_ids)

            # 该类别标注的数量
            num_cls_anns = len(gts.getAnnIds(imgIds=img_ids, catIds=cls_id, iscrowd=None))

            # 用于打印
            class_line = [cls_name, str(num_cls_imgs), str(num_cls_anns)]

            # 计算该类别的ap
            precision = precisions[:, :, idx, 0, -1]   # [10, 101]
            precision = precision[precision > -1]

            if precision.size:
                cls_ap = np.mean(precision)
            else:
                cls_ap = float('nan')

            class_line.append(str(self._round(cls_ap)))
            class_metric['ap'] = cls_ap

            # 计算该类别的ar
            recall = recalls[:, idx, 0, -1]  # [10]
            recall = recall[recall > -1]

            if recall.size:
                cls_ar = np.mean(recall)
            else:
                cls_ar = float('nan')

            class_line.append(str(self._round(cls_ar)))
            class_metric['ar'] = cls_ar

            # 根据 ap & ar 计算 f1
            cls_f1 = self._get_f1_score(cls_ap, cls_ar)
            class_line.append(str(self._round(cls_f1)))
            class_metric['f1'] = cls_f1

            # ap@.5 & ap@.75
            for iou_idx, iou_type in zip([0, 5], ['AP@.5', 'AP@.75']):
                precision = precisions[iou_idx, :, idx, 0, -1]
                precision = precision[precision > -1]

                if precision.size:
                    cls_iou_ap = np.mean(precision)
                    self._classwise_precisions[cls_name][iou_type] = precision.tolist()
                else:
                    cls_iou_ap = float('nan')
                    self._classwise_precisions[cls_name][iou_type] = [0] * 101

                class_line.append(str(self._round(cls_iou_ap)))
                class_metric[iou_type] = cls_iou_ap
            
            # indexes of area of small, median and large
            for area_idx, area_flag in enumerate(['AP@small', 'AP@medium', 'AP@large']):
                area_idx = area_idx + 1

                precision = precisions[:, :, idx, area_idx, -1]
                precision = precision[precision > -1]
                
                if precision.size:
                    cls_area_ap = np.mean(precision)
                else:
                    cls_area_ap = float('nan')

                class_line.append(str(self._round(cls_area_ap)))
                class_metric[area_flag] = cls_area_ap
            
            classwise_metric_table.append(tuple(class_line))
            self._classwise_metrics[cls_name] = class_metric
        
        # 打印类别指标
        num_columns = len(classwise_metric_table[0])
        classwise_metrics_flatten = list(itertools.chain(*classwise_metric_table))
        classwise_metrics = itertools.zip_longest(*[classwise_metrics_flatten[i::num_columns] for i in range(num_columns)])
        
        classwise_table = [self.classwise_metric_header]
        classwise_table += [metric for metric in classwise_metrics]
        classwise_table = AsciiTable(classwise_table)

        logger.info('{:-^110}'.format(' Classwise Metric '))
        for classwise in classwise_table.table.split('\n'):
            logger.info(classwise)

        # average
        logger.info('{:-^110}'.format(' Average Metric '))
        average_metric_header = self.average_metric_header
        average_metric_header += [f'mAR@{max_dets}' for max_dets in evaluator.params.maxDets]
        average_metric_mapping = {k: idx for idx, k in enumerate(average_metric_header)}

        for name, metric_idx in average_metric_mapping.items():
            metric_value = evaluator.stats[metric_idx]
            self._metrics[name] = metric_value

            # best metric
            if name not in self._best_metric:
                self._best_metric[name] = (self.epoch, metric_value)
            else:
                if metric_value > self._best_metric[name][-1]:
                    self._best_metric[name] = (self.epoch, metric_value)
            
            current_str = '{: <20}'.format(f'{name}: {self._round(metric_value)}')
            best_str = f'(best: {self._round(self._best_metric[name][-1])} [{self._best_metric[name][0]}])'
            logger.info(f'{current_str}{best_str}')
        
        f1 = self._get_f1_score(evaluator.stats[0], evaluator.stats[8])
        self._metrics['mF1'] = f1

        if 'mF1' not in self._best_metric:
            self._best_metric['mF1'] = (self.epoch, f1)
        else:
            if f1 > self._best_metric['mF1'][-1]:
                self._best_metric['mF1'] = (self.epoch, f1)
        
        current_str = '{: <20}'.format(f'mF1: {self._round(f1)}')
        best_str = f'(best: {self._round(self._best_metric["mF1"][-1])} [{self._best_metric["mF1"][0]}])'
        logger.info(f'{current_str}{best_str}')

        return self.metrics, self.classwise_metrics




