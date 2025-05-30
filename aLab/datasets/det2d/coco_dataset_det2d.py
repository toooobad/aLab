# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import itertools
import numpy as np
import os.path as osp

from loguru import logger
from typing import (Tuple, List)
from pycocotools.coco import COCO
from terminaltables import AsciiTable

from aLab.register import DATASET
from .base_dataset_det2d import BaseDatasetDet2D



__all__ = ['CocoDatasetDet2D']


@DATASET.register_module()
class CocoDatasetDet2D(BaseDatasetDet2D):
    def __init__(self,
                 img_dir: str,
                 ann_file: str,
                 max_fetch: int = 100,
                 test_mode: bool = False,
                 preprocesses: List[dict] = None) -> None:
        super(CocoDatasetDet2D, self).__init__(test_mode=test_mode, preprocesses=preprocesses, max_fetch=max_fetch)
        
        self.img_dir = img_dir
        ann_file = osp.expanduser(osp.abspath(ann_file))

        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat_mapping = {cat_id: idx for idx, cat_id in enumerate(self.cat_ids)}
        self.ids = list(sorted(self.coco.imgs.keys()))

        # get class name
        # coco中cat_id可能不是连续的, 需要映射一下
        cats = self.coco.loadCats(self.cat_ids)
        self.class_names = dict()
        for cat in cats:
            cat_id = int(cat['id'])
            cls_id = self.cat_mapping[cat_id]
            self.class_names[cls_id] = cat['name']

        # 获取coco格式数据集的信息
        self._get_dataset_infos()

    def _get_dataset_infos(self):
        dataset_header = ['index', 'origin', 'name', 'images', 'instances']
        dataset_infos = []

        origin_inds = self.coco.getCatIds()

        total_images = len(self)
        total_instances = 0
        
        for origin_index in origin_inds:
            cls_idx = self.cat_mapping[origin_index]
            name = self.coco.loadCats(origin_index)[0]['name']
            img_ids = self.coco.getImgIds(catIds=origin_index)
            num_images = len(img_ids)
            num_instances = len(self.coco.getAnnIds(imgIds=img_ids, catIds=origin_index, iscrowd=None))
            dataset_infos.append([cls_idx, origin_index, name, num_images, num_instances])
            total_instances += num_instances
        
        # 计算各个类别的占比
        for info in dataset_infos:
            info[-2] = f'{info[-2]} | ({round((info[-2] / total_images) * 100, 2)}%)'
            info[-1] = f'{info[-1]} | ({round((info[-1] / total_instances) * 100, 2)}%)'
        
        num_columns = len(dataset_infos[0])
        dataset_infos_flatten = list(itertools.chain(*dataset_infos))
        dataset_infos = itertools.zip_longest(*[dataset_infos_flatten[i::num_columns] for i in range(num_columns)])
        
        dataset_table = [dataset_header]
        dataset_table += [metric for metric in dataset_infos]
        dataset_table = AsciiTable(dataset_table)

        for line in dataset_table.table.split('\n'):
            logger.info(line)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index) -> dict:
        img_id = self.ids[index]

        if self.test_mode:
            return self._get_test_data(img_id)
        else:
            return self._get_train_data(img_id)
        
    def _init_data(self, img_id: int) -> dict:
        return dict(img_id=img_id,
                    filename=osp.join(self.img_dir,  self.coco.loadImgs(img_id)[0]['file_name']))
    
    def _load_anns(self, img_id: int) -> Tuple[np.ndarray, ...]:
        anns = self.coco.loadAnns(self.coco.getAnnIds(img_id))
        
        gt_bboxes = []
        gt_labels = []

        for ann in anns:
            if ann.get('ignore', False):
                continue
            
            # bbox
            x1, y1, w, h = ann['bbox']
            gt_bboxes.append([x1, y1, x1 + w, y1 + h])

            # label
            gt_labels.append(self.cat_mapping[int(ann['category_id'])])
        
        if len(gt_bboxes):
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int32)
        else:
            gt_bboxes = np.zeros([0, 4], dtype=np.float32)
            gt_labels = np.zeros([0, ], dtype=np.int32)

        return (gt_bboxes, gt_labels)
    
    def _get_test_data(self, img_id: int) -> dict:
        data = self._init_data(img_id)
        return self.preprocesses(data)
    
    def _get_train_data(self, img_id: int) -> dict:
        for _ in range(self.max_fetch):
            data = self._init_data(img_id)

            # load anns
            gt_bboxes, gt_labels = self._load_anns(img_id)
            data['gt_bboxes'] = gt_bboxes
            data['gt_labels'] = gt_labels

            # transform
            data = self.preprocesses(data)
            if data is None:
                img_id = self._rand_another()
                continue
        
            return data
