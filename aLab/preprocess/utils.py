# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

"""
训练时, 预处理需要处理的ground-truth的keys
"""

__all__ = ['GroundTruth2DKeys']


GTKeys = dict(
    gt_bboxes=('gt_labels', ),  # det2d
    
    # tlftr
    gt_bboxes30=('gt_labels30', ),
    gt_bboxes120=('gt_labels120', ),
)