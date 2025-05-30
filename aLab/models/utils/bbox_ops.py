# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import torch
from torch import Tensor


__all__ = ['bbox_xyxy_to_cxcywh', 'bbox_cxcywh_to_xyxy', 'bbox_ious']


def bbox_xyxy_to_cxcywh(bbox: Tensor) -> Tensor:
    """Convert bbox coordinates from (x1, y1, x2, y2) to (cx, cy, w, h).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    x1, y1, x2, y2 = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)]
    return torch.cat(bbox_new, dim=-1)


def bbox_cxcywh_to_xyxy(bbox: Tensor) -> Tensor:
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    cx, cy, w, h = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.cat(bbox_new, dim=-1)


def bbox_ious(bboxes1: Tensor, bboxes2: Tensor, is_aligned: bool = False, eps: float = 1e-6) -> Tensor:
    # Either the boxes are empty or the length of boxes last dimension is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)    # number of bboxes1
    cols = bboxes2.size(-2)    # number of bboxes2
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))
        
    # Areas
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * ((bboxes1[..., 3] - bboxes1[..., 1]))
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * ((bboxes2[..., 3] - bboxes2[..., 1]))
    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])   # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])   # [B, rows, 2]

        wh = torch.clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        union = area1 + area2 - overlap
    else:
        lt = torch.max(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.max(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = torch.clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        union = area1[..., None] + area2[..., None, :] - overlap
    
    eps = union.new_tensor([eps])
    union = torch.max(union, eps)

    ious = overlap / union

    return ious