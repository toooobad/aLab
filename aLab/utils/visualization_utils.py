# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import os
import json
import imageio
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
from loguru import logger
from typing import (Optional, Union, List, Tuple)

from aLab.structures import (InstanceData, DetDataSample)


__all__ = ['get_adaptive_font_size', 'get_adaptive_dpi_linewidth', 'VisualizeDet2D']


NUM_COLORS = 100
COLORS = sns.color_palette('husl', NUM_COLORS)


def get_adaptive_font_size(area: float, min_area: int = 800, max_area: int = 30000) -> int:
    font_size = 0.5 + (area - min_area) // (max_area - min_area)
    font_size = int(np.clip(font_size, 0.5, 1.0) * 13)
    return font_size


def get_adaptive_dpi_linewidth(area: float) -> tuple:
    pixel200 = 1600 * 1200
    pixel400 = 2240 * 1680

    if 0 < area <= pixel200:  # 200w
        dpi = 150
        linewidth = 1
    elif pixel200 < area <= pixel400:  # 400w
        dpi = 200
        linewidth = 2
    else:
        dpi = 300
        linewidth = 3

    return (dpi, linewidth)


class VisualizeDet2D:
    def __init__(self, 
                 work_dir: str,
                 fps: int = 25,
                 video_name: Optional[str] = None, 
                 save_preds: Optional[bool] = False,
                 draw_gts: Optional[bool] = False,
                 draw_preds: Optional[bool] = True,
                 class_names: Optional[Union[List[str], Tuple[str]]] = None) -> None:
        
        self.draw_gts = draw_gts
        self.draw_preds = draw_preds
        assert self.draw_gts or self.draw_preds

        self.video_name = video_name
        self.save_preds = save_preds
        self.class_names = class_names
        
        self.save_image_dir = os.path.join(work_dir, 'visualized_images')
        if not os.path.exists(self.save_image_dir):
            os.makedirs(self.save_image_dir, exist_ok=True)

        if self.save_preds:
            self.save_preds_dir = os.path.join(work_dir, 'pred_results')
            if not os.path.exists(self.save_preds_dir):
                os.makedirs(self.save_preds_dir, exist_ok=True)
        
        if self.video_name is not None:
            self.video_writer = imageio.get_writer(os.path.join(work_dir, f'{video_name}.mp4'), fps=fps)
        else:
            self.video_writer = None

        self.ax = None
        self.fig = None

    def close(self):
        if self.video_writer is not None:
            self.video_writer.close()

    def visualize(self, data_samples: List[DetDataSample]):
        for data_sample in data_samples:
            filepath = data_sample.metainfo['filename']
            filename = os.path.basename(filepath)

            # step1. load image
            image = np.array(Image.open(filepath))
            image = np.ascontiguousarray(np.array(image))
            image_h, image_w = image.shape[:-1]

            # 根据图像尺寸设置线宽 & dpi
            dpi, linewidth = get_adaptive_dpi_linewidth(image_h * image_w)

            # step2. 创建一个 Matplotlib 图形
            self.fig, self.ax = plt.subplots(figsize=(image_w / dpi, image_h / dpi))
            self.ax.imshow(image, extent=(0, image_w, image_h, 0), interpolation='none')

            # step3. draw preds
            if self.draw_preds:
                pred_instances = data_sample.pred_instances
                self._draw_bboxes(linewidth=linewidth,
                                  draw_format='pred',
                                  instance_data=pred_instances)
            
            # step4. draw gts
            if self.draw_gts:
                gt_instances = data_sample.gt_instances
                self._draw_bboxes(linewidth=linewidth,
                                  draw_format='gt',
                                  instance_data=gt_instances,
                                  scale_factors=data_sample.metainfo['scale_factors'])

            # step5. 设置图形范围
            self.ax.set_xlim(0, image_w)
            self.ax.set_ylim(image_h, 0)  # 注意 Matplotlib 的 y 轴方向与图像坐标相反

            # step6. 去掉坐标轴
            self.ax.axis('off')

            # step7. 调整布局以确保没有多余的空白区域
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            # step8. 保存图片
            save_path=os.path.join(self.save_image_dir, filename)
            if save_path is not None:
                plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0)  # 保存图片，去掉多余的空白区域

            # setp9. 将 Matplotlib 图形转换为 PIL 图像
            if self.video_writer is not None:
                self.fig.canvas.draw()
                visualized_image = Image.frombytes('RGB', self.fig.canvas.get_width_height(), self.fig.canvas.tostring_rgb())
                visualized_image = np.array(visualized_image)
                self.video_writer.append_data(visualized_image)
            
            plt.close(self.fig)

            # step10. save preds
            if self.save_preds:
                basename = os.path.splitext(filename)[0]
                pred_dict = dict(filename=filename, predictions=[], img_id=data_sample.metainfo['img_id'])

                obj_id = 0
                pred_instances = data_sample.pred_instances
                for (bbox, label, score) in zip(pred_instances.bboxes, pred_instances.labels, pred_instances.scores):
                    obj = dict(bbox=bbox.tolist(), label=int(label), score=float(score), obj_id=obj_id)
                    if self.class_names is not None:
                        obj['class_name'] = self.class_names[int(label)]

                    pred_dict['predictions'].append(obj)
                    obj_id += 1
                
                with open(os.path.join(self.save_preds_dir, f'{basename}.json'), 'w', encoding='utf-8') as file:
                    json.dump(pred_dict, file, ensure_ascii=False, indent=4)

    def _draw_bboxes(self, linewidth: int, instance_data: InstanceData, draw_format: str = 'pred', scale_factors: Union[List[float], Tuple[float]] = None) -> None:
        # 设置字体样式
        font = {
            'size': 12,          # 字体大小
            'color': 'white',    # 字体颜色
            'weight': 'normal',  # 字体粗细
            'family': 'sans-serif',   # 字体类型
            }
        if draw_format == 'gt': 
            font['color'] = 'green'

        # step3. 遍历每个bbox并绘制
        bboxes = instance_data.get('bboxes', None)
        if bboxes is not None:
            labels = instance_data.get('labels', None)
            if labels is None:
                labels = [-1] * len(bboxes)

            scores = instance_data.get('scores', None)

            # rescale
            if scale_factors is not None:
                bboxes[:, [0, 2]] = bboxes[:, [0, 2]] / scale_factors[0]
                bboxes[:, [1, 3]] = bboxes[:, [1, 3]] / scale_factors[1]

            for ins_idx, bbox in enumerate(bboxes):
                # bbox
                x1, y1, x2, y2 = bbox
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                
                # cls
                cls_id = int(labels[ins_idx])

                # color
                if draw_format == 'pred':
                    color = COLORS[cls_id]
                else:
                    color = (0., 1., 0.)
                
                # 绘制边界框
                rect = patches.Rectangle((x1, y1),
                                        bbox_w,
                                        bbox_h,
                                        linewidth=linewidth,
                                        edgecolor=color,
                                        facecolor='none')
                self.ax.add_patch(rect)
                
                # 根据边界框的大小动态调整字体大小
                font['size'] = get_adaptive_font_size(bbox_w * bbox_h)
                
                # 创建文本内容
                if draw_format == 'pred':
                    text_x1 = x1
                    text_y1 = y1
                    bbox_bg = dict(facecolor='black', alpha=0.8, edgecolor='none', pad=0.7)
                else:
                    text_x1 = x1
                    text_y1 = y1 - font['size'] * 2 - linewidth * 2
                    bbox_bg = dict(facecolor='white', alpha=0.9, edgecolor='none', pad=0.5)

                if self.class_names is not None:
                    text = f'{self.class_names[cls_id]}'
                else:
                    text = f'cls{cls_id}'
                
                if scores is not None:
                    text += f": {scores[ins_idx]:.2f}"
                
                # 在边界框的左上角添加文本
                self.ax.text(text_x1,
                             text_y1,
                             text, 
                             fontdict=font,
                             verticalalignment='top',
                             horizontalalignment='left',
                             bbox=bbox_bg)
        else:
            logger.warning('No BBoxes!')
            logger.info(instance_data)
