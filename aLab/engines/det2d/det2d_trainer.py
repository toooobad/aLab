# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import os
import sys
import math
import time
import torch
import shutil
import datetime
import torch.nn as nn
import os.path as osp

from copy import deepcopy
from loguru import logger
from typing import (Tuple, Any)
from argparse import Namespace
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, BatchSampler)

from aLab.register import TRAINER
from aLab.utils import (heading, load_checkpoint, parameter, StatusLogger, SmoothedValue)
from aLab.builder import (build_model, build_optimizer, build_lr_scheduler, build_dataset, build_metric)


__all__ = ['Det2dTrainer']


@TRAINER.register_module(name='det2d')
class Det2dTrainer:
    def __init__(self, cfgs: Namespace) -> None:
        # init
        self.cfgs = cfgs

        self.cur_epoch = 1
        self.iterations = 1

        self.ckpt_dir = osp.join(self.cfgs.work_dir, 'checkpoints')
        self.metric_dir = osp.join(self.cfgs.work_dir, 'metrics')

        # load from
        self.load_from = self.cfgs.load_from

        # max norm
        self.max_norm = self.cfgs.max_norm

        # device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # model
        self.model = self._build_model()

        # optimizer
        self.optimizer = self._build_optimizer()

        # lr scheduler
        self.lr_scheduler = self._build_lr_scheduler()

        # datasets
        self.train_loader, self.val_lodaer, self.train_class_names = self._build_datasets()

        # resume
        self.start_epoch = self._resume()

        # metric
        self.metric = self._build_metric()

        # tensorboard
        self.tb_writer = SummaryWriter(osp.join(self.cfgs.work_dir, 'tensorboard'))

        # misc
        self.delimiter = ' | '
        self.metric_fmt = '{value:.6f}'
        self.window_size = self.cfgs.log_interval
    
    def _build_model(self) -> nn.Module:
        # init model
        model = build_model(self.cfgs)

        # to device
        model = model.to(self.device)

        # load checkpoint
        model = self._load_checkpoints(model)

        return model

    def _load_checkpoints(self, model: nn.Module) -> nn.Module:
        if self.load_from is not None:
            parameter(f'load model checkpoint from', self.load_from)
            assert osp.exists(self.load_from), ValueError(f'"{self.load_from}" is not exists.')

            checkpoints = torch.load(self.load_from, map_location='cpu')
            weights = load_checkpoint(model, checkpoints)

            model.load_state_dict(weights)

            if 'class_names' in checkpoints:
                parameter('model class names', checkpoints['class_names'])
        
        return model
    
    def _build_optimizer(self) -> torch.optim.Optimizer:
        optimizer = build_optimizer(self.model, self.cfgs)

        return optimizer

    def _build_lr_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        lr_scheduler = build_lr_scheduler(self.optimizer, self.cfgs)

        return lr_scheduler

    def _build_datasets(self) -> Tuple[DataLoader, DataLoader, Any]:
        num_workers = self.cfgs.num_workers

        # train
        train_dataset = build_dataset(self.cfgs, mode='train')

        if len(train_dataset) < self.cfgs.batch_size:
            logger.info(f'train batch size will be set to {len(train_dataset)}, because there are only {len(train_dataset)} samples in the train dataset (origin={self.cfgs.batch_size}).')
            train_batch_size = len(train_dataset)
        else:
            train_batch_size = self.cfgs.batch_size
        self.cfgs.train_batch_size = train_batch_size

        train_loader = DataLoader(train_dataset,
                                  batch_sampler=BatchSampler(
                                      RandomSampler(train_dataset),
                                      train_batch_size,
                                      drop_last=True),
                                  num_workers=num_workers,
                                  collate_fn=train_dataset.collate_fn)

        train_class_names = train_dataset.class_names

        # val
        val_dataset = build_dataset(self.cfgs, mode='val')

        if hasattr(self.cfgs, 'val_batch_size'):
            val_batch_size = self.cfgs.val_batch_size
        else:
            val_batch_size = 1
        self.cfgs.val_batch_size = val_batch_size
        
        logger.info(f'valid batch size will be set to {val_batch_size} during validation. To change this setting, please set "val_batch_size" in the yaml file.')
        
        val_loader = DataLoader(val_dataset,
                                val_batch_size,
                                sampler=SequentialSampler(val_dataset),
                                drop_last=False,
                                num_workers=num_workers,
                                collate_fn=val_dataset.collate_fn)

        return train_loader, val_loader, train_class_names
    
    def _build_metric(self) -> object:
        metric = build_metric(self.cfgs)

        return metric

    def _resume(self) -> int:
        start_epoch = 1

        if self.load_from is not None:
            basename = osp.basename(self.load_from)

            checkpoints = torch.load(self.load_from, map_location='cpu')

            if self.cfgs.resume:
                heading('resume')
                logger.info(f'resume from "{self.load_from}"')

                if 'optimizer' in checkpoints:
                    logger.info(f'load optimizer chekpoint.')
                    self.optimizer.load_state_dict(checkpoints['optimizer'])
                else:
                    logger.info(f'"optimizer" not in checkpoints.')

                if 'lr_scheduler' in checkpoints:
                    logger.info(f'load lr_scheduler chekpoint.')
                    self.lr_scheduler.load_state_dict(checkpoints['lr_scheduler'])
                else:
                    logger.info(f'"lr_scheduler" not in checkpoints.')
                
                if 'epoch' in checkpoints:
                    checkpoint_epoch = checkpoints['epoch']
                    start_epoch = checkpoint_epoch + 1
                    logger.info(f'checkpoint epoch is {checkpoint_epoch}, start epoch set to {start_epoch}.')
                else:
                    logger.info(f'"epoch" not in checkpoints, "start_epoch" will be set to {start_epoch}')
                
                shutil.copy(self.load_from, osp.join(self.cfgs.work_dir, f'resume_from-{basename}'))
            else:
                shutil.copy(self.load_from, osp.join(self.cfgs.work_dir, f'load_from-{basename}'))

        return start_epoch
    
    def _save_weights(self, filename: str, only_model: bool = False, save_latest: bool = False, prefix: str = '') -> None:        
        ckpt_dir = osp.join(self.ckpt_dir, prefix)
        if not osp.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)

        model_dict = {
            'cfgs': deepcopy(vars(self.cfgs)),
            'epoch': self.cur_epoch,
            'model': self.model.state_dict(),
            'class_names': self.train_class_names}

        if not only_model:
            model_dict.update({'optimizer': self.optimizer.state_dict(),
                               'lr_scheduler': self.lr_scheduler.state_dict()})
            
        torch.save(model_dict, osp.join(ckpt_dir, f'{filename}.pth'))
        logger.info(f'Saving {filename} checkpoints to "{ckpt_dir}"')

        if save_latest:
            torch.save(model_dict, osp.join(ckpt_dir, 'latest.pth'))

    def _train_one_epoch(self) -> None:
        heading(f'epoch-{self.cur_epoch}')

        self.model.train()
        
        train_status = StatusLogger(delimiter=self.delimiter,
                                    metric_fmt=self.metric_fmt,
                                    window_size=self.window_size)
        train_status.add_meter('lr', SmoothedValue(window_size=1, fmt=self.metric_fmt))

        for batch_samples in train_status.train_log_every(self.train_loader, f'Epoch[{self.cur_epoch}]'):
            inputs = batch_samples.pop('inputs')

            # to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            batch_data_samples = batch_samples.pop('data_samples')
            batch_data_samples = [data_sample.to(self.device) for data_sample in batch_data_samples]
   
            # inference & losses
            loss_dict = self.model(inputs, batch_data_samples)
            losses = sum(loss_dict.values())

            if not math.isfinite(losses):
                logger.warning(f'Loss is {losses}, stopping training')
                logger.warning(loss_dict)
                sys.exit(1)
            
            # 更新梯度
            self.optimizer.zero_grad()
            losses.backward()
            
            grad_norm = None
            if self.max_norm > 0:
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            self.optimizer.step()

            train_status.update(loss=losses, **loss_dict)
            train_status.update(lr=self.optimizer.param_groups[0]['lr'])
            if grad_norm is not None:
                train_status.update(grad_norm=grad_norm.item())
            
            # iteration tensorboard
            self.tb_writer.add_scalar(f'iterations/losses', float(losses), self.iterations)
            for k, v in loss_dict.items():
                self.tb_writer.add_scalar(f'iterations/{k}', v, self.iterations)
            
            self.tb_writer.add_scalar(f'iterations/lr', self.optimizer.param_groups[0]['lr'], self.iterations)
            
            if grad_norm is not None:
                self.tb_writer.add_scalar('iterations/grad_norm', grad_norm, self.iterations)
            
            self.iterations += 1
        
        # epoch tensorboard
        for k, meter in train_status.meters.items():
            self.tb_writer.add_scalar(f'epochs/{k}', meter.global_avg, self.cur_epoch)

        # 更新学习率
        self.lr_scheduler.step()

        # 保存模型权重
        self._save_weights(f'epoch_{self.cur_epoch}', save_latest=True)
    
    @torch.no_grad()
    def _validation(self) -> None:
        heading('validation')

        self.model.eval()

        if self.cfgs.val_batch_size == 1:
            window_size = len(self.val_lodaer.dataset) // 4
        else:
            window_size = self.window_size

        eval_status = StatusLogger(delimiter=self.delimiter,
                                   metric_fmt=self.metric_fmt,
                                   window_size=window_size)

        for batch_samples in eval_status.log_every(self.val_lodaer, 'Val'):
            inputs = batch_samples.pop('inputs')
            batch_data_samples = batch_samples.pop('data_samples')

            # to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # forward
            predictions = self.model(inputs, batch_data_samples=batch_data_samples, training=False)

            # process
            self.metric.process_preds(predictions)
        
        # metric
        metric, classwise_metric = self.metric.compute_metric()

        # tensorboard
        for k, m in metric.items():
            self.tb_writer.add_scalar(f'metric/{k}', m, self.cur_epoch)

        for cls_name, cls_metric in classwise_metric.items():
            for k, m in cls_metric.items():
                self.tb_writer.add_scalar(f'classwise_metric/{cls_name}/{k}', m, self.cur_epoch)
    
        # save best checkpoints
        for metric_name, best_info in self.metric.best_metric.items():
            if best_info[0] == self.cur_epoch:
                self._save_weights(f'best_{metric_name}', only_model=True, prefix='best')
        
        # save metric
        self.metric.save_metric(self.metric_dir)

        # reset
        self.metric.reset()
    
    def train(self) -> None:
        heading('start det2d training')
        parameter('max grad', self.max_norm)
        parameter('start epoch', self.start_epoch)
        parameter('totoal epoch', self.cfgs.epochs)
        parameter('valid batch size', self.cfgs.val_batch_size)
        parameter('train batch size', self.cfgs.train_batch_size)

        start_time = time.time()

        for self.cur_epoch in range(self.start_epoch, self.cfgs.epochs + 1):
            # train one epoch
            self._train_one_epoch()

            # validation
            if (self.cur_epoch % self.cfgs.eval_interval == 0) or (self.cur_epoch == self.cfgs.epochs):
                self.metric.epoch = self.cur_epoch
                self._validation()
        
        # end training
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))

        logger.success(f'Training completed, total time spent {total_time_str}.')
