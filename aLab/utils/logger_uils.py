# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import os
import sys
import time
import torch
import inspect
import datetime

from loguru import logger
from collections import deque
from typing import (Any, Optional, Iterable)


__all__ = ['setup_logger', 'heading', 'parameter', 'SmoothedValue', 'StatusLogger']


MB = 1024.0 * 1024.0


class StreamToLoguru:
    """
    stream object that redirects writes to a logger instance.
    """

    def __init__(self, level: str = 'INFO', caller_names: tuple = ('apex', 'pycocotools')) -> None:
        """
        Args:
            level(str): log level string of loguru. Default value: 'INFO'.
            caller_names(tuple): caller names of redirected module. Default value: (apex, pycocotools).
        """
        self.level = level
        self.linebuf = ''
        self.caller_names = caller_names
    
    @staticmethod
    def get_caller_name(depth: int = 0) -> str:
        """
        Args:
            depth (int): Depth of caller conext, use 0 for caller depth. Default value: 0.

        Returns:
            str: module name of the caller
        """
        # the following logic is a little bit faster than inspect.stack() logic
        frame = inspect.currentframe().f_back
        for _ in range(depth):
            frame = frame.f_back

        return frame.f_globals['__name__']

    def write(self, buf: Any) -> None:
        full_name = self.get_caller_name(depth=1)
        module_name = full_name.rsplit('.', maxsplit=-1)[0]

        if module_name in self.caller_names:
            for line in buf.rstrip().splitlines():
                # use caller level log
                logger.opt(depth=2).log(self.level, line.rstrip())
        else:
            sys.__stdout__.write(buf)

    def flush(self) -> None:
        pass


def setup_logger(work_dir: str, filename: str = 'log.log', mode: str = 'a') -> None:
    """setup logger for training and testing.
    Args:
        filename (string): log save name.
        work_dir(str): location to save log file
        mode(str): log file write mode, `append` or `override`. default is `a`.

    Return:
        logger instance.
    """
    # step1. 删除loguru默认处理程序的配置, 其id=0
    logger.remove()

    # step2. 创建日志文件
    save_file = os.path.join(work_dir, filename)
    if mode == 'o' and os.path.exists(save_file):
        os.remove(save_file)

    # ---------------- logger.add 参数说明 ----------------
    # sink: 为记录器生成的每条记录指定目的地, 默认情况下，它设置为 sys.stderr输出到终端
    # level: 指定记录器的最低日志级别
    # filter: 用于确定一条记录是否应该被记录
    # ----------------------------------------------------
    base_log_level(save_file)
    custom_log_level(save_file)

    # redirect stdout/stderr to loguru
    redirect_logger = StreamToLoguru('INFO')
    sys.stderr = redirect_logger
    sys.stdout = redirect_logger


def base_log_level(save_file: str) -> None:
    """
    重定义基础loguru的信息格式
    包括: INFO, WARNING, ERROR, DEBUG, SUCCESS, TRACE, CRITICAL

    Args:
        save_file (str): 输出的日志文件
    """
    # 设置info-level日志处理程序 (INFO (20): 用于记录描述程序正常操作的信息消息)
    info_format = (
        '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | '
        '<white>{message}</white>'
        )
    logger.add(
        sink=sys.stderr,
        level='INFO',
        filter=lambda record: record['level'].name == 'INFO', format=info_format)
    logger.add(
        sink=save_file,
        colorize=False,
        level='INFO',
        filter=lambda record: record['level'].name == 'INFO', format=info_format)
    
    # 设置warning-level日志处理程序 (WARNING (30): 警告类型，用于指示可能需要进一步调查的不寻常事件)
    warning_format = (
        '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | '
        '<yellow><level>{module}</level></yellow>:<yellow><level>{line}</level></yellow> - <yellow><level>{message}</level></yellow>'
        )
    logger.add(
        sink=sys.stderr,
        level='WARNING',
        filter=lambda record: record['level'].name == 'WARNING', format=warning_format)
    logger.add(
        sink=save_file,
        colorize=False,
        level='WARNING',
        filter=lambda record: record['level'].name == 'WARNING', format=warning_format)
    
    # 设置error-level日志处理程序 (ERROR (40): 错误类型，用于记录影响特定操作的错误条件) (格式里面添加了process和thread记录，方便查看多进程和线程程序)
    error_format = (
        '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | '
        '<magenta>{process}</magenta>:<yellow>{thread}</yellow> | '
        '<red><level>{module}</level></red>:<red><level>{line}</level></red> - <red><level>{message}</level></red>'
        )
    logger.add(
        sink=sys.stderr,
        level='ERROR',
        filter=lambda record: record['level'].name == 'ERROR', format=error_format)
    logger.add(
        sink=save_file,
        colorize=False,
        level='ERROR',
        filter=lambda record: record['level'].name == 'ERROR', format=error_format)
    
    # 设置debug-level日志处理程序 (DEBUG (10): 开发人员使用该工具记录调试信息)
    debug_format = (
        '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | '
        '<blue>{module}</blue>:<blue>{line}</blue> - <blue>{message}</blue>'
        )
    logger.add(
        sink=sys.stderr,
        level='DEBUG',
        filter=lambda record: record['level'].name == 'DEBUG', format=debug_format)
    logger.add(
        sink=save_file,
        colorize=False,
        level='DEBUG',
        filter=lambda record: record['level'].name == 'DEBUG', format=debug_format)
    
    # 设置success-level日志处理程序 (SUCCESS (25): 类似于INFO，用于指示操作成功的情况。)
    success_format = (
        '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | '
        '<green>{message}</green>'
        )
    logger.add(
        sink=sys.stderr,
        level='SUCCESS',
        filter=lambda record: record['level'].name == 'SUCCESS', format=success_format)
    logger.add(
        sink=save_file,
        colorize=False,
        level='SUCCESS',
        filter=lambda record: record['level'].name == 'SUCCESS', format=success_format)

    # 设置trace-level日志处理程序 (TRACE (5): 用于记录程序执行路径的细节信息，以进行诊断) (格式里面添加了process和thread记录，方便查看多进程和线程程序)
    trace_format = (
        '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | '
        '<magenta>{process}</magenta>:<magenta>{thread}</magenta> | '
        '<magenta><level>{module}</level></magenta>:<magenta><level>{line}</level></magenta> - <magenta><level>{message}</level></magenta>'
        )
    logger.add(
        sink=sys.stderr,
        level='TRACE',
        filter=lambda record: record['level'].name == 'TRACE', format=trace_format)
    logger.add(
        sink=save_file,
        colorize=False,
        level='TRACE',
        filter=lambda record: record['level'].name == 'TRACE', format=trace_format)
    
    # 设置critical-level日志处理程序 (CRITICAL (50): 严重类型，用于记录阻止核心功能正常工作的错误条件) (格式里面添加了process和thread记录，方便查看多进程和线程程序)
    critical_format = (
        '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | '
        '<red>{process}</red>:<red>{thread}</red> | '
        '<red><level>{module}</level></red>:<red><level>{line}</level></red> - <red><level>{message}</level></red>'
        )
    logger.add(
        sink=sys.stderr,
        level='CRITICAL',
        filter=lambda record: record['level'].name == 'CRITICAL', format=critical_format)
    logger.add(
        sink=save_file,
        colorize=False,
        level='CRITICAL',
        filter=lambda record: record['level'].name == 'CRITICAL', format=critical_format)
    

def custom_log_level(save_file: str) -> None:
    """
    自定义loguru日志格式
    包括: HEADING, SUBHEADING, PARAMETER

    Args:
        save_file (str): 输出的日志文件
    """
    # 设置subheading日志处理程序 (SUBHEADING (24): 副标题)
    logger.level('HEADING', no=24, color='<blue>', icon='@')
    heading_fromat = (
        '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | '
        '<b><magenta>{message}</magenta></b>'
        )
    logger.add(
        sink=sys.stderr,
        level='HEADING',
        filter=lambda record: record['level'].name == 'HEADING', format=heading_fromat)
    logger.add(
        sink=save_file,
        colorize=False,
        level='HEADING',
        filter=lambda record: record['level'].name == 'HEADING', format=heading_fromat)
    
    # 设置key-info日志处理程序 (PARAMETER (22): 用于高亮显示一些关键信息)
    logger.level('PARAMETER', no=22, color='<yellow>', icon='@')
    keyinfo_format = (
        '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | '
        '<blue>===> {message}:</blue> <white>{extra[value]}</white>'
    )
    logger.add(
        sink=sys.stderr,
        level='PARAMETER',
        filter=lambda record: record['level'].name == 'PARAMETER', format=keyinfo_format)
    logger.add(
        sink=save_file,
        level='PARAMETER',
        colorize=False,
        filter=lambda record: record['level'].name == 'PARAMETER', format=keyinfo_format)
    

def heading(string: str) -> None:
    """打印标题
    目前支持: HEADING

    Args:
        string (str): 标题
    """
    logger.log('HEADING', f' {string.upper()} '.center(120, '-'))


def parameter(p: str, v: str = '') -> None:
    """打印关键信息

    Args:
        p (str): 参数名称
        v (str, optional): 参数值. Defaults to ''.
    """
    logger.log('PARAMETER', p, value=v)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size: Optional[int] = 20, fmt: Optional[str] = None):
        if fmt is None:
            fmt = "{global_avg:.4f}"

        self.fmt = fmt
        
        self.count = 0
        self.total = 0.0

        self.deque = deque(maxlen=window_size)

    def update(self, value, n=1) -> None:
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self) -> float:
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self) -> float:
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self) -> float:
        return self.total / self.count

    @property
    def max(self) -> float:
        return max(self.deque)

    @property
    def value(self) -> float:
        return self.deque[-1]

    def __str__(self) -> str:
        return self.fmt.format(
            value=self.value,
            max=self.max,
            avg=self.avg,
            median=self.median,
            global_avg=self.global_avg)
    

class StatusLogger(object):
    def __init__(self,
                 delimiter: Optional[str] = ' | ',
                 metric_fmt: Optional[str] = None,
                 window_size: Optional[int] = 20):
        
        self.meters = dict()
        self.delimiter = delimiter

        self.metric_fmt = metric_fmt
        self.window_size = window_size

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        
        if attr in self.__dict__:
            return self.__dict__[attr]
        
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))
    
    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)
    
    def add_meter(self, name, meter) -> None:
        self.meters[name] = meter

    def update(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))

            if k not in self.meters:
               self.meters[k] = SmoothedValue(window_size=self.window_size, 
                                              fmt=self.metric_fmt)
            
            self.meters[k].update(v)

    def train_log_every(self, iterable: Iterable, header: str = ''):
        iterations = 1
        num_iterations = len(iterable)

        iter_time = SmoothedValue(window_size=self.window_size, fmt='{avg:.4f}')   # 记录每次迭代的时间
        data_time = SmoothedValue(window_size=self.window_size, fmt='{avg:.4f}')   # 记录每次数据加载的时间

        space_fmt = f':{str(len(str(len(iterable))))}d'
        log_fmt = [header, 
                   '[{iterations' + space_fmt + '}/{num_iterations}]',
                   'eta: {eta}',
                   '{meters}',
                   'time: {time}',
                   'data: {data}']
        
        if torch.cuda.is_available():
            log_fmt.append('max mem: {memory:.0f}')
        
        log_msg = self.delimiter.join(log_fmt)
        
        # 迭代
        stime = time.time()
        etime = time.time()

        for obj in iterable:
            # 记录加载数据的时间
            data_time.update(time.time() - etime)

            # 推理 & 更新梯度
            yield obj

            # 记录一次迭代的时间
            iter_time.update(time.time() - etime)  # 包含加载数据的时间

            # 打印日志
            if iterations % self.window_size == 0 or iterations == num_iterations:
                eta_seconds = iter_time.global_avg * (num_iterations - iterations)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                log_values = dict(
                    iterations=iterations,
                    num_iterations=num_iterations,
                    eta=eta_string,
                    meters=str(self),
                    time=str(iter_time), 
                    data=str(data_time))

                if torch.cuda.is_available():
                    log_values['memory'] = torch.cuda.max_memory_allocated() / MB
                
                logger.info(log_msg.format(**log_values))
            
            iterations += 1
            etime = time.time()

        total_time = time.time() - stime
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.success('{} total time: {} ({:.4f} s / iter)'.format(header, total_time_str, total_time / num_iterations))
    
    def log_every(self, iterable: Iterable, header: str = ''):
        iterations = 1
        num_iterations = len(iterable)

        iter_time = SmoothedValue(window_size=self.window_size, fmt='{avg:.4f}')   # 记录每次迭代的时间
        data_time = SmoothedValue(window_size=self.window_size, fmt='{avg:.4f}')   # 记录每次数据加载的时间

        space_fmt = f':{str(len(str(len(iterable))))}d'
        log_fmt = [header, 
                   '[{iterations' + space_fmt + '}/{num_iterations}]',
                   'eta: {eta}',
                   'time: {time}',
                   'data: {data}']
        
        if torch.cuda.is_available():
            log_fmt.append('max mem: {memory:.0f}')
        
        log_msg = self.delimiter.join(log_fmt)
        
        # 迭代
        stime = time.time()
        etime = time.time()

        for obj in iterable:
            # 记录加载数据的时间
            data_time.update(time.time() - etime)

            # 推理 & 更新梯度
            yield obj

            # 记录一次迭代的时间
            iter_time.update(time.time() - etime)  # 包含加载数据的时间

            # 打印日志
            if iterations % self.window_size == 0 or iterations == num_iterations:
                eta_seconds = iter_time.global_avg * (num_iterations - iterations)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                log_values = dict(
                    iterations=iterations,
                    num_iterations=num_iterations,
                    eta=eta_string,
                    time=str(iter_time), 
                    data=str(data_time))

                if torch.cuda.is_available():
                    log_values['memory'] = torch.cuda.max_memory_allocated() / MB
                
                logger.info(log_msg.format(**log_values))
            
            iterations += 1
            etime = time.time()

        total_time = time.time() - stime
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.success('{} total time: {} ({:.4f} s / iter)'.format(header, total_time_str, total_time / num_iterations))