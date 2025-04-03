import os
import sys
import logging
from typing import Optional, Dict, Any, Union, List, Tuple
from pathlib import Path


def setup_directories(directories: List[str]) -> None:
    """创建指定的日志目录列表
    
    Args:
        directories: 需要创建的目录路径列表
    """
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


def get_logger(
    name: str = None,
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
    console_level: int = logging.INFO,
    error_level: int = logging.WARNING,
    formatter_str: str = "%(asctime)s - %(levelname)s - %(message)s",
    file_mode: str = "w",
    propagate: bool = False,
    clear_existing_handlers: bool = True
) -> logging.Logger:
    """配置并返回一个日志记录器
    
    Args:
        name: 日志记录器的名称，默认为root logger
        log_file: 日志文件路径，如果不提供则只输出到控制台
        log_level: 日志记录的最低级别
        console_level: 控制台输出的最低级别
        error_level: 错误输出(stderr)的最低级别
        formatter_str: 日志格式字符串
        file_mode: 文件写入模式，'w'表示覆盖，'a'表示追加
        propagate: 是否传播日志到父记录器
        clear_existing_handlers: 是否清除现有的处理程序
        
    Returns:
        配置好的日志记录器
    """
    # 获取logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = propagate
    
    # 清除现有handlers
    if clear_existing_handlers:
        if name is None:
            # 对于root logger
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
        else:
            # 对于命名logger
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
    
    # 创建formatter
    formatter = logging.Formatter(formatter_str)
    
    # 添加文件处理程序
    if log_file:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file, mode=file_mode)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # 添加控制台处理程序
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(console_level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    
    # 添加错误处理程序
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(error_level)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)
    
    return logger


def configure_project_logger(
    project_name: str,
    log_dir: str = "./logs",
    backtest_dir: str = "./backtest",
    only_backtest: bool = False,
    universe: str = "default",
    log_level: int = logging.INFO,
    clear_handlers: bool = True
) -> logging.Logger:
    """项目专用日志配置，模拟原始main.py的日志设置
    
    Args:
        project_name: 项目名称，用于日志前缀
        log_dir: 常规日志目录
        backtest_dir: 回测日志目录
        only_backtest: 是否只使用回测日志
        universe: 市场标识，用于日志文件命名
        log_level: 日志级别
        clear_handlers: 是否清除已有的handlers
        
    Returns:
        配置好的日志记录器
    """
    # 创建必要的目录
    setup_directories([log_dir, backtest_dir])
    
    # 确定日志文件路径
    if only_backtest:
        log_file = f"{backtest_dir}/{universe}.log"
    else:
        log_file = f"{log_dir}/{universe}.log"
    
    # 配置并返回日志记录器
    return get_logger(
        name=project_name,
        log_file=log_file,
        log_level=log_level,
        console_level=log_level,
        error_level=logging.WARNING,
        clear_existing_handlers=clear_handlers
    )


def redirect_print_to_logger(logger: logging.Logger) -> callable:
    """重定向print函数输出到logger
    
    Args:
        logger: 目标日志记录器
        
    Returns:
        原始的print函数，可用于恢复
    """
    import builtins
    orig_print = print
    
    def custom_print(*args, **kwargs):
        msg = ' '.join(map(str, args))
        logger.info(msg)
        # 如果需要同时保留控制台输出，取消注释下一行
        # orig_print(*args, **kwargs)
    
    builtins.print = custom_print
    return orig_print


def restore_print(orig_print: callable) -> None:
    """恢复原始的print函数
    
    Args:
        orig_print: 原始的print函数
    """
    import builtins
    builtins.print = orig_print


class LoggerContext:
    """日志上下文管理器，用于临时重定向print到logger"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.orig_print = None
    
    def __enter__(self):
        self.orig_print = redirect_print_to_logger(self.logger)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        restore_print(self.orig_print) 