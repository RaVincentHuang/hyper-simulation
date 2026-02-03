# log_config.py
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys

def setup(level: str = "INFO", log_dir: str = "logs") -> None:
    """
    一次性配置全局日志：控制台 + 轮转文件
    
    Args:
        level: "DEBUG" / "INFO" / "WARNING"
        log_dir: 日志文件存储目录
    """
    # 防止重复配置（避免多次import时重复添加handler）
    if getattr(setup, "_configured", False):
        return
    setup._configured = True
    
    # 解析日志级别
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR
    }
    log_level = level_map.get(level.upper(), logging.INFO)
    
    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # 格式化器
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台输出（始终显示INFO及以上）
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    
    file_handler = RotatingFileHandler(
        filename=log_path / "hyper_simulation.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # 配置根日志器
    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(file_handler)
    
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    
    logging.getLogger(__name__).info(f"✓ 日志系统初始化完成 (level={level}, dir={log_dir})")