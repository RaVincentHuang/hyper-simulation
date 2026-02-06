import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from tqdm import tqdm
from contextvars import ContextVar

current_task: ContextVar[str] = ContextVar("task", default="hotpotqa")
current_query_id: ContextVar[str] = ContextVar("query_id", default="")

# 1. 定义一个自定义 Handler - 与tqdm兼容
class TqdmLoggingHandler(logging.StreamHandler):
    """
    与tqdm兼容的日志处理器，将日志输出到stderr，避免与进度条混乱
    """
    def __init__(self, level=logging.NOTSET):
        super().__init__(sys.stderr)
        self.setLevel(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            # 使用tqdm.write确保在进度条上方输出
            tqdm.write(msg, file=sys.stderr)
            self.flush()
        except Exception:
            self.handleError(record)

def getLogger(name: str, level: str = "INFO", log_dir: str = "logs") -> logging.Logger:
    """
    配置全局日志：Tqdm控制台兼容 + 轮转文件
    使用stderr进行日志输出，确保不与tqdm进度条混乱
    """
    # 解析日志级别
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR
    }
    log_level = level_map.get(level.upper(), logging.INFO)
    qid = current_query_id.get()
    task = current_task.get()
    log_path = Path(log_dir) / task
    if qid:
        log_path = log_path / qid
    log_path.mkdir(exist_ok=True, parents=True)
    
    # 通用格式化器
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)-8s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S'
    )
    
    # 控制台处理器 - 使用自定义的TqdmLoggingHandler输出到stderr
    console = TqdmLoggingHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    # 文件 Handler (保持不变)
    file_handler = RotatingFileHandler(
        filename=log_path / f"{name}.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # 配置日志器
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # 清除旧 handler 避免重复
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.addHandler(console)
    logger.addHandler(file_handler)
    
    # 防止向上传播到root logger，避免重复记录
    logger.propagate = False
    
    return logger
