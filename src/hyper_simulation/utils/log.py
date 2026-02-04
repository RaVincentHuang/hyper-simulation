import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler
from tqdm import tqdm

# 1. 定义一个自定义 Handler
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            # 核心：使用 tqdm.write 替代 print 或 sys.stdout.write
            # 这能确保日志打印在进度条上方，而不会打断进度条
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

def getLogger(name: str, query_id: str = "", dataset: str = "hotpotqa", level: str = "INFO", log_dir: str = "logs") -> logging.Logger:
    """
    配置全局日志：Tqdm控制台兼容 + 轮转文件
    """
    # 使用函数属性实现单例模式（防止重复添加 Handler）
    
    # 解析日志级别
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR
    }
    log_level = level_map.get(level.upper(), logging.INFO)
    
    if query_id:
        log_path = Path(log_dir, dataset) / query_id
    else:
        log_path = Path(log_dir, dataset)
    log_path.mkdir(exist_ok=True, parents=True) # parents=True防止父目录不存在报错
    
    # 通用格式化器
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)-10s | %(levelname)-7s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # --- 修改点开始 ---
    # 移除原有的 StreamHandler(sys.stdout)
    # 替换为自定义的 TqdmLoggingHandler
    console = TqdmLoggingHandler()
    console.setLevel(logging.INFO) # 控制台通常只看 INFO
    console.setFormatter(formatter)
    # --- 修改点结束 ---

    # 文件 Handler (保持不变)
    file_handler = RotatingFileHandler(
        filename=log_path / f"{name}.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(log_level) # 文件记录详细等级
    file_handler.setFormatter(formatter)
    
    # 配置根日志器
    root = logging.getLogger(name) # 获取当前模块logger，或者使用 logging.getLogger() 获取根 logger
    root.setLevel(log_level)
    
    # 清除旧 handler 避免重复
    if root.hasHandlers():
        root.handlers.clear()
        
    root.addHandler(console)
    root.addHandler(file_handler)
    
    return root

# --- 测试代码 ---
if __name__ == "__main__":
    import time
    
    # 初始化 logger
    logger = getLogger(level="DEBUG")
    
    logger.info("开始任务...")
    
    # 测试 Tqdm 兼容性
    for i in tqdm(range(10), desc="模拟进度"):
        time.sleep(0.5)
        if i == 5:
            # 这条日志会出现在进度条上方，且进度条不会断裂
            logger.warning(f"检测到异常值在索引 {i}")
            
    logger.info("任务完成")