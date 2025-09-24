# src/core/logger.py
"""日志模块"""

import logging
from datetime import datetime


class Logger:
    """日志管理类"""

    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.timestamp_format = "%Y-%m-%d %H:%M:%S"

    def log_message(self, message: str, level: str = "INFO") -> None:
        """统一日志输出格式"""
        timestamp = datetime.now().strftime(self.timestamp_format)
        print(f"[{timestamp}] [{level}] {message}")
