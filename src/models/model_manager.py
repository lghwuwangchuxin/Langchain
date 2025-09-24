# src/models/model_manager.py
"""模型管理模块"""

from typing import Tuple, Optional
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
# noinspection PyUnresolvedReferences
from config.config_manager import ConfigManager
# noinspection PyUnresolvedReferences
from core.logger import Logger


class ModelManager:
    """模型管理类"""

    def __init__(self, logger: Logger, config_manager: ConfigManager):
        self.logger = logger
        self.config_manager = config_manager

    def setup_models(self) -> Tuple[Optional[Ollama], Optional[OllamaEmbeddings]]:
        """设置Ollama模型"""
        try:
            self.logger.log_message("开始配置 Ollama 模型...")

            # 获取配置
            ollama_config = self.config_manager.get_ollama_config()

            # 配置语言模型
            self.logger.log_message("配置语言模型...")
            llm = Ollama(
                model=ollama_config["llm_model"],
                base_url=ollama_config["base_url"],
                temperature=ollama_config["temperature"]
            )
            self.logger.log_message("语言模型配置完成")

            # 配置嵌入模型
            self.logger.log_message("配置嵌入模型...")
            embeddings = OllamaEmbeddings(
                model=ollama_config["embedding_model"],
                base_url=ollama_config["base_url"]
            )
            self.logger.log_message("嵌入模型配置完成")

            self.logger.log_message("Ollama模型配置成功")
            return llm, embeddings
        except Exception as e:
            self.logger.log_message(f"Ollama模型配置失败: {e}", "ERROR")
            return None, None
