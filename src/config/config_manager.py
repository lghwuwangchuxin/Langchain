# src/config/config_manager.py
"""配置管理模块"""

import configparser
import os
from typing import Dict, Any, Optional


class ConfigManager:
    """配置管理类"""

    def __init__(self, config_path: str = "/Users/liuguanghu/PythonPorject/Langchain/config/config.ini"):
        self.config = configparser.ConfigParser()
        # 检查配置文件是否存在
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        self.config.read(config_path)

        # 缓存配置以避免重复读取
        self._ollama_config: Optional[Dict[str, Any]] = None
        self._milvus_config: Optional[Dict[str, Any]] = None
        self._document_config: Optional[Dict[str, Any]] = None
        self._qa_config: Optional[Dict[str, Any]] = None

    def get_ollama_config(self) -> Dict[str, Any]:
        """获取Ollama配置"""
        if self._ollama_config is None:
            self._ollama_config = {
                "llm_model": self.config.get("ollama", "llm_model"),
                "embedding_model": self.config.get("ollama", "embedding_model"),
                "base_url": self.config.get("ollama", "base_url"),
                "temperature": self.config.getfloat("ollama", "temperature")
            }
        return self._ollama_config

    def get_milvus_config(self) -> Dict[str, Any]:
        """获取Milvus配置"""
        if self._milvus_config is None:
            self._milvus_config = {
                "host": self.config.get("milvus", "host"),
                "port": self.config.get("milvus", "port"),
                "collection_name": self.config.get("milvus", "collection_name"),
                "database_name": self.config.get("milvus", "database_name")
            }
        return self._milvus_config

    def get_document_config(self) -> Dict[str, Any]:
        """获取文档配置"""
        if self._document_config is None:
            file_paths = self.config.get("document", "file_paths", fallback="")
            # 解析多行文件路径
            paths = [path.strip() for path in file_paths.split(",") if path.strip()]
            directory_path = self.config.get("document", "directory_path", fallback=None)
            file_glob = self.config.get("document", "file_glob", fallback="*")
            exclude_pattern = self.config.get("document", "exclude_pattern", fallback=None)
            # 获取分割器类型配置，默认为 recursive_character
            text_splitter = self.config.get("document", "text_splitter", fallback="recursive_character")
            self._document_config = {
                "file_paths": paths,
                "directory_path": directory_path,
                "file_glob": file_glob,
                "exclude_pattern": exclude_pattern,
                "text_splitter": text_splitter
            }
        return self._document_config

    def get_qa_config(self) -> Dict[str, Any]:
        """获取QA配置"""
        if self._qa_config is None:
            self._qa_config = {
                "retriever_k": self.config.getint("qa", "retriever_k"),
                "prompt_template": self.config.get("qa", "prompt_template"),
                "use_hub_prompt": self.config.getboolean("qa", "use_hub_prompt", fallback=False)
            }
        return self._qa_config
