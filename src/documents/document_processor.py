# src/documents/document_processor.py
"""文档处理模块"""
import os
from typing import List, Any
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# noinspection PyUnresolvedReferences
from config.config_manager import ConfigManager
# noinspection PyUnresolvedReferences
from core.logger import Logger


class DocumentProcessor:
    """文档处理类"""

    # 文档分割参数常量
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 20

    def __init__(self, logger: Logger, config_manager: ConfigManager):
        self.logger = logger
        self.config_manager = config_manager
        self.doc_config = self.config_manager.get_document_config()

    def load_documents(self) -> List[Any]:
        """加载与读取文档"""
        try:
            self.logger.log_message("开始加载文档...")
            documents = []

            # 如果配置了目录路径，使用DirectoryLoader加载
            if (self.doc_config["directory_path"] and
                    self.doc_config["directory_path"].strip() and
                    os.path.exists(self.doc_config["directory_path"]) and
                    os.path.isdir(self.doc_config["directory_path"])):
                self.logger.log_message(f"从目录加载文档: {self.doc_config['directory_path']}")
                loader_kwargs = {"glob": self.doc_config["file_glob"]}
                if self.doc_config["exclude_pattern"]:
                    loader_kwargs["exclude"] = self.doc_config["exclude_pattern"]

                try:
                    loader = DirectoryLoader(
                        self.doc_config["directory_path"],
                        loader_cls=TextLoader,
                        **loader_kwargs
                    )
                    docs = loader.load()
                    documents.extend(docs)
                    self.logger.log_message(f"从目录加载了 {len(docs)} 个文档")
                except Exception as e:
                    self.logger.log_message(f"从目录加载文档失败: {e}", "ERROR")

            # 加载指定的文件路径
            for file_path in self.doc_config["file_paths"]:
                if not os.path.exists(file_path):
                    self.logger.log_message(f"文件不存在: {file_path}", "WARNING")
                    continue

                try:
                    if file_path.endswith('.pdf'):
                        loader = PyPDFLoader(file_path)
                    else:
                        loader = TextLoader(file_path, encoding='utf-8')

                    docs = loader.load()
                    documents.extend(docs)
                    self.logger.log_message(f"成功加载文件: {file_path}")
                except Exception as e:
                    self.logger.log_message(f"加载文件失败 {file_path}: {e}", "ERROR")
                    continue

            if not documents:
                self.logger.log_message("警告: 未加载到任何文档", "WARNING")
                # 不再抛出异常，而是返回空列表

            self.logger.log_message(f"成功加载 {len(documents)} 个文档")

            # 输出文档内容预览
            if documents:
                self.logger.log_message("文档内容预览:")
                for i, doc in enumerate(documents[:3]):  # 只显示前3个文档
                    self.logger.log_message(f"文档 {i + 1}: {doc.page_content[:200]}...")  # 显示前200个字符

            return documents
        except Exception as e:
            self.logger.log_message(f"加载文档失败: {e}", "ERROR")
            return []  # 返回空列表而不是抛出异常

    def split_documents(self, documents: List[Any]) -> List[Any]:
        """分割文档"""
        try:
            self.logger.log_message("开始分割文档...")
            if not documents:
                self.logger.log_message("没有文档可供分割", "WARNING")
                return []

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.CHUNK_SIZE,
                chunk_overlap=self.CHUNK_OVERLAP,
                length_function=len
            )
            splits = text_splitter.split_documents(documents)

            if not splits:
                self.logger.log_message("警告: 文档分割后未生成任何节点", "WARNING")
                return []

            self.logger.log_message(f"文档已分割为 {len(splits)} 个节点")

            # 输出分割后的内容
            self.logger.log_message("文档分割结果:")
            for i, doc in enumerate(splits[:10]):  # 显示前10个分割结果
                self.logger.log_message(f"节点 {i + 1}: {doc.page_content[:200]}...")  # 显示前200个字符

            return splits
        except Exception as e:
            self.logger.log_message(f"分割文档失败: {e}", "ERROR")
            return []
