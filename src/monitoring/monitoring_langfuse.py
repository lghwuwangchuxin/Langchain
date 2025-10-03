# src/monitoring/monitoring_langfuse.py
"""Langfuse RAG流程监控模块 - 优化版"""

from typing import Optional, Any, Dict, List
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langchain_core.runnables import Runnable
from langchain_core.callbacks import BaseCallbackHandler
import uuid
import logging
import os
# noinspection PyUnresolvedReferences
from config.config_manager import ConfigManager
# noinspection PyUnresolvedReferences
from core.logger import Logger


class RAGMonitor:
    """RAG流程监控器 - 修复优化版"""

    def __init__(self, logger: Logger, config_manager: ConfigManager):
        """
        初始化RAG监控器

        Args:
            logger: 日志记录器实例
            config_manager: 配置管理器实例
        """
        self.logger = logger
        self.config_manager = config_manager
        self.langfuse = None
        self._is_enabled = False  # 默认禁用

        # 获取Langfuse配置
        langfuse_config = self._get_langfuse_config()

        # 检查必要的配置项
        if not langfuse_config["public_key"] or not langfuse_config["secret_key"]:
            self.logger.log_message("Langfuse公钥或私钥未配置，监控功能将被禁用", level=logging.WARNING)
            self._is_enabled = False
            return

        try:
            # 初始化Langfuse客户端 - 使用正确的参数名
            self.langfuse = Langfuse(
                public_key=langfuse_config["public_key"],
                secret_key=langfuse_config["secret_key"],
                host=langfuse_config["host"],
            )
            self._is_enabled = True

            # 测试连接
            self._test_connection()
            self.logger.log_message("RAG监控器初始化完成 - Langfuse已启用")

        except Exception as e:
            self.logger.log_message(f"Langfuse初始化失败: {e}", level=logging.ERROR)
            self._is_enabled = False

    def _get_langfuse_config(self) -> Dict[str, Any]:
        """
        获取Langfuse配置 - 修复版

        Returns:
            包含Langfuse配置的字典
        """
        config = self.config_manager.config

        # 检查配置节是否存在
        if not config.has_section("Langfuse"):
            self.logger.log_message("未找到Langfuse配置节，监控功能将被禁用", level=logging.WARNING)
            return {
                "public_key": "",
                "secret_key": "",
                "host": "https://cloud.langfuse.com",
                "enabled": False
            }

        # 从配置文件读取配置
        return {
            "public_key": config.get("Langfuse", "public_key", fallback=""),
            "secret_key": config.get("Langfuse", "secret_key", fallback=""),
            "host": config.get("Langfuse", "host", fallback="https://cloud.langfuse.com"),
            "enabled": config.getboolean("Langfuse", "enabled", fallback=True)
        }

    def _test_connection(self):
        """测试Langfuse连接"""
        if not self._is_enabled:
            return

        try:
            # 简单的连接测试
            self.langfuse.auth_check()
            self.logger.log_message("Langfuse连接测试成功")
        except Exception as e:
            self.logger.log_message(f"Langfuse连接测试失败: {e}", level=logging.WARNING)
            self._is_enabled = False

    def get_callback_handler(self, trace_name: str = "RAG_Process",
                             user_id: str = None,
                             session_id: str = None,
                             metadata: Optional[Dict] = None) -> Optional[CallbackHandler]:
        """
        获取Langfuse回调处理器 - 修复版

        Args:
            trace_name: 追踪名称
            user_id: 用户ID
            session_id: 会话ID
            metadata: 附加元数据

        Returns:
            Langfuse回调处理器实例或None（如果监控被禁用）
        """
        if not self._is_enabled:
            return None

        try:
            return CallbackHandler()
        except Exception as e:
            self.logger.log_message(f"创建回调处理器失败: {e}", level=logging.ERROR)
            return None

    def monitor_rag_chain(self, rag_chain: Runnable, query: str,
                          session_id: Optional[str] = None,
                          user_id: Optional[str] = None,
                          metadata: Optional[Dict] = None) -> Any:
        """
        监控RAG链的执行过程 - 修复版

        Args:
            rag_chain: RAG链实例
            query: 用户查询
            session_id: 会话ID（可选）
            user_id: 用户ID（可选）
            metadata: 附加元数据

        Returns:
            RAG链的执行结果
        """
        if not self._is_enabled:
            self.logger.log_message("Langfuse监控未启用，直接执行查询")
            return rag_chain.invoke(query)

        callback_handler = None
        try:
            # 创建回调处理器
            callback_handler = self.get_callback_handler(
                trace_name="RAG_QA_Process",
                user_id=user_id,
                session_id=session_id,
                metadata=metadata
            )

            if not callback_handler:
                self.logger.log_message("无法创建回调处理器，直接执行查询", level=logging.WARNING)
                return rag_chain.invoke(query)

            self.logger.log_message(f"开始执行RAG查询: {query}")

            # 执行RAG链并传递回调处理器
            result = rag_chain.invoke(
                query
            )

            self.logger.log_message("RAG查询执行完成")
            return result

        except Exception as e:
            self.logger.log_message(f"RAG监控执行失败: {e}", level=logging.ERROR)
            # 回退到无监控执行
            return rag_chain.invoke(query)

    def create_custom_trace(self, name: str, input_data: Any = None,
                            user_id: str = "default_user",
                            session_id: str = None,
                            metadata: Optional[Dict] = None) -> Optional[str]:
        """
        创建自定义追踪 - 修复版

        Args:
            name: 追踪名称
            input_data: 输入数据
            user_id: 用户ID
            session_id: 会话ID
            metadata: 元数据

        Returns:
            追踪ID或None（如果失败）
        """
        if not self._is_enabled:
            return None

        try:
            # 使用正确的API创建追踪
            trace = self.langfuse.trace(
                name=name,
                user_id=user_id,
                session_id=session_id or self._generate_session_id(),
                input=input_data,
                metadata=metadata or {}
            )
            return trace.id
        except Exception as e:
            self.logger.log_message(f"创建自定义追踪失败: {e}", level=logging.ERROR)
            return None

    def update_trace_output(self, trace_id: str, output_data: Any) -> bool:
        """
        更新追踪的输出数据 - 修复版

        Args:
            trace_id: 追踪ID
            output_data: 输出数据

        Returns:
            是否成功更新
        """
        if not self._is_enabled:
            return False

        try:
            # 获取现有追踪并更新
            trace = self.langfuse.trace(id=trace_id)
            if trace:
                trace.update(output=output_data)
                return True
            return False
        except Exception as e:
            self.logger.log_message(f"更新追踪输出失败: {e}", level=logging.ERROR)
            return False

    def get_trace_url(self, trace_id: str) -> str:
        """
        获取追踪URL用于查看监控结果

        Args:
            trace_id: 追踪ID

        Returns:
            追踪详情页面URL
        """
        if not self._is_enabled:
            return "监控未启用"

        try:
            return self.langfuse.get_trace_url(trace_id)
        except Exception as e:
            self.logger.log_message(f"获取追踪URL失败: {e}", level=logging.WARNING)
            return f"无法生成URL: {str(e)}"

    def _generate_session_id(self) -> str:
        """
        生成会话ID

        Returns:
            会话ID字符串
        """
        return f"session_{uuid.uuid4().hex[:8]}"

    def is_enabled(self) -> bool:
        """
        检查监控是否启用

        Returns:
            布尔值表示是否启用
        """
        return self._is_enabled

    def flush(self):
        """强制刷新所有待处理事件到Langfuse"""
        if self._is_enabled and self.langfuse:
            try:
                self.langfuse.flush()
            except Exception as e:
                self.logger.log_message(f"刷新数据失败: {e}", level=logging.WARNING)