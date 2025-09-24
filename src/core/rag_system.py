# src/core/rag_system.py
"""RAG系统主模块"""

# 条件导入hub模块
try:
    from langchain import hub
    HUB_AVAILABLE = True
except ImportError:
    HUB_AVAILABLE = False
    print("警告: 无法导入 langchain.hub，将使用自定义提示模板")

from typing import Optional
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
# noinspection PyUnresolvedReferences
from config.config_manager import ConfigManager
# noinspection PyUnresolvedReferences
from core.logger import Logger
# noinspection PyUnresolvedReferences
from models.model_manager import ModelManager
# noinspection PyUnresolvedReferences
from vectorstore.milvus_manager import MilvusManager
# noinspection PyUnresolvedReferences
from documents.document_processor import DocumentProcessor


class RAGSystem:
    """RAG系统主类"""

    def __init__(self, config_path: str = "/Users/liuguanghu/PythonPorject/Langchain/config/config.ini"):
        self.logger = Logger()
        self.config_manager = ConfigManager(config_path)
        self.model_manager = ModelManager(self.logger, self.config_manager)
        self.milvus_manager = MilvusManager(self.logger, self.config_manager)
        self.document_processor = DocumentProcessor(self.logger, self.config_manager)
        self.vector_store = None
        self.rag_chain = None

    def initialize_system(self) -> bool:
        """初始化RAG系统"""
        try:
            # 1. 设置模型
            llm, embeddings = self.model_manager.setup_models()
            if llm is None or embeddings is None:
                return False

            # 2. 连接Milvus
            if not self.milvus_manager.connect():
                return False

            # 3. 加载和处理文档
            documents = self.document_processor.load_documents()
            splits = self.document_processor.split_documents(documents)

            # 检查是否有文档被加载和分割
            if not splits:
                self.logger.log_message("没有文档可供处理，系统初始化失败", "ERROR")
                return False

            # 4. 设置向量存储
            self.vector_store = self.milvus_manager.setup_vector_store(embeddings)

            # 5. 将文档添加到向量存储
            self.logger.log_message("开始将文档添加到向量存储...")
            # 为每个文档生成唯一ID以避免Milvus的auto_id问题
            ids = [f"doc_{i}" for i in range(len(splits))]
            self.vector_store.add_documents(splits, ids=ids)
            self.logger.log_message("文档添加完成")

            # 6. 创建检索器
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.config_manager.get_qa_config()["retriever_k"]}
            )

            # 7. 创建RAG链
            self.logger.log_message("开始构造RAG链...")

            # 获取QA配置
            qa_config = self.config_manager.get_qa_config()

            # 选择使用hub提示模板还是自定义模板
            if qa_config["use_hub_prompt"] and HUB_AVAILABLE:
                prompt = hub.pull("rlm/rag-prompt")
            else:
                prompt = PromptTemplate(
                    template=qa_config["prompt_template"],
                    input_variables=["context", "question"]
                )

            # 使用现代的LangChain表达式语言(LCEL)构建RAG链
            self.rag_chain = (
                    {
                        "context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
                        "question": RunnablePassthrough()
                    }
                    | prompt
                    | llm
                    | StrOutputParser()
            )

            self.logger.log_message("RAG链已准备就绪")

            return True

        except Exception as e:
            self.logger.log_message(f"系统初始化失败: {e}", "ERROR")
            import traceback
            self.logger.log_message(f"详细错误信息: {traceback.format_exc()}", "ERROR")
            return False

    def run_interactive_qa(self) -> None:
        """运行交互式问答"""
        if not self.rag_chain:
            self.logger.log_message("RAG链未初始化，无法进行问答", "ERROR")
            return

        self.logger.log_message("欢迎使用AI助手！输入 'exit' 退出程序。")
        while True:
            try:
                user_input = input("\n问题：").strip()
                if user_input.lower() == "exit":
                    self.logger.log_message("用户退出程序")
                    print("再见！")
                    break

                if not user_input:
                    print("请输入有效问题")
                    continue

                self.logger.log_message(f"正在处理用户问题: {user_input}")

                # 执行问答
                response = self.rag_chain.invoke(user_input)
                print("AI助手：", response)
                print("-" * 50)

            except KeyboardInterrupt:
                self.logger.log_message("程序被用户中断")
                print("\n程序被用户中断")
                break
            except Exception as e:
                self.logger.log_message(f"查询过程中出错: {e}", "ERROR")
                print(f"查询出错：{e}")

    def cleanup(self) -> None:
        """清理资源"""
        try:
            self.logger.log_message("正在断开 Milvus 连接...")
            self.milvus_manager.disconnect()
            self.logger.log_message("已断开 Milvus 连接")
        except Exception as e:
            self.logger.log_message(f"断开连接时发生错误: {e}", "ERROR")

        self.logger.log_message("程序执行完成")
