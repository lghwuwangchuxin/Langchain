from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_milvus.vectorstores.milvus import Milvus as MilvusVectorStore

# 条件导入hub模块
try:
    from langchain import hub

    HUB_AVAILABLE = True
except ImportError:
    HUB_AVAILABLE = False
    print("警告: 无法导入 langchain.hub，将使用自定义提示模板")

from pymilvus import connections, utility
import logging
from datetime import datetime
import os
import configparser
from typing import Dict, List, Tuple, Optional, Any


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
            self._document_config = {
                "file_paths": paths,
                "directory_path": directory_path,
                "file_glob": file_glob,
                "exclude_pattern": exclude_pattern
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


class MilvusManager:
    """Milvus数据库管理类"""

    def __init__(self, logger: Logger, config_manager: ConfigManager):
        self.logger = logger
        self.config_manager = config_manager
        milvus_config = self.config_manager.get_milvus_config()
        self.host = milvus_config["host"]
        self.port = milvus_config["port"]
        self.collection_name = milvus_config["collection_name"]
        self.db_name = milvus_config["database_name"]
        self._connection = None

    def connect(self) -> bool:
        """连接到Milvus服务"""
        try:
            self.logger.log_message("开始连接 Milvus 服务...")
            self._connection = connections.connect("default", host=self.host, port=self.port)
            self.logger.log_message("成功连接到 Milvus")
            return True
        except Exception as e:
            self.logger.log_message(f"连接 Milvus 失败: {e}", "ERROR")
            return False

    def disconnect(self) -> None:
        """断开Milvus连接"""
        try:
            if self._connection:
                connections.disconnect("default")
                self._connection = None
                self.logger.log_message("已断开 Milvus 连接")
        except Exception as e:
            self.logger.log_message(f"断开 Milvus 连接失败: {e}", "ERROR")

    def check_database_support(self) -> bool:
        """检查是否支持多数据库功能"""
        self.logger.log_message("检查 Milvus 数据库支持功能...")
        has_list = hasattr(utility, 'list_database')
        has_create = hasattr(utility, 'create_database')
        support = has_list and has_create
        self.logger.log_message(
            f"数据库支持检查结果: list_database={has_list}, create_database={has_create}, 支持多数据库={support}")
        return support

    def create_database(self) -> bool:
        """创建数据库"""
        self.logger.log_message(f"开始创建数据库: {self.db_name}")
        if not self.check_database_support():
            self.logger.log_message("当前版本不支持多数据库功能，使用默认数据库", "WARNING")
            return False

        try:
            existing_databases = utility.list_database()
            self.logger.log_message(f"现有数据库列表: {existing_databases}")

            if self.db_name not in existing_databases:
                utility.create_database(db_name=self.db_name)
                self.logger.log_message(f"数据库 '{self.db_name}' 创建成功")
                return True
            else:
                self.logger.log_message(f"数据库 '{self.db_name}' 已存在")
                return True
        except Exception as e:
            self.logger.log_message(f"创建数据库失败: {e}", "ERROR")
            raise

    def setup_vector_store(self, embeddings: OllamaEmbeddings) -> MilvusVectorStore:
        """设置向量存储"""
        try:
            self.logger.log_message("开始准备 Milvus 向量存储...")

            # 检查并创建数据库（如果支持）
            if self.check_database_support():
                self.create_database()
                # 切换到指定数据库
                self.logger.log_message("断开当前连接...")
                self.disconnect()
                self.logger.log_message("重新连接到指定数据库...")
                self._connection = connections.connect("default", host=self.host, port=self.port, db_name=self.db_name)
                self.logger.log_message(f"成功切换到数据库 '{self.db_name}'")

            # 检查集合是否存在，如果存在则删除
            self.logger.log_message("检查并清理现有集合...")
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                self.logger.log_message(f"已删除现有的集合 {self.collection_name}")
            else:
                self.logger.log_message(f"集合 {self.collection_name} 不存在，无需删除")

            # 创建 Milvus 向量存储
            self.logger.log_message("创建 Milvus 向量存储...")
            vector_store = MilvusVectorStore(
                embedding_function=embeddings,
                connection_args={"host": self.host, "port": self.port},
                collection_name=self.collection_name,
                drop_old=True
            )
            self.logger.log_message(f"成功创建 Milvus 集合 {self.collection_name}")
            return vector_store
        except Exception as e:
            self.logger.log_message(f"创建 Milvus 向量存储失败: {e}", "ERROR")
            raise


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


def main() -> None:
    """主函数"""
    try:
        rag_system = RAGSystem()

        # 初始化系统
        if rag_system.initialize_system():
            # 运行交互式问答
            rag_system.run_interactive_qa()
        else:
            print("系统初始化失败，无法启动问答功能")
    except Exception as e:
        print(f"程序启动失败: {e}")
    finally:
        # 清理资源
        if 'rag_system' in locals():
            rag_system.cleanup()


if __name__ == "__main__":
    main()
