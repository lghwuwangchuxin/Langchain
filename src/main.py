# src/main.py
"""主程序入口"""

from core.rag_system import RAGSystem


def main():
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
