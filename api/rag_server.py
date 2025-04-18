from flask import Flask, Response, stream_with_context
from flask_cors import CORS
import threading
import json

from models.llm import init_llm
from models.retriever import load_or_create_retriever
from models import create_enhanced_rag_chain
from routes import register_routes

# 创建Flask应用
app = Flask(__name__)
CORS(app)  # 启用CORS支持跨域请求

global_rag_chain = None
initialization_complete = False
initialization_error = None

def initialize_rag_system():
    """初始化RAG系统"""
    global global_rag_chain, initialization_complete, initialization_error
    
    try:
        print("初始化RAG系统...")
        
        # 加载或创建检索器
        vectorstore, base_retriever = load_or_create_retriever()
        
        # 初始化LLM
        llm = init_llm()
        
        # 创建RAG链（使用非流式版本）
        # 流式处理将在routes.py中实现
        global_rag_chain = create_enhanced_rag_chain(vectorstore, base_retriever, llm)
        
        print("RAG系统初始化完成!")
        initialization_complete = True
        
        # 注册所有路由（包含流式路由）
        register_routes(app, global_rag_chain, initialization_complete, initialization_error)
        
    except Exception as e:
        initialization_error = str(e)
        print(f"初始化RAG系统时出错: {e}")
        initialization_complete = False
        register_routes(app, global_rag_chain, initialization_complete, initialization_error)

# 在后台线程中初始化RAG系统
threading.Thread(target=initialize_rag_system).start()

if __name__ == "__main__":
    # 启动Flask应用
    app.run(host='0.0.0.0', port=5000, debug=False)