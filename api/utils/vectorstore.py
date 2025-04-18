import os
import pickle
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_core.documents import Document
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser.text import TokenTextSplitter

from config import (
    VECTOR_STORE_PATH, 
    CHUNKS_CACHE_PATH, 
    RETRIEVER_CONFIG_PATH, 
    DOCS_DIR,
    MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DEFAULT_RETRIEVER_CONFIG
)

def load_or_create_vectorstore():
    """加载或创建向量存储和检索器"""
    # 检查是否所有持久化文件都存在
    all_files_exist = (os.path.exists(VECTOR_STORE_PATH) and 
                      os.path.isdir(VECTOR_STORE_PATH) and 
                      len(os.listdir(VECTOR_STORE_PATH)) > 0 and
                      os.path.exists(CHUNKS_CACHE_PATH) and
                      os.path.exists(RETRIEVER_CONFIG_PATH))

    if all_files_exist:
        print("从磁盘加载向量存储、文档块和检索策略...")
        try:
            # 1. 加载向量存储
            embeddings = OllamaEmbeddings(model=MODEL_NAME)
            dense_vectorstore = FAISS.load_local(
                VECTOR_STORE_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            
            # 2. 加载文档块
            with open(CHUNKS_CACHE_PATH, 'rb') as f:
                chunks = pickle.load(f)
            print(f"成功加载 {len(chunks)} 个文档块")
            
            # 3. 加载检索策略配置
            with open(RETRIEVER_CONFIG_PATH, 'rb') as f:
                retriever_config = pickle.load(f)
            
            # 4. 重建检索器
            sparse_retriever = BM25Retriever.from_documents(
                chunks, 
                k=retriever_config['sparse_k']
            )
            
            dense_retriever = dense_vectorstore.as_retriever(
                search_type=retriever_config['dense_search_type'],
                search_kwargs=retriever_config['dense_search_kwargs'],
            )
            
            base_retriever = EnsembleRetriever(
                retrievers=[sparse_retriever, dense_retriever],
                weights=retriever_config['weights'],
                top_k=retriever_config['top_k']
            )
            
            print("成功加载完整检索策略!")
            return dense_vectorstore, base_retriever
            
        except Exception as e:
            print(f"加载失败: {e}，将重新创建所有组件")
            all_files_exist = False

    if not all_files_exist:
        print("创建新的向量存储和检索策略...")
        
        # 1. 加载并处理文档
        llama_docs = SimpleDirectoryReader(
            input_dir=DOCS_DIR,
            recursive=True,
            required_exts=[".txt"] 
        ).load_data()
        
        splitter = TokenTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separator=" ",
        )
        
        nodes = splitter.get_nodes_from_documents(llama_docs)
        
        chunks = []
        for i, node in enumerate(nodes):
            metadata = node.metadata.copy() if node.metadata else {}
            metadata["chunk_id"] = i
            metadata["source"] = os.path.basename(metadata.get("source", f"chunk_{i}"))
            chunks.append(
                Document(
                    page_content=node.text,
                    metadata=metadata
                )
            )
        
        # 2. 创建向量存储
        embeddings = OllamaEmbeddings(model=MODEL_NAME)
        dense_vectorstore = FAISS.from_documents(chunks, embeddings)
        
        # 3. 定义检索策略配置
        retriever_config = DEFAULT_RETRIEVER_CONFIG
        
        # 4. 创建检索器
        sparse_retriever = BM25Retriever.from_documents(
            chunks, 
            k=retriever_config['sparse_k']
        )
        
        base_retriever = EnsembleRetriever(
            retrievers=[
                sparse_retriever,
                dense_vectorstore.as_retriever(
                    search_type=retriever_config['dense_search_type'],
                    search_kwargs=retriever_config['dense_search_kwargs']
                )
            ],
            weights=retriever_config['weights'],
            top_k=retriever_config['top_k']
        )
        
        # 5. 保存所有组件
        try:
            # 保存向量存储
            os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
            dense_vectorstore.save_local(VECTOR_STORE_PATH)
            
            # 保存文档块
            with open(CHUNKS_CACHE_PATH, 'wb') as f:
                pickle.dump(chunks, f)
            
            # 保存检索配置
            with open(RETRIEVER_CONFIG_PATH, 'wb') as f:
                pickle.dump(retriever_config, f)
                
            print("成功保存所有组件到磁盘!")
        except Exception as e:
            print(f"保存组件时出错: {e}")
            
        return dense_vectorstore, base_retriever