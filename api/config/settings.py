# 配置文件
import os

# 文件路径配置
VECTOR_STORE_PATH = r"C:\Users\86185\Desktop\vector_store"
CHUNKS_CACHE_PATH = r"C:\Users\86185\Desktop\chunks_cache.pkl"
RETRIEVER_CONFIG_PATH = r"C:\Users\86185\Desktop\retriever_config.pkl"
DOCS_DIR = r"C:\Users\86185\Desktop\计设\mingshi"

# API配置
DEEPSEEK_API_KEY = "sk-8ec64ee2711d4df294b31c22e758ff30"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# 模型配置
LLM_MODEL = "deepseek-r1:7b"
MODEL_NAME = "deepseek-r1:7b"
TEMPERATURE = 0.7

# 检索配置
DEFAULT_RETRIEVER_CONFIG = {
    'sparse_k': 6,
    'dense_search_type': "similarity",
    'dense_search_kwargs': {"k": 6, "fetch_k": 12},
    'weights': [0.1, 0.9],
    'top_k': 10
}

# 分块配置
CHUNK_SIZE = 1536
CHUNK_OVERLAP = 50