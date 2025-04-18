from langchain_ollama import OllamaLLM
from config.settings import MODEL_NAME, TEMPERATURE

def init_llm(temperature=TEMPERATURE):
    """初始化LLM模型"""
    return OllamaLLM(model=MODEL_NAME, temperature=temperature,streaming=True)