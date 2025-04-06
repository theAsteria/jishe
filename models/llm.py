from langchain_community.llms import Ollama as LangChainOllama
from config.settings import LLM_MODEL, LLM_TEMPERATURE

def init_llm(temperature=LLM_TEMPERATURE):
    """初始化LLM模型"""
    return LangChainOllama(model=LLM_MODEL, temperature=temperature)