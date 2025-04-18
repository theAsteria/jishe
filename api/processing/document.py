from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

def create_context_compressor(llm, base_retriever):
    """创建上下文压缩器"""
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    return compression_retriever

def docs_to_str(docs):
    """将文档转换为字符串格式"""
    if not docs:
        return "没有找到相关信息。"
    
    formatted_docs = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "未知来源")
        formatted_docs.append(f"[文档 {i+1}] 来源: {source}\n{doc.page_content}")
    
    return "\n\n".join(formatted_docs)