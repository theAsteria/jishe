from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def create_self_evaluator(llm):
    """创建自我评估器"""
    eval_template = """评估以下回答的质量，并提供改进建议:
    
    问题: {query}
    回答: {response}
    
    评分(1-10):
    改进建议:"""
    
    eval_prompt = PromptTemplate.from_template(eval_template)
    eval_chain = (
        eval_prompt 
        | llm 
        | StrOutputParser()
    )
    
    return eval_chain

def create_context_compressor(llm, base_retriever):
    """创建上下文压缩器"""
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor
    
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    return compression_retriever