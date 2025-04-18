from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from typing import Iterator, Dict, Any, AsyncIterator

from utils.query_rewriter import create_query_rewriter
from utils.reranker import create_reranker
from utils.evaluator import create_self_evaluator

def create_enhanced_rag_chain(vectorstore, base_retriever, llm):
    """创建增强的RAG链（支持流式输出）"""
    # 初始化各组件
    query_rewriter = create_query_rewriter()
    reranker = create_reranker()
    self_evaluator = create_self_evaluator(llm)
    
    # 增强的提示模板
    template = """你是一个专业的历史学者和教育者。基于以下上下文信息回答用户的问题。
    
    上下文信息:
    {context}
    
    用户问题: {question}
    
    回答要求:
    1. 使用上下文中的信息并结合自己所知内容综合分析，进行回答
    2. 回答应该结构清晰，使用适当的段落和标点
    3. 对于相关内容，需要详细说明
    4. 回答应该符合中文的语法和用词习惯
    5. 回答应该符合中文的文体规范
    回答:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # 文档处理
    def docs_to_str(docs):
        if not docs:
            return "没有找到相关信息。"
        
        formatted_docs = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "未知来源")
            formatted_docs.append(f"[文档 {i+1}] 来源: {source}\n{doc.page_content}")
        
        return "\n\n".join(formatted_docs)
    
    # 创建完整的RAG链（支持流式）
    def process_query(query: str) -> Dict[str, Any]:
        print(f"原始查询: {query}")
        
        # 1. 查询重写
        rewritten_query = query_rewriter(query)
        print(f"重写后的查询: {rewritten_query}")
        
        # 2. 检索文档
        raw_docs = base_retriever.get_relevant_documents(rewritten_query)
        print(f"检索到 {len(raw_docs)} 个文档")
        
        # 3. 重排序
        reranked_docs = reranker(raw_docs, query)
        print(f"重排序完成")
        
        # 4. 生成回答的上下文
        context = docs_to_str(reranked_docs[:8])
        return {"context": context, "question": query}
    
    # 创建流式处理链
    rag_chain = (
        RunnableLambda(process_query)
        | prompt
        | llm.with_config({"stream": True})  # 确保LLM启用流式
        | StrOutputParser()
    )
    
    return rag_chain


