from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama as LangChainOllama  
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from llama_index.core.node_parser.text import (
     SemanticSplitterNodeParser,
     SentenceSplitter,
     TokenTextSplitter,
)
from llama_index.llms.ollama import Ollama as LlamaIndexOllama  
from llama_index.embeddings.ollama import OllamaEmbedding
from langchain_core.documents import Document  
from llama_index.core import SimpleDirectoryReader
import os
import re

# 增强的向量数据库初始化 
def init_vectorstore(docs_dir=r"C:\Users\86185\Desktop\计设\mingshi", file_pattern="*.txt"):

    llama_docs = SimpleDirectoryReader(
        input_dir=docs_dir,
        recursive=True,
        required_exts=[".txt"] 
    ).load_data()
    
    print(f"加载了 {len(llama_docs)} 个文档")
    
    
#    embed_model = OllamaEmbedding(model_name="deepseek-r1:7b",base_url="http://localhost:11434")
        
    # splitter = SemanticSplitterNodeParser(
    #     embed_model=embed_model,  
    #     buffer_size=3, 
    #     breakpoint_percentile_threshold=95,
    #     paragraph_separator="\n",
    # )   
    splitter = TokenTextSplitter(
        chunk_size=1024,
        chunk_overlap=20,
        separator=" ",
    )
    # 进行语义分割
    nodes = splitter.get_nodes_from_documents(llama_docs)
    print(f"创建了 {len(nodes)} 个语义文本块")
    
    # 将LlamaIndex节点转换回LangChain文档
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
    
    # 创建嵌入模型
    embeddings = OllamaEmbeddings(model="deepseek-r1:7b")
    
    
    # 创建向量存储（密集检索）
    dense_vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # 创建BM25检索器（稀疏检索）
    from langchain_community.retrievers import BM25Retriever
    sparse_retriever = BM25Retriever.from_documents(chunks, k=15)
    
    # 创建混合检索器
    from langchain.retrievers.ensemble import EnsembleRetriever
    
    # 基础检索器 - 混合稀疏检索和密集检索
    base_retriever = EnsembleRetriever(
        retrievers=[sparse_retriever, dense_vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8, "fetch_k": 12}
        )],
        weights=[0.5, 0.5],  # 稀疏检索权重0.5，密集检索权重0.5
        top_k=10
    )
    
    return dense_vectorstore, base_retriever

def init_llm(temperature=0.5):
    return LangChainOllama(model="deepseek-r1:7b", temperature=temperature)  # 使用LangChain的Ollama

# 2. 查询重写器 - 将用户查询扩展为更有效的搜索查询
def create_query_rewriter(llm):
    query_rewrite_template = """你是一个查询重写专家。你的任务是将用户的原始查询重写为更有效的向量搜索查询。
    
    原始查询: {query}
    
    请重写这个查询，使其更适合向量检索系统。你应该:
    1. 提取关键概念和实体
    2. 添加同义词或相关术语
    3. 移除不必要的词语
    4. 确保查询简洁明了
    
    重写后的查询:"""
    
    query_rewrite_prompt = PromptTemplate.from_template(query_rewrite_template)
    query_rewriter_chain = (
        query_rewrite_prompt 
        | llm 
        | StrOutputParser()
    )
    
    return query_rewriter_chain

# 3. 上下文压缩器 
def create_context_compressor(llm, base_retriever):
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    return compression_retriever

# 4. 结果重排序器 
def create_reranker(llm):
    rerank_template = """你是一个文档重排序专家。评估以下文档与查询的相关性，并给出0-10的分数。
    
    查询: {query}
    
    文档: {document}
    
    相关性分数(0-10，只返回数字):"""
    
    rerank_prompt = PromptTemplate.from_template(rerank_template)
    rerank_chain = (
        rerank_prompt 
        | llm 
        | StrOutputParser()
    )
    
    def rerank_docs(docs, query):
        if not docs:
            return []
        
        # 为每个文档评分
        scores = []
        for doc in docs:
            try:
                score_text = rerank_chain.invoke({"query": query, "document": doc.page_content})
                # 提取数字
                score_match = re.search(r'(\d+(\.\d+)?)', score_text)
                if score_match:
                    score = float(score_match.group(1))
                else:
                    score = 5.0  # 默认中等相关性
            except Exception as e:
                print(f"重排序错误: {e}")
                score = 5.0
            scores.append(score)
        
        # 根据分数重新排序
        sorted_pairs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        sorted_docs = [doc for doc, _ in sorted_pairs]
        
        return sorted_docs
    
    return rerank_docs

# 5. 自我评估和反馈循环
def create_self_evaluator(llm):
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

#RAG链
def create_enhanced_rag_chain(vectorstore, base_retriever, llm):
    # 初始化各组件
    query_rewriter = create_query_rewriter(llm)
    # 上下文压缩器
    #compression_retriever = create_context_compressor(llm, base_retriever)
    reranker = create_reranker(llm)
    self_evaluator = create_self_evaluator(llm)
    
    # 增强的提示模板
    template = """你是一个专业的历史学者和教育者。基于以下上下文信息回答用户的问题。
    
    上下文信息:
    {context}
    
    用户问题: {question}
    
    回答要求:
    1. 如果上下文中提供了信息，请使用上下文中提供的信息
    2. 如果上下文中没有足够信息，请明确说明，但仍要根据已有的信息进行回答
    3. 保持客观、准确，避免添加未在上下文中的信息
    4. 回答应该结构清晰，使用适当的段落和标点
    5. 对于相关内容，需要详细说明
    6. 回答应该简洁明了，使用恰当的词汇
    7. 回答应该符合中文的语法和用词习惯
    8. 回答应该符合中文的文体规范
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
    
    # RAP pipeline
    def rag_pipeline(query: str):
        print(f"原始查询: {query}")
        
        # 1. 查询重写
        rewritten_query = query_rewriter.invoke({"query": query})
        print(f"重写后的查询: {rewritten_query}")
        
        # 2. 检索文档
        raw_docs = base_retriever.get_relevant_documents(rewritten_query)
        print(f"检索到 {len(raw_docs)} 个文档")
        
        # 3. 上下文压缩
        # compressed_docs = compression_retriever.get_relevant_documents(rewritten_query)
        # print(f"压缩后剩余 {len(compressed_docs)} 个文档")
        
        # 4. 重排序
        # reranked_docs = reranker(raw_docs, query)
        # print(f"重排序完成")
        reranked_docs = raw_docs 
        # 5. 生成回答
        context = docs_to_str(reranked_docs[:5])  # 只使用前5个最相关文档
        chain_input = {"context": context, "question": query}
        
        prompt_value = prompt.invoke(chain_input)
        answer = llm.invoke(prompt_value.to_string()).strip()
        
        # 6. 自我评估
        evaluation = self_evaluator.invoke({"query": query, "response": answer})
        print(f"自我评估: {evaluation}")
        
        # 7. 评分低，尝试改进
        if "评分: " in evaluation and int(evaluation.split("评分: ")[1][0]) < 6:
            print("尝试改进回答...")
            improved_prompt = f"""基于以下评估改进你的回答:
            
            原问题: {query}
            原回答: {answer}
            评估: {evaluation}
            
            改进后的回答:"""
            
            improved_answer = llm.invoke(improved_prompt)
            return improved_answer
        
        return answer
    
    return RunnableLambda(rag_pipeline)

def main():
    print("初始化RAG系统...")
    vectorstore, base_retriever = init_vectorstore()
    llm = init_llm()
    enhanced_rag_chain = create_enhanced_rag_chain(vectorstore, base_retriever, llm)
    
    print("RAG系统初始化完成!")
    
    query = "详细讲讲朱元璋的功绩"
    print("\n处理查询中...")
    
    response = enhanced_rag_chain.invoke(query)
    
    print("\n最终回答:")
    print(response)

if __name__ == "__main__":
    main()