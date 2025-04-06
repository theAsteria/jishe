from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama as LangChainOllama  
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from llama_index.core.node_parser.text import TokenTextSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from langchain_core.documents import Document  
from llama_index.core import SimpleDirectoryReader
import os
import re
import pickle
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading

# 创建Flask应用
app = Flask(__name__)
CORS(app)  # 启用CORS支持跨域请求

# 全局变量存储RAG链和初始化状态
global_rag_chain = None
initialization_complete = False
initialization_error = None

def init_llm(temperature=0.7):
    return LangChainOllama(model="deepseek-r1:7b", temperature=temperature)

def create_query_rewriter():
    from openai import OpenAI
    
    client = OpenAI(
        api_key="sk-8ec64ee2711d4df294b31c22e758ff30",  
        base_url="https://api.deepseek.com/v1"  
    )
    
    # 创建查询缓存
    query_cache = {}
    
    def rewrite_query(query):
        # 使用查询哈希作为缓存键
        cache_key = hash(query)
        
        # 检查缓存
        if cache_key in query_cache:
            return query_cache[cache_key]
        
        try:
            # 构建提示
            prompt = f"""你是一个查询重写专家。你的任务是将用户的原始查询重写为更有效的向量搜索查询。
            
            原始查询: {query}
            
            请重写这个查询，使其更适合向量检索系统。你应该:
            1. 提取关键概念和实体
            3. 扩充代表性词汇
            4. 关键词控制在四个以内
            
            重写后的查询:"""
            
            # 调用API
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100
            )
            
            # 提取重写后的查询
            rewritten_query = response.choices[0].message.content.strip()
            
            # 更新缓存
            query_cache[cache_key] = rewritten_query
            if len(query_cache) > 100:  # 限制缓存大小
                query_cache.clear()
                
            return rewritten_query
        except Exception as e:
            print(f"API查询重写错误: {e}")
            return query  # 如果出错，返回原始查询
    
    return rewrite_query

def create_context_compressor(llm, base_retriever):
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    return compression_retriever

def create_reranker():
    from openai import OpenAI
    
    client = OpenAI(
        api_key="sk-8ec64ee2711d4df294b31c22e758ff30",  
        base_url="https://api.deepseek.com/v1"  
    )
    
    # 创建评分缓存
    score_cache = {}
    
    def score_document(doc_content, query):
        # 使用内容哈希和查询哈希作为缓存键
        cache_key = (hash(doc_content[:100]), hash(query))
        
        # 检查缓存
        if cache_key in score_cache:
            return score_cache[cache_key]
        
        try:
            # 构建提示
            prompt = f"""你是一个文档重排序专家。评估以下文档与查询的相关性，并给出0-10的分数。
            
            查询: {query}
            文档: {doc_content}
            
            相关性分数(0-10，只返回数字):"""
            
            # 调用API
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=10
            )
            
            # 提取分数
            score_text = response.choices[0].message.content.strip()
            score_match = re.search(r'(\d+(\.\d+)?)', score_text)
            score = float(score_match.group(1)) if score_match else 5.0
            score = max(0.0, min(10.0, score))  # 确保分数在0-10之间
            
            # 更新缓存
            score_cache[cache_key] = score
            if len(score_cache) > 200:
                score_cache.clear()
                
            return score
        except Exception as e:
            print(f"API评分错误: {e}")
            return 5.0  # 默认值
    
    def rerank_docs(docs, query, max_docs=4):
        if not docs:
            return []
        
        # 限制处理文档数量
        docs_subset = docs[:max_docs]
        
        # 为每个文档评分
        scored_docs = []
        for doc in docs_subset:
            score = score_document(doc.page_content, query)
            scored_docs.append((doc, score))
        
        # 根据分数排序
        sorted_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in sorted_docs]
    
    return rerank_docs

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

def create_enhanced_rag_chain(vectorstore, base_retriever, llm):
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
    1. 如果上下文中提供了信息，请使用上下文中提供的信息
    2. 如果上下文中没有足够信息，请明确说明，但仍要根据已有的信息进行回答
    3. 保持客观、准确，避免添加未在上下文中的信息
    4. 回答应该结构清晰，使用适当的段落和标点
    5. 对于相关内容，需要详细说明
    6. 回答应该简洁明了，使用恰当的词汇
    7. 回答应该符合中文的语法和用词习惯
    8. 回答应该符合中文的文体规范
    9. 要准确回答问题
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
    
    # 创建可流式处理的RAG链
    def process_query(query):
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
        context = docs_to_str(reranked_docs[:5])
        chain_input = {"context": context, "question": query}
        
        prompt_value = prompt.invoke(chain_input)
        
        return prompt_value.to_string()
    
    # 创建完整的RAG链
    rag_chain = RunnableLambda(process_query) | llm | StrOutputParser()
    
    return rag_chain

def initialize_rag_system():
    global global_rag_chain, initialization_complete, initialization_error
    
    try:
        print("初始化RAG系统...")
        
        # 定义持久化文件路径
        vector_store_path = r"C:\Users\86185\Desktop\vector_store"
        chunks_cache_path = r"C:\Users\86185\Desktop\chunks_cache.pkl"
        retriever_config_path = r"C:\Users\86185\Desktop\retriever_config.pkl"
        docs_dir = r"C:\Users\86185\Desktop\计设\mingshi"

        # 检查是否所有持久化文件都存在
        all_files_exist = (os.path.exists(vector_store_path) and 
                          os.path.isdir(vector_store_path) and 
                          len(os.listdir(vector_store_path)) > 0 and
                          os.path.exists(chunks_cache_path) and
                          os.path.exists(retriever_config_path))

        if all_files_exist:
            print("从磁盘加载向量存储、文档块和检索策略...")
            try:
                # 1. 加载向量存储
                embeddings = OllamaEmbeddings(model="deepseek-r1:7b")
                dense_vectorstore = FAISS.load_local(
                    vector_store_path, 
                    embeddings, 
                    allow_dangerous_deserialization=True
                )
                
                # 2. 加载文档块
                with open(chunks_cache_path, 'rb') as f:
                    chunks = pickle.load(f)
                print(f"成功加载 {len(chunks)} 个文档块")
                
                # 3. 加载检索策略配置
                with open(retriever_config_path, 'rb') as f:
                    retriever_config = pickle.load(f)
                
                # 4. 重建检索器
                sparse_retriever = BM25Retriever.from_documents(
                    chunks, 
                    k=retriever_config['sparse_k']
                )
                
                dense_retriever = dense_vectorstore.as_retriever(
                    search_type=retriever_config['dense_search_type'],
                    search_kwargs=retriever_config['dense_search_kwargs']
                )
                
                base_retriever = EnsembleRetriever(
                    retrievers=[sparse_retriever, dense_retriever],
                    weights=retriever_config['weights'],
                    top_k=retriever_config['top_k']
                )
                
                print("成功加载完整检索策略!")
                
            except Exception as e:
                print(f"加载失败: {e}，将重新创建所有组件")
                all_files_exist = False

        if not all_files_exist:
            print("创建新的向量存储和检索策略...")
            
            # 1. 加载并处理文档
            llama_docs = SimpleDirectoryReader(
                input_dir=docs_dir,
                recursive=True,
                required_exts=[".txt"] 
            ).load_data()
            
            splitter = TokenTextSplitter(
                chunk_size=1536,
                chunk_overlap=50,
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
            embeddings = OllamaEmbeddings(model="deepseek-r1:7b")
            dense_vectorstore = FAISS.from_documents(chunks, embeddings)
            
            # 3. 定义检索策略配置
            retriever_config = {
                'sparse_k': 4,
                'dense_search_type': "mmr",
                'dense_search_kwargs': {"k": 4, "fetch_k": 10},
                'weights': [0.5, 0.5],
                'top_k': 8
            }
            
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
                os.makedirs(vector_store_path, exist_ok=True)
                dense_vectorstore.save_local(vector_store_path)
                
                # 保存文档块
                with open(chunks_cache_path, 'wb') as f:
                    pickle.dump(chunks, f)
                
                # 保存检索配置
                with open(retriever_config_path, 'wb') as f:
                    pickle.dump(retriever_config, f)
                    
                print("成功保存所有组件到磁盘!")
            except Exception as e:
                print(f"保存组件时出错: {e}")

        # 初始化LLM和RAG链
        llm = init_llm(temperature=0.7)
        global_rag_chain = create_enhanced_rag_chain(dense_vectorstore, base_retriever, llm)
        
        print("RAG系统初始化完成!")
        initialization_complete = True
        
    except Exception as e:
        initialization_error = str(e)
        print(f"初始化RAG系统时出错: {e}")
        initialization_complete = False

# 在后台线程中初始化RAG系统
threading.Thread(target=initialize_rag_system).start()

@app.route('/api/rag/status', methods=['GET'])
def get_status():
    """获取RAG系统初始化状态"""
    return jsonify({
        'initialized': initialization_complete,
        'error': initialization_error
    })

@app.route('/api/rag/ask', methods=['POST'])
def ask_rag():
    """使用RAG系统回答问题"""
    if not initialization_complete:
        return jsonify({
            'success': False,
            'error': '系统正在初始化，请稍后再试',
            'initialized': False
        })
    
    try:
        data = request.json
        question = data.get('question', '')
        
        if not question:
            return jsonify({
                'success': False,
                'error': '问题不能为空'
            })
        
        # 使用RAG链处理问题
        answer = global_rag_chain.invoke(question)
        
        # 截断<think></think>标签及其内容
        import re
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
        
        return jsonify({
            'success': True,
            'data': answer
        })
        
    except Exception as e:
        print(f"处理问题时出错: {e}")
        return jsonify({
            'success': False,
            'error': f'处理问题时出错: {str(e)}'
        })

if __name__ == "__main__":
    # 启动Flask应用
    app.run(host='0.0.0.0', port=5000, debug=False)