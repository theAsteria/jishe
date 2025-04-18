import re
from openai import OpenAI
from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

def create_reranker():
    """创建文档重排序器"""
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL
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