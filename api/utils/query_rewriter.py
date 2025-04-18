from openai import OpenAI
from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

from config.historical_terms import expand_query_with_historical_terms



# 在create_query_rewriter函数中修改rewrite_query函数
def create_query_rewriter():
    
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL
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
            # 首先使用历史术语扩展查询
            expanded_query = expand_query_with_historical_terms(query)
            
            # 构建提示
            prompt = f"""你是一个专门处理明史文献查询的专家。你的任务是将用户的原始查询重写为更有效的向量搜索查询。
            
            原始查询: {expanded_query}
            
            请重写这个查询，使其更适合检索明史文献。你应该:
            1. 提取关键概念和实体（人名、地名、官职、事件）
            2. 保留专有名词和关键术语
            3. 确保查询简洁明了
            4. 只给出一个回答
            5. 保证回答不偏离主题

            重写后的查询:"""
            
            # 调用API
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
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
            return expanded_query  # 如果出错，返回已扩展的查询
    
    return rewrite_query