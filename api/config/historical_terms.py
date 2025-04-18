# 明史特有术语同义词表

# 官职同义词
OFFICIAL_TITLES = {
    "丞相": ["宰相", "相国", "辅臣"],
    "太师": ["三公", "太傅", "太保"],
    "尚书": ["九卿", "六部尚书"],
    "侍郎": ["次官", "副官"],
    "御史": ["言官", "监察官"],
    "巡抚": ["抚台", "封疆大吏"],
    "总督": ["督抚", "封疆大吏"],
    # 可根据需要扩展
}

# 地名同义词
PLACE_NAMES = {
    "京师": ["北京", "顺天府", "燕京"],
    "南京": ["应天府", "建康"],
    "杭州": ["临安", "钱塘"],
    "苏州": ["姑苏", "吴县"],
    # 可根据需要扩展
}

# 朝代/年号同义词
PERIOD_NAMES = {
    "洪武": ["明太祖", "朱元璋时期"],
    "永乐": ["明成祖", "朱棣时期"],
    "嘉靖": ["明世宗", "朱厚熜时期"],
    "万历": ["明神宗", "朱翊钧时期"],
    # 可根据需要扩展
}

# 事件同义词
EVENT_NAMES = {
    "靖难之役": ["靖难", "朱棣夺位"],
    "土木之变": ["土木堡之变", "也先之变"],
    "大礼议": ["大礼之争", "嘉靖改制"],
    # 可根据需要扩展
}

def get_synonyms(term):
    """获取历史术语的同义词"""
    for category in [OFFICIAL_TITLES, PLACE_NAMES, PERIOD_NAMES, EVENT_NAMES]:
        if term in category:
            return category[term]
    return []

def expand_query_with_historical_terms(query):
    """使用历史术语扩展查询"""
    expanded_terms = []
    
    # 简单分词（实际应用中可使用更复杂的分词）
    words = query.split()
    
    for word in words:
        synonyms = get_synonyms(word)
        if synonyms:
            expanded_terms.extend(synonyms)
    
    # 如果找到同义词，将它们添加到原始查询中
    if expanded_terms:
        expanded_query = query + " " + " ".join(expanded_terms)
        return expanded_query
    
    return query