import re
from flask import request, jsonify

def register_routes(app, global_rag_chain, initialization_complete, initialization_error):
    """注册API路由"""
    
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