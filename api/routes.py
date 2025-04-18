import re
import json  # 添加这行导入
from flask import request, jsonify, Response, stream_with_context
from flask import Flask, request, Response, stream_with_context
import json

def register_routes(app, global_rag_chain, initialization_complete, initialization_error):
    """注册所有API路由"""
    
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

    @app.route('/api/rag/ask/stream', methods=['GET'])
    def ask_rag_stream():
        question = request.args.get('question')
        if not question:
            return "问题参数缺失", 400

        def generate():
            buffer = ""  # 缓冲区，累积接收到的文本
            response = global_rag_chain.invoke(question, {"response_mode": "streaming"})

            for chunk in response:
                buffer += chunk  # 将新片段追加到缓冲区

                while True:
                    # 查找第一个出现的 </think> 标签的位置
                    pos = buffer.find('</think>')
                    if pos == -1:
                        break  # 未找到标签，继续累积

                    remaining = buffer[pos + len('</think>'):]
                    # 更新缓冲区为剩余部分，继续处理后续内容
                    buffer = remaining
            
            # 处理剩余未发送的缓冲区内容（如果有的话）
            if buffer:
                # 修改：不再遍历buffer中的每个字符，而是作为整体发送
                print(f"发送数据块: {buffer}")  # 添加日志记录
                # 确保使用ensure_ascii=False以正确处理UTF-8字符
                yield f"data: {json.dumps({'data': buffer}, ensure_ascii=False)}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )