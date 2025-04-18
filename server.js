// 后端代码 server.js
const express = require('express');
const cors = require('cors');
const OpenAI = require('openai');
const axios = require('axios');

const app = express();
const port = 3000;

// 配置中间件
app.use(cors());
app.use(express.json());

const openai = new OpenAI({
    baseURL: 'https://api.deepseek.com/v1',
    apiKey: 'sk-8ec64ee2711d4df294b31c22e758ff30'
});

// 配置RAG服务器地址
const RAG_SERVER_URL = 'http://127.0.0.1:5000';
const USE_RAG = true; // 设置为true使用RAG系统，false使用普通API

// 创建GET端点，支持EventSource连接
app.get('/api/ask', async (req, res) => {
    try {
        const { question } = req.query;

        if (!question) {
            return res.status(400).json({
                success: false,
                error: '请提供问题参数'
            });
        }

        // 处理RAG请求
        if (USE_RAG) {
            handleRagRequest(question, res);
        } else {
            // 对于GET请求，如果不使用RAG，返回错误
            res.status(400).json({
                success: false,
                error: '非RAG模式下不支持GET请求'
            });
        }
    } catch (error) {
        console.error('API Error:', error);
        res.status(500).json({
            success: false,
            error: '服务器处理请求时发生错误'
        });
    }
});

// API端点
app.post('/api/ask', async (req, res) => {
    try {
        const { question } = req.body;

        if (USE_RAG) {
            // 检查RAG系统是否已初始化
            try {
                const statusResponse = await axios.get(`${RAG_SERVER_URL}/api/rag/status`);
                const { initialized, error } = statusResponse.data;

                if (!initialized) {
                    return res.json({
                        success: true,
                        data: `RAG系统正在初始化中，请稍后再试。(${error || '加载知识库...'})`
                    });
                }

                // 处理RAG请求
                handleRagRequest(question, res);
            } catch (error) {
                console.error('RAG服务器连接失败:', error);
                console.log('回退到普通API...');
                handleNormalRequest(question, res);
            }
        } else {
            // 使用普通API
            handleNormalRequest(question, res);
        }
    } catch (error) {
        console.error('API Error:', error);
        res.status(500).json({
            success: false,
            error: '服务器处理请求时发生错误'
        });
    }
});

// 处理RAG请求的函数
async function handleRagRequest(question, res) {
    try {
        // 使用流式API
        const url = new URL(`${RAG_SERVER_URL}/api/rag/ask/stream`);
        url.searchParams.append('question', question);

        const response = await fetch(url, {
            method: 'GET',
            headers: {
                'Accept': 'text/event-stream'
            }
        });

        // 创建流式响应
        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');
        res.setHeader('Access-Control-Allow-Origin', '*'); // 添加CORS支持

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { done, value } = await reader.read();
            if (done) {
                // 发送完成事件
                res.write(`event: done\ndata: {}\n\n`);
                break;
            }

            const chunk = decoder.decode(value);
            // 正确格式化SSE消息
            const jsonData = JSON.stringify({
                success: true,
                data: chunk.trim()
            });

            // 按照SSE标准格式发送
            res.write(`event: message\ndata: ${jsonData}\n\n`);

            // 确保数据立即发送
            if (res.flush) res.flush();
        }

        return res.end();
    } catch (error) {
        console.error('处理RAG请求失败:', error);
        res.write(`event: error\ndata: ${JSON.stringify({ success: false, error: '处理请求失败' })}\n\n`);
        return res.end();
    }
}

// 处理普通API请求的函数
async function handleNormalRequest(question, res) {
    const completion = await openai.chat.completions.create({
        messages: [
            { role: "system", content: "你是一个专业的历史研究助手" },
            { role: "user", content: question }
        ],
        model: "deepseek-chat",
    });

    res.json({
        success: true,
        data: completion.choices[0].message.content
    });
}

// 启动服务器
app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
    console.log(`使用${USE_RAG ? 'RAG系统' : '普通API'}回答问题`);
});