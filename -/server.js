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
const RAG_SERVER_URL = 'http://localhost:5000';
const USE_RAG = true; // 设置为true使用RAG系统，false使用普通API

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
        
        // 使用RAG系统回答问题
        const ragResponse = await axios.post(`${RAG_SERVER_URL}/api/rag/ask`, {
          question: question
        });
        
        return res.json({
          success: true,
          data: ragResponse.data.success ? ragResponse.data.data : `RAG系统回答失败：${ragResponse.data.error}`
        });
        
      } catch (error) {
        console.error('RAG服务器连接失败:', error);
        // 如果RAG服务器不可用，回退到普通API
        console.log('回退到普通API...');
      }
    }

    // 使用普通API回答问题
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
  } catch (error) {
    console.error('API Error:', error);
    res.status(500).json({
      success: false,
      error: '服务器处理请求时发生错误'
    });
  }
});

// 启动服务器
app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
  console.log(`使用${USE_RAG ? 'RAG系统' : '普通API'}回答问题`);
});