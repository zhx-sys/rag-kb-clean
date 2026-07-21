# RAG 私有知识库问答系统

基于 LangChain + ChromaDB + Gradio 的本地 RAG（检索增强生成）系统。上传文档后，AI 基于知识库内容回答你的问题。

## 技术栈

| 层 | 技术 |
|---|---|
| LLM 问答 | 硅基流动 deepseek-ai/DeepSeek-V3（免费） |
| 向量化 | 硅基流动 BAAI/bge-m3（免费） |
| 向量库 | ChromaDB（本地持久化） |
| 框架 | LangChain |
| 界面 | Gradio |
| 文档解析 | PyPDF + docx2txt + TextLoader |

## 项目结构

```
rag-kb/
├── app.py          # Gradio 主入口
├── chain.py        # RAG 问答链（LCEL 管道）
├── config.py       # API Key 与参数配置
├── loader.py       # 文档加载与分割
├── vectorstore.py  # ChromaDB 向量库操作
├── requirements.txt
└── chroma_db/      # 向量库持久化目录（自动生成）
```

## 安装

```bash
pip install -r requirements.txt
```

## 配置

编辑 `config.py`，填入你的硅基流动 API Key（[免费注册](https://siliconflow.cn)）：

```python
LLM_API_KEY = "your-api-key"
LLM_BASE_URL = "https://api.siliconflow.cn/v1"
LLM_MODEL = "deepseek-ai/DeepSeek-V3"

SILICONFLOW_API_KEY = "your-api-key"
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
SILICONFLOW_EMBEDDING_MODEL = "BAAI/bge-m3"
```

## 启动

```bash
python app.py
```

浏览器打开 `http://127.0.0.1:7860`

## 使用流程

1. 左侧上传 PDF / DOCX / TXT / MD 文档，点击「入库」
2. 通过下拉框选择检索范围（全部文档或指定文档）
3. 右侧输入问题，AI 基于知识库内容回答
