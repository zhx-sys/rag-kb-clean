# RAG 知识库配置文件
# 请填写你的 API Key

# DeepSeek API（兼容 OpenAI 格式）
DEEPSEEK_API_KEY = "your-deepseek-api-key"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

# 硅基流动 Embedding API（免费，兼容 OpenAI 格式）
SILICONFLOW_API_KEY = "your-siliconflow-api-key"
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
SILICONFLOW_EMBEDDING_MODEL = "BAAI/bge-m3"

# ChromaDB 持久化路径（相对于项目根目录）
CHROMA_PERSIST_DIR = "E:\\rag-kb\\chroma_db"

# 文本分割参数
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# 检索参数
TOP_K = 4
