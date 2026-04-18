import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class Config:
    # PostgreSQL 配置（仅用于 SQLAlchemy 存储会话历史，不做向量搜索）
    POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "unibrain_db")

    # SQLAlchemy 异步连接字符串（会话历史使用）
    DATABASE_URL = f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

    # Elasticsearch 配置（用于向量知识库存储）
    ES_URL = os.getenv("ES_URL", "http://localhost:9200")
    ES_INDEX_NAME = os.getenv("ES_INDEX_NAME", "institution_docs")
    ES_USERNAME = os.getenv("ES_USERNAME", "")  # 如无认证可留空
    ES_PASSWORD = os.getenv("ES_PASSWORD", "")  # 如无认证可留空

    # LLM Settings (现已配置为智谱 AI)
    # 智谱提供兼容 OpenAI 格式的 API，所以可直接复用现有 OpenAI 代码架构
    LLM_API_BASE = os.getenv("LLM_API_BASE", "https://open.bigmodel.cn/api/paas/v4/")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "702beb3d010b4d30949c19a4f14ef375.7NLh9GlER4EM6Iw9") 
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "glm-4.7")
    GLM_OCR_MODEL_NAME = os.getenv("GLM_OCR_MODEL_NAME", "glm-ocr")

    # Embeddings Settings 
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "embedding-3")
    
    # Reranker Settings
    RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "rerank")
    RAG_RETRIEVAL_K = int(os.getenv("RAG_RETRIEVAL_K", "15"))
    RAG_RERANK_TOP_N = int(os.getenv("RAG_RERANK_TOP_N", "5"))
    
settings = Config()
