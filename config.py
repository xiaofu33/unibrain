import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class Config:
    # 从环境中读取配置设置
    POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "unibrain_db")
    
    # SQLAlchemy & pgvector 的数据库连接字符串配置
    DATABASE_URL = f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

    # LLM Settings (现已配置为智谱 AI)
    # 智谱提供兼容 OpenAI 格式的 API，所以可直接复用现有 OpenAI 代码架构
    LLM_API_BASE = os.getenv("LLM_API_BASE", "https://open.bigmodel.cn/api/paas/v4/")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "702beb3d010b4d30949c19a4f14ef375.7NLh9GlER4EM6Iw9") 
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "glm-4.7")

    # Embeddings Settings 
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "embedding-3")
    
settings = Config()
