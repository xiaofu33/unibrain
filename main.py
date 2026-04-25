from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import api
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from contextlib import asynccontextmanager
from services.cache_service import cache_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动：初始化向量索引结构内容封装完毕。
    await cache_service.init_indices()
    yield
    # 关闭：清理连接内容封装完毕。
    await cache_service.close()
# Setup Huggingface Mirror to prevent download failures in China
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


app = FastAPI(
    title="Intelligent Q&A Microservice",
    description="RAG-based intelligent Q&A module for Institution Management System",
    version="1.0.0",
    lifespan=lifespan
)

# 为前端调用和主 Java 系统配置跨域资源共享 (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载核心 API 路由
app.include_router(api.router, prefix="/api/v1")

# 确保静态文件目录存在
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Service is running"}
