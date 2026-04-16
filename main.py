from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import api
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
# Setup Huggingface Mirror to prevent download failures in China
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


app = FastAPI(
    title="Intelligent Q&A Microservice",
    description="RAG-based intelligent Q&A module for Institution Management System",
    version="1.0.0",
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
