# Unibrain: 制度管理系统 - 智能问答微服务

![Python Version](https://img.shields.io/badge/python-3.13+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135+-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-1.2+-yellow.svg)

> 一个基于 RAG (检索增强生成) 架构的智能问答微服务，通过 FastAPI 向上游业务系统提供纯 RESTful API 接口，快速实现企业制度文档的解析、向量化存储和智能问答能力。

## 🌟 核心特性

- **文档智能处理**: 支持 PDF 等格式制度文档的上传、解析、文本切分与向量化入库。
- **RAG 智能问答**: 基于 LangChain 结合大语言模型，提供依据本地制度文档的精准问答。
- **平滑系统对接**: 提供标准 REST API，对原 Java 或业务微服务系统零侵入，轻松实现知识库能力外包。
- **多端同步能力**: 支持通过接口触发机制，从现有系统 URL 拉取并同步知识文本。
- **极速依赖管理**: 基于现代化工具 `uv` 提供高效的包及虚拟环境管理。

## 🛠️ 技术栈

- **Web 框架**: FastAPI + Uvicorn
- **大模型核心**: LangChain 生态 (LangChain-OpenAI 等)
- **向量存储**: PostgreSQL (pgvector) / Elasticsearch
- **环境管理**: uv

## 🚀 快速开始

### 1. 环境准备
- 确保系统已安装 Python >= 3.13。
- 配置好相关的数据库实例（PostgreSQL + `pgvector` 插件，或 Elasticsearch 实例）。
- 可用的本地或局域网 LLM 接口服务（如 Ollama，或兼容 OpenAI API 格式的在线大模型接口）。
- 建议全局安装新一代 Python 环境工具 [uv](https://github.com/astral-sh/uv)。

### 2. 安装与运行

```bash
# 1. 克隆或进入项目代码目录
cd unibrain

# 2. 同步项目依赖 (如果不使用 uv，也可以使用 pip install -r requirements.txt)
uv sync

# 3. 环境变量配置 (请参考配置模板并填写您的密钥/数据库链接)
cp .env.example .env

# 4. 启动后端服务 (服务默认挂载在 http://0.0.0.0:8000)
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 📚 API 接口指南

以下接口主要供 Java 端或其他业务端调用。完整、可交互的接口定义请在启动服务后访问 Swagger UI 面板：[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)  或访问 `/redoc` 离线风格文档。

### A. 文档上传与解析入库
**Endpoint:** `POST /api/v1/documents/upload`  
**Content-Type:** `multipart/form-data`  

**请求示例 (Java Spring RestTemplate):**
```java
HttpHeaders headers = new HttpHeaders();
headers.setContentType(MediaType.MULTIPART_FORM_DATA);

MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
body.add("file", new FileSystemResource(new File("/path/to/制度文档.pdf")));

HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);
RestTemplate restTemplate = new RestTemplate();
ResponseEntity<String> response = restTemplate.postForEntity(
    "http://127.0.0.1:8000/api/v1/documents/upload", 
    requestEntity, 
    String.class
);
```

### B. 与现有知识库同步
**Endpoint:** `POST /api/v1/documents/sync?source_url={url}`  
用于触发定时任务或其他 Webhook，将企业系统中已有的文本数据拉取到并同步入向量数据库中。

### C. 智能问答 (RAG)
**Endpoint:** `POST /api/v1/qa/ask`  
**Content-Type:** `application/json`

**请求参数 Body:**
```json
{
  "question": "员工请假应该怎么计算考勤？",
  "session_id": "user-123-session" 
}
```

**响应结果示例:**
```json
{
  "answer": "根据《考勤管理制度》第 x 条规定，员工请假期间的考勤……",
  "source_documents": ["《考勤管理规范.pdf》"]
}
```

## 🏗️ 架构设计简述

1. **统一模型交互**: 大量采用 `langchain` 包进行底层代理操作，包含 `langchain_postgres` / `langchain_elasticsearch` 等组件，无缝衔接 Embedding 与文本搜索。
2. **松耦合架构**: 避免对原本 Java 业务进行大规模重构，而是让此 Python 微服务仅且专门负责计算张量以及自然语言处理（NLP）工作。
3. **扩展性预留**: `services/document_service.py` 或同级的处理器预留了接口封装层。当业务需要解析更复杂的富文本、接入 OCR（如识别图片版 PDF）或者读取 Confluence API 时，可平稳进行代码扩展。
