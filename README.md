# 制度管理系统 - 智能问答微服务 (Python)

该项目是“制度管理系统”的智能问答微服务，提供基于文档向量化的 RAG (检索增强生成) 能力。通过 FastAPI 向上游 Java 系统提供纯 RESTful API 接口。

## 1. 快速启动指南

### 环境准备
1. 确保已安装 Python 3.9+。
2. 配置 PostgreSQL 数据库，并安装 `pgvector` 插件 (`CREATE EXTENSION vector;`)。
3. 请配置本地或局域网的 LLM 接口服务（如 Ollama / vLLM 兼容 OpenAI API 格式）。

### 安装与运行
```bash
# 1. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量 (可选，不配置则使用默认值)
cp .env.example .env

# 4. 启动服务 (默认将在 http://127.0.0.1:8000 开放)
uvicorn main:app --reload
```

## 2. API 接口指引 (供 Java 端调用)

### A. 文档上传与解析入库
**Endpoint:** `POST /api/v1/documents/upload`  
**Content-Type:** `multipart/form-data`  

**Java 端调用示例 (使用 Spring RestTemplate):**
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
用于触发定时任务或其他钩子，将 Java 系统中已有的文本同步到向量数据库。

### C. 智能问答 (RAG)
**Endpoint:** `POST /api/v1/qa/ask`  
**Content-Type:** `application/json`

**请求 Body:**
```json
{
  "question": "员工请假应该怎么计算考勤？",
  "session_id": "user-123-session" 
}
```

**Java 端调用示例:**
```java
HttpHeaders headers = new HttpHeaders();
headers.setContentType(MediaType.APPLICATION_JSON);

String requestJson = "{\"question\": \"员工请假应该怎么计算考勤？\"}";
HttpEntity<String> entity = new HttpEntity<>(requestJson, headers);

RestTemplate restTemplate = new RestTemplate();
ResponseEntity<String> response = restTemplate.postForEntity(
    "http://127.0.0.1:8000/api/v1/qa/ask", 
    entity, 
    String.class
);

System.out.println(response.getBody()); 
// 返回结果中包含基于制度生成的答案 
// {"answer": "根据《考勤管理制度》第X条规定..."}
```

## 3. 架构设计说明
1. **Langchain 集成**: 采用了 `langchain_postgres` 模块作为存储引擎，无缝衔接模型 Embedding 和搜索。
2. **轻量与隔离**: 避免对原本 Java 系统做重构，仅把涉及到张量计算与自然语言处理的环节托管给此 Python 系统。
3. **扩展性**: `services/document_service.py` 中的 `process_file_upload` 可拓展以支持 OCR 引擎，或解析更多复杂的 Markdown, Confluence API 数据等。
