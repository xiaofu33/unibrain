from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import json
from typing import Optional
from services.document_service import document_service
from services.qa_service import qa_service
from services.history_service import history_service
import shutil
import os
from pydantic import BaseModel

router = APIRouter()

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

@router.post("/documents/upload", tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
        
    upload_dir = "static/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        chunks = await document_service.process_file_upload(file_path, file.filename)
        # 成功后在硬盘保留文件，便于前台引用溯源查看
        return {"message": "Upload successful", "chunks_processed": len(chunks)}
    except Exception as e:
        import traceback
        traceback.print_exc()
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/documents/sync", tags=["Documents"])
async def sync_knowledge_base(source_url: str):
    await document_service.sync_external_knowledge(source_url)
    return {"message": "Sync triggered successfully"}

@router.post("/conversations")
async def create_conversation():
    session_id = await history_service.create_conversation("新对话")
    return {"session_id": session_id}

@router.get("/conversations")
async def list_conversations():
    convs = await history_service.get_conversations()
    return {"conversations": convs}

@router.get("/conversations/{session_id}/messages")
async def get_messages(session_id: str):
    msgs = await history_service.get_messages(session_id)
    return {"messages": msgs}

@router.post("/chat")
async def chat(request: ChatRequest):
    try:
        session_id = request.session_id
        if not session_id:
            session_id = await history_service.create_conversation("新对话")
            
        # 存入用户消息
        await history_service.add_message(session_id, "user", request.question)
        
        async def event_generator():
            full_answer = ""
            # 持续监听并获取流式返回
            async for chunk in qa_service.ask_question_stream(request.question, session_id=session_id):
                full_answer += chunk
                
                # 遵循 Server-Sent Events (SSE) 数据格式抛出数据段
                payload = json.dumps({"delta": chunk, "session_id": session_id}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
                
            # 当所有文本块迭代生成完毕后，触发本地 Postgres 历史保存入库
            await history_service.add_message(session_id, "ai", full_answer)

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents")
async def list_documents():
    """返回当前数据库中已经存入的所有知识库文档列表内容封装完毕。"""
    try:
        docs = await document_service.get_uploaded_documents()
        return {"documents": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/documents/{filename}", tags=["Documents"])
async def delete_document(filename: str):
    """
    [企业运维] 根据文件名从知识库中彻底删除文档及其向量片段内容封装完毕。
    """
    try:
        await document_service.delete_document(filename)
        return {"message": f"Document {filename} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cache/clear", tags=["System"])
async def clear_cache():
    """[企业运维] 手动强制清空所有层级的语义缓存内容封装完毕。"""
    from services.cache_service import cache_service
    await cache_service.clear_all_cache()
    return {"message": "All semantic caches have been cleared."}
