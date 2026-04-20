import pytest
import asyncio
from services.document_service import document_service
from langchain_core.documents import Document
from config import settings

@pytest.mark.asyncio
async def test_hybrid_search_functionality():
    """
    测试混合检索功能：验证关键词匹配是否在某些场景下优于或补充纯向量检索
    """
    # 确保混合检索已开启
    assert settings.ENABLE_HYBRID_SEARCH is True
    
    test_filename = "hybrid_test_doc.txt"
    unique_term = "UniBrain-Super-Special-X100"
    content = f"这是一个关于 {unique_term} 的专用文档。该产品在 2026 年发布。"
    
    # 模拟上传文档
    # 为了测试，我们直接注入一个文档
    test_doc = Document(page_content=content, metadata={"source": test_filename, "type": "upload"})
    document_service.vector_store.add_documents([test_doc])
    
    # 给 ES 一点索引时间
    await asyncio.sleep(2)
    
    try:
        # 使用唯一关键词进行检索
        retriever = document_service.vector_store.as_retriever(search_kwargs={"k": 5})
        results = await retriever.ainvoke(unique_term)
        
        # 验证是否召回了该文档
        found = any(unique_term in doc.page_content for doc in results)
        assert found, f"混合检索未能根据唯一关键词 '{unique_term}' 召回文档"
        print(f"\n[成功] 混合检索召回了包含唯一关键词的文档: {len(results)} 个结果")
        
    finally:
        # 清理测试数据 (ES 原生清理比较麻烦，这里暂时跳过或使用特定索引)
        pass

if __name__ == "__main__":
    asyncio.run(test_hybrid_search_functionality())
