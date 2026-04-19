import sys
from unittest.mock import MagicMock, AsyncMock, patch
import pytest
import os

# 屏蔽 ES
sys.modules["elasticsearch"] = MagicMock()
sys.modules["langchain_elasticsearch"] = MagicMock()

from langchain_core.documents import Document

@pytest.fixture
def mock_qa_service():
    with patch("langchain_openai.ChatOpenAI"), \
         patch("services.document_service.document_service"), \
         patch("services.reranker_service.ZhipuAIReranker"):
        from services.qa_service import QAService
        service = QAService()
        service.llm = AsyncMock()
        return service

@pytest.mark.asyncio
async def test_rewrite_and_expand_query(mock_qa_service):
    # 模拟生成 4 路查询
    mock_response = "北京的加班费怎么算？\n加班费计算标准\n工作日加班补偿\n周末加班费规定"
    mock_qa_service.llm.ainvoke = AsyncMock(return_value=AsyncMock(content=mock_response))
    
    with patch("services.history_service.history_service.get_messages", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = [{"sender": "user", "content": "加班费", "created_at": None}]
        
        queries = await mock_qa_service._rewrite_and_expand_query("怎么算？", session_id="test")
        
        assert len(queries) == 4
        assert queries[0] == "北京的加班费怎么算？"
        assert "加班费计算标准" in queries

def test_deduplicate_docs(mock_qa_service):
    doc1 = Document(page_content="内容A", metadata={"source": "file1.pdf"})
    doc2 = Document(page_content="内容A", metadata={"source": "file1.pdf"}) # 重复
    doc3 = Document(page_content="内容B", metadata={"source": "file2.pdf"})
    
    docs_list = [
        [doc1, doc2],
        [doc1, doc3]
    ]
    
    unique = mock_qa_service._deduplicate_docs(docs_list)
    
    assert len(unique) == 2
    contents = [d.page_content for d in unique]
    assert "内容A" in contents
    assert "内容B" in contents


