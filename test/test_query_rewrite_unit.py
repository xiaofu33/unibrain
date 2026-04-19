import sys
from unittest.mock import MagicMock

# 彻底拦截 elasticsearch 模块防止连接尝试
mock_es = MagicMock()
sys.modules["elasticsearch"] = mock_es
sys.modules["langchain_elasticsearch"] = MagicMock()

import pytest
from unittest.mock import AsyncMock, patch
import os

# 设置环境变量
os.environ["LLM_API_KEY"] = "mock-key"
os.environ["ES_URL"] = "http://mock-es:9200"

@pytest.fixture
def mock_qa_deps():
    # 在导入 QAService 之前 Mock 掉 langchain_openai
    with patch("langchain_openai.ChatOpenAI"), \
         patch("services.reranker_service.ZhipuAIReranker"):
        from services.qa_service import QAService
        from config import settings
        
        qa_service = QAService()
        qa_service.llm = AsyncMock()
        yield qa_service, settings

@pytest.mark.asyncio
async def test_rewrite_query_logic(mock_qa_deps):
    qa_service, _ = mock_qa_deps
    
    # 模拟有历史记录的情况
    mock_msgs = [
        {"sender": "user", "content": "公司关于交通费报销有什么规定？", "created_at": None},
        {"sender": "ai", "content": "公司交通费报销规定如下...", "created_at": None}
    ]
    
    session_id = "test-session-123"
    
    with patch("services.history_service.history_service.get_messages", new_callable=AsyncMock) as mock_get_msgs:
        mock_get_msgs.return_value = mock_msgs
        
        # 模拟 LLM 返回重写后的结果
        rewritten_val = "公司关于交通费报销的标准是多少？"
        qa_service.llm.ainvoke = AsyncMock(return_value=AsyncMock(content=rewritten_val))
        
        question = "它的标准是多少？"
        result = await qa_service._rewrite_query(question, session_id=session_id)
        
        assert result == rewritten_val
        
        # 验证提示词内容
        call_args = qa_service.llm.ainvoke.call_args[0][0]
        assert "交通费报销" in call_args
        assert "补全专家" in call_args

@pytest.mark.asyncio
async def test_rewrite_query_no_history(mock_qa_deps):
    qa_service, _ = mock_qa_deps
    
    with patch("services.history_service.history_service.get_messages", new_callable=AsyncMock) as mock_get_msgs:
        mock_get_msgs.return_value = []
        
        question = "你好"
        result = await qa_service._rewrite_query(question, session_id="test-id")
        assert result == question
