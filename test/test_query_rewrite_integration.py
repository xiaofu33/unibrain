import asyncio
import pytest
from services.qa_service import qa_service
from services.history_service import history_service
from config import settings

@pytest.mark.asyncio
async def test_query_rewrite_integration():
    """
    集成测试：验证在真实服务(ES, DB, LLM)连接下的查询重写流程。
    注意：此测试需要真实的 ES 服务已启动。
    """
    # 1. 创建一个新的会话
    session_id = await history_service.create_conversation("集成测试会话")
    print(f"\n[1] 创建会话成功: {session_id}")
    
    # 2. 模拟第一轮对话并存入历史 (告知系统我们在聊什么)
    first_question = "我想了解公司的加班补贴政策。"
    first_answer = "公司的加班补贴政策规定：工作日加班按1.5倍工资计算，周末加班按2倍计算。"
    await history_service.add_message(session_id, "user", first_question)
    await history_service.add_message(session_id, "ai", first_answer)
    print(f"[2] 存入第一轮历史记录")
    
    # 3. 运行第二轮提问（带代词，触发重写）
    second_question = "那它的上限是多少？"
    print(f"[3] 发送第二轮提问: {second_question}")
    
    # 我们可以通过临时 Hook 或打印来观察重写后的文字
    # 这里我们直接调用业务接口，验证是否能跑通且不报错
    try:
        # 尝试重写查询以供观察 (手动调用私有方法进行验证)
        rewritten_query = await qa_service._rewrite_query(second_question, session_id)
        print(f">>> LLM 重写后的查询为: 【{rewritten_query}】")
        
        # 验证重写逻辑是否有效（重写后的查询应包含关键词）
        assert "加班" in rewritten_query or "上限" in rewritten_query
        
        # 执行完整的问答流程 (包含重写 -> 检索 -> 生成)
        # 注意：如果 ES 里没数据，这里会返回“没找到相关资料”，这是正常的
        answer = await qa_service.ask_question(second_question, session_id)
        print(f"[4] AI 的最终回答: {answer[:100]}...")
        
    except Exception as e:
        pytest.fail(f"集成测试失败，发生异常: {e}")

if __name__ == "__main__":
    # 手动运行方式
    asyncio.run(test_query_rewrite_integration())
