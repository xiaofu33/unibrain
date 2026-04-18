import asyncio
from langchain_core.documents import Document
from services.reranker_service import ZhipuAIReranker

async def test_reranker():
    print("开始测试智谱 Reranker...")
    
    # 模拟一些候选文档
    docs = [
        Document(page_content="企业差旅费报销标准是每天300元。", metadata={"source": "制度A"}),
        Document(page_content="公司团建活动通常安排在周末。", metadata={"source": "制度B"}),
        Document(page_content="员工加班补偿按照国家法定节假日规定执行。", metadata={"source": "制度C"}),
        Document(page_content="报销流程需要主管和财务部双重审核。", metadata={"source": "制度A"}),
    ]
    
    query = "差旅费报销怎么领？"
    
    reranker = ZhipuAIReranker(top_n=2)
    print(f"正在对查询 '{query}' 进行重排...")
    
    try:
        reranked_docs = reranker.compress_documents(docs, query)
        
        print("\n--- 重排结果 ---")
        for i, doc in enumerate(reranked_docs):
            score = doc.metadata.get("rerank_score", "N/A")
            print(f"[{i+1}] Score: {score} | 内容: {doc.page_content[:50]}... [来源: {doc.metadata['source']}]")
            
        if len(reranked_docs) > 0 and "差旅" in reranked_docs[0].page_content:
            print("\n✅ 验证通过：相关文档已排在首位。")
        else:
            print("\n❌ 验证失败：相关文档未排在首位或未返回结果。")
            
    except Exception as e:
        print(f"\n❌ 测试出错: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_reranker())
