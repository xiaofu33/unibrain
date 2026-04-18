from config import settings
from langchain_openai import ChatOpenAI
from services.document_service import document_service
from langchain_classic.retrievers import ContextualCompressionRetriever
from services.reranker_service import ZhipuAIReranker

class QAService:
    def __init__(self):
        # 初始化并连接到大语言模型 (API格式兼容的开源端点)
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL_NAME,
            api_key=settings.LLM_API_KEY, # type: ignore
            base_url=settings.LLM_API_BASE,
            streaming=True
        )
        
    async def ask_question(self, question: str, session_id: str = None) -> str:
        """
        通过 RAG 流程处理问答:
        1. 问题文本向量化
        2. 从本地 PostgreSQL (pgvector) 等向量数据库检索出相关制度原文片段
        3. 将检索到的上下文及用户问题共同构成 Prompt 提示词
        4. 请求 LLM 进行思考及回复
        """
        # 从 PostgreSQL pgvector 检索符合语义的关联知识库块
        # 升级为双阶段检索逻辑: 
        # 1. 第一阶段召回 (k=15)
        base_retriever = document_service.vector_store.as_retriever(search_kwargs={"k": settings.RAG_RETRIEVAL_K})
        # 2. 第二阶段重排 (top_n=5)
        compressor = ZhipuAIReranker(top_n=settings.RAG_RERANK_TOP_N)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=base_retriever
        )
        docs = compression_retriever.invoke(question)
        
        if not docs:
            context = "没有找到与您问题相关的制度资料。"
        else:
            context = "\n\n".join([f"[来源: {doc.metadata.get('source', '未知')}]\n{doc.page_content}" for doc in docs])
        
        # 加载历史对话上下文 (如果存在)
        chat_history_str = ""
        if session_id:
            from services.history_service import history_service
            msgs = await history_service.get_messages(session_id)
            if msgs:
                chat_history_str = "\n".join([f"{'用户' if m['sender']=='user' else 'AI'}: {m['content']}" for m in msgs])
        
        prompt = f"""
        你是一个企业制度管理系统中的智能问答助手。
        请严格根据提供的相关制度片段回答问题。如果制度中找不到答案，请诚实说明不清楚。
        
        【历史对话记录】
        {chat_history_str if chat_history_str else "无（全新对话）"}
        
        【相关制度片段】
        {context}
        
        用户当前的新问题: {question}
        你的回答:"""
        
        response = await self.llm.ainvoke(prompt)
        return response.content

    async def ask_question_stream(self, question: str, session_id: str = None):
        """
        与 ask_question 逻辑完全一致，但使用流式生成器返回每个 Chunk
        """
        unique_sources = []
        if document_service.vector_store is None:
            context = "当前系统的制度库尚未上传任何文档，请先在左侧界面上传一份制度资料。"
        else:
            # 升级为流式上下文重排逻辑
            base_retriever = document_service.vector_store.as_retriever(search_kwargs={"k": settings.RAG_RETRIEVAL_K})
            compressor = ZhipuAIReranker(top_n=settings.RAG_RERANK_TOP_N)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, 
                base_retriever=base_retriever
            )
            docs = compression_retriever.invoke(question)
            
            if not docs:
                context = "没有找到与您问题相关的制度资料。"
            else:
                context = "\n\n".join([f"[来源: {doc.metadata.get('source', '未知')}]\n{doc.page_content}" for doc in docs])
                # 提取用于引用的原文件列表
                for doc in docs:
                    src = doc.metadata.get('source')
                    if src and src not in unique_sources:
                        unique_sources.append(src)
        
        chat_history_str = ""
        if session_id:
            from services.history_service import history_service
            msgs = await history_service.get_messages(session_id)
            if msgs:
                chat_history_str = "\n".join([f"{'用户' if m['sender']=='user' else 'AI'}: {m['content']}" for m in msgs])
        
        prompt = f"""
        你是一个企业制度管理系统中的智能问答助手。
        请严格根据提供的相关制度片段回答问题。如果制度中找不到答案，请诚实说明不清楚。
        
        【历史对话记录】
        {chat_history_str if chat_history_str else "无（全新对话）"}
        
        【相关制度片段】
        {context}
        
        用户当前的新问题: {question}
        你的回答:"""
        
        async for chunk in self.llm.astream(prompt):
            if chunk.content:
                yield chunk.content
        
        # 最后，自动附加参考文档的文件预览链接
        if unique_sources:
            import urllib.parse
            yield "\n\n<hr/>\n**📚 参阅相关原文件：**\n"
            for src in unique_sources:
                safe_src = urllib.parse.quote(src)
                yield f"- [{src}](/static/uploads/{safe_src})\n"

qa_service = QAService()
