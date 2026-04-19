from config import settings
from langchain_openai import ChatOpenAI
from services.document_service import document_service
import asyncio
from typing import List, Set
from langchain_core.documents import Document
from services.reranker_service import ZhipuAIReranker

class QAService:
    def __init__(self):
        # 初始化并连接到大语言模型 (API格式兼容的开源端点)
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL_NAME,
            api_key=settings.LLM_API_KEY, # type: ignore
            base_url=settings.LLM_API_BASE,
            streaming=True,
            max_tokens=4096
        )
        
    async def _rewrite_and_expand_query(self, question: str, session_id: str = None) -> List[str]:
        """
        利用 LLM 和对话历史，重写当前问题并生成多个检索变体 (Multi-Query)
        """
        if not session_id or not settings.ENABLE_QUERY_REWRITE:
            return [question]
            
        from services.history_service import history_service
        msgs = await history_service.get_messages(session_id)
        if not msgs:
            return [question]
            
        recent_msgs = msgs[-settings.QUERY_REWRITE_HISTORY_COUNT:]
        history_str = "\n".join([f"{'用户' if m['sender']=='user' else 'AI'}: {m['content']}" for m in recent_msgs])
        
        # 定义变体数量
        variant_count = settings.MULTI_QUERY_COUNT if settings.ENABLE_MULTI_QUERY else 0
        
        prompt = f"""角色：对话补全与检索专家
任务：根据对话历史，补全当前提问语义，并生成 {variant_count} 个语义相关的中文检索变体。
规则：
1. 第一行：输出补全后的完整问题（消除代词歧义，保持原意）。
2. 后续每行输出一个检索变体（更换术语、视角或侧重点，利于向量检索）。
3. 只输出最终文本，每行一个，禁止编号和解释。

对话历史：
{history_str}

用户当前提议：{question}
输出："""
        
        try:
            response = await self.llm.ainvoke(prompt)
            lines = [line.strip() for line in response.content.strip().split("\n") if line.strip()]
            return lines if lines else [question]
        except Exception as e:
            print(f"Query expansion failed: {e}")
            return [question]

    def _deduplicate_docs(self, docs_list: List[List[Document]]) -> List[Document]:
        """对多路检索回来的文档按内容去重"""
        seen_content = set()
        unique_docs = []
        for sub_list in docs_list:
            for doc in sub_list:
                # 使用内容作为唯一标识 (也可以结合 metadata 中的 source)
                content_id = f"{doc.metadata.get('source', '')}_{doc.page_content}"
                if content_id not in seen_content:
                    seen_content.add(content_id)
                    unique_docs.append(doc)
        return unique_docs

    async def ask_question(self, question: str, session_id: str = None) -> str:
        """
        通过增强的 RAG 流程处理问答 (Multi-Query + Rerank)
        """
        # 1. 查询重写与扩展
        queries = await self._rewrite_and_expand_query(question, session_id)
        
        # 2. 多路并发检索
        base_retriever = document_service.vector_store.as_retriever(search_kwargs={"k": settings.RAG_RETRIEVAL_K})
        # 使用 ainvoke 并行并发召回
        retrieval_tasks = [base_retriever.ainvoke(q) for q in queries]
        all_docs_list = await asyncio.gather(*retrieval_tasks)
        
        # 3. 去重与合并
        unique_docs = self._deduplicate_docs(all_docs_list)
        
        # 4. 统一重排 (Rerank)
        if not unique_docs:
            context = "没有找到与您问题相关的制度资料。"
        else:
            reranker = ZhipuAIReranker(top_n=settings.RAG_RERANK_TOP_N)
            # 对去重后的所有片段执行精排
            ranked_docs = reranker.compress_documents(unique_docs, queries[0]) # 使用补全后的主 Query 排序
            context = "\n\n".join([f"[来源: {doc.metadata.get('source', '未知')}]\n{doc.page_content}" for doc in ranked_docs])
        
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
        流式问答逻辑：多路查询 -> 并行召回 -> 去重 -> 重排 -> 生成
        """
        unique_sources = []
        # 1. 查询重写与扩展
        queries = await self._rewrite_and_expand_query(question, session_id)

        if document_service.vector_store is None:
            context = "当前系统的制度库尚未上传任何文档，请先在左侧界面上传一份制度资料。"
        else:
            # 2. 多路并发检索
            base_retriever = document_service.vector_store.as_retriever(search_kwargs={"k": settings.RAG_RETRIEVAL_K})
            retrieval_tasks = [base_retriever.ainvoke(q) for q in queries]
            all_docs_list = await asyncio.gather(*retrieval_tasks)
            
            # 3. 去重与合并
            unique_docs = self._deduplicate_docs(all_docs_list)
            
            if not unique_docs:
                context = "没有找到与您问题相关的制度资料。"
            else:
                # 4. 统一重排 (Rerank)
                reranker = ZhipuAIReranker(top_n=settings.RAG_RERANK_TOP_N)
                ranked_docs = reranker.compress_documents(unique_docs, queries[0])
                
                context = "\n\n".join([f"[来源: {doc.metadata.get('source', '未知')}]\n{doc.page_content}" for doc in ranked_docs])
                # 提取用于引用的原文件列表
                for doc in ranked_docs:
                    src = doc.metadata.get('source')
                    if src and src not in unique_sources:
                        unique_sources.append(src)
            
            # 已完成重排与上下文构建
        
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
