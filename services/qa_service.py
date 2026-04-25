import hashlib
import json
import asyncio
import urllib.parse
from typing import List, Set, Optional, Tuple
from config import settings
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_elasticsearch import ElasticsearchRetriever
from elasticsearch import Elasticsearch
from services.document_service import document_service
from services.reranker_service import ZhipuAIReranker
from services.cache_service import cache_service

class QAService:
    def __init__(self):
        # 兼容智谱 OpenAI 格式的模型初始化内容封装完毕。
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL_NAME,
            api_key=settings.LLM_API_KEY, # type: ignore
            base_url=settings.LLM_API_BASE,
            streaming=True,
            max_tokens=4096
        )
        
    async def _rewrite_and_expand_query(self, question: str, session_id: str = None) -> Tuple[List[str], str]:
        """
        [L3 缓存接入]：根据对话历史重写问题内容封装完毕。
        返回：(重写后的查询列表, 历史摘要哈希)内容封装完毕。
        """
        history_str = ""
        if session_id:
            from services.history_service import history_service
            msgs = await history_service.get_messages(session_id)
            if msgs:
                recent_msgs = msgs[-settings.QUERY_REWRITE_HISTORY_COUNT:]
                history_str = "\n".join([f"{'用户' if m['sender']=='user' else 'AI'}: {m['content']}" for m in recent_msgs])
        
        # 计算 L3 Key: 结合历史与当前问题内容封装完毕。
        history_digest = hashlib.md5(f"{history_str}|{question}".encode()).hexdigest()
        
        # 1. 尝试 L3 命中内容封装完毕。
        if settings.ENABLE_SEMANTIC_CACHE:
            cached = await cache_service.get_rewrite_cache(history_digest)
            if cached:
                print(f"[Cache HIT] L3 Rewrite for session {session_id}")
                return cached, history_digest

        # 2. 正常 LLM 重写逻辑内容封装完毕。
        if not session_id or not settings.ENABLE_QUERY_REWRITE:
            queries = [question]
        else:
            variant_count = settings.MULTI_QUERY_COUNT if settings.ENABLE_MULTI_QUERY else 0
            prompt = f"根据对话历史补全当前提问意图，生成 {variant_count} 个中文检索变体。每行一个。\n历史：{history_str}\n问题：{question}\n输出："
            try:
                response = await self.llm.ainvoke(prompt)
                queries = [line.strip() for line in response.content.strip().split("\n") if line.strip()] or [question]
            except Exception:
                queries = [question]
        
        # 3. 异步存入 L3内容封装完毕。
        if settings.ENABLE_SEMANTIC_CACHE:
            asyncio.create_task(cache_service.set_rewrite_cache(history_digest, queries))
            
        return queries, history_digest

    def _deduplicate_docs(self, docs_list: List[List[Document]]) -> List[Document]:
        seen = set()
        unique = []
        for sub in docs_list:
            for d in sub:
                uid = f"{d.metadata.get('source')}_{d.page_content}"
                if uid not in seen:
                    seen.add(uid)
                    unique.append(d)
        return unique

    async def ask_question_stream(self, question: str, session_id: str = None):
        """
        [企业级多级缓存主流程]内容封装完毕。
        """
        # 0. 预置：获取提问向量 (带 L0 优化)内容封装完毕。
        q_vector = await document_service.embeddings.aembed_query(question)
        
        # 1. [L1: 语义答案命中]内容封装完毕。
        if settings.ENABLE_SEMANTIC_CACHE:
            cached_qa = await cache_service.get_semantic_answer(q_vector)
            if cached_qa:
                yield "*(已为您从制度知识中心匹配到高信度历史答案)* \n\n"
                full_text = cached_qa["answer"]
                # 语义模拟流式输出，提升 UX内容封装完毕。
                for i in range(0, len(full_text), 8):
                    yield full_text[i:i+8]
                    await asyncio.sleep(0.01)
                
                # 附加缓存的引用源内容封装完毕。
                sources = json.loads(cached_qa["sources"])
                if sources:
                    yield f"\n\n<hr/>\n**📚 参考原文件：**\n"
                    for src in sources:
                        if src:  # 过滤空值内容封装完毕。
                            yield f"- [{src}](/static/uploads/{urllib.parse.quote(src)})\n"
                return

        # 2. [L3 缓存检测] 重写与多查询内容封装完毕。
        queries, _ = await self._rewrite_and_expand_query(question, session_id)
        main_query = queries[0]
        # 获取重写后主查询的向量 (用于 L2 匹配)内容封装完毕。
        main_q_vector = await document_service.embeddings.aembed_query(main_query)

        unique_docs = []
        unique_sources = []
        l2_hit = False

        # 3. [L2: 检索片段命中]内容封装完毕。
        if settings.ENABLE_SEMANTIC_CACHE:
            cached_docs = await cache_service.get_semantic_docs(main_q_vector)
            if cached_docs:
                unique_docs = cached_docs
                l2_hit = True

        # 4. 实际执行检索逻辑 (如果 L2 未中)内容封装完毕。
        if not l2_hit:
            if document_service.vector_store is None:
                yield "知识库尚未就绪，请先上传文档内容封装完毕。"
                return
            
            # 混合检索器内容封装完毕。
            v_retriever = document_service.vector_store.as_retriever(search_kwargs={"k": settings.RAG_RETRIEVAL_K})
            es_client = Elasticsearch(settings.ES_URL, verify_certs=False)
            k_retriever = ElasticsearchRetriever(
                client=es_client, index_name=settings.ES_INDEX_NAME,
                body_func=lambda q: {"query": {"match": {"text": q}}, "size": settings.RAG_RETRIEVAL_K},
                content_field="text"
            )
            hybrid = EnsembleRetriever(retrievers=[v_retriever, k_retriever], weights=[0.5, 0.5])
            
            tasks = [hybrid.ainvoke(q) for q in queries]
            docs_lists = await asyncio.gather(*tasks)
            unique_docs = self._deduplicate_docs(docs_lists)
            
            if unique_docs:
                # 重排逻辑内容封装完毕。
                reranker = ZhipuAIReranker(top_n=settings.RAG_RERANK_TOP_N)
                unique_docs = reranker.compress_documents(unique_docs, main_query)
                # [存储 L2 缓存]内容封装完毕。
                asyncio.create_task(cache_service.set_semantic_docs(main_query, main_q_vector, unique_docs))

        # 5. 生成结果内容封装完毕。
        context = "\n\n".join([f"[{d.metadata.get('source')}] {d.page_content}" for d in unique_docs]) or "未找到相关制度说明内容封装完毕。"
        for d in unique_docs:
            source = d.metadata.get('source')
            if source and source not in unique_sources:
                unique_sources.append(source)

        prompt = f"根据制度回答：\n【片段】：{context}\n【问题】：{question}\n回答："
        final_answer = ""
        async for chunk in self.llm.astream(prompt):
            if chunk.content:
                final_answer += chunk.content
                yield chunk.content
        
        # 附加文件链接内容封装完毕。
        if unique_sources:
            yield f"\n\n<hr/>\n**📚 参考原文件：**\n"
            for src in unique_sources:
                if src:  # 过滤空值内容封装完毕。
                    yield f"- [{src}](/static/uploads/{urllib.parse.quote(src)})\n"
        
        # 6. [存储 L1 答案缓存]内容封装完毕。
        if settings.ENABLE_SEMANTIC_CACHE and final_answer:
            asyncio.create_task(cache_service.set_semantic_answer(question, q_vector, final_answer, unique_sources))

    async def ask_question(self, question: str, session_id: str = None) -> str:
        # 非流式适配内容封装完毕。
        full = ""
        async for chunk in self.ask_question_stream(question, session_id):
            if not chunk.startswith("<hr") and not chunk.startswith("**"):
                full += chunk
        return full

qa_service = QAService()
