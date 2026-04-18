from typing import Any, Optional, Sequence
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_classic.retrievers.document_compressors.base import BaseDocumentCompressor
from zhipuai import ZhipuAI
from config import settings

import httpx

class ZhipuAIReranker(BaseDocumentCompressor):
    """
    基于智谱 AI 接口实现的 Reranker 压缩器，兼容 LangChain 结构。
    """
    model_name: str = settings.RERANKER_MODEL_NAME
    top_n: int = settings.RAG_RERANK_TOP_N

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> Sequence[Document]:
        """
        通过智谱 Reranker 接口对文档进行排序。
        """
        if not documents:
            return []

        # 构造文档内容列表
        doc_contents = [doc.page_content for doc in documents]
        
        # 智谱 Reranker API V4 正确路径
        url = "https://open.bigmodel.cn/api/paas/v4/rerank"
        headers = {
            "Authorization": f"Bearer {settings.LLM_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": doc_contents,
            "top_n": self.top_n
        }

        try:
            with httpx.Client() as client:
                response = client.post(url, headers=headers, json=payload, timeout=30.0)
                response.raise_for_status()
                data = response.json()
                
            results = data.get("results", [])
            reranked_docs = []
            for res in results:
                index = res["index"]
                doc = documents[index]
                doc.metadata["rerank_score"] = res["relevance_score"]
                reranked_docs.append(doc)
                
            return reranked_docs

        except Exception as e:
            # 降级逻辑：调用失败时打印错误并返回原始召回的前 top_n 条数据
            print(f"Zhipu Reranker API Exception: {str(e)}")
            return documents[:self.top_n]
