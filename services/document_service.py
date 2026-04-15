from typing import List
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_elasticsearch import ElasticsearchStore
from elasticsearch import Elasticsearch
from config import settings
import httpx


class ZhipuEmbeddings(Embeddings):
    """
    直接调用智谱 Embedding API，完全绕过 tiktoken。
    使用 trust_env=False 禁止 httpx 读取系统代理环境变量（HTTPS_PROXY 等），
    避免代理未启动时阻塞请求。
    """

    def __init__(self, model: str, api_key: str, api_base: str):
        self.model = model
        self._base_url = api_base.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _embed(self, texts: List[str]) -> List[List[float]]:
        # trust_env=False：忽略 HTTPS_PROXY 等系统代理环境变量
        with httpx.Client(trust_env=False, timeout=30.0) as client:
            resp = client.post(
                f"{self._base_url}/embeddings",
                headers=self._headers,
                json={"model": self.model, "input": texts},
            )
            resp.raise_for_status()
            sorted_data = sorted(resp.json()["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in sorted_data]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]


class DocumentService:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        # 使用自定义 ZhipuEmbeddings，绕过 tiktoken 和系统代理
        self.embeddings = ZhipuEmbeddings(
            model=settings.EMBEDDING_MODEL_NAME,
            api_key=settings.LLM_API_KEY,
            api_base=settings.LLM_API_BASE,
        )

        # 构建 ES 认证参数（无认证时二者均为空字符串则跳过）
        es_kwargs = {"es_url": settings.ES_URL}
        if settings.ES_USERNAME and settings.ES_PASSWORD:
            es_kwargs["es_user"] = settings.ES_USERNAME
            es_kwargs["es_password"] = settings.ES_PASSWORD

        # 实例化 ElasticsearchStore 作为向量引擎
        self.vector_store = ElasticsearchStore(
            index_name=settings.ES_INDEX_NAME,
            embedding=self.embeddings,
            **es_kwargs,
        )

        # 原生 ES 客户端，用于聚合查询已上传的文件列表
        client_kwargs = {"hosts": [settings.ES_URL]}
        if settings.ES_USERNAME and settings.ES_PASSWORD:
            client_kwargs["http_auth"] = (settings.ES_USERNAME, settings.ES_PASSWORD)
        self._es_client = Elasticsearch(**client_kwargs)

    async def get_uploaded_documents(self) -> List[str]:
        """
        获取所有已持久化到 Elasticsearch 中的源文件名称（去重）。
        """
        try:
            resp = self._es_client.search(
                index=settings.ES_INDEX_NAME,
                body={
                    "size": 0,
                    "query": {"term": {"metadata.type.keyword": "upload"}},
                    "aggs": {
                        "unique_sources": {
                            "terms": {"field": "metadata.source.keyword", "size": 1000}
                        }
                    }
                }
            )
            buckets = resp["aggregations"]["unique_sources"]["buckets"]
            return [b["key"] for b in buckets]
        except Exception as e:
            print("Error retrieving documents from ES:", e)
            return []

    async def process_file_upload(self, file_path: str, filename: str) -> List[Document]:
        """
        处理单个上传的文件，将其解析为 LangChain Document 格式，
        并向量化后写入 Elasticsearch。
        """
        extension = filename.split(".")[-1].lower()
        if extension == "pdf":
            loader = PyPDFLoader(file_path)
        elif extension in ["doc", "docx"]:
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            loader = TextLoader(file_path, encoding="utf-8")

        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)

        for chunk in chunks:
            chunk.metadata["source"] = filename
            chunk.metadata["type"] = "upload"

        if chunks:
            self.vector_store.add_documents(chunks)
            print(f"[文档服务] 已写入 {len(chunks)} 个片段到 ES: {filename}")

        return chunks

    async def sync_external_knowledge(self, source_url: str):
        pass


document_service = DocumentService()
