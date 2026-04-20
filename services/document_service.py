from typing import List
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
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
        # 基础递归字符分割器，用于对长分块进行二次细分
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            length_function=len,
        )
        # Markdown 标题分割器，用于保持自然结构
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False # 保留标题文字在正文中
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
            try:
                from zhipuai import ZhipuAI
                import base64

                client = ZhipuAI(api_key=settings.LLM_API_KEY)
                
                # GLM-OCR 能够直接处理文件或图片。
                # 由于我们这里通常是本地文件，我们将其读取并调用 GLM-OCR。
                # 注意：GLM-OCR 最佳实践是对每页进行处理或直接上传 PDF。
                # 这里我们采用 PDF 直接处理方案 (如果模型支持) 或简单的单次调用。
                
                with open(file_path, "rb") as f:
                    file_b64 = base64.b64encode(f.read()).decode("utf-8")

                response = client.chat.completions.create(
                    model=settings.GLM_OCR_MODEL_NAME,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "请对该文档进行 OCR 解析，并以 Markdown 格式输出内容。保持原有的标题层级、列表和表格结构。"
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:application/pdf;base64,{file_b64}"
                                    }
                                }
                            ]
                        }
                    ],
                )
                
                content = response.choices[0].message.content
                # 将提取的内容包装成 LangChain Document
                documents = [Document(page_content=content, metadata={"source": filename, "page": 1})]
                # 跳过后面的 loader.load()
                loader = None 
            except Exception as e:
                print(f"[OCR 错误] GLM-OCR 解析失败: {e}")
                loader = PyPDFLoader(file_path)
        elif extension in ["doc", "docx"]:
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            loader = TextLoader(file_path, encoding="utf-8")

        if loader:
            documents = loader.load()
        
        final_chunks = []
        
        # 处理逻辑：如果是 Markdown (来自 OCR 或其他)，使用结构化分割
        # 否则回退到普通分割
        for doc in documents:
            # 判断是否包含 Markdown 特征或强制对 OCR 结果使用
            if extension == "pdf" or "# " in doc.page_content:
                # 1. 结构化分割
                header_splits = self.markdown_splitter.split_text(doc.page_content)
                
                for h_split in header_splits:
                    # 提取标题路径作为上下文
                    headers = [h_split.metadata.get(f"Header {i}") for i in range(1, 4)]
                    header_path = " > ".join([h for h in headers if h])
                    
                    # 2. 如果片段过长，进行二次分割
                    if len(h_split.page_content) > self.text_splitter._chunk_size:
                        sub_splits = self.text_splitter.split_text(h_split.page_content)
                        for i, sub_content in enumerate(sub_splits):
                            # 注入上下文前缀
                            context_prefix = f"[章节: {header_path}] " if header_path else ""
                            enriched_content = f"{context_prefix}{sub_content}"
                            
                            new_doc = Document(
                                page_content=enriched_content,
                                metadata={**doc.metadata, **h_split.metadata, "chunk_index": i, "header_path": header_path}
                            )
                            final_chunks.append(new_doc)
                    else:
                        # 注入上下文前缀
                        context_prefix = f"[章节: {header_path}] " if header_path else ""
                        new_doc = Document(
                            page_content=f"{context_prefix}{h_split.page_content}",
                            metadata={**doc.metadata, **h_split.metadata, "header_path": header_path}
                        )
                        final_chunks.append(new_doc)
            else:
                # 非 Markdown 文档走普通逻辑
                final_chunks.extend(self.text_splitter.split_documents([doc]))

        chunks = final_chunks

        # 过滤 metadata，严格保留安全的基本字段，避免抛出不可预见的 BulkIndexError
        for chunk in chunks:
            chunk.metadata["source"] = filename
            chunk.metadata["type"] = "upload"
            
            safe_metadata = {}
            for k in ["source", "type", "page"]:
                if k in chunk.metadata:
                    safe_metadata[k] = chunk.metadata[k]
            chunk.metadata = safe_metadata

        if chunks:
            self.vector_store.add_documents(chunks)
            print(f"[文档服务] 已写入 {len(chunks)} 个片段到 ES: {filename}")

        return chunks

    async def sync_external_knowledge(self, source_url: str):
        pass

document_service = DocumentService()
