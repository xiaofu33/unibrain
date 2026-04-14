from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
import psycopg
from config import settings

class DocumentService:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        # 初始化 Embedding 向量模型，通过本地或智谱等兼容端点连接
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL_NAME,
            openai_api_base=settings.LLM_API_BASE,
            openai_api_key=settings.LLM_API_KEY
        )
        
        # 实例化 PGVector 向量引擎，挂载于 PostgreSQL
        sync_url = settings.DATABASE_URL.replace("postgresql+psycopg_async", "postgresql+psycopg")
        self.sync_url = sync_url
        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name="institution_docs",
            connection=self.sync_url,
            use_jsonb=True,
        )
        
    async def get_uploaded_documents(self) -> List[str]:
        """
        获取所有已持久化到 PgVector 库中的源文件名称。
        """
        try:
            with psycopg.connect(self.sync_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT DISTINCT cmetadata->>'source' FROM langchain_pg_embedding WHERE cmetadata->>'type' = 'upload';")
                    rows = cur.fetchall()
                    return [row[0] for row in rows if row[0]]
        except Exception as e:
            print("Error retrieving documents from DB:", e)
            return []

    async def process_file_upload(self, file_path: str, filename: str) -> List[Document]:
        """
        处理单个上传的文件，将其解析为 LangChain 所需的 Document 文档格式。
        """
        extension = filename.split(".")[-1].lower()
        if extension == "pdf":
            loader = PyPDFLoader(file_path)
        elif extension in ["doc", "docx"]:
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            # 后备措施：默认使用纯文本解析器
            loader = TextLoader(file_path)
            
        documents = loader.load()
        # 将长文档拆分成短的文本片段
        chunks = self.text_splitter.split_documents(documents)
        
        # 为文本片段打上来源标识和类型的元数据
        for chunk in chunks:
            chunk.metadata["source"] = filename
            chunk.metadata["type"] = "upload"
            
        # 写入 PGVector PostgreSQL 数据库完成持久化
        self.vector_store.add_documents(chunks)
        
        return chunks
        
    async def sync_external_knowledge(self, source_url: str):
        """
        从原有系统的外部知识库（如 Confluence 或 Notion）同步文档。
        """
        # 具体的接口防伪校验和 Webhook 逻辑将在这里实现
        pass
        
document_service = DocumentService()

