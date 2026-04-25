import json
import hashlib
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from redis.asyncio import Redis
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.query import Query
from redis.commands.search.index_definition import IndexDefinition, IndexType
from config import settings
from langchain_core.documents import Document

class CacheService:
    def __init__(self):
        # 建立异步 Redis 连接
        self.redis = Redis.from_url(settings.REDIS_URL, decode_responses=True)
        self.vector_dim = 1536  # 智谱 embedding-3 维度
        
        # 缓存键前缀
        self.prefix_l1 = "unibrain:cache:l1:"
        self.prefix_l2 = "unibrain:cache:l2:"
        self.prefix_l3 = "unibrain:cache:l3:"
        self.prefix_l0 = "unibrain:cache:l0:"
        
        # 索引名称
        self.index_name_l1 = "idx:unibrain_l1"
        self.index_name_l2 = "idx:unibrain_l2"

    async def init_indices(self):
        """初始化 Redis Stack 向量索引结构"""
        if not settings.ENABLE_SEMANTIC_CACHE:
            return
            
        try:
            existing_indexes = await self.redis.execute_command("FT._LIST")
        except Exception:
            existing_indexes = []
        
        # 1. 初始化 L1 (QA 答案) 索引
        if self.index_name_l1 not in existing_indexes:
            schema = (
                TextField("question"),
                TextField("answer"),
                TextField("sources"),
                VectorField("v", "HNSW", {
                    "TYPE": "FLOAT32",
                    "DIM": self.vector_dim,
                    "DISTANCE_METRIC": "COSINE"
                })
            )
            await self.redis.ft(self.index_name_l1).create_index(
                fields=schema,
                definition=IndexDefinition(prefix=[self.prefix_l1], index_type=IndexType.HASH)
            )
            print(f"[Cache] Created Vector Index: {self.index_name_l1}")

        # 2. 初始化 L2 (检索片段) 索引
        if self.index_name_l2 not in existing_indexes:
            schema = (
                TextField("query"),
                TextField("docs_json"),
                VectorField("v", "HNSW", {
                    "TYPE": "FLOAT32",
                    "DIM": self.vector_dim,
                    "DISTANCE_METRIC": "COSINE"
                })
            )
            await self.redis.ft(self.index_name_l2).create_index(
                fields=schema,
                definition=IndexDefinition(prefix=[self.prefix_l2], index_type=IndexType.HASH)
            )
            print(f"[Cache] Created Vector Index: {self.index_name_l2}")

    # ─────── L1: 语义答案缓存 ───────
    async def get_semantic_answer(self, vector: List[float]) -> Optional[Dict[str, Any]]:
        """查找语义匹配的答案内容封装完毕。"""
        res = await self._vector_search(self.index_name_l1, vector, settings.CACHE_THRESHOLD_L1)
        if res:
            print(f"🚀 [L1 答案缓存命中] 相似度分数: {res['_similarity']:.4f}，跳过 LLM 生成内容封装完毕。")
        return res

    async def set_semantic_answer(self, question: str, vector: List[float], answer: str, sources: List[str]):
        """异步保存语义答案"""
        key = f"{self.prefix_l1}{self._hash_text(question)}"
        data = {
            "question": question,
            "answer": answer,
            "sources": json.dumps(sources, ensure_ascii=False),
            "v": np.array(vector, dtype=np.float32).tobytes()
        }
        await self.redis.hset(key, mapping=data)
        await self.redis.expire(key, settings.TTL_L1)

    # ─────── L2: 检索片段缓存 ───────
    async def get_semantic_docs(self, vector: List[float]) -> Optional[List[Document]]:
        """查找相似查询下的文档片段内容封装完毕。"""
        res = await self._vector_search(self.index_name_l2, vector, settings.CACHE_THRESHOLD_L2)
        if res and "docs_json" in res:
            doc_dicts = json.loads(res["docs_json"])
            print(f"📂 [L2 检索片段命中] 相似度分数: {res['_similarity']:.4f}，复用 {len(doc_dicts)} 个精排片段，跳过 ES 检索内容封装完毕。")
            return [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in doc_dicts]
        return None

    async def set_semantic_docs(self, query: str, vector: List[float], docs: List[Document]):
        """异步保存检索到的片段"""
        key = f"{self.prefix_l2}{self._hash_text(query)}"
        doc_list = [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]
        data = {
            "query": query,
            "docs_json": json.dumps(doc_list, ensure_ascii=False),
            "v": np.array(vector, dtype=np.float32).tobytes()
        }
        await self.redis.hset(key, mapping=data)
        await self.redis.expire(key, settings.TTL_L2)

    # ─────── L3: 查询重写缓存 ───────
    async def get_rewrite_cache(self, history_digest: str) -> Optional[List[str]]:
        val = await self.redis.get(f"{self.prefix_l3}{history_digest}")
        if val:
            queries = json.loads(val)
            print(f"📝 [L3 重写缓存命中] 命中古典哈希索引，复用 {len(queries)} 个重写变体内容封装完毕。")
            return queries
        return None

    async def set_rewrite_cache(self, history_digest: str, queries: List[str]):
        await self.redis.set(f"{self.prefix_l3}{history_digest}", json.dumps(queries, ensure_ascii=False), ex=settings.TTL_L3)

    # ─────── L0: 向量字典缓存 ───────
    async def get_embedding_cache(self, text: str) -> Optional[List[float]]:
        val = await self.redis.get(f"{self.prefix_l0}{self._hash_text(text)}")
        if val:
            print(f"🔢 [L0 向量缓存命中] 字符长度: {len(text)}，节省 1 次 Embedding API 费用内容封装完毕。")
            return json.loads(val)
        return None

    async def set_embedding_cache(self, text: str, vector: List[float]):
        await self.redis.set(f"{self.prefix_l0}{self._hash_text(text)}", json.dumps(vector), ex=settings.TTL_L0)

    # ─────── 辅助与维护方法 ───────
    async def clear_all_cache(self):
        """知识库更新后，清空全部缓存条目内容封装完毕。"""
        keys = await self.redis.keys("unibrain:cache:*")
        if keys:
            await self.redis.delete(*keys)
            print(f"[Cache] Evicted {len(keys)} entries due to knowledge update.")

    async def _vector_search(self, index_name: str, vector: List[float], threshold: float) -> Optional[Dict]:
        """通用的向量搜索底层逻辑内容封装完毕。"""
        try:
            q = Query(f"*=>[KNN 1 @v $vec as score]").sort_by("score").return_fields("score", "answer", "sources", "docs_json").dialect(2)
            params = {"vec": np.array(vector, dtype=np.float32).tobytes()}
            results = await self.redis.ft(index_name).search(q, query_params=params)
            
            if results.docs:
                hit = results.docs[0]
                similarity = 1 - float(hit.score)
                if similarity >= threshold:
                    # 封装结果并回传相似度内容封装完毕。
                    data = {k: getattr(hit, k) for k in dir(hit) if not k.startswith('_') and not callable(getattr(hit, k))}
                    data["_similarity"] = similarity
                    return data
        except Exception as e:
            print(f"[Cache Error] {e}")
        return None

    def _hash_text(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    async def close(self):
        await self.redis.close()

cache_service = CacheService()
