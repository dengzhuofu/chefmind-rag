"""
向量数据库配置模块
使用ChromaDB作为本地向量数据库（替代Milvus Lite）
"""

from app.core.config import settings
import logging
import os

logger = logging.getLogger(__name__)

# ChromaDB客户端实例
_client = None
_collection = None


def get_chroma_client():
    """获取ChromaDB客户端"""
    global _client
    if _client is None:
        import chromadb

        # 确保数据目录存在
        db_path = "./chroma_data"
        os.makedirs(db_path, exist_ok=True)

        # 创建持久化客户端
        _client = chromadb.PersistentClient(path=db_path)
        logger.info(f"Connected to ChromaDB: {db_path}")

    return _client


def connect_milvus():
    """连接向量数据库（兼容接口）"""
    client = get_chroma_client()
    return client


def create_recipe_collection():
    """创建菜谱向量集合"""
    client = get_chroma_client()
    collection_name = settings.MILVUS_COLLECTION_NAME

    # 获取或创建集合
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"description": "Recipe chunks with embeddings"}
    )

    logger.info(f"Created/Loaded collection '{collection_name}'")
    logger.info(f"   当前记录数: {collection.count()}")

    return collection_name


def get_collection():
    """获取菜谱集合"""
    client = get_chroma_client()
    return client.get_collection(settings.MILVUS_COLLECTION_NAME)


def drop_collection():
    """删除集合(用于测试)"""
    client = get_chroma_client()
    try:
        client.delete_collection(settings.MILVUS_COLLECTION_NAME)
        logger.info(f"Dropped collection '{settings.MILVUS_COLLECTION_NAME}'")
    except Exception:
        pass


def insert_data(data: list):
    """插入数据到ChromaDB"""
    client = get_chroma_client()
    collection_name = settings.MILVUS_COLLECTION_NAME
    collection = client.get_collection(collection_name)

    # 准备数据格式
    ids = [d["id"] for d in data]
    embeddings = [d["embedding"] for d in data]
    documents = [d["content"] for d in data]
    metadatas = []
    for d in data:
        meta = {k: v for k, v in d.items() if k not in ["id", "embedding", "content"]}
        # ChromaDB要求metadata值必须是str/int/float/bool
        for k, v in meta.items():
            if isinstance(v, list):
                meta[k] = str(v)
            elif v is None:
                meta[k] = ""
        metadatas.append(meta)

    # 批量插入
    batch_size = 500
    total_inserted = 0
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i+batch_size]
        batch_embeddings = embeddings[i:i+batch_size]
        batch_documents = documents[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]

        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_documents,
            metadatas=batch_metadatas
        )
        total_inserted += len(batch_ids)

    return {"insert_count": total_inserted}


def search_vectors(query_embedding: list, top_k: int = 5, filter_expr: str = None):
    """向量检索"""
    client = get_chroma_client()
    collection_name = settings.MILVUS_COLLECTION_NAME
    collection = client.get_collection(collection_name)

    # 构建查询参数
    query_params = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"]
    }

    # 添加过滤条件
    if filter_expr:
        # ChromaDB使用不同的过滤语法
        # 这里简化处理，后续可以扩展
        pass

    results = collection.query(**query_params)

    # 转换为统一格式
    formatted_results = []
    if results and len(results["ids"]) > 0:
        for i in range(len(results["ids"][0])):
            hit = {
                "id": results["ids"][0][i],
                "distance": results["distances"][0][i],
                "entity": {
                    "content": results["documents"][0][i],
                    **results["metadatas"][0][i]
                }
            }
            formatted_results.append(hit)

    return [formatted_results]


def get_collection_stats():
    """获取集合统计信息"""
    client = get_chroma_client()
    collection_name = settings.MILVUS_COLLECTION_NAME

    try:
        collection = client.get_collection(collection_name)
        return {
            "row_count": collection.count(),
            "collection_name": collection_name
        }
    except Exception:
        return {
            "row_count": 0,
            "collection_name": collection_name
        }
