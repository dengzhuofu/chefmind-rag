"""
数据库初始化脚本
初始化SQLite和Milvus Lite，并索引所有菜谱数据
"""

import sys
import os
import asyncio
import logging
from pathlib import Path

# 添加backend目录到路径
sys.path.insert(0, os.path.dirname(__file__))

# 加载环境变量
from dotenv import load_dotenv
load_dotenv('.env')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def init_sqlite():
    """初始化SQLite数据库"""
    logger.info("=" * 50)
    logger.info("Step 1: 初始化SQLite数据库")
    logger.info("=" * 50)

    from app.core.database import engine, Base
    from app.models.recipe import Recipe, RecipeChunk, Document, Conversation, UserFeedback

    # 创建所有表
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("✅ SQLite数据库表创建成功")
    logger.info("   - recipes")
    logger.info("   - recipe_chunks")
    logger.info("   - documents")
    logger.info("   - conversations")
    logger.info("   - user_feedback")

    return True


def init_milvus():
    """初始化Milvus Lite向量数据库"""
    logger.info("=" * 50)
    logger.info("Step 2: 初始化Milvus Lite向量数据库")
    logger.info("=" * 50)

    from app.core.milvus import connect_milvus, create_recipe_collection, drop_collection

    # 确保数据目录存在
    os.makedirs('./milvus_data', exist_ok=True)

    # 连接Milvus
    connect_milvus()

    # 先删除旧集合，确保数据干净
    drop_collection()
    logger.info("   旧集合已清空")

    # 创建集合
    collection_name = create_recipe_collection()

    logger.info(f"✅ Milvus集合创建成功: {collection_name}")

    return collection_name


def parse_and_index_recipes(collection_name):
    """解析并索引菜谱数据"""
    logger.info("=" * 50)
    logger.info("Step 3: 解析并索引菜谱数据")
    logger.info("=" * 50)

    from app.services.recipe_parser import recipe_parser
    from app.services.recipe_chunker import recipe_chunker
    from app.services.embedding_service import get_embedding_service
    from app.core.milvus import insert_data

    # 1. 解析菜谱
    logger.info("\n3.1 解析菜谱文件...")
    data_dir = '../data/cook/dishes'
    recipes = recipe_parser.parse_directory(data_dir)
    logger.info(f"   解析完成: {len(recipes)} 个菜谱")

    # 2. 分块
    logger.info("\n3.2 分块处理...")
    all_chunks = []
    for recipe in recipes:
        chunks = recipe_chunker.chunk_recipe(recipe)
        all_chunks.extend(chunks)

    logger.info(f"   分块完成: {len(all_chunks)} 个块")

    # 统计块类型
    chunk_types = {}
    for chunk in all_chunks:
        chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1

    logger.info("   块类型分布:")
    for ctype, count in chunk_types.items():
        logger.info(f"     - {ctype}: {count}")

    # 3. 生成嵌入
    logger.info("\n3.3 生成嵌入向量...")
    embedding_service = get_embedding_service()

    # 准备文本
    texts = [chunk.content for chunk in all_chunks]
    logger.info(f"   待嵌入文本数: {len(texts)}")

    # 批量嵌入
    embeddings = embedding_service.embed_documents(texts, batch_size=32)
    logger.info(f"   嵌入完成: {len(embeddings)} 个向量")
    logger.info(f"   向量维度: {len(embeddings[0])}")

    # 4. 存储到Milvus
    logger.info("\n3.4 存储到Milvus...")

    # 准备数据（字典列表格式）
    data = []
    for i, chunk in enumerate(all_chunks):
        data.append({
            "id": chunk.chunk_id,
            "embedding": embeddings[i],
            "content": chunk.content,
            "recipe_id": chunk.recipe_id,
            "recipe_title": chunk.recipe_title,
            "chunk_type": chunk.chunk_type,
            "chunk_id": chunk.chunk_id,
            "step_number": chunk.step_number or 0,
            "document_source": chunk.document_source,
            "cooking_time": chunk.cooking_time or "",
            "difficulty": chunk.difficulty or "",
            "tags": ",".join(chunk.tags) if chunk.tags else ""
        })

    # 批量插入
    result = insert_data(data)

    logger.info(f"✅ 数据插入成功: {result['insert_count']} 条记录")

    return collection_name


def test_retrieval(collection_name):
    """测试检索功能"""
    logger.info("=" * 50)
    logger.info("Step 4: 测试检索功能")
    logger.info("=" * 50)

    from app.services.embedding_service import get_embedding_service
    from app.core.milvus import search_vectors, get_collection_stats

    embedding_service = get_embedding_service()

    # 获取集合统计
    stats = get_collection_stats()
    logger.info(f"集合统计: {stats}")

    # 测试查询
    test_queries = [
        "清蒸鲈鱼怎么做",
        "红烧肉的食材",
        "川菜有哪些"
    ]

    for query in test_queries:
        logger.info(f"\n查询: {query}")

        # 生成查询向量
        query_embedding = embedding_service.embed_query(query)

        # 向量检索
        results = search_vectors(query_embedding, top_k=3)

        logger.info("检索结果:")
        if results and len(results) > 0:
            for i, hit in enumerate(results[0], 1):
                entity = hit.get('entity', {})
                logger.info(f"  {i}. {entity.get('recipe_title')} - {entity.get('chunk_type')}")
                logger.info(f"     相似度: {hit.get('distance', 0):.4f}")
                content = entity.get('content', '')[:60]
                logger.info(f"     内容: {content}...")

    logger.info("\n✅ 检索测试完成")


async def save_to_sqlite():
    """将菜谱元数据保存到SQLite"""
    logger.info("=" * 50)
    logger.info("Step 5: 保存菜谱元数据到SQLite")
    logger.info("=" * 50)

    from app.core.database import async_session_factory
    from app.models.recipe import Recipe, Document
    from app.services.recipe_parser import recipe_parser

    # 解析菜谱
    data_dir = '../data/cook/dishes'
    recipes = recipe_parser.parse_directory(data_dir)

    async with async_session_factory() as session:
        # 保存文档记录
        doc = Document(
            filename="data/cook/dishes",
            file_path=data_dir,
            status="done",
            recipe_count=len(recipes)
        )
        session.add(doc)

        # 保存菜谱元数据（前10个作为示例）
        for recipe in recipes[:10]:
            recipe_model = Recipe(
                id=recipe_parser.generate_recipe_id(recipe.title),
                title=recipe.title,
                description=recipe.description,
                ingredients=recipe.ingredients,
                steps=recipe.steps,
                tips=recipe.tips,
                cooking_time=recipe.cooking_time,
                difficulty=recipe.difficulty,
                tags=recipe.tags,
                source_file=recipe.source_file,
                images=recipe.images
            )
            session.add(recipe_model)

        await session.commit()

    logger.info(f"✅ 保存了 1 个文档记录和 10 个菜谱元数据")


async def main():
    """主函数"""
    logger.info("🚀 ChefMind 数据库初始化开始\n")

    try:
        # 1. 初始化SQLite
        await init_sqlite()

        # 2. 初始化Milvus
        collection_name = init_milvus()

        # 3. 解析并索引菜谱
        parse_and_index_recipes(collection_name)

        # 4. 测试检索
        test_retrieval(collection_name)

        # 5. 保存元数据到SQLite
        await save_to_sqlite()

        logger.info("\n" + "=" * 50)
        logger.info("🎉 数据库初始化完成！")
        logger.info("=" * 50)

        # 获取最终统计
        from app.core.milvus import get_collection_stats
        stats = get_collection_stats()
        logger.info(f"\n统计信息:")
        logger.info(f"  - Milvus集合: {collection_name}")
        logger.info(f"  - 向量记录数: {stats.get('row_count', 'unknown')}")
        logger.info(f"  - SQLite数据库: chefmind.db")

    except Exception as e:
        logger.error(f"❌ 初始化失败: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
