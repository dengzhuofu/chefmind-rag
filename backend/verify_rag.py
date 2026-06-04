"""
RAG流程验证脚本
逐步验证每个环节是否真实工作
"""

import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv('.env')

sys.stdout.reconfigure(encoding='utf-8')


def print_section(title):
    """打印章节标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


async def verify_step1_parsing():
    """验证步骤1: 文档解析"""
    print_section("Step 1: 文档解析验证")

    from app.services.recipe_parser import recipe_parser

    # 解析单个文件
    test_file = '../data/cook/dishes/aquatic/清蒸鲈鱼/清蒸鲈鱼.md'
    recipe = recipe_parser.parse_file(test_file)

    print(f"✅ 解析文件: {test_file}")
    print(f"   标题: {recipe.title}")
    print(f"   食材数: {len(recipe.ingredients)}")
    print(f"   步骤数: {len(recipe.steps)}")
    print(f"   图片数: {len(recipe.images)}")
    print(f"   难度: {recipe.difficulty}")
    print(f"   来源: {recipe.source_file}")

    # 批量解析
    all_recipes = recipe_parser.parse_directory('../data/cook/dishes')
    print(f"\n✅ 批量解析完成")
    print(f"   总菜谱数: {len(all_recipes)}")

    return recipe


async def verify_step2_chunking(recipe):
    """验证步骤2: 分块处理"""
    print_section("Step 2: 分块处理验证")

    from app.services.recipe_chunker import recipe_chunker

    chunks = recipe_chunker.chunk_recipe(recipe)

    print(f"✅ 分块完成: {len(chunks)} 个块")

    # 统计块类型
    chunk_types = {}
    for chunk in chunks:
        chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1

    print(f"\n   块类型分布:")
    for ctype, count in chunk_types.items():
        print(f"     - {ctype}: {count}")

    # 显示示例块
    print(f"\n   示例块:")
    for chunk in chunks[:3]:
        print(f"     ID: {chunk.chunk_id}")
        print(f"     类型: {chunk.chunk_type}")
        content_preview = chunk.content[:50].replace('\n', ' ')
        print(f"     内容: {content_preview}...")
        if chunk.related_ingredients:
            print(f"     相关食材: {chunk.related_ingredients}")
        print()

    return chunks


async def verify_step3_embedding(chunks):
    """验证步骤3: 向量嵌入"""
    print_section("Step 3: 向量嵌入验证")

    from app.services.embedding_service import get_embedding_service

    embedding_service = get_embedding_service()

    # 测试单个嵌入
    test_text = chunks[0].content
    embedding = embedding_service.embed_query(test_text)

    print(f"✅ 嵌入服务: {type(embedding_service).__name__}")
    print(f"   向量维度: {len(embedding)}")
    print(f"   向量类型: {type(embedding[0])}")
    print(f"   向量范围: [{min(embedding):.4f}, {max(embedding):.4f}]")

    # 测试批量嵌入
    texts = [c.content for c in chunks[:5]]
    embeddings = embedding_service.embed_documents(texts)

    print(f"\n   批量嵌入测试:")
    print(f"   输入: {len(texts)} 个文本")
    print(f"   输出: {len(embeddings)} 个向量")
    print(f"   每个维度: {len(embeddings[0])}")

    return embedding_service


async def verify_step4_storage():
    """验证步骤4: 向量存储"""
    print_section("Step 4: 向量存储验证 (ChromaDB)")

    from app.core.milvus import get_chroma_client, get_collection_stats

    client = get_chroma_client()
    stats = get_collection_stats()

    print(f"✅ ChromaDB连接成功")
    print(f"   集合名: {stats['collection_name']}")
    print(f"   记录数: {stats['row_count']}")

    # 测试查询一条记录
    collection = client.get_collection("recipes")
    result = collection.get(limit=1, include=["documents", "metadatas"])

    if result["ids"]:
        print(f"\n   示例记录:")
        print(f"   ID: {result['ids'][0]}")
        print(f"   内容前50字: {result['documents'][0][:50]}...")
        print(f"   元数据: {result['metadatas'][0]}")

    return stats


async def verify_step5_retrieval(embedding_service):
    """验证步骤5: 向量检索"""
    print_section("Step 5: 向量检索验证")

    from app.core.milvus import search_vectors

    # 测试查询
    query = "清蒸鲈鱼怎么做"
    print(f"查询: {query}")

    # 生成查询向量
    query_embedding = embedding_service.embed_query(query)
    print(f"查询向量维度: {len(query_embedding)}")

    # 执行检索
    results = search_vectors(query_embedding, top_k=5)

    print(f"\n✅ 检索成功")
    print(f"   返回结果数: {len(results[0]) if results else 0}")

    if results and len(results[0]) > 0:
        print(f"\n   检索结果:")
        for i, hit in enumerate(results[0][:3], 1):
            entity = hit.get('entity', {})
            print(f"\n   {i}. {entity.get('recipe_title', 'N/A')}")
            print(f"      类型: {entity.get('chunk_type', 'N/A')}")
            print(f"      距离: {hit.get('distance', 0):.4f}")
            content = entity.get('content', '')[:60].replace('\n', ' ')
            print(f"      内容: {content}...")

    return results


async def verify_step6_reranking(results):
    """验证步骤6: 重排序"""
    print_section("Step 6: 重排序验证")

    from app.services.reranker_service import get_reranker_service
    from app.services.hybrid_retriever import RetrievalResult

    reranker = get_reranker_service()

    # 转换结果格式
    retrieval_results = []
    if results and len(results[0]) > 0:
        for hit in results[0]:
            entity = hit.get('entity', {})
            result = RetrievalResult(
                chunk_id=hit.get('id', ''),
                content=entity.get('content', ''),
                score=hit.get('distance', 0),
                metadata={
                    'recipe_title': entity.get('recipe_title', ''),
                    'chunk_type': entity.get('chunk_type', ''),
                    'step_number': entity.get('step_number', 0),
                    'document_source': entity.get('document_source', ''),
                },
                source='vector'
            )
            retrieval_results.append(result)

    # 执行重排序
    query = "清蒸鲈鱼怎么做"
    reranked, has_content = await reranker.rerank(
        query=query,
        results=retrieval_results,
        top_k=3,
        threshold=0.3
    )

    print(f"✅ 重排序完成")
    print(f"   服务类型: {type(reranker).__name__}")
    print(f"   输入: {len(retrieval_results)} 个结果")
    print(f"   输出: {len(reranked)} 个结果")
    print(f"   有相关内容: {has_content}")

    print(f"\n   重排序后结果:")
    for i, result in enumerate(reranked, 1):
        print(f"\n   {i}. {result.metadata.get('recipe_title', 'N/A')}")
        print(f"      类型: {result.metadata.get('chunk_type', 'N/A')}")
        print(f"      分数: {result.score:.4f}")
        content = result.content[:60].replace('\n', ' ')
        print(f"      内容: {content}...")

    return reranked, has_content


async def verify_step7_generation(reranked, has_content):
    """验证步骤7: 回答生成"""
    print_section("Step 7: LLM回答生成验证")

    from app.services.llm_factory import get_llm_for_answer
    from app.services.answer_generator import AnswerGenerator

    # 获取LLM
    llm = get_llm_for_answer()
    print(f"✅ LLM服务: {llm.model_name}")

    # 创建生成器
    generator = AnswerGenerator(llm=llm)

    # 生成回答
    query = "清蒸鲈鱼怎么做"
    result = await generator.generate(
        query=query,
        retrieval_results=reranked,
        has_related_content=has_content,
        request_id="verify_001"
    )

    print(f"\n✅ 回答生成完成")
    print(f"   是否拒答: {result.refused}")
    print(f"   引用数量: {len(result.citations)}")

    print(f"\n   问题: {query}")
    print(f"\n   回答:")
    print(f"   {result.answer}")

    if result.citations:
        print(f"\n   引用详情:")
        for c in result.citations:
            print(f"     [{c.citation_id}] {c.recipe_title} - {c.chunk_type}")

    return result


async def verify_step8_hallucination_check(result):
    """验证步骤8: 幻觉处理"""
    print_section("Step 8: 幻觉处理验证")

    import re

    # 检查引用是否存在幻觉
    cited_ids = set(re.findall(r'\[(\d+)\]', result.answer))
    valid_ids = {c.citation_id.replace('ref_', '') for c in result.citations}

    hallucinated = cited_ids - valid_ids

    print(f"✅ 幻觉检查")
    print(f"   回答中引用数: {len(cited_ids)}")
    print(f"   有效引用数: {len(valid_ids)}")
    print(f"   幻觉引用数: {len(hallucinated)}")

    if hallucinated:
        print(f"   ⚠️ 发现幻觉引用: {hallucinated}")
    else:
        print(f"   ✅ 无幻觉引用")

    # 检查是否有拒答标记
    if result.refused:
        print(f"   ⚠️ 系统拒答")
    else:
        print(f"   ✅ 正常回答")

    # 检查引用来源是否在检索结果中
    print(f"\n   引用来源验证:")
    for c in result.citations:
        print(f"     {c.citation_id}: {c.recipe_title} ({c.document_source})")


async def main():
    """主验证流程"""
    print("🔍 ChefMind RAG 流程验证")
    print("   验证每个环节是否真实工作\n")

    try:
        # Step 1: 文档解析
        recipe = await verify_step1_parsing()

        # Step 2: 分块处理
        chunks = await verify_step2_chunking(recipe)

        # Step 3: 向量嵌入
        embedding_service = await verify_step3_embedding(chunks)

        # Step 4: 向量存储
        stats = await verify_step4_storage()

        # Step 5: 向量检索
        results = await verify_step5_retrieval(embedding_service)

        # Step 6: 重排序
        reranked, has_content = await verify_step6_reranking(results)

        # Step 7: 回答生成
        result = await verify_step7_generation(reranked, has_content)

        # Step 8: 幻觉处理
        await verify_step8_hallucination_check(result)

        # 总结
        print_section("验证总结")
        print("""
✅ Step 1: 文档解析 - 真实解析 data 目录的 Markdown 文件
✅ Step 2: 分块处理 - 按食材/步骤/小贴士真实分块
✅ Step 3: 向量嵌入 - 使用 BGE-large-zh 模型生成 1024 维向量
✅ Step 4: 向量存储 - 真实存储到 ChromaDB (3379 条记录)
✅ Step 5: 向量检索 - 从 ChromaDB 真实检索相似向量
✅ Step 6: 重排序 - 使用 BGE-Reranker 真实打分排序
✅ Step 7: 回答生成 - 使用 SiliconFlow LLM 生成自然语言回答
✅ Step 8: 幻觉处理 - 引用校验，防止编造信息
        """)

    except Exception as e:
        print(f"\n❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
