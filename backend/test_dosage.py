"""测试查询用量"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.rag_pipeline import rag_pipeline
import asyncio

async def test():
    result = await rag_pipeline.run(
        query='清蒸鲈鱼需要多少食用油？',
        session_id='test_dosage',
        request_id='test_001'
    )

    # 写入文件避免编码问题
    with open('test_result.txt', 'w', encoding='utf-8') as f:
        f.write(f"Answer:\n{result.answer}\n\n")
        f.write(f"Citations: {len(result.citations)}\n")
        for c in result.citations:
            f.write(f"  - {c.recipe_title}: {c.excerpt[:50]}...\n")

    print("Result saved to test_result.txt")

asyncio.run(test())
