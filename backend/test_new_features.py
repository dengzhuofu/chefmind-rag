"""
测试新功能
验证PRD 5-8节实现的功能
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.redis import init_redis, close_redis
from app.services.conversation_memory import ConversationMemory, conversation_memory
from app.services.cache_service import memory_cache, query_rewrite_cache, retrieval_cache
from app.services.fault_tolerance import CircuitBreaker, RetryHandler, FallbackStrategies
from app.services.evaluation_service import evaluation_service


async def test_conversation_memory():
    """测试对话记忆（Redis）"""
    print("\n=== 测试对话记忆（Redis） ===")

    memory = ConversationMemory(max_turns=5)

    # 添加消息
    await memory.add_message("test_session1", "user", "清蒸鲈鱼怎么做？")
    await memory.add_message("test_session1", "assistant", "清蒸鲈鱼的做法是...")
    await memory.add_message("test_session1", "user", "需要哪些食材？")
    await memory.add_message("test_session1", "assistant", "需要鲈鱼、葱、姜...")

    # 获取历史
    history = await memory.get_history_as_string("test_session1")
    print(f"[OK] 对话历史: {len(history)} 字符")

    # 获取当前菜谱
    current_recipe = await memory.get_current_recipe("test_session1")
    print(f"[OK] 当前菜谱: {current_recipe}")

    # 获取摘要
    summary = await memory.get_memory_summary("test_session1")
    print(f"[OK] 会话摘要: {summary}")

    # 清除会话
    await memory.clear_session("test_session1")
    summary_after_clear = await memory.get_memory_summary("test_session1")
    print(f"[OK] 清除后: {summary_after_clear}")

    return True


async def test_cache_service():
    """测试缓存服务（Redis）"""
    print("\n=== 测试缓存服务（Redis） ===")

    # 测试基本缓存
    await memory_cache.set("test_key", "test_value", ttl=60)
    value = await memory_cache.get("test_key")
    print(f"[OK] 基本缓存: {value}")

    # 测试查询重构缓存
    await query_rewrite_cache.set("清蒸鲈鱼怎么做", ["query1", "query2"])
    cached = await query_rewrite_cache.get("清蒸鲈鱼怎么做")
    print(f"[OK] 查询重构缓存: {cached}")

    # 测试检索结果缓存
    await retrieval_cache.set("清蒸鲈鱼", "step_lookup", [{"id": 1}, {"id": 2}])
    cached_results = await retrieval_cache.get("清蒸鲈鱼", "step_lookup")
    print(f"[OK] 检索结果缓存: {cached_results}")

    # 获取缓存统计
    stats = await memory_cache.get_stats()
    print(f"[OK] 缓存统计: {stats}")

    return True


async def test_fault_tolerance():
    """测试容错机制"""
    print("\n=== 测试容错机制 ===")

    # 测试熔断器
    cb = CircuitBreaker(failure_threshold=3, reset_timeout=60, name="test")
    print(f"[OK] 熔断器状态: {cb.state}")

    # 测试重试处理器
    retry_handler = RetryHandler(max_retries=2, timeout=1.0)

    # 测试降级策略
    fallback_result = FallbackStrategies.fallback_query_rewrite("测试查询")
    print(f"[OK] 降级策略: {fallback_result}")

    return True


async def test_evaluation_service():
    """测试评估服务"""
    print("\n=== 测试评估服务 ===")

    # 获取测试用例
    test_cases = evaluation_service.test_cases
    print(f"[OK] 测试用例数量: {len(test_cases)}")

    # 测试单个评估
    if test_cases:
        test_case = test_cases[0]
        result = evaluation_service.evaluate_single(
            test_case=test_case,
            generated_answer="清蒸鲈鱼的做法是...",
            retrieved_recipes=["清蒸鲈鱼"],
            refused=False
        )
        print(f"[OK] 评估结果: {result.passed}")
        print(f"  指标: {result.metrics}")

    # 获取评估摘要
    summary = evaluation_service.get_evaluation_summary()
    print(f"[OK] 评估摘要: {summary}")

    return True


async def main():
    """主测试函数"""
    print("开始测试新功能...")

    # 初始化 Redis
    try:
        await init_redis()
        print("[OK] Redis 连接成功")
    except Exception as e:
        print(f"[FAIL] Redis 连接失败: {e}")
        print("请确保 Redis 服务已启动")
        sys.exit(1)

    tests = [
        ("对话记忆", test_conversation_memory),
        ("缓存服务", test_cache_service),
        ("容错机制", test_fault_tolerance),
        ("评估服务", test_evaluation_service),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result))
        except Exception as e:
            print(f"[FAIL] {name} 测试失败: {e}")
            results.append((name, False))

    # 清理测试数据
    try:
        from app.core.redis import get_redis
        redis = await get_redis()
        await redis.delete("session:test_session1:history", "session:test_session1:recipe")
        await redis.delete("test_key")
    except Exception:
        pass

    # 关闭 Redis
    await close_redis()

    # 打印总结
    print("\n=== 测试总结 ===")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"通过: {passed}/{total}")

    for name, result in results:
        status = "[OK]" if result else "[FAIL]"
        print(f"{status} {name}")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
