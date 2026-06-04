"""
LLM配置测试脚本
"""

import sys
import os
import asyncio

# 添加backend目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# 手动加载.env文件
from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(__file__), 'backend', '.env')
load_dotenv(env_path)

sys.stdout.reconfigure(encoding='utf-8')

# 重新加载配置
from app.core.config import Settings
settings = Settings()


def check_env():
    """检查环境变量"""
    print("=== 环境变量检查 ===")
    print(f"SILICONFLOW_API_KEY: {settings.SILICONFLOW_API_KEY[:10]}..." if settings.SILICONFLOW_API_KEY else "❌ 未配置")
    print(f"SILICONFLOW_TEXT_MODEL: {settings.SILICONFLOW_TEXT_MODEL}")
    print(f"SILICONFLOW_VISION_MODEL: {settings.SILICONFLOW_VISION_MODEL}")
    print()


def test_llm_creation():
    """测试LLM创建"""
    print("=== LLM创建测试 ===")

    from app.services.llm_factory import (
        create_siliconflow_llm,
        create_siliconflow_vision_llm,
    )

    # 测试SiliconFlow文本模型
    print("\n1. 测试SiliconFlow文本模型 (DeepSeek-V3.2):")
    text_llm = create_siliconflow_llm()
    if text_llm:
        print(f"   ✅ 创建成功: {text_llm.model_name}")
    else:
        print("   ❌ 创建失败")

    # 测试SiliconFlow多模态模型
    print("\n2. 测试SiliconFlow多模态模型 (DeepSeek-OCR):")
    vision_llm = create_siliconflow_vision_llm()
    if vision_llm:
        print(f"   ✅ 创建成功: {vision_llm.model_name}")
    else:
        print("   ❌ 创建失败")

    return text_llm, vision_llm


async def test_llm_call(text_llm, vision_llm):
    """测试LLM调用"""
    print("\n=== LLM调用测试 ===")

    # 测试文本模型
    if text_llm:
        print(f"\n1. 测试文本模型 ({text_llm.model_name}):")
        try:
            from langchain_core.prompts import ChatPromptTemplate

            prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一个食谱助手。"),
                ("human", "清蒸鲈鱼需要哪些食材？请用中文简短回答。")
            ])

            chain = prompt | text_llm
            result = await chain.ainvoke({})

            print(f"   ✅ 调用成功!")
            print(f"   回答: {result.content[:200]}...")

        except Exception as e:
            print(f"   ❌ 调用失败: {e}")

    # 测试多模态模型（纯文本）
    if vision_llm:
        print(f"\n2. 测试多模态模型 ({vision_llm.model_name}):")
        try:
            from langchain_core.prompts import ChatPromptTemplate

            prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一个食谱助手。"),
                ("human", "请用中文简短介绍清蒸鲈鱼这道菜。")
            ])

            chain = prompt | vision_llm
            result = await chain.ainvoke({})

            print(f"   ✅ 调用成功!")
            print(f"   回答: {result.content[:200]}...")

        except Exception as e:
            print(f"   ❌ 调用失败: {e}")


def main():
    print("🔍 ChefMind LLM配置检查\n")

    # 检查环境变量
    check_env()

    # 测试LLM创建
    text_llm, vision_llm = test_llm_creation()

    # 测试LLM调用
    if text_llm or vision_llm:
        asyncio.run(test_llm_call(text_llm, vision_llm))

    print("\n" + "="*50)
    if text_llm:
        print("✅ LLM配置完成，可以启动完整RAG系统")
        print(f"   文本模型: {settings.SILICONFLOW_TEXT_MODEL}")
        print(f"   多模态模型: {settings.SILICONFLOW_VISION_MODEL}")
    else:
        print("❌ LLM配置失败，请检查SILICONFLOW_API_KEY")


if __name__ == "__main__":
    main()
