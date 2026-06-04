"""
LLM工厂
创建和管理LLM实例
"""

import logging
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


def create_llm(provider: str = "siliconflow", temperature: float = 0.1):
    """
    创建LLM实例

    Args:
        provider: 提供商 (siliconflow/deepseek/qwen/openai)
        temperature: 温度参数

    Returns:
        LLM实例
    """
    if provider == "siliconflow":
        return create_siliconflow_llm(temperature)
    elif provider == "deepseek":
        return create_deepseek_llm(temperature)
    elif provider == "qwen":
        return create_qwen_llm(temperature)
    elif provider == "openai":
        return create_openai_llm(temperature)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def create_siliconflow_llm(temperature: float = 0.1, model: Optional[str] = None):
    """
    创建SiliconFlow LLM

    SiliconFlow兼容OpenAI API格式
    需要配置: SILICONFLOW_API_KEY
    """
    if not settings.SILICONFLOW_API_KEY:
        logger.warning("SILICONFLOW_API_KEY not configured")
        return None

    try:
        from langchain_openai import ChatOpenAI

        model_name = model or settings.SILICONFLOW_TEXT_MODEL

        llm = ChatOpenAI(
            model=model_name,
            api_key=settings.SILICONFLOW_API_KEY,
            base_url=settings.SILICONFLOW_API_BASE,
            temperature=temperature,
            max_tokens=800,
            streaming=True,
        )
        logger.info(f"SiliconFlow LLM created: {model_name}")
        return llm
    except Exception as e:
        logger.error(f"Failed to create SiliconFlow LLM: {e}")
        return None


def create_siliconflow_vision_llm(temperature: float = 0.1):
    """
    创建SiliconFlow多模态LLM（用于图片处理）

    使用DeepSeek-OCR模型
    """
    return create_siliconflow_llm(temperature, model=settings.SILICONFLOW_VISION_MODEL)


def create_deepseek_llm(temperature: float = 0.1):
    """
    创建DeepSeek LLM

    需要配置: DEEPSEEK_API_KEY
    """
    if not settings.DEEPSEEK_API_KEY:
        logger.warning("DEEPSEEK_API_KEY not configured")
        return None

    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_API_BASE,
            temperature=temperature,
            max_tokens=800,
            streaming=True,
        )
        logger.info("DeepSeek LLM created")
        return llm
    except Exception as e:
        logger.error(f"Failed to create DeepSeek LLM: {e}")
        return None


def create_qwen_llm(temperature: float = 0.1):
    """
    创建通义千问 LLM

    需要配置: DASHSCOPE_API_KEY
    """
    if not settings.DASHSCOPE_API_KEY:
        logger.warning("DASHSCOPE_API_KEY not configured")
        return None

    try:
        from langchain_community.chat_models import ChatTongyi

        llm = ChatTongyi(
            model_name="qwen-max",
            dashscope_api_key=settings.DASHSCOPE_API_KEY,
            temperature=temperature,
            max_tokens=800,
            streaming=True,
        )
        logger.info("Qwen LLM created")
        return llm
    except Exception as e:
        logger.error(f"Failed to create Qwen LLM: {e}")
        return None


def create_openai_llm(temperature: float = 0.1):
    """
    创建OpenAI LLM

    需要配置: OPENAI_API_KEY
    """
    if not settings.OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not configured")
        return None

    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=settings.OPENAI_API_KEY,
            temperature=temperature,
            max_tokens=800,
            streaming=True,
        )
        logger.info("OpenAI LLM created")
        return llm
    except Exception as e:
        logger.error(f"Failed to create OpenAI LLM: {e}")
        return None


def get_llm_for_answer():
    """获取用于回答生成的LLM"""
    # 优先使用SiliconFlow
    llm = create_siliconflow_llm(temperature=0.1)
    if llm:
        return llm

    # 备选：DeepSeek
    llm = create_deepseek_llm(temperature=0.1)
    if llm:
        return llm

    # 备选：通义千问
    llm = create_qwen_llm(temperature=0.1)
    if llm:
        return llm

    # 备选：OpenAI
    llm = create_openai_llm(temperature=0.1)
    if llm:
        return llm

    logger.warning("No LLM available, using rule-based generation")
    return None


def get_llm_for_rewrite():
    """获取用于查询重构的LLM（低延迟）"""
    # 使用SiliconFlow
    llm = create_siliconflow_llm(temperature=0.3)
    if llm:
        return llm

    # 备选：DeepSeek
    llm = create_deepseek_llm(temperature=0.3)
    if llm:
        return llm

    return None


def get_llm_for_vision():
    """获取用于多模态处理的LLM"""
    # 使用SiliconFlow的DeepSeek-OCR
    llm = create_siliconflow_vision_llm(temperature=0.1)
    if llm:
        return llm

    # 备选：OpenAI
    llm = create_openai_llm(temperature=0.1)
    if llm:
        return llm

    return None
