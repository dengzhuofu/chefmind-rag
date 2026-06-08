"""
多模态处理服务
实现PRD第6节定义的多模态处理流程
"""

import logging
import os
import uuid
from typing import Dict, Any, Optional, List
from pathlib import Path
import base64

from app.core.config import settings
from app.services.llm_factory import get_llm_for_vision

logger = logging.getLogger(__name__)


class MultimodalService:
    """
    多模态处理服务

    PRD规范：
    1. 图片上传：通过/recipes/upload接收，存至MinIO（本地开发用文件系统）
    2. 异步任务：调用多模态模型生成图片描述
    3. 生成结构化JSON：{"title": "...", "ingredients": [...], "description": "..."}
    4. 将描述文本作为image_desc类型的chunk进行嵌入和索引
    5. 多模态模型降级：若API调用失败，回退为OCR文本
    """

    # 图片描述提示词
    IMAGE_DESCRIPTION_PROMPT = """请详细描述这张图片中的菜品，包括：
1. 菜品名称（如果是可识别的菜品）
2. 外观特征（颜色、摆盘、装饰等）
3. 可能的食材（从图片中可见的）
4. 可能的口味特点
5. 烹饪方式（如果可以判断）

请以JSON格式返回：
{
    "title": "菜品名称",
    "ingredients": ["食材1", "食材2", ...],
    "description": "详细描述",
    "appearance": "外观描述",
    "cooking_method": "可能的烹饪方式"
}

如果无法识别为菜品，请返回：
{
    "title": "未知",
    "ingredients": [],
    "description": "图片描述",
    "appearance": "外观描述",
    "cooking_method": "未知"
}"""

    def __init__(self):
        """初始化多模态服务"""
        self.upload_dir = Path(settings.UPLOAD_DIR if hasattr(settings, 'UPLOAD_DIR') else "./uploads")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.vision_llm = None

    def _get_vision_llm(self):
        """获取视觉LLM"""
        if self.vision_llm is None:
            try:
                self.vision_llm = get_llm_for_vision()
            except Exception as e:
                logger.error(f"Failed to initialize vision LLM: {e}")
        return self.vision_llm

    async def process_image(
        self,
        image_path: str,
        recipe_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        处理图片，生成描述

        Args:
            image_path: 图片路径
            recipe_id: 关联的菜谱ID（可选）

        Returns:
            图片描述结果
        """
        try:
            # 读取图片
            with open(image_path, "rb") as f:
                image_data = f.read()

            # 尝试使用多模态LLM生成描述
            description = await self._generate_description_with_llm(image_data)

            if description:
                return {
                    "success": True,
                    "image_path": image_path,
                    "recipe_id": recipe_id,
                    "description": description,
                    "method": "vision_llm"
                }
            else:
                # 降级：返回基本描述
                return {
                    "success": True,
                    "image_path": image_path,
                    "recipe_id": recipe_id,
                    "description": {
                        "title": "菜品图片",
                        "ingredients": [],
                        "description": f"图片文件：{os.path.basename(image_path)}",
                        "appearance": "无法识别",
                        "cooking_method": "未知"
                    },
                    "method": "fallback"
                }

        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return {
                "success": False,
                "image_path": image_path,
                "error": str(e)
            }

    async def _generate_description_with_llm(
        self,
        image_data: bytes
    ) -> Optional[Dict[str, Any]]:
        """
        使用LLM生成图片描述

        Args:
            image_data: 图片数据

        Returns:
            描述结果
        """
        try:
            vision_llm = self._get_vision_llm()
            if not vision_llm:
                logger.warning("Vision LLM not available")
                return None

            # 将图片转为base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')

            # 构建消息
            from langchain_core.messages import HumanMessage

            message = HumanMessage(
                content=[
                    {"type": "text", "text": self.IMAGE_DESCRIPTION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            )

            # 调用LLM
            response = await vision_llm.ainvoke([message])

            # 解析JSON响应
            import json
            try:
                # 尝试从响应中提取JSON
                content = response.content
                # 查找JSON块
                start = content.find('{')
                end = content.rfind('}') + 1
                if start != -1 and end != -1:
                    json_str = content[start:end]
                    description = json.loads(json_str)
                    return description
            except json.JSONDecodeError:
                logger.warning("Failed to parse vision LLM response as JSON")

            return None

        except Exception as e:
            logger.error(f"Vision LLM generation failed: {e}")
            return None

    def save_uploaded_image(
        self,
        file_content: bytes,
        filename: str,
        recipe_id: Optional[str] = None
    ) -> str:
        """
        保存上传的图片

        Args:
            file_content: 文件内容
            filename: 文件名
            recipe_id: 关联的菜谱ID

        Returns:
            保存的文件路径
        """
        # 生成唯一文件名
        ext = Path(filename).suffix or ".jpg"
        unique_filename = f"{uuid.uuid4().hex}{ext}"

        # 创建目录
        if recipe_id:
            save_dir = self.upload_dir / recipe_id
        else:
            save_dir = self.upload_dir / "general"
        save_dir.mkdir(parents=True, exist_ok=True)

        # 保存文件
        file_path = save_dir / unique_filename
        with open(file_path, "wb") as f:
            f.write(file_content)

        logger.info(f"Saved image: {file_path}")
        return str(file_path)

    async def process_and_index_image(
        self,
        image_path: str,
        recipe_id: Optional[str] = None,
        recipe_title: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        处理图片并索引到向量数据库

        Args:
            image_path: 图片路径
            recipe_id: 菜谱ID
            recipe_title: 菜谱名称

        Returns:
            处理结果
        """
        # 处理图片
        result = await self.process_image(image_path, recipe_id)

        if not result["success"]:
            return result

        # 生成chunk内容
        description = result["description"]
        chunk_content = f"菜品图片描述：{description.get('title', '未知')}。"
        chunk_content += f"外观：{description.get('appearance', '未知')}。"
        chunk_content += f"食材：{'、'.join(description.get('ingredients', []))}。"
        chunk_content += f"描述：{description.get('description', '')}。"

        # 返回chunk数据（由调用者决定是否存储到向量数据库）
        return {
            "success": True,
            "image_path": image_path,
            "recipe_id": recipe_id,
            "recipe_title": recipe_title or description.get("title", "未知"),
            "chunk_content": chunk_content,
            "chunk_type": "image_desc",
            "metadata": {
                "image_path": image_path,
                "recipe_id": recipe_id,
                "recipe_title": recipe_title,
                "document_source": os.path.basename(image_path),
                "description": description
            }
        }

    # 食材识别提示词
    INGREDIENT_IDENTIFICATION_PROMPT = """请识别这张图片中的食材或菜品，并列出可用于烹饪的食材。

请以JSON格式返回：
{
    "ingredients": ["食材1", "食材2", ...],
    "title": "识别到的菜品或主要食材",
    "description": "简要描述图片内容"
}

要求：
- ingredients 列出所有可识别的食材名称（如：鸡蛋、番茄、牛肉、青椒等）
- 如果图片是一道成品菜，列出该菜的主要食材
- 如果图片是生食材，直接列出食材名称
- 只返回JSON，不要其他文字"""

    async def identify_ingredients(self, image_base64: str) -> Dict[str, Any]:
        """
        识别图片中的食材

        Args:
            image_base64: Base64编码的图片数据

        Returns:
            {"ingredients": [...], "title": "...", "description": "..."}
        """
        try:
            vision_llm = self._get_vision_llm()
            if not vision_llm:
                logger.warning("Vision LLM not available for ingredient identification")
                return {"ingredients": [], "title": "未知", "description": "视觉模型不可用"}

            from langchain_core.messages import HumanMessage

            message = HumanMessage(
                content=[
                    {"type": "text", "text": self.INGREDIENT_IDENTIFICATION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            )

            response = await vision_llm.ainvoke([message])

            # 解析JSON响应
            import json
            content = response.content
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end != -1:
                json_str = content[start:end]
                result = json.loads(json_str)
                return {
                    "ingredients": result.get("ingredients", []),
                    "title": result.get("title", "未知"),
                    "description": result.get("description", "")
                }

            return {"ingredients": [], "title": "未知", "description": "无法解析识别结果"}

        except Exception as e:
            logger.error(f"Ingredient identification failed: {e}")
            return {"ingredients": [], "title": "未知", "description": f"识别失败: {str(e)}"}

    def get_image_url(self, image_path: str) -> str:
        """
        获取图片URL

        Args:
            image_path: 图片路径

        Returns:
            图片URL
        """
        # 本地开发：返回相对路径
        # 生产环境：返回MinIO URL
        return f"/uploads/{os.path.basename(image_path)}"


# 全局实例
multimodal_service = MultimodalService()
