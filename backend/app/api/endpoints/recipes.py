"""
菜谱API端点
实现PRD第6节定义的多模态处理流程
"""

import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from typing import Optional, List
import os

from app.services.multimodal_service import multimodal_service
from app.services.rag_pipeline import rag_pipeline

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/recipes")
async def get_recipes(
    search: Optional[str] = Query(None, description="搜索关键词"),
    difficulty: Optional[str] = Query(None, description="难度筛选"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量")
):
    """
    获取食谱列表

    从ChromaDB中获取所有食谱信息，去重并返回

    Args:
        search: 搜索关键词（菜谱名称或食材）
        difficulty: 难度筛选
        page: 页码
        page_size: 每页数量

    Returns:
        食谱列表
    """
    try:
        from app.core.milvus import get_chroma_client

        client = get_chroma_client()
        collection = client.get_collection("recipes")

        # 获取所有数据
        results = collection.get(
            include=["documents", "metadatas"]
        )

        # 提取唯一的菜谱
        recipes_map = {}
        if results and results["ids"]:
            for i, doc_id in enumerate(results["ids"]):
                metadata = results["metadatas"][i] if results["metadatas"] else {}
                document = results["documents"][i] if results["documents"] else ""

                recipe_title = metadata.get("recipe_title", "未知菜谱")
                recipe_id = metadata.get("recipe_id", doc_id)

                # 去重：每个菜谱只保留一次
                if recipe_id not in recipes_map:
                    # 从文档路径提取额外信息
                    doc_source = metadata.get("document_source", "")

                    # 提取食材（从ingredient类型的chunk）
                    ingredients = []
                    if metadata.get("chunk_type") == "ingredient":
                        # 解析食材
                        lines = document.split("\n")
                        for line in lines:
                            line = line.strip()
                            if line.startswith("- ") or line.startswith("• "):
                                ingredients.append(line[2:].strip())

                    recipes_map[recipe_id] = {
                        "id": recipe_id,
                        "title": recipe_title,
                        "description": "",
                        "ingredients": ingredients,
                        "cooking_time": metadata.get("cooking_time", ""),
                        "difficulty": metadata.get("difficulty", "中等"),
                        "tags": metadata.get("tags", []),
                        "source_file": doc_source,
                        "chunk_count": 0
                    }

                # 更新chunk计数
                recipes_map[recipe_id]["chunk_count"] += 1

                # 如果当前chunk是step类型，提取描述
                if metadata.get("chunk_type") == "step" and not recipes_map[recipe_id]["description"]:
                    # 提取前100个字符作为描述
                    desc = document[:150].replace("\n", " ").strip()
                    if desc:
                        recipes_map[recipe_id]["description"] = desc + "..."

        # 转换为列表
        recipes_list = list(recipes_map.values())

        # 应用搜索筛选
        if search:
            search_lower = search.lower()
            recipes_list = [
                r for r in recipes_list
                if search_lower in r["title"].lower()
                or any(search_lower in ing.lower() for ing in r["ingredients"])
                or search_lower in r.get("description", "").lower()
            ]

        # 应用难度筛选
        if difficulty and difficulty != "all":
            recipes_list = [
                r for r in recipes_list
                if r.get("difficulty") == difficulty
            ]

        # 计算总数
        total = len(recipes_list)

        # 应用分页
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_recipes = recipes_list[start_idx:end_idx]

        return {
            "recipes": paginated_recipes,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size
        }

    except Exception as e:
        logger.error(f"Get recipes error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recipes/{recipe_id}")
async def get_recipe_detail(recipe_id: str):
    """
    获取菜谱详情

    Args:
        recipe_id: 菜谱ID

    Returns:
        菜谱详情
    """
    try:
        from app.core.milvus import get_chroma_client

        client = get_chroma_client()
        collection = client.get_collection("recipes")

        # 获取该菜谱的所有chunks
        results = collection.get(
            where={"recipe_id": recipe_id},
            include=["documents", "metadatas"]
        )

        if not results or not results["ids"]:
            raise HTTPException(status_code=404, detail="Recipe not found")

        # 组织数据
        recipe_info = {
            "id": recipe_id,
            "title": "",
            "description": "",
            "ingredients": [],
            "steps": [],
            "tips": [],
            "images": [],
            "cooking_time": "",
            "difficulty": "中等",
            "tags": [],
            "source_file": ""
        }

        for i, doc_id in enumerate(results["ids"]):
            metadata = results["metadatas"][i] if results["metadatas"] else {}
            document = results["documents"][i] if results["documents"] else ""

            if not recipe_info["title"]:
                recipe_info["title"] = metadata.get("recipe_title", "未知菜谱")

            if not recipe_info["source_file"]:
                recipe_info["source_file"] = metadata.get("document_source", "")

            chunk_type = metadata.get("chunk_type", "")

            if chunk_type == "ingredient":
                # 解析食材
                lines = document.split("\n")
                for line in lines:
                    line = line.strip()
                    if line.startswith("- ") or line.startswith("• "):
                        ingredient = line[2:].strip()
                        if ingredient and ingredient not in recipe_info["ingredients"]:
                            recipe_info["ingredients"].append(ingredient)

            elif chunk_type == "step":
                step_num = metadata.get("step_number", len(recipe_info["steps"]) + 1)
                # 提取步骤内容（去掉标题）
                lines = document.split("\n")
                step_content = ""
                for line in lines:
                    if not line.startswith("#") and not line.startswith("步骤"):
                        step_content += line.strip() + " "

                recipe_info["steps"].append({
                    "number": step_num,
                    "content": step_content.strip()
                })

                # 用第一个步骤作为描述
                if not recipe_info["description"]:
                    recipe_info["description"] = step_content.strip()[:150] + "..."

            elif chunk_type == "tip":
                recipe_info["tips"].append(document.strip())

            elif chunk_type == "image_desc":
                image_path = metadata.get("image_path", "")
                if image_path:
                    recipe_info["images"].append({
                        "path": image_path,
                        "description": document.strip()
                    })

        # 按步骤号排序
        recipe_info["steps"].sort(key=lambda x: x["number"])

        return recipe_info

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get recipe detail error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recipes/upload")
async def upload_recipe_image(
    file: UploadFile = File(...),
    recipe_id: Optional[str] = Form(None),
    recipe_title: Optional[str] = Form(None)
):
    """
    上传菜谱图片

    PRD规范：
    1. 通过/recipes/upload接收图片
    2. 存储到MinIO（本地开发用文件系统）
    3. 异步处理：调用多模态模型生成图片描述
    4. 生成结构化JSON
    5. 将描述文本作为image_desc类型的chunk进行嵌入和索引

    Args:
        file: 上传的图片文件
        recipe_id: 关联的菜谱ID（可选）
        recipe_title: 菜谱名称（可选）

    Returns:
        上传结果
    """
    try:
        # 验证文件类型
        allowed_types = ["image/jpeg", "image/png", "image/webp", "image/gif"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}. Allowed: {allowed_types}"
            )

        # 读取文件内容
        file_content = await file.read()

        # 检查文件大小（最大10MB）
        max_size = 10 * 1024 * 1024  # 10MB
        if len(file_content) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {max_size / 1024 / 1024}MB"
            )

        # 保存图片
        image_path = multimodal_service.save_uploaded_image(
            file_content=file_content,
            filename=file.filename,
            recipe_id=recipe_id
        )

        # 处理图片并生成描述
        result = await multimodal_service.process_and_index_image(
            image_path=image_path,
            recipe_id=recipe_id,
            recipe_title=recipe_title
        )

        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"Image processing failed: {result.get('error', 'Unknown error')}"
            )

        return {
            "status": "success",
            "image_path": image_path,
            "recipe_id": recipe_id,
            "recipe_title": result.get("recipe_title"),
            "chunk_content": result.get("chunk_content"),
            "message": "Image uploaded and processed successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recipes/{recipe_id}/images")
async def get_recipe_images(recipe_id: str):
    """
    获取菜谱的图片列表

    Args:
        recipe_id: 菜谱ID

    Returns:
        图片列表
    """
    try:
        upload_dir = os.path.join(multimodal_service.upload_dir, recipe_id)
        if not os.path.exists(upload_dir):
            return {"images": []}

        images = []
        for filename in os.listdir(upload_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.gif')):
                image_path = os.path.join(upload_dir, filename)
                images.append({
                    "filename": filename,
                    "path": image_path,
                    "url": multimodal_service.get_image_url(image_path)
                })

        return {"images": images}

    except Exception as e:
        logger.error(f"Get images error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
