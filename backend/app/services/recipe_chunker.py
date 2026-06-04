"""
菜谱领域专用分块器
实现PRD 4.2节定义的Chunk切分策略
"""

import re
from typing import List, Optional, Any, Dict

from app.schemas.parsed_recipe import ParsedRecipe, RecipeChunk
from app.core.config import settings

# LangChain导入（可选）
try:
    from langchain_core.documents import Document
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False


class RecipeChunker:
    """
    菜谱领域专用分块器

    继承LangChain的BaseDocumentTransformer理念，实现食谱专用分块策略

    分块类型：
    - ingredient: 食材块，每项食材一个chunk
    - step: 步骤块，每个步骤一个chunk
    - tip: 小贴士块
    - image_desc: 图片描述块
    """

    # 最大token长度（PRD规定512）
    MAX_TOKENS = settings.CHUNK_MAX_TOKENS

    def chunk_recipe(self, recipe: ParsedRecipe) -> List[RecipeChunk]:
        """
        将解析后的菜谱分块

        Args:
            recipe: 解析后的菜谱数据

        Returns:
            分块列表
        """
        chunks = []
        recipe_id = self._generate_recipe_id(recipe.title)

        # 1. 生成食材块
        ingredient_chunks = self._create_ingredient_chunks(recipe, recipe_id)
        chunks.extend(ingredient_chunks)

        # 2. 生成步骤块
        step_chunks = self._create_step_chunks(recipe, recipe_id)
        chunks.extend(step_chunks)

        # 3. 生成小贴士块
        if recipe.tips:
            tip_chunk = self._create_tip_chunk(recipe, recipe_id)
            chunks.append(tip_chunk)

        # 4. 生成图片描述块（如果有图片）
        image_chunks = self._create_image_desc_chunks(recipe, recipe_id)
        chunks.extend(image_chunks)

        return chunks

    def chunk_to_langchain_documents(self, recipe: ParsedRecipe) -> List[Any]:
        """
        将菜谱转换为LangChain Document对象

        Args:
            recipe: 解析后的菜谱数据

        Returns:
            LangChain Document列表（如果langchain可用）或字典列表
        """
        chunks = self.chunk_recipe(recipe)
        documents = []

        for chunk in chunks:
            metadata = {
                "chunk_id": chunk.chunk_id,
                "recipe_id": chunk.recipe_id,
                "recipe_title": chunk.recipe_title,
                "chunk_type": chunk.chunk_type,
                "step_number": chunk.step_number,
                "document_source": chunk.document_source,
                "cooking_time": chunk.cooking_time,
                "difficulty": chunk.difficulty,
                "tags": ",".join(chunk.tags) if chunk.tags else "",
                "ingredient_list": ",".join(chunk.ingredient_list) if chunk.ingredient_list else "",
                "truncated": chunk.truncated,
            }

            if HAS_LANGCHAIN:
                doc = Document(
                    page_content=chunk.content,
                    metadata=metadata
                )
            else:
                # 返回字典格式作为备选
                doc = {
                    "page_content": chunk.content,
                    "metadata": metadata
                }

            documents.append(doc)

        return documents

    def _generate_recipe_id(self, title: str) -> str:
        """生成菜谱ID"""
        import hashlib
        hash_obj = hashlib.md5(title.encode('utf-8'))
        return f"rec_{hash_obj.hexdigest()[:8]}"

    def _create_ingredient_chunks(
        self, recipe: ParsedRecipe, recipe_id: str
    ) -> List[RecipeChunk]:
        """
        创建食材块

        PRD规范：
        - 每条前自动拼接菜名与类别（如"《麻婆豆腐》食材：豆腐 200g"）
        - 元数据包含ingredient_list数组，供关键词精确匹配
        - 内容包含难度和烹饪时长，供LLM生成回答时参考
        """
        chunks = []

        # 创建一个包含所有食材的chunk
        if recipe.ingredients:
            # 构建元数据头
            meta_parts = []
            if recipe.difficulty:
                meta_parts.append(f"难度：{recipe.difficulty}")
            if recipe.cooking_time:
                meta_parts.append(f"约{recipe.cooking_time}")
            meta_header = f"[{' | '.join(meta_parts)}]" if meta_parts else ""

            # 构建食材内容，添加菜名前缀和元数据
            ingredients_text = "\n".join([f"- {ing}" for ing in recipe.ingredients])
            content = f"《{recipe.title}》{meta_header}\n食材：\n{ingredients_text}"

            # 截断处理
            content, truncated = self._truncate_content(content)

            chunk = RecipeChunk(
                chunk_id=f"{recipe_id}_ingredient_all",
                recipe_id=recipe_id,
                recipe_title=recipe.title,
                chunk_type="ingredient",
                content=content,
                document_source=recipe.source_file,
                cooking_time=recipe.cooking_time,
                difficulty=recipe.difficulty,
                tags=recipe.tags,
                ingredient_list=recipe.ingredients,
                truncated=truncated,
            )
            chunks.append(chunk)

        return chunks

    def _create_step_chunks(
        self, recipe: ParsedRecipe, recipe_id: str
    ) -> List[RecipeChunk]:
        """
        创建步骤块

        优化策略：
        - 每个步骤 chunk 开头包含完整食材清单（带用量），使 chunk 自包含
        - 包含难度和烹饪时长元数据
        - 合并过短的连续步骤，避免碎片化
        - 步骤中的嵌套子步骤保留原始格式
        """
        chunks = []

        if not recipe.steps:
            return chunks

        # 构建食材摘要（带用量），用于嵌入每个步骤 chunk
        ingredient_summary = self._build_ingredient_summary(recipe)

        # 构建元数据头
        meta_parts = []
        if recipe.difficulty:
            meta_parts.append(f"难度：{recipe.difficulty}")
        if recipe.cooking_time:
            meta_parts.append(f"约{recipe.cooking_time}")
        meta_header = f"[{' | '.join(meta_parts)}]" if meta_parts else ""

        # 预处理：合并过短的连续步骤
        merged_steps = self._merge_short_steps(recipe.steps)

        step_number = 0
        for merged_step in merged_steps:
            step_number += 1

            # 构建步骤内容，包含完整食材清单
            content = f"《{recipe.title}》{meta_header}\n"
            content += f"食材：{ingredient_summary}\n\n"
            content += f"步骤 {step_number}：{merged_step}"

            # 附加用到的食材
            related_ingredients = self._extract_related_ingredients(merged_step, recipe.ingredients)
            if related_ingredients:
                content += f"\n用到的食材：{', '.join(related_ingredients)}"

            # 截断处理
            content, truncated = self._truncate_content(content)

            chunk = RecipeChunk(
                chunk_id=f"{recipe_id}_step_{step_number}",
                recipe_id=recipe_id,
                recipe_title=recipe.title,
                chunk_type="step",
                content=content,
                step_number=step_number,
                document_source=recipe.source_file,
                cooking_time=recipe.cooking_time,
                difficulty=recipe.difficulty,
                tags=recipe.tags,
                related_ingredients=related_ingredients,
                truncated=truncated,
            )
            chunks.append(chunk)

        return chunks

    def _build_ingredient_summary(self, recipe: ParsedRecipe) -> str:
        """
        构建精简的食材摘要（带用量），用于嵌入步骤 chunk
        例如：虾 10只, 花椒 5g, 葱 50g, 姜 20g, 黄酒 30g, 盐 3g, 冰糖 10g, 植物油

        策略：去掉括号内的详细说明，只保留食材名+用量
        """
        if not recipe.ingredients:
            return "无"

        summary_parts = []
        total_len = 0
        for ing in recipe.ingredients:
            # 去掉括号内的详细说明（中英文括号）
            cleaned = re.sub(r'[（(][^）)]*[）)]', '', ing).strip()
            # 去掉多余空格
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            # 去掉末尾的标点
            cleaned = cleaned.rstrip('，,、')

            if not cleaned:
                continue

            if total_len + len(cleaned) > 150:
                summary_parts.append("等")
                break
            summary_parts.append(cleaned)
            total_len += len(cleaned) + 2  # +2 for ", "

        return ", ".join(summary_parts)

    def _merge_short_steps(self, steps: List[str], min_len: int = 10) -> List[str]:
        """
        合并过短的连续步骤

        当连续多个步骤都很短（如"浇汁"、"完成"、"开吃"）时，
        合并为一个步骤，避免碎片化。

        Args:
            steps: 原始步骤列表
            min_len: 最短步骤长度（字符数），低于此值的步骤会被合并

        Returns:
            合并后的步骤列表
        """
        if not steps:
            return steps

        merged = []
        buffer = []

        for step in steps:
            # 计算步骤的有效内容长度（去掉缩进前缀）
            effective_len = len(step.strip())

            if effective_len < min_len:
                # 短步骤，放入缓冲区
                buffer.append(step)
            else:
                # 长步骤
                if buffer:
                    # 先处理缓冲区中的短步骤
                    if len(buffer) == 1:
                        # 只有一个短步骤，检查是否可以与当前步骤合并
                        merged.append(buffer[0])
                    else:
                        # 多个短步骤，合并为一个
                        merged.append("\n  ".join(buffer))
                    buffer = []

                merged.append(step)

        # 处理末尾的缓冲区
        if buffer:
            if len(buffer) == 1:
                merged.append(buffer[0])
            else:
                merged.append("\n  ".join(buffer))

        return merged

    def _create_tip_chunk(
        self, recipe: ParsedRecipe, recipe_id: str
    ) -> RecipeChunk:
        """
        创建小贴士块

        PRD规范：
        - 独立索引，关联所属菜谱ID
        - 包含菜名和元数据，提供完整上下文
        """
        # 构建元数据头
        meta_parts = []
        if recipe.difficulty:
            meta_parts.append(f"难度：{recipe.difficulty}")
        if recipe.cooking_time:
            meta_parts.append(f"约{recipe.cooking_time}")
        meta_header = f"[{' | '.join(meta_parts)}]" if meta_parts else ""

        # 构建小贴士内容
        content = f"《{recipe.title}》{meta_header}\n小贴士：\n{recipe.tips}"

        # 截断处理
        content, truncated = self._truncate_content(content)

        return RecipeChunk(
            chunk_id=f"{recipe_id}_tip",
            recipe_id=recipe_id,
            recipe_title=recipe.title,
            chunk_type="tip",
            content=content,
            document_source=recipe.source_file,
            cooking_time=recipe.cooking_time,
            difficulty=recipe.difficulty,
            tags=recipe.tags,
            truncated=truncated,
        )

    def _create_image_desc_chunks(
        self, recipe: ParsedRecipe, recipe_id: str
    ) -> List[RecipeChunk]:
        """
        创建图片描述块

        PRD规范：
        - 描述由多模态模型生成
        - 文本后附加原图路径
        - 元数据含image_path和image_type（成品图/步骤图）
        """
        chunks = []

        for i, image_path in enumerate(recipe.images):
            # 判断图片类型
            image_type = self._classify_image(image_path, i, len(recipe.images))

            # 构建图片描述内容
            # 注意：实际的图片描述需要调用多模态模型，这里先使用占位描述
            description = self._generate_image_placeholder(recipe.title, image_type)
            content = f"《{recipe.title}》{image_type}：{description}\n图片路径：{image_path}"

            chunk = RecipeChunk(
                chunk_id=f"{recipe_id}_image_{i}",
                recipe_id=recipe_id,
                recipe_title=recipe.title,
                chunk_type="image_desc",
                content=content,
                document_source=recipe.source_file,
                cooking_time=recipe.cooking_time,
                difficulty=recipe.difficulty,
                tags=recipe.tags,
                image_path=image_path,
                image_type=image_type,
            )
            chunks.append(chunk)

        return chunks

    def _classify_image(self, image_path: str, index: int, total: int) -> str:
        """分类图片类型"""
        path_lower = image_path.lower()

        # 根据文件名判断
        if any(keyword in path_lower for keyword in ['成品', '示例', '完成']):
            return "成品图"
        elif any(keyword in path_lower for keyword in ['步骤', '过程', '操作']):
            return "步骤图"
        elif any(keyword in path_lower for keyword in ['改刀', '摆盘', '切']):
            return "步骤图"
        else:
            # 默认根据位置判断：最后一张通常是成品图
            return "成品图" if index == total - 1 else "步骤图"

    def _generate_image_placeholder(self, title: str, image_type: str) -> str:
        """生成图片占位描述（实际应调用多模态模型）"""
        if image_type == "成品图":
            return f"{title}的成品展示图"
        else:
            return f"{title}的制作过程图"

    def _extract_related_ingredients(self, step: str, all_ingredients: List[str]) -> List[str]:
        """提取步骤中涉及的食材"""
        related = []
        step_lower = step.lower()

        for ingredient in all_ingredients:
            # 提取食材的主要名称（中文字符部分，去掉数量、括号等）
            main_name = re.match(r'^([一-鿿]+)', ingredient)
            if main_name:
                main_name = main_name.group(1)
            else:
                # 回退：取第一个空格前的部分
                main_name = ingredient.split()[0] if ingredient else ingredient

            if main_name and len(main_name) >= 1 and main_name.lower() in step_lower:
                related.append(main_name)

        return related

    def _get_step_summary(self, steps: List[str], index: int) -> Optional[str]:
        """获取步骤摘要（用于上下文增强）"""
        if 0 <= index < len(steps):
            step = steps[index]
            # 截取前50个字符作为摘要
            if len(step) > 50:
                return step[:50] + "..."
            return step
        return None

    def _truncate_content(self, content: str) -> tuple[str, bool]:
        """
        截断内容到最大token长度

        PRD规定：最大长度512 tokens，自动截断，标记truncated: true
        """
        # 简单估算：1个中文字符约等于1.5个token
        estimated_tokens = len(content) * 1.5

        if estimated_tokens <= self.MAX_TOKENS:
            return content, False

        # 截断到合适的长度
        max_chars = int(self.MAX_TOKENS / 1.5)
        truncated_content = content[:max_chars] + "..."
        return truncated_content, True


# 创建全局分块器实例
recipe_chunker = RecipeChunker()
