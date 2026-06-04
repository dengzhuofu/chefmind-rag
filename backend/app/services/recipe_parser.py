
"""
菜谱Markdown解析器
实现PRD 4.1节定义的文档处理流水线
"""

import re
import os
from pathlib import Path
from typing import List, Optional, Tuple
import hashlib

from app.schemas.parsed_recipe import ParsedRecipe


class RecipeParser:
    """
    Markdown菜谱解析器
    支持解析标准格式的菜谱文件，提取结构化信息
    """

    # 难度映射
    DIFFICULTY_MAP = {
        "★": "简单",
        "★★": "简单",
        "★★★": "中等",
        "★★★★": "困难",
        "★★★★★": "困难",
    }

    # 图片正则
    IMAGE_PATTERN = re.compile(r'!\[.*?\]\((.*?)\)')

    # 步骤序号正则
    STEP_NUMBER_PATTERN = re.compile(r'^\d+[.、)]\s*')

    def parse_file(self, file_path: str) -> ParsedRecipe:
        """
        解析单个Markdown菜谱文件

        Args:
            file_path: Markdown文件路径

        Returns:
            解析后的结构化菜谱数据
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return self.parse_content(content, file_path)

    def parse_content(self, content: str, source_file: str) -> ParsedRecipe:
        """
        解析Markdown内容

        Args:
            content: Markdown文本内容
            source_file: 来源文件名

        Returns:
            解析后的结构化菜谱数据
        """
        # 提取各个部分
        title = self._extract_title(content, source_file)
        description = self._extract_description(content)
        difficulty = self._extract_difficulty(content)
        ingredients = self._extract_ingredients(content)
        steps = self._extract_steps(content)
        tips = self._extract_tips(content)
        images = self._extract_images(content)
        cooking_time = self._estimate_cooking_time(steps, content)
        tags = self._generate_tags(title, ingredients, content)

        return ParsedRecipe(
            title=title,
            description=description,
            ingredients=ingredients,
            steps=steps,
            tips=tips,
            cooking_time=cooking_time,
            difficulty=difficulty,
            tags=tags,
            raw_text=content,
            images=images,
            source_file=source_file
        )

    def parse_directory(self, directory: str) -> List[ParsedRecipe]:
        """
        递归解析目录下所有Markdown文件

        Args:
            directory: 目录路径

        Returns:
            解析后的菜谱列表
        """
        recipes = []
        dir_path = Path(directory)

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        # 递归查找所有.md文件
        for md_file in dir_path.rglob("*.md"):
            try:
                recipe = self.parse_file(str(md_file))
                recipes.append(recipe)
            except Exception as e:
                print(f"Error parsing {md_file}: {e}")
                continue

        return recipes

    def generate_recipe_id(self, title: str) -> str:
        """
        生成菜谱唯一ID

        Args:
            title: 菜谱名称

        Returns:
            唯一ID
        """
        # 使用标题的MD5哈希作为ID
        hash_obj = hashlib.md5(title.encode('utf-8'))
        return f"rec_{hash_obj.hexdigest()[:8]}"

    def _extract_title(self, content: str, source_file: str) -> str:
        """提取标题"""
        # 尝试从H1标题提取
        h1_match = re.search(r'^#\s+(.+?)(?:\s*的做法)?$', content, re.MULTILINE)
        if h1_match:
            return h1_match.group(1).strip()

        # 尝试从H2标题提取
        h2_match = re.search(r'^##\s+(.+?)(?:\s*的做法)?$', content, re.MULTILINE)
        if h2_match:
            return h2_match.group(1).strip()

        # 从文件名提取
        filename = Path(source_file).stem
        return filename.replace("_", " ").replace("-", " ")

    def _extract_description(self, content: str) -> Optional[str]:
        """提取描述/简介"""
        # 移除标题行
        lines = content.split('\n')
        desc_lines = []
        in_header = True

        for line in lines:
            stripped = line.strip()

            # 跳过标题行和图片行
            if in_header:
                if stripped.startswith('#') or not stripped:
                    continue
                if stripped.startswith('!['):
                    continue
                if '预估烹饪难度' in stripped:
                    in_header = False
                    continue
                in_header = False

            # 遇到章节标题停止
            if stripped.startswith('##'):
                break

            # 收集描述行
            if stripped and not stripped.startswith('!['):
                desc_lines.append(stripped)

        return '\n'.join(desc_lines) if desc_lines else None

    def _extract_difficulty(self, content: str) -> Optional[str]:
        """提取难度等级"""
        # 匹配 "预估烹饪难度：★★★" 格式
        match = re.search(r'预估烹饪难度[：:]\s*(★+)', content)
        if match:
            stars = match.group(1)
            return self.DIFFICULTY_MAP.get(stars, "中等")

        # 匹配 "难度：简单/中等/困难" 格式
        match = re.search(r'难度[：:]\s*(简单|中等|困难)', content)
        if match:
            return match.group(1)

        return None

    def _extract_ingredients(self, content: str) -> List[str]:
        """提取食材列表（包含用量）"""
        ingredients = []

        # 优先查找"计算"章节（包含用量信息）
        calc_match = re.search(
            r'##\s*计算\s*\n(.*?)(?=\n##[^#]|\Z)',
            content,
            re.DOTALL
        )

        if calc_match:
            section = calc_match.group(1)
            # 提取列表项（保留用量）
            # 支持两种格式：
            # 1. - 鲈鱼 一条
            # 2. - 手枪腿（或者鸡胸脯肉） = 1 支（约 350g）
            in_subsection = False
            for line in section.split('\n'):
                line = line.strip()
                # 跳过说明文字和空行
                if not line or line.startswith('注意') or line.startswith('使用') or line.startswith('每份'):
                    continue
                # 处理缩进的子列表项（如：  - 大葱 = 1 根）
                if line.startswith('  -') or line.startswith('  *'):
                    item = line.lstrip(' -*').strip()
                    if item and not item.startswith('必须') and not item.startswith('进阶') and not item.startswith('可选'):
                        ingredients.append(item)
                elif line.startswith('-') or line.startswith('*'):
                    item = line.lstrip('-* ').strip()
                    # 跳过分类标题和说明文字
                    if item and not item.startswith('注意') and not item.startswith('使用'):
                        # 检查是否是分类标题（如"必须配料"、"进阶配料"）
                        is_category = item in ['必须配料', '进阶配料', '可选配料']
                        if not is_category:
                            ingredients.append(item)

        # 如果"计算"章节没有找到或为空，回退到"必备原料和工具"章节
        if not ingredients:
            section_match = re.search(
                r'##\s*(?:必备原料和工具|食材|原料|材料)\s*\n(.*?)(?=\n##[^#]|\Z)',
                content,
                re.DOTALL
            )

            if section_match:
                section = section_match.group(1)
                # 提取列表项
                for line in section.split('\n'):
                    line = line.strip()
                    if line.startswith('-') or line.startswith('*'):
                        # 移除列表标记
                        item = line.lstrip('-* ').strip()
                        # 移除括号内的说明
                        item = re.sub(r'[（(].*?[）)]', '', item).strip()
                        if item:
                            ingredients.append(item)

        return ingredients

    def _clean_ingredient(self, item: str) -> str:
        """清理食材名称，移除数量描述"""
        # 移除数量和单位
        patterns = [
            r'\d+\.?\d*\s*(克|g|kg|毫升|ml|L|升|个|根|块|片|瓣|只|条|勺|调羹|适量)',
            r'[（(].*?[）)]',
            r'（别称：.*?）',
            r'\*.*$',
        ]

        for pattern in patterns:
            item = re.sub(pattern, '', item)

        # 移除"可选"标记
        item = item.replace('（可选）', '').replace('(可选)', '')

        return item.strip()

    def _extract_steps(self, content: str) -> List[str]:
        """
        提取操作步骤（支持嵌套列表项合并）

        算法：
        1. 逐行扫描 ## 操作 章节
        2. 识别 ### 子标题 → 记录当前阶段名
        3. 顶级列表项（无缩进）→ 创建新步骤
        4. 缩进列表项（2空格/4空格）→ 追加到当前父步骤
        5. 过滤纯图片行
        6. 清理 markdown 格式
        """
        steps = []

        # 查找"操作"或"做法"章节
        # 使用 (?=\n##[^#]|\Z) 避免匹配到 ### 子标题
        section_match = re.search(
            r'##\s*(?:操作|做法|步骤|烹饪步骤)\s*\n(.*?)(?=\n##[^#]|\Z)',
            content,
            re.DOTALL
        )

        if not section_match:
            return steps

        section = section_match.group(1)
        current_step = None  # 当前正在构建的步骤

        for line in section.split('\n'):
            # 跳过空行
            if not line.strip():
                continue

            # 计算缩进层级
            indent = len(line) - len(line.lstrip())
            stripped = line.strip()

            # 跳过 ### 子标题（作为阶段分隔，不影响步骤）
            if stripped.startswith('###'):
                continue

            # 跳过纯图片行
            if re.match(r'^!\[.*?\]\(.*?\)$', stripped):
                continue

            # 跳过引用块中的提示（> 开头的行归入上一步）
            if stripped.startswith('>'):
                if current_step is not None:
                    note = stripped.lstrip('> ').strip()
                    if note:
                        current_step += f"\n  {note}"
                continue

            # 判断是否是列表项
            is_list_item = False
            item_text = None

            # 无缩进的列表项 → 新步骤
            if indent == 0:
                if stripped.startswith(('- ', '* ')):
                    is_list_item = True
                    item_text = stripped[2:].strip()
                elif re.match(r'^\d+[.、)]\s*', stripped):
                    is_list_item = True
                    item_text = re.sub(r'^\d+[.、)]\s*', '', stripped).strip()

            # 缩进的列表项 → 子步骤，追加到父步骤
            elif indent >= 2:
                if stripped.startswith(('- ', '* ')):
                    is_list_item = True
                    item_text = stripped[2:].strip()
                elif re.match(r'^\d+[.、)]\s*', stripped):
                    is_list_item = True
                    item_text = re.sub(r'^\d+[.、)]\s*', '', stripped).strip()

            if is_list_item and item_text:
                # 移除图片标记
                item_text = re.sub(r'!\[.*?\]\(.*?\)', '', item_text).strip()
                # 移除 markdown 链接格式，保留文字
                item_text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', item_text)
                # 移除加粗标记
                item_text = item_text.replace('**', '')

                if not item_text:
                    continue

                if indent == 0:
                    # 顶级列表项 → 保存上一步，开始新步骤
                    if current_step is not None:
                        steps.append(current_step)
                    current_step = item_text
                else:
                    # 缩进列表项 → 追加到当前父步骤
                    if current_step is not None:
                        current_step += f"\n  {item_text}"

        # 保存最后一个步骤
        if current_step is not None:
            steps.append(current_step)

        return steps

    def _extract_tips(self, content: str) -> Optional[str]:
        """提取小贴士/附加内容"""
        # 查找"附加内容"或"小贴士"章节
        section_match = re.search(
            r'##\s*(?:附加内容|小贴士|注意|提示|技术总结)\s*\n(.*?)(?=\n##[^#]|\Z)',
            content,
            re.DOTALL
        )

        if section_match:
            section = section_match.group(1).strip()
            # 清理格式
            lines = []
            for line in section.split('\n'):
                stripped = line.strip()
                if stripped.startswith('-') or stripped.startswith('*'):
                    stripped = stripped.lstrip('-* ').strip()
                if stripped:
                    lines.append(stripped)
            return '\n'.join(lines)

        return None

    def _extract_images(self, content: str) -> List[str]:
        """提取图片路径"""
        return self.IMAGE_PATTERN.findall(content)

    def _estimate_cooking_time(self, steps: List[str], content: str) -> Optional[str]:
        """估算烹饪时长"""
        # 尝试从内容中提取明确的时间
        time_match = re.search(r'(?:烹饪时间|时长|用时)[：:]\s*(\d+\s*(?:分钟|小时))', content)
        if time_match:
            return time_match.group(1)

        # 根据步骤数量估算
        if len(steps) <= 3:
            return "15分钟"
        elif len(steps) <= 6:
            return "30分钟"
        elif len(steps) <= 10:
            return "45分钟"
        else:
            return "60分钟"

    def _generate_tags(self, title: str, ingredients: List[str], content: str) -> List[str]:
        """生成标签"""
        tags = []

        # 根据食材生成标签
        ingredient_tags = {
            "鱼": "海鲜",
            "虾": "海鲜",
            "蟹": "海鲜",
            "鸡": "禽类",
            "鸭": "禽类",
            "猪": "猪肉",
            "牛": "牛肉",
            "羊": "羊肉",
            "豆腐": "素食",
            "蔬菜": "素食",
        }

        content_lower = content.lower()
        for keyword, tag in ingredient_tags.items():
            if keyword in content_lower:
                tags.append(tag)

        # 根据烹饪方式生成标签
        cooking_methods = {
            "蒸": "清蒸",
            "炒": "炒菜",
            "煮": "煮菜",
            "炖": "炖菜",
            "烤": "烧烤",
            "煎": "煎炸",
            "炸": "煎炸",
            "凉拌": "凉菜",
        }

        for keyword, tag in cooking_methods.items():
            if keyword in content_lower:
                tags.append(tag)

        # 去重
        return list(set(tags))


# 创建全局解析器实例
recipe_parser = RecipeParser()
