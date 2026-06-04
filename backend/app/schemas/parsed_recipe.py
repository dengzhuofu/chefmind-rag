"""解析后的菜谱结构化数据模型 - 符合PRD 4.1节规范"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class ParsedRecipe(BaseModel):
    """
    解析后的菜谱结构化JSON
    符合PRD 4.1节定义的标准格式
    """
    title: str = Field(..., description="菜谱名称")
    description: Optional[str] = Field(None, description="菜谱简介/描述")
    ingredients: List[str] = Field(default_factory=list, description="食材列表")
    steps: List[str] = Field(default_factory=list, description="步骤列表")
    tips: Optional[str] = Field(None, description="小贴士/附加内容")
    cooking_time: Optional[str] = Field(None, description="烹饪时长")
    difficulty: Optional[str] = Field(None, description="难度等级")
    tags: List[str] = Field(default_factory=list, description="标签")
    raw_text: str = Field(..., description="原始文本")
    images: List[str] = Field(default_factory=list, description="图片路径列表")
    source_file: str = Field(..., description="来源文件名")

    class Config:
        json_schema_extra = {
            "example": {
                "title": "清蒸鲈鱼",
                "description": "一道简单的清蒸鲈鱼做法",
                "ingredients": ["鲈鱼", "香葱", "姜", "食用油", "蒸鱼豉油", "料酒", "食用盐"],
                "steps": [
                    "姜切片切丝、香葱的葱白切段，葱绿切丝，切丝后放入冷水浸泡备用",
                    "鲈鱼处理好后洗净，用厨房纸擦干，两面分别划几刀",
                    "鱼肚内塞上姜和葱白，鱼身也撒上姜和葱白",
                    "水烧热感觉到水温后放进入鱼，大火清蒸10分钟"
                ],
                "tips": "关键点在于火候，鱼的大小跟火候都会相关",
                "cooking_time": "15分钟",
                "difficulty": "中等",
                "tags": ["海鲜", "清蒸", "家常菜"],
                "raw_text": "# 清蒸鲈鱼的做法\n\n...",
                "images": ["清蒸鲈鱼.jpg", "改刀.jpg", "摆盘.jpg"],
                "source_file": "清蒸鲈鱼.md"
            }
        }


class RecipeChunk(BaseModel):
    """
    分块后的数据单元
    符合PRD 4.2节定义的通用元数据字段
    """
    # 唯一标识
    chunk_id: str = Field(..., description="全局唯一的chunk标识，如rec_042_step_3")

    # 所属菜谱信息
    recipe_id: str = Field(..., description="所属菜谱唯一ID")
    recipe_title: str = Field(..., description="菜谱名称")

    # 分块类型
    chunk_type: str = Field(
        ...,
        description="块类型：ingredient/step/tip/image_desc"
    )

    # 内容
    content: str = Field(..., description="分块后的文本内容")

    # 元数据
    step_number: Optional[int] = Field(None, description="步骤序号（仅step类型）")
    document_source: str = Field(..., description="来源文件名")
    cooking_time: Optional[str] = Field(None, description="烹饪时长")
    difficulty: Optional[str] = Field(None, description="难度等级")
    tags: List[str] = Field(default_factory=list, description="标签数组")
    ingredient_list: Optional[List[str]] = Field(None, description="食材列表（仅ingredient类型）")
    image_path: Optional[str] = Field(None, description="图片路径（仅image_desc类型）")
    image_type: Optional[str] = Field(None, description="图片类型：成品图/步骤图")
    truncated: bool = Field(False, description="是否被截断")
    prev_step_summary: Optional[str] = Field(None, description="上一步摘要")
    next_step_summary: Optional[str] = Field(None, description="下一步摘要")
    related_ingredients: Optional[List[str]] = Field(None, description="步骤用到的食材")

    class Config:
        json_schema_extra = {
            "example": {
                "chunk_id": "rec_清蒸鲈鱼_step_3",
                "recipe_id": "rec_清蒸鲈鱼",
                "recipe_title": "清蒸鲈鱼",
                "chunk_type": "step",
                "content": "鱼肚内塞上姜和葱白，鱼身也撒上姜和葱白，量为备用的一半",
                "step_number": 3,
                "document_source": "清蒸鲈鱼.md",
                "cooking_time": "15分钟",
                "difficulty": "中等",
                "tags": ["海鲜", "清蒸"],
                "related_ingredients": ["姜", "香葱"]
            }
        }
