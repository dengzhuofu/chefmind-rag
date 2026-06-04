"""菜谱相关的Pydantic模型"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class ChunkType(str, Enum):
    """分块类型"""
    INGREDIENT = "ingredient"
    STEP = "step"
    TIP = "tip"
    IMAGE_DESC = "image_desc"


class Difficulty(str, Enum):
    """难度等级"""
    EASY = "简单"
    MEDIUM = "中等"
    HARD = "困难"


class DocumentStatus(str, Enum):
    """文档处理状态"""
    PENDING = "pending"
    PARSING = "parsing"
    CHUNKING = "chunking"
    INDEXING = "indexing"
    DONE = "done"
    ERROR = "error"


# ==================== 请求模型 ====================

class ChatRequest(BaseModel):
    """聊天请求"""
    query: str = Field(..., description="用户问题")
    session_id: Optional[str] = Field(None, description="会话ID")


class UploadRequest(BaseModel):
    """上传请求"""
    pass  # 文件通过multipart上传


# ==================== 响应模型 ====================

class Citation(BaseModel):
    """引用信息"""
    citation_id: str = Field(..., description="引用ID")
    recipe_title: str = Field(..., description="菜谱名称")
    chunk_type: ChunkType = Field(..., description="分块类型")
    step_number: Optional[int] = Field(None, description="步骤序号")
    excerpt: str = Field(..., description="原文摘录")
    document_source: str = Field(..., description="文档来源")


class RetrievalDetail(BaseModel):
    """检索结果详情"""
    chunk_id: str = Field("", description="分块ID")
    score: float = Field(0, description="相关性分数")
    source: str = Field("", description="来源: vector/bm25/hybrid")
    recipe_title: str = Field("", description="菜谱名称")
    chunk_type: str = Field("", description="分块类型")
    step_number: Optional[int] = Field(None, description="步骤序号")
    content_preview: str = Field("", description="内容预览(前200字)")


class PipelineDebug(BaseModel):
    """管道调试信息"""
    query_type: str = Field("", description="查询分类类型")
    query_weights: dict = Field(default_factory=dict, description="检索权重配置")
    rewritten_queries: List[str] = Field(default_factory=list, description="改写后的查询列表")
    retrieval_count: int = Field(0, description="检索候选数量")
    retrieval_results: List[RetrievalDetail] = Field(default_factory=list, description="检索结果详情")
    rerank_results: List[RetrievalDetail] = Field(default_factory=list, description="重排序结果详情")
    has_related_content: bool = Field(True, description="是否有相关内容")
    generation_time_ms: int = Field(0, description="生成耗时(毫秒)")
    total_time_ms: int = Field(0, description="总耗时(毫秒)")


class ChatResponse(BaseModel):
    """聊天响应"""
    request_id: str = Field(..., description="请求ID")
    answer: str = Field(..., description="回答内容")
    citations: List[Citation] = Field(default_factory=list, description="引用列表")
    has_related_content: bool = Field(True, description="是否有相关内容")
    refused: bool = Field(False, description="是否拒答")
    pipeline_debug: Optional[PipelineDebug] = Field(None, description="管道调试信息")


class RecipeBase(BaseModel):
    """菜谱基础信息"""
    title: str = Field(..., description="菜谱名称")
    description: Optional[str] = Field(None, description="菜谱描述")
    ingredients: List[str] = Field(default_factory=list, description="食材列表")
    steps: List[str] = Field(default_factory=list, description="步骤列表")
    tips: Optional[str] = Field(None, description="小贴士")
    cooking_time: Optional[str] = Field(None, description="烹饪时长")
    difficulty: Optional[Difficulty] = Field(None, description="难度")
    tags: List[str] = Field(default_factory=list, description="标签")
    source_file: Optional[str] = Field(None, description="来源文件")


class RecipeCreate(RecipeBase):
    """创建菜谱"""
    pass


class RecipeResponse(RecipeBase):
    """菜谱响应"""
    id: str = Field(..., description="菜谱ID")
    images: List[str] = Field(default_factory=list, description="图片列表")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: Optional[datetime] = Field(None, description="更新时间")

    class Config:
        from_attributes = True


class DocumentResponse(BaseModel):
    """文档响应"""
    id: str = Field(..., description="文档ID")
    filename: str = Field(..., description="文件名")
    status: DocumentStatus = Field(..., description="处理状态")
    recipe_count: int = Field(0, description="菜谱数量")
    error_message: Optional[str] = Field(None, description="错误信息")
    created_at: datetime = Field(..., description="创建时间")

    class Config:
        from_attributes = True


class TaskStatus(BaseModel):
    """任务状态"""
    task_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="任务状态")
    progress: Optional[float] = Field(None, description="进度(0-100)")
    result: Optional[dict] = Field(None, description="结果")
    error: Optional[str] = Field(None, description="错误信息")


class FeedbackRequest(BaseModel):
    """反馈请求"""
    conversation_id: str = Field(..., description="对话ID")
    rating: int = Field(..., ge=1, le=5, description="评分(1-5)")
    comment: Optional[str] = Field(None, description="评论")


class SessionResponse(BaseModel):
    """会话响应"""
    session_id: str = Field(..., description="会话ID")
    message_count: int = Field(0, description="消息数量")
    current_recipe: Optional[str] = Field(None, description="当前讨论的菜谱")
    last_message: Optional[dict] = Field(None, description="最后一条消息")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "ok"
    version: str = "1.0.0"
    milvus_connected: bool = False
    database_connected: bool = False
