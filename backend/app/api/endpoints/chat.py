"""
聊天API端点
实现PRD 7.4节定义的API响应结构
"""

import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional
import json

from app.schemas.recipe import ChatRequest, ChatResponse, Citation, PipelineDebug, RetrievalDetail
from app.services.rag_pipeline import rag_pipeline
from app.services.conversation_service import conversation_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    聊天接口

    实现PRD 7.4节定义的响应结构：
    {
        "request_id": "uuid",
        "answer": "回答内容",
        "citations": [...],
        "has_related_content": true,
        "refused": false
    }
    """
    try:
        # 执行RAG管道（使用session_id管理对话记忆）
        result, debug_info = await rag_pipeline.run(
            query=request.query,
            session_id=request.session_id or "default",
            request_id=None
        )

        # 转换为响应格式
        citations = [
            Citation(
                citation_id=c.citation_id,
                recipe_title=c.recipe_title,
                chunk_type=c.chunk_type,
                step_number=c.step_number,
                excerpt=c.excerpt,
                document_source=c.document_source,
            )
            for c in result.citations
        ]

        # 构建调试信息
        pipeline_debug = PipelineDebug(
            query_type=debug_info.get("query_type", ""),
            query_weights=debug_info.get("query_weights", {}),
            rewritten_queries=debug_info.get("rewritten_queries", []),
            retrieval_count=debug_info.get("retrieval_count", 0),
            retrieval_results=[
                RetrievalDetail(**r) for r in debug_info.get("retrieval_results", [])
            ],
            rerank_results=[
                RetrievalDetail(**r) for r in debug_info.get("rerank_results", [])
            ],
            has_related_content=debug_info.get("has_related_content", True),
            generation_time_ms=debug_info.get("generation_time_ms", 0),
            total_time_ms=debug_info.get("total_time_ms", 0),
        )

        return ChatResponse(
            request_id=result.request_id,
            answer=result.answer,
            citations=citations,
            has_related_content=result.has_related_content,
            refused=result.refused,
            pipeline_debug=pipeline_debug,
        )

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    流式聊天接口

    使用SSE（Server-Sent Events）实现流式输出
    """
    async def generate():
        try:
            async for chunk in rag_pipeline.run_with_streaming(
                query=request.query,
                session_id=request.session_id or "default",
                request_id=None
            ):
                if isinstance(chunk, dict):
                    # 元数据事件（citations + pipeline_debug）
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                else:
                    # 文本片段（LangChain TextAccessor 需要显式 str() 转换）
                    text = str(chunk)
                    if text:
                        yield f"data: {json.dumps({'text': text}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.post("/new_session")
async def new_session(session_id: str = "default"):
    """
    创建新会话（清除对话记忆）

    PRD规范：前端可调用/new_session接口，触发记忆清除

    Args:
        session_id: 会话ID

    Returns:
        操作结果
    """
    try:
        await rag_pipeline.clear_session(session_id)
        return {"status": "success", "message": f"Session {session_id} cleared"}
    except Exception as e:
        logger.error(f"New session error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}")
async def get_session(session_id: str):
    """
    获取会话信息

    Args:
        session_id: 会话ID

    Returns:
        会话摘要
    """
    try:
        summary = await rag_pipeline.get_session_summary(session_id)
        return summary
    except Exception as e:
        logger.error(f"Get session error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations")
async def get_conversations():
    """
    获取所有对话列表

    Returns:
        对话摘要列表
    """
    try:
        conversations = await conversation_service.get_conversation_list()
        return {"conversations": conversations}
    except Exception as e:
        logger.error(f"Get conversations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations/{session_id}")
async def get_conversation_messages(session_id: str):
    """
    获取指定对话的完整消息历史

    Args:
        session_id: 会话ID

    Returns:
        消息列表
    """
    try:
        messages = await conversation_service.get_messages(session_id)
        return {"session_id": session_id, "messages": messages}
    except Exception as e:
        logger.error(f"Get conversation messages error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/conversations/{session_id}")
async def delete_conversation(session_id: str):
    """
    删除指定对话

    Args:
        session_id: 会话ID

    Returns:
        操作结果
    """
    try:
        # 同时清除内存中的记忆
        await rag_pipeline.clear_session(session_id)
        # 删除数据库中的记录
        deleted = await conversation_service.delete_conversation(session_id)
        return {"status": "success", "deleted": deleted}
    except Exception as e:
        logger.error(f"Delete conversation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
