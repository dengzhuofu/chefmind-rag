"""
对话持久化服务
负责对话历史的数据库读写
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy import select, delete, func
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import async_session_factory
from app.models.recipe import Conversation

logger = logging.getLogger(__name__)


class ConversationService:
    """对话持久化服务"""

    async def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        citations: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        保存单条消息到数据库

        Args:
            session_id: 会话ID
            role: 角色 (user/assistant)
            content: 消息内容
            citations: 引用列表
        """
        async with async_session_factory() as session:
            conversation = Conversation(
                session_id=session_id,
                role=role,
                content=content,
                citations=citations,
            )
            session.add(conversation)
            await session.commit()
            logger.debug(f"Saved message to DB: session={session_id}, role={role}")

    async def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """
        获取指定会话的所有消息

        Args:
            session_id: 会话ID

        Returns:
            消息列表
        """
        async with async_session_factory() as session:
            stmt = (
                select(Conversation)
                .where(Conversation.session_id == session_id)
                .order_by(Conversation.created_at)
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()

            return [
                {
                    "id": row.id,
                    "role": row.role,
                    "content": row.content,
                    "citations": row.citations,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
                for row in rows
            ]

    async def get_conversation_list(self) -> List[Dict[str, Any]]:
        """
        获取所有对话的摘要列表

        Returns:
            对话摘要列表，按最后更新时间倒序
        """
        async with async_session_factory() as session:
            # 获取每个session_id的最新消息时间和第一条用户消息作为标题
            subq = (
                select(
                    Conversation.session_id,
                    func.max(Conversation.created_at).label("last_message_at"),
                    func.count(Conversation.id).label("message_count"),
                )
                .group_by(Conversation.session_id)
                .subquery()
            )

            # 获取每个session的第一条用户消息作为标题预览
            stmt = select(
                subq.c.session_id,
                subq.c.last_message_at,
                subq.c.message_count,
            ).order_by(subq.c.last_message_at.desc())

            result = await session.execute(stmt)
            sessions = result.all()

            conversations = []
            for s in sessions:
                # 获取第一条用户消息作为标题
                first_msg_stmt = (
                    select(Conversation.content)
                    .where(
                        Conversation.session_id == s.session_id,
                        Conversation.role == "user",
                    )
                    .order_by(Conversation.created_at)
                    .limit(1)
                )
                first_msg_result = await session.execute(first_msg_stmt)
                first_msg = first_msg_result.scalar_one_or_none()

                # 获取最后一条消息作为预览
                last_msg_stmt = (
                    select(Conversation.content)
                    .where(Conversation.session_id == s.session_id)
                    .order_by(Conversation.created_at.desc())
                    .limit(1)
                )
                last_msg_result = await session.execute(last_msg_stmt)
                last_msg = last_msg_result.scalar_one_or_none()

                title = first_msg[:30] + "..." if first_msg and len(first_msg) > 30 else (first_msg or "新对话")
                preview = last_msg[:50] + "..." if last_msg and len(last_msg) > 50 else (last_msg or "")

                conversations.append({
                    "session_id": s.session_id,
                    "title": title,
                    "preview": preview,
                    "message_count": s.message_count,
                    "last_message_at": s.last_message_at.isoformat() if s.last_message_at else None,
                })

            return conversations

    async def delete_conversation(self, session_id: str) -> bool:
        """
        删除指定会话的所有消息

        Args:
            session_id: 会话ID

        Returns:
            是否删除成功
        """
        async with async_session_factory() as session:
            stmt = delete(Conversation).where(Conversation.session_id == session_id)
            result = await session.execute(stmt)
            await session.commit()
            logger.info(f"Deleted conversation: {session_id}, rows={result.rowcount}")
            return result.rowcount > 0

    async def conversation_exists(self, session_id: str) -> bool:
        """检查会话是否存在"""
        async with async_session_factory() as session:
            stmt = select(func.count(Conversation.id)).where(
                Conversation.session_id == session_id
            )
            result = await session.execute(stmt)
            return result.scalar() > 0


# 全局实例
conversation_service = ConversationService()
