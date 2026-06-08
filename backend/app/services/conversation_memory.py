"""
对话记忆服务
实现PRD第5节定义的对话记忆设计
使用 Redis 持久化对话历史，服务重启后不丢失
"""

import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from app.core.redis import get_redis

logger = logging.getLogger(__name__)


class Message:
    """对话消息"""

    def __init__(
        self,
        role: str,
        content: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else None,
            metadata=data.get("metadata", {})
        )


class ConversationMemory:
    """
    对话记忆管理器（Redis 持久化）

    PRD规范：
    - 短期记忆：保留最近5轮对话
    - 实体追踪：提取最近一次提及的recipe_title作为current_recipe
    - 记忆清理：支持/new_session接口触发记忆清除
    """

    def __init__(self, max_turns: int = 5):
        """
        初始化对话记忆

        Args:
            max_turns: 最大保留对话轮数
        """
        self.max_turns = max_turns
        self._recipe_keywords = ["菜谱", "做法", "步骤", "食材", "调料"]

    def _history_key(self, session_id: str) -> str:
        """Redis key for session message history"""
        return f"session:{session_id}:history"

    def _recipe_key(self, session_id: str) -> str:
        """Redis key for current recipe"""
        return f"session:{session_id}:recipe"

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        添加消息到记忆

        Args:
            session_id: 会话ID
            role: 角色（user/assistant）
            content: 消息内容
            metadata: 元数据
        """
        redis = await get_redis()
        message = Message(role=role, content=content, metadata=metadata)
        await redis.rpush(self._history_key(session_id), json.dumps(message.to_dict(), ensure_ascii=False))

        # 修剪超出 max_turns 的旧消息（保留最近 max_turns*2 条）
        max_messages = self.max_turns * 2
        await redis.ltrim(self._history_key(session_id), -max_messages, -1)

        # 追踪实体
        if role == "user":
            await self._extract_current_recipe(session_id, content)

        logger.debug(f"Added message to session {session_id}: {role}")

    async def get_history(
        self,
        session_id: str,
        max_turns: Optional[int] = None
    ) -> List[Message]:
        """
        获取对话历史

        Args:
            session_id: 会话ID
            max_turns: 最大轮数（可选）

        Returns:
            消息列表
        """
        redis = await get_redis()
        count = (max_turns or self.max_turns) * 2
        raw_list = await redis.lrange(self._history_key(session_id), -count, -1)
        return [Message.from_dict(json.loads(raw)) for raw in raw_list]

    async def get_history_as_string(
        self,
        session_id: str,
        max_turns: Optional[int] = None
    ) -> str:
        """
        获取对话历史（字符串格式）

        Args:
            session_id: 会话ID
            max_turns: 最大轮数

        Returns:
            格式化的对话历史
        """
        messages = await self.get_history(session_id, max_turns)
        if not messages:
            return ""

        history_parts = []
        for msg in messages:
            if msg.role == "user":
                history_parts.append(f"用户: {msg.content}")
            else:
                history_parts.append(f"助手: {msg.content}")

        return "\n".join(history_parts)

    async def get_current_recipe(self, session_id: str) -> Optional[str]:
        """
        获取当前讨论的菜谱

        Args:
            session_id: 会话ID

        Returns:
            当前菜谱名称
        """
        redis = await get_redis()
        return await redis.get(self._recipe_key(session_id))

    async def _extract_current_recipe(self, session_id: str, content: str):
        """
        从用户消息中提取菜谱名称

        Args:
            session_id: 会话ID
            content: 用户消息
        """
        import re
        for keyword in self._recipe_keywords:
            if keyword in content:
                patterns = [
                    rf'([一-龥]+){keyword}',
                    rf'{keyword}([一-龥]+)',
                ]
                for pattern in patterns:
                    match = re.search(pattern, content)
                    if match:
                        recipe_name = match.group(1).strip()
                        if 2 <= len(recipe_name) <= 10:
                            redis = await get_redis()
                            await redis.set(self._recipe_key(session_id), recipe_name)
                            logger.info(f"Extracted current recipe: {recipe_name}")
                            return

    async def clear_session(self, session_id: str):
        """
        清除会话记忆

        Args:
            session_id: 会话ID
        """
        redis = await get_redis()
        await redis.delete(self._history_key(session_id), self._recipe_key(session_id))
        logger.info(f"Cleared session: {session_id}")

    async def get_memory_summary(self, session_id: str) -> Dict[str, Any]:
        """
        获取记忆摘要

        Args:
            session_id: 会话ID

        Returns:
            记忆摘要
        """
        messages = await self.get_history(session_id)
        return {
            "session_id": session_id,
            "message_count": len(messages),
            "current_recipe": await self.get_current_recipe(session_id),
            "last_message": messages[-1].to_dict() if messages else None
        }


class ConversationBufferWindowMemory:
    """
    LangChain风格的ConversationBufferWindowMemory

    PRD规范：保留最近5轮对话
    """

    def __init__(self, k: int = 5, memory_key: str = "chat_history"):
        """
        初始化

        Args:
            k: 保留的对话轮数
            memory_key: 记忆键名
        """
        self.k = k
        self.memory_key = memory_key
        self.messages: List[Dict[str, str]] = []

    def add_user_message(self, message: str):
        """添加用户消息"""
        self.messages.append({"role": "user", "content": message})
        self._trim_messages()

    def add_ai_message(self, message: str):
        """添加AI消息"""
        self.messages.append({"role": "assistant", "content": message})
        self._trim_messages()

    def _trim_messages(self):
        """保留最近k轮对话"""
        max_messages = self.k * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def get_messages(self) -> List[Dict[str, str]]:
        """获取消息列表"""
        return self.messages

    def get_history_string(self) -> str:
        """获取历史字符串"""
        if not self.messages:
            return ""

        parts = []
        for msg in self.messages:
            if msg["role"] == "user":
                parts.append(f"用户: {msg['content']}")
            else:
                parts.append(f"助手: {msg['content']}")
        return "\n".join(parts)

    def clear(self):
        """清除记忆"""
        self.messages = []


class InMemoryConversationMemory:
    """
    内存版对话记忆（不需要Redis）
    用于评估脚本和测试场景
    """

    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self._sessions: Dict[str, List[Message]] = {}
        self._recipes: Dict[str, str] = {}

    async def add_message(self, session_id: str, role: str, content: str, metadata=None):
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append(Message(role=role, content=content, metadata=metadata))
        # 保留最近 max_turns*2 条
        max_messages = self.max_turns * 2
        if len(self._sessions[session_id]) > max_messages:
            self._sessions[session_id] = self._sessions[session_id][-max_messages:]

    async def get_history(self, session_id: str, max_turns=None) -> List[Message]:
        count = (max_turns or self.max_turns) * 2
        return self._sessions.get(session_id, [])[-count:]

    async def get_history_as_string(self, session_id: str, max_turns=None) -> str:
        messages = await self.get_history(session_id, max_turns)
        if not messages:
            return ""
        parts = []
        for msg in messages:
            prefix = "用户" if msg.role == "user" else "助手"
            parts.append(f"{prefix}: {msg.content}")
        return "\n".join(parts)

    async def get_current_recipe(self, session_id: str) -> Optional[str]:
        return self._recipes.get(session_id)

    async def clear_session(self, session_id: str):
        self._sessions.pop(session_id, None)
        self._recipes.pop(session_id, None)

    async def get_memory_summary(self, session_id: str) -> Dict[str, Any]:
        messages = self._sessions.get(session_id, [])
        return {
            "session_id": session_id,
            "message_count": len(messages),
            "current_recipe": self._recipes.get(session_id),
            "last_message": messages[-1].to_dict() if messages else None,
        }


# 全局实例：优先使用Redis，不可用时回退到内存版
conversation_memory = InMemoryConversationMemory(max_turns=5)

async def _init_conversation_memory():
    """尝试切换到Redis版对话记忆"""
    global conversation_memory
    try:
        from app.core.redis import get_redis
        redis = await get_redis()
        await redis.ping()
        conversation_memory = ConversationMemory(max_turns=5)
        logger.info("Using Redis-backed conversation memory")
    except Exception:
        logger.info("Redis not available, using in-memory conversation memory")
