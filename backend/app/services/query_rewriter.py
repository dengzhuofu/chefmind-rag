"""
查询重构服务
实现PRD 4.4.1节定义的查询重构策略
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# LLM导入（可选）
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False


@dataclass
class RewrittenQuery:
    """重构后的查询"""
    original: str  # 原始查询
    rewritten: str  # 重构后的查询
    strategy: str  # 使用的策略
    confidence: float  # 置信度


class QueryRewriter:
    """
    查询重构服务

    实现PRD 4.4.1节规范：
    - 指代消解与上下文补全
    - 复合意图分解
    - 多视角生成
    - HyDE（可选）
    """

    # 指代消解提示词
    RESOLVE_PROMPT = """你是一个食谱查询助手。请根据对话历史，将用户的当前问题改写为独立完整的查询。

## 对话历史
{chat_history}

## 当前问题
{query}

## 要求
1. 消除代词（它、这个、那个等），替换为具体食材或菜名
2. 补充缺失的上下文信息
3. 保持查询意图不变
4. 输出纯文本，不要添加解释

## 改写后的查询："""

    # 复合意图分解提示词
    DECOMPOSE_PROMPT = """你是一个食谱查询助手。请判断用户的问题是否包含多个意图，如果是，请分解为1-3个独立的子查询。

## 用户问题
{query}

## 要求
1. 如果问题只包含一个意图，返回原始问题
2. 如果问题包含多个意图（如"怎么做红烧肉和糖醋排骨"），分解为独立查询
3. 每个子查询应该完整独立
4. 返回JSON格式：{{"queries": ["查询1", "查询2"]}}

## 输出："""

    # 多视角生成提示词
    MULTI_VIEW_PROMPT = """你是一个食谱查询助手。请为用户的查询生成2-3种不同的表达方式，以提高检索覆盖率。

## 原始查询
{query}

## 要求
1. 保持查询意图不变
2. 使用不同的词汇和句式
3. 包含同义词、近义词
4. 返回JSON格式：{{"views": ["视角1", "视角2", "视角3"]}}

## 输出："""

    def __init__(self, llm=None):
        """
        初始化查询重构服务

        Args:
            llm: 语言模型实例（可选）
        """
        self.llm = llm

    async def rewrite_query(
        self,
        query: str,
        chat_history: Optional[str] = None
    ) -> List[RewrittenQuery]:
        """
        重构查询

        Args:
            query: 原始查询
            chat_history: 对话历史（可选）

        Returns:
            重构后的查询列表
        """
        results = []

        # 1. 指代消解（如果有对话历史）
        if chat_history:
            resolved = await self._resolve_references(query, chat_history)
            if resolved and resolved != query:
                results.append(RewrittenQuery(
                    original=query,
                    rewritten=resolved,
                    strategy="resolve",
                    confidence=0.9
                ))

        # 2. 复合意图分解
        decomposed = await self._decompose_intent(query)
        if len(decomposed) > 1:
            for i, sub_query in enumerate(decomposed):
                results.append(RewrittenQuery(
                    original=query,
                    rewritten=sub_query,
                    strategy="decompose",
                    confidence=0.85
                ))

        # 3. 多视角生成
        views = await self._generate_multi_views(query)
        for view in views:
            results.append(RewrittenQuery(
                original=query,
                rewritten=view,
                strategy="multi_view",
                confidence=0.8
            ))

        # 如果没有重构结果，返回原始查询
        if not results:
            results.append(RewrittenQuery(
                original=query,
                rewritten=query,
                strategy="original",
                confidence=1.0
            ))

        return results

    async def _resolve_references(self, query: str, chat_history: str) -> Optional[str]:
        """
        指代消解

        Args:
            query: 当前查询
            chat_history: 对话历史

        Returns:
            消解后的查询
        """
        if not self.llm:
            # 无LLM时，使用规则进行简单消解
            return self._simple_resolve(query, chat_history)

        try:
            prompt = ChatPromptTemplate.from_template(self.RESOLVE_PROMPT)
            chain = prompt | self.llm
            result = await chain.ainvoke({
                "query": query,
                "chat_history": chat_history
            })
            return result.content.strip()
        except Exception as e:
            logger.warning(f"Reference resolution failed: {e}")
            return None

    def _simple_resolve(self, query: str, chat_history: str) -> str:
        """
        简单的规则消解（无LLM时使用）

        Args:
            query: 当前查询
            chat_history: 对话历史

        Returns:
            消解后的查询
        """
        # 提取最近的菜名
        import re
        recipe_pattern = r'[《](.*?)[》]'
        recipes = re.findall(recipe_pattern, chat_history)

        if recipes and any(word in query for word in ['它', '这个', '那个', '怎么做', '怎么弄']):
            # 替换代词为最近的菜名
            latest_recipe = recipes[-1]
            for pronoun in ['它', '这个', '那个']:
                query = query.replace(pronoun, latest_recipe)

        return query

    async def _decompose_intent(self, query: str) -> List[str]:
        """
        复合意图分解

        Args:
            query: 查询

        Returns:
            子查询列表
        """
        if not self.llm:
            # 无LLM时，使用规则进行简单分解
            return self._simple_decompose(query)

        try:
            prompt = ChatPromptTemplate.from_template(self.DECOMPOSE_PROMPT)
            chain = prompt | self.llm | JsonOutputParser()
            result = await chain.ainvoke({"query": query})
            return result.get("queries", [query])
        except Exception as e:
            logger.warning(f"Intent decomposition failed: {e}")
            return [query]

    def _simple_decompose(self, query: str) -> List[str]:
        """
        简单的规则分解（无LLM时使用）

        Args:
            query: 查询

        Returns:
            子查询列表
        """
        # 检测并列关系
        import re

        # 匹配"A和B怎么做"、"A、B、C的做法"等模式
        patterns = [
            r'(.+?)和(.+?)(?:怎么做|怎么弄|的做法|如何做)',
            r'(.+?)[、,](.+?)(?:怎么做|怎么弄|的做法|如何做)',
        ]

        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                items = [g.strip() for g in match.groups() if g.strip()]
                if len(items) > 1:
                    # 为每个食材生成独立查询
                    suffix = re.sub(r'^.*?(怎么做|怎么弄|的做法|如何做)', r'\1', query)
                    return [f"{item}{suffix}" for item in items]

        return [query]

    async def _generate_multi_views(self, query: str) -> List[str]:
        """
        多视角生成

        Args:
            query: 查询

        Returns:
            多视角查询列表
        """
        if not self.llm:
            # 无LLM时，使用规则生成简单变体
            return self._simple_multi_view(query)

        try:
            prompt = ChatPromptTemplate.from_template(self.MULTI_VIEW_PROMPT)
            chain = prompt | self.llm | JsonOutputParser()
            result = await chain.ainvoke({"query": query})
            return result.get("views", [])
        except Exception as e:
            logger.warning(f"Multi-view generation failed: {e}")
            return []

    def _simple_multi_view(self, query: str) -> List[str]:
        """
        简单的规则多视角生成（无LLM时使用）

        Args:
            query: 查询

        Returns:
            多视角查询列表
        """
        views = []

        # 同义词替换
        synonyms = {
            '怎么做': ['如何制作', '做法', '烹饪方法'],
            '怎么弄': ['如何制作', '做法', '烹饪方法'],
            '食材': ['原料', '材料', '配料'],
            '步骤': ['做法', '过程', '流程'],
        }

        for original, alternatives in synonyms.items():
            if original in query:
                for alt in alternatives[:2]:  # 最多取2个
                    views.append(query.replace(original, alt))

        return views

    async def generate_hyde(self, query: str) -> Optional[str]:
        """
        HyDE（假设性文档嵌入）

        让LLM生成假设的菜谱摘要文本，用于向量检索

        Args:
            query: 查询

        Returns:
            假设的文档内容
        """
        if not self.llm:
            return None

        hyde_prompt = """你是一个食谱专家。请根据用户的查询，生成一段可能包含答案的菜谱摘要文本。

## 用户查询
{query}

## 要求
1. 生成的内容应该像真实的菜谱描述
2. 包含相关的食材、步骤或技巧
3. 长度约100-200字
4. 使用中文

## 输出："""

        try:
            prompt = ChatPromptTemplate.from_template(hyde_prompt)
            chain = prompt | self.llm
            result = await chain.ainvoke({"query": query})
            return result.content.strip()
        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}")
            return None


# 创建全局实例
query_rewriter = QueryRewriter()
