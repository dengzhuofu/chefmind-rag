"""
查询分类与路由服务
实现PRD 4.4.2节定义的查询分类策略
"""

import re
import logging
from typing import Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """
    查询类型枚举

    PRD 4.4.2节定义的类型：
    """
    INGREDIENT_SEARCH = "ingredient_search"  # 基于冰箱食材的匹配
    STEP_LOOKUP = "step_lookup"  # 询问具体步骤
    SUBSTITUTION = "substitution"  # 食材替换建议
    TIP = "tip"  # 烹饪技巧
    IMAGE_REQUEST = "image_request"  # 要求看图
    GENERATE_RECIPE = "generate_recipe"  # 直接生成菜谱
    GENERAL = "general"  # 通用查询


class RouteConfig:
    """路由配置"""

    # 检索权重配置（PRD 4.4.3节）
    WEIGHTS = {
        QueryType.INGREDIENT_SEARCH: {"semantic": 0.4, "bm25": 0.6},
        QueryType.STEP_LOOKUP: {"semantic": 0.7, "bm25": 0.3},
        QueryType.TIP: {"semantic": 0.8, "bm25": 0.2},
        QueryType.SUBSTITUTION: {"semantic": 0.5, "bm25": 0.5},
        QueryType.IMAGE_REQUEST: {"semantic": 0.6, "bm25": 0.4},
        QueryType.GENERATE_RECIPE: {"semantic": 0.7, "bm25": 0.3},
        QueryType.GENERAL: {"semantic": 0.5, "bm25": 0.5},
    }

    # 返回的chunk类型过滤
    CHUNK_TYPE_FILTER = {
        QueryType.INGREDIENT_SEARCH: ["ingredient"],
        QueryType.STEP_LOOKUP: ["step", "ingredient"],  # 步骤查询也需要食材信息
        QueryType.TIP: ["tip", "step"],  # 技巧查询也需要步骤上下文
        QueryType.IMAGE_REQUEST: ["image_desc"],
        QueryType.SUBSTITUTION: ["ingredient", "step", "tip"],
        QueryType.GENERATE_RECIPE: ["ingredient", "step", "tip"],
        QueryType.GENERAL: None,  # 不过滤
    }

    @classmethod
    def get_weights(cls, query_type: QueryType) -> Dict[str, float]:
        """获取检索权重"""
        return cls.WEIGHTS.get(query_type, cls.WEIGHTS[QueryType.GENERAL])

    @classmethod
    def get_chunk_type_filter(cls, query_type: QueryType) -> Optional[list]:
        """获取chunk类型过滤"""
        return cls.CHUNK_TYPE_FILTER.get(query_type)


class QueryClassifier:
    """
    查询分类器

    实现PRD 4.4.2节规范：
    - 使用零样本分类
    - 输出严格JSON
    - 支持6种查询类型
    """

    # 关键词模式
    PATTERNS = {
        QueryType.INGREDIENT_SEARCH: [
            r'(?:冰箱|家里|现有|剩余|还有).*?(?:食材|原料|材料)',
            r'(?:有什么|有哪些|什么菜|做什么菜)',
            r'(?:用|拿|靠).*?(?:做|煮|炒|炖)',
        ],
        QueryType.STEP_LOOKUP: [
            r'(?:步骤|做法|过程|流程|怎么)',
            r'(?:第[一二三四五六七八九十\d]步|下一步|上一步)',
            r'(?:如何|怎样).*?(?:操作|处理|制作)',
        ],
        QueryType.SUBSTITUTION: [
            r'(?:替换|代替|替代|换)',
            r'(?:没有|缺少|缺|少).*?(?:怎么办|可以用)',
            r'(?:可以.*?代替|能.*?替代)',
        ],
        QueryType.TIP: [
            r'(?:技巧|窍门|秘诀|小贴士|注意事项)',
            r'(?:怎样.*?(?:更好|更嫩|更入味))',
            r'(?:如何.*?(?:避免|防止))',
        ],
        QueryType.IMAGE_REQUEST: [
            r'(?:图片|照片|图|样子|外观)',
            r'(?:看看|显示|展示|给我看)',
            r'(?:长什么样|什么样子)',
        ],
        QueryType.GENERATE_RECIPE: [
            r'(?:推荐|建议|生成|创建).*?(?:菜谱|食谱)',
            r'(?:发明|创造|设计).*?(?:菜|料理)',
            r'(?:新|原创).*?(?:做法|食谱)',
        ],
    }

    def __init__(self, llm=None):
        """
        初始化分类器

        Args:
            llm: 语言模型实例（可选）
        """
        self.llm = llm

    async def classify(self, query: str) -> QueryType:
        """
        分类查询

        Args:
            query: 查询文本

        Returns:
            查询类型
        """
        # 1. 尝试使用LLM分类
        if self.llm:
            try:
                return await self._classify_with_llm(query)
            except Exception as e:
                logger.warning(f"LLM classification failed: {e}")

        # 2. 使用规则分类
        return self._classify_with_rules(query)

    def _classify_with_rules(self, query: str) -> QueryType:
        """
        规则分类

        Args:
            query: 查询文本

        Returns:
            查询类型
        """
        query_lower = query.lower()

        # 计算每种类型的匹配分数
        scores = {}
        for query_type, patterns in self.PATTERNS.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            scores[query_type] = score

        # 返回得分最高的类型
        if scores:
            best_type = max(scores, key=scores.get)
            if scores[best_type] > 0:
                return best_type

        # 默认返回通用类型
        return QueryType.GENERAL

    async def _classify_with_llm(self, query: str) -> QueryType:
        """
        LLM分类

        Args:
            query: 查询文本

        Returns:
            查询类型
        """
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import JsonOutputParser

        prompt = ChatPromptTemplate.from_template("""你是一个食谱查询分类器。请判断用户查询属于以下哪种类型：

## 查询类型
- ingredient_search: 基于现有食材寻找菜谱
- step_lookup: 询问具体菜谱的步骤
- substitution: 食材替换建议
- tip: 烹饪技巧和窍门
- image_request: 要求查看图片
- generate_recipe: 请求生成或推荐菜谱
- general: 其他通用查询

## 用户查询
{query}

## 要求
返回JSON格式：{{"type": "查询类型", "confidence": 0.9}}

## 输出：""")

        chain = prompt | self.llm | JsonOutputParser()
        result = await chain.ainvoke({"query": query})

        query_type = result.get("type", "general")
        try:
            return QueryType(query_type)
        except ValueError:
            return QueryType.GENERAL

    async def classify_and_get_config(self, query: str) -> Dict[str, Any]:
        """
        分类并获取路由配置

        Args:
            query: 查询文本

        Returns:
            包含类型和配置的字典
        """
        query_type = await self.classify(query)

        return {
            "type": query_type,
            "weights": RouteConfig.get_weights(query_type),
            "chunk_type_filter": RouteConfig.get_chunk_type_filter(query_type),
        }


# 创建全局实例
query_classifier = QueryClassifier()
