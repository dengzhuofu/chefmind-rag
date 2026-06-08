"""
回答生成服务
实现PRD 4.6节定义的回答生成策略
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from app.core.config import settings
from app.services.hybrid_retriever import RetrievalResult

logger = logging.getLogger(__name__)

# LLM导入（可选）
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False


@dataclass
class Citation:
    """引用信息"""
    citation_id: str
    recipe_title: str
    chunk_type: str
    step_number: Optional[int]
    excerpt: str
    document_source: str
    image_path: Optional[str] = None


@dataclass
class AnswerResult:
    """回答结果"""
    answer: str
    citations: List[Citation]
    has_related_content: bool
    refused: bool
    request_id: str


class AnswerGenerator:
    """
    回答生成服务

    实现PRD 4.6节规范：
    - 提示词设计：强制引用、禁止编造
    - 上下文构建：格式化检索结果
    - 生成参数：temperature=0.1，max_tokens=800
    - 引用强制：后处理校验
    """

    # 系统提示词（PRD 4.6节）
    SYSTEM_PROMPT = """/no_think
你是一个严谨的食谱问答助手。你只能依据下方提供的"参考食谱片段"来回答问题。

## 回答要求
1. **必须标注引用**：每个基于片段的内容，请在句末标注引用来源，格式为 `[来源编号]`。
2. **引用编号对应**：来源编号必须与下方片段前的编号严格对应。
3. **不得编造**：如果片段信息不足，请直接回复"根据现有食谱库，无法回答此问题"，绝对禁止编造任何步骤、食材或技巧。
4. **保留用量**：回答中必须保留食材的具体用量（如"盐 3g"、"酱油 15ml"），不要省略。
5. **步骤完整**：如果用户询问做法，请按步骤顺序完整回答，不要遗漏关键步骤。
6. **格式示例**：
   用户问：麻婆豆腐怎么做？
   回答：
   麻婆豆腐是一道经典川菜，以下是详细做法：

   **食材准备**：豆腐 300g、牛肉末 100g、豆瓣酱 30g...

   **烹饪步骤**：
   1. 将豆腐切成2cm见方的小块，放入沸水中焯水2分钟，捞出沥干备用[1]。
   2. 热锅倒油，放入牛肉末煸炒至变色出油[2]。
   3. 加入豆瓣酱炒出红油，放入豆腐轻轻翻煮至入味[3]。

> 📎 引用来源：
> [1] 《麻婆豆腐》(川菜经典菜谱.pdf) - 步骤1
> [2] 《麻婆豆腐》(川菜经典菜谱.pdf) - 步骤2
> [3] 《麻婆豆腐》(川菜经典菜谱.pdf) - 步骤3"""

    # 人类提示词模板
    HUMAN_PROMPT = """{chat_history_section}
## 参考食谱片段

{context}

## 用户问题

{query}"""

    def __init__(self, llm=None):
        """
        初始化回答生成服务

        Args:
            llm: 语言模型实例
        """
        self.llm = llm

    async def generate(
        self,
        query: str,
        retrieval_results: List[RetrievalResult],
        has_related_content: bool,
        chat_history: Optional[str] = None,
        request_id: str = ""
    ) -> AnswerResult:
        """
        生成回答

        Args:
            query: 用户查询
            retrieval_results: 检索结果
            has_related_content: 是否有相关内容
            chat_history: 对话历史
            request_id: 请求ID

        Returns:
            回答结果
        """
        # 检查是否有相关内容
        if not has_related_content or not retrieval_results:
            return AnswerResult(
                answer="根据现有食谱库，无法回答此问题。请尝试换个问题或提供更多细节。",
                citations=[],
                has_related_content=False,
                refused=True,
                request_id=request_id
            )

        # 构建上下文和引用映射
        context, citation_map = self._build_context(retrieval_results)

        # 生成回答
        if self.llm:
            answer = await self._generate_with_llm(query, context, chat_history)
        else:
            # 无LLM时，返回简单的格式化回答
            answer = self._generate_simple_answer(query, retrieval_results)

        # 后处理：校验引用
        answer = self._validate_citations(answer, citation_map)

        # 构建结构化引用
        citations = self._build_citations(answer, citation_map)

        return AnswerResult(
            answer=answer,
            citations=citations,
            has_related_content=True,
            refused=False,
            request_id=request_id
        )

    def _build_context(
        self, results: List[RetrievalResult]
    ) -> tuple[str, Dict[str, Dict]]:
        """
        构建上下文

        PRD规范：
        [1] 来源《麻婆豆腐》(川菜经典菜谱.pdf) 步骤1：
        将豆腐切成2cm见方的小块...

        Args:
            results: 检索结果

        Returns:
            (格式化上下文, 引用映射)
        """
        formatted = []
        citation_map = {}

        for idx, result in enumerate(results, 1):
            meta = result.metadata
            recipe_title = meta.get("recipe_title", "未知")
            document_source = meta.get("document_source", "未知来源")
            chunk_type = meta.get("chunk_type", "")
            step_number = meta.get("step_number")

            # 构建来源描述
            source = f"[{idx}] 来源《{recipe_title}》({document_source})"
            if chunk_type == "step" and step_number:
                source += f" 步骤{step_number}"
            elif chunk_type == "ingredient":
                source += " 食材"
            elif chunk_type == "tip":
                source += " 小贴士"
            elif chunk_type == "image_desc":
                source += " 图片描述"

            source += f"：\n{result.content}"
            formatted.append(source)

            # 保存引用映射
            citation_map[str(idx)] = {
                "recipe_title": recipe_title,
                "chunk_type": chunk_type,
                "step_number": step_number,
                "excerpt": result.content[:100],
                "document_source": document_source,
                "image_path": meta.get("image_path"),
            }

        context = "\n\n".join(formatted)
        return context, citation_map

    async def _generate_with_llm(
        self,
        query: str,
        context: str,
        chat_history: Optional[str]
    ) -> str:
        """
        使用LLM生成回答

        Args:
            query: 查询
            context: 上下文
            chat_history: 对话历史

        Returns:
            生成的回答
        """
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.SYSTEM_PROMPT),
                ("human", self.HUMAN_PROMPT)
            ])

            chain = prompt | self.llm | StrOutputParser()

            # 构建对话历史部分
            chat_history_section = ""
            if chat_history:
                chat_history_section = f"## 对话历史\n{chat_history}\n"

            # 构建输入
            input_data = {
                "query": query,
                "context": context,
                "chat_history_section": chat_history_section,
            }

            # 生成回答（流式）
            answer = ""
            async for chunk in chain.astream(input_data):
                answer += chunk

            return answer.strip()

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._generate_simple_answer(query, [])

    async def _generate_streaming(
        self,
        query: str,
        context: str,
        chat_history: Optional[str]
    ):
        """
        使用LLM流式生成回答（逐chunk yield）

        Args:
            query: 查询
            context: 上下文
            chat_history: 对话历史

        Yields:
            文本片段
        """
        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser

            prompt = ChatPromptTemplate.from_messages([
                ("system", self.SYSTEM_PROMPT),
                ("human", self.HUMAN_PROMPT)
            ])

            chain = prompt | self.llm | StrOutputParser()

            chat_history_section = ""
            if chat_history:
                chat_history_section = f"## 对话历史\n{chat_history}\n"

            input_data = {
                "query": query,
                "context": context,
                "chat_history_section": chat_history_section,
            }

            async for chunk in chain.astream(input_data):
                text = str(chunk)
                if text:
                    yield text

        except Exception as e:
            logger.error(f"LLM streaming generation failed: {e}")
            yield self._generate_simple_answer(query, [])

    def _generate_simple_answer(
        self,
        query: str,
        results: List[RetrievalResult]
    ) -> str:
        """
        简单回答生成（无LLM时使用）
        按菜谱分组，去重，展示完整步骤

        Args:
            query: 查询
            results: 检索结果

        Returns:
            格式化的回答
        """
        if not results:
            return "根据现有食谱库，无法回答此问题。"

        # 按菜谱分组
        recipes = {}  # recipe_title -> {ingredient, steps, source}
        for result in results:
            meta = result.metadata
            title = meta.get("recipe_title", "未知")
            chunk_type = meta.get("chunk_type", "")
            source = meta.get("document_source", "未知来源")

            if title not in recipes:
                recipes[title] = {"ingredient": None, "steps": [], "source": source, "difficulty": "", "time": ""}

            if chunk_type == "ingredient" and recipes[title]["ingredient"] is None:
                recipes[title]["ingredient"] = result.content
            elif chunk_type == "step":
                step_num = meta.get("step_number", 0)
                recipes[title]["steps"].append((step_num, result.content))
            elif chunk_type == "tip":
                recipes[title]["steps"].append((999, result.content))

        # 构建回答
        answer_parts = []
        citation_idx = 0
        citations = []

        for title, info in list(recipes.items())[:3]:  # 最多3个菜谱
            # 提取食材信息（从step chunk的头部获取）
            ingredient_text = ""
            if info["ingredient"]:
                # 从ingredient chunk提取食材列表
                lines = info["ingredient"].split("\n")
                ingredient_lines = [l.strip("- ").strip() for l in lines if l.strip().startswith("- ")]
                if ingredient_lines:
                    ingredient_text = "、".join(ingredient_lines[:8])  # 最多8种食材

            # 排序步骤
            sorted_steps = sorted(info["steps"], key=lambda x: x[0])
            # 去重（按步骤号）
            seen_steps = set()
            unique_steps = []
            for step_num, content in sorted_steps:
                if step_num not in seen_steps:
                    seen_steps.add(step_num)
                    unique_steps.append((step_num, content))

            # 构建菜谱部分
            part = f"**《{title}》**"
            if ingredient_text:
                part += f"\n食材：{ingredient_text}"

            # 提取步骤文本（去掉元数据头）
            step_texts = []
            for step_num, content in unique_steps[:6]:  # 最多6步
                # 去掉 [难度：xxx] 和 食材：xxx 行
                lines = content.split("\n")
                step_line = ""
                for line in lines:
                    if line.strip().startswith("步骤"):
                        step_line = line.strip()
                        break
                    elif line.strip() and not line.strip().startswith("[") and not line.strip().startswith("食材") and not line.strip().startswith("《"):
                        step_line = line.strip()
                        break
                if step_line:
                    step_texts.append(step_line)
                    citation_idx += 1

            if step_texts:
                part += "\n" + "\n".join(step_texts)

            answer_parts.append(part)
            citations.append(f"[{len(citations)+1}] 《{title}》({info['source']})")

        answer = "根据食谱库，找到以下相关信息：\n\n" + "\n\n".join(answer_parts)

        if citations:
            answer += "\n\n> 📎 引用来源：\n> " + "\n> ".join(citations)

        return answer

    def _validate_citations(
        self, answer: str, citation_map: Dict[str, Dict]
    ) -> str:
        """
        校验引用

        PRD规范：
        - 若LLM输出中包含[引用编号]但该编号在citation_map中不存在，则判定为幻觉引用

        Args:
            answer: LLM生成的回答
            citation_map: 引用映射

        Returns:
            校验后的回答
        """
        # 提取回答中的引用编号
        cited_ids = set(re.findall(r'\[(\d+)\]', answer))

        # 检查是否有幻觉引用
        hallucinated = []
        for cid in cited_ids:
            if cid not in citation_map:
                hallucinated.append(cid)

        if hallucinated:
            logger.warning(f"Hallucinated citations detected: {hallucinated}")
            # 移除幻觉引用
            for cid in hallucinated:
                answer = answer.replace(f"[{cid}]", "")

            # 添加不确定性声明
            if not answer.strip().endswith("。"):
                answer += "。"
            answer += "\n\n⚠️ 注意：部分引用可能不准确，请以原始食谱为准。"

        return answer

    def _build_citations(
        self, answer: str, citation_map: Dict[str, Dict]
    ) -> List[Citation]:
        """
        构建结构化引用

        Args:
            answer: 回答文本
            citation_map: 引用映射

        Returns:
            引用列表
        """
        cited_ids = set(re.findall(r'\[(\d+)\]', answer))
        citations = []

        for cid in sorted(cited_ids, key=int):
            if cid in citation_map:
                meta = citation_map[cid]
                citations.append(Citation(
                    citation_id=f"ref_{cid}",
                    recipe_title=meta["recipe_title"],
                    chunk_type=meta["chunk_type"],
                    step_number=meta.get("step_number"),
                    excerpt=meta["excerpt"],
                    document_source=meta["document_source"],
                    image_path=meta.get("image_path"),
                ))

        return citations


def build_citation_map(retrieved_docs: List[RetrievalResult]) -> Dict[str, Dict]:
    """
    为检索到的文档构建引用映射

    Args:
        retrieved_docs: 检索结果列表

    Returns:
        引用映射字典
    """
    citation_map = {}
    for idx, doc in enumerate(retrieved_docs, start=1):
        meta = doc.metadata
        citation_map[str(idx)] = {
            "recipe_title": meta.get("recipe_title", "未知菜谱"),
            "chunk_type": meta.get("chunk_type", "unknown"),
            "step_number": meta.get("step_number"),
            "excerpt": doc.page_content[:100] if hasattr(doc, 'page_content') else doc.content[:100],
            "document_source": meta.get("document_source", "未知来源"),
        }
    return citation_map


def format_final_response(llm_output: str, citation_map: Dict[str, Dict]) -> Dict[str, Any]:
    """
    将LLM输出与引用映射合并为结构化响应

    Args:
        llm_output: LLM输出文本
        citation_map: 引用映射

    Returns:
        结构化响应
    """
    cited_ids = set(re.findall(r'\[(\d+)\]', llm_output))

    citations = []
    for cid in sorted(cited_ids, key=int):
        if cid in citation_map:
            citations.append({
                "citation_id": f"ref_{cid}",
                "recipe_title": citation_map[cid]["recipe_title"],
                "document_source": citation_map[cid]["document_source"],
                "chunk_type": citation_map[cid]["chunk_type"],
                "step_number": citation_map[cid]["step_number"],
                "excerpt": citation_map[cid]["excerpt"],
                "image_path": citation_map[cid].get("image_path"),
            })

    return {
        "answer": llm_output,
        "citations": citations,
        "has_related_content": len(citations) > 0,
    }


# 创建全局实例
answer_generator = AnswerGenerator()
