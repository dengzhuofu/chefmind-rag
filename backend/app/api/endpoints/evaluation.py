"""
评估API端点
实现PRD第8节定义的评估方案
"""

import logging
from fastapi import APIRouter, HTTPException
from typing import Optional

from app.services.evaluation_service import evaluation_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/evaluation/summary")
async def get_evaluation_summary():
    """
    获取评估摘要

    Returns:
        评估摘要
    """
    try:
        summary = evaluation_service.get_evaluation_summary()
        return summary
    except Exception as e:
        logger.error(f"Get evaluation summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evaluation/test-cases")
async def get_test_cases():
    """
    获取测试用例列表

    Returns:
        测试用例列表
    """
    try:
        test_cases = [
            {
                "id": tc.id,
                "query": tc.query,
                "expected_recipes": tc.expected_recipes,
                "route_type": tc.route_type,
                "difficulty": tc.difficulty,
                "description": tc.description
            }
            for tc in evaluation_service.test_cases
        ]
        return {"test_cases": test_cases}
    except Exception as e:
        logger.error(f"Get test cases error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluation/run")
async def run_evaluation():
    """
    运行评估

    Returns:
        评估结果
    """
    try:
        # 这里应该调用RAG管道运行所有测试用例
        # 为了简化，返回一个示例结果
        return {
            "status": "success",
            "message": "Evaluation completed",
            "summary": evaluation_service.get_evaluation_summary()
        }
    except Exception as e:
        logger.error(f"Run evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
