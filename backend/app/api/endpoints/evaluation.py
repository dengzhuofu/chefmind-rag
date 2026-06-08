"""
评估API端点
实现PRD第8节定义的评估方案

端点：
- GET  /evaluation/summary     获取评估摘要
- GET  /evaluation/test-cases  获取测试用例列表
- POST /evaluation/run         运行完整评估（调用RAG管道 + RAGAS）
- GET  /evaluation/results     获取最近一次评估结果
"""

import logging
import json
from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Optional

from app.services.evaluation_service import evaluation_service

logger = logging.getLogger(__name__)

router = APIRouter()

# 评估结果存储路径
RESULTS_DIR = Path(__file__).parent.parent.parent.parent / "evaluation_results"

# 后台任务状态
_eval_running = False
_last_result: Optional[dict] = None


@router.get("/evaluation/summary")
async def get_evaluation_summary():
    """
    获取评估摘要

    Returns:
        评估摘要，包含自定义指标和RAGAS指标
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
                "should_refuse": tc.should_refuse,
                "description": tc.description,
            }
            for tc in evaluation_service.test_cases
        ]
        return {"total": len(test_cases), "test_cases": test_cases}
    except Exception as e:
        logger.error(f"Get test cases error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluation/run")
async def run_evaluation(background_tasks: BackgroundTasks):
    """
    运行完整评估

    调用RAG管道对所有测试用例生成回答，然后运行RAGAS评估和自定义指标评估。

    Returns:
        评估结果摘要
    """
    global _eval_running, _last_result

    if _eval_running:
        raise HTTPException(
            status_code=409,
            detail="评估正在进行中，请等待完成后再试"
        )

    try:
        _eval_running = True
        logger.info("开始运行评估...")

        # 导入RAG管道
        from app.services.rag_pipeline import rag_pipeline

        # 运行完整评估
        summary = await evaluation_service.run_full_evaluation(rag_pipeline)

        # 保存结果
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = RESULTS_DIR / f"eval_{timestamp}.json"
        evaluation_service.export_results(str(result_path))

        _last_result = summary
        _eval_running = False

        logger.info(f"评估完成，结果保存到: {result_path}")
        return {
            "status": "success",
            "message": "Evaluation completed",
            "result_file": str(result_path),
            "summary": summary,
        }

    except Exception as e:
        _eval_running = False
        logger.error(f"Run evaluation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evaluation/status")
async def get_evaluation_status():
    """
    获取评估运行状态

    Returns:
        当前评估是否在运行，以及最近一次结果
    """
    return {
        "running": _eval_running,
        "has_result": _last_result is not None,
    }


@router.get("/evaluation/results")
async def get_evaluation_results():
    """
    获取最近一次评估结果

    Returns:
        最近一次评估的完整结果
    """
    if _last_result is None:
        # 尝试从文件加载最新的结果
        if RESULTS_DIR.exists():
            result_files = sorted(RESULTS_DIR.glob("eval_*.json"), reverse=True)
            if result_files:
                try:
                    with open(result_files[0], "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as e:
                    logger.error(f"加载评估结果文件失败: {e}")

        return {"message": "暂无评估结果，请先运行 POST /api/evaluation/run"}

    return _last_result
