"""Agent 编排引擎 - 参考 OpenClaw 设计

职责：
1. 接收用户请求，匹配对应 SKILL
2. 通过 SmartRouter 决策路由（本地 / 云端）
3. 调用 ModelHub 或 CloudClient 执行
4. 记录路由结果到记忆
5. 支持多技能组合（多智能体协作）
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from .cloud_client import CloudModelClient
from .config import PlatformConfig
from .modelhub import InferenceTask, ModelHub, TaskStatus
from .router import SmartRouter
from .skills import SkillSpec, get_skill, list_skills

logger = logging.getLogger(__name__)


@dataclass
class AgentRequest:
    """Agent 请求"""

    request_id: str = field(default_factory=lambda: f"req-{uuid.uuid4().hex[:8]}")
    skill_name: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    # 可选：直接指定模型（跳过 SKILL 匹配）
    model_name: str = ""
    task_type: str = ""
    complexity: str = ""
    data_sensitivity: str = ""
    priority: str = ""


@dataclass
class AgentResponse:
    """Agent 响应"""

    request_id: str
    skill_name: str = ""
    status: str = ""  # success / failed
    routed_to: str = ""  # LOCAL / CLOUD / LOCAL_DEGRADE
    result: Any = None
    error: str = ""
    latency_ms: float = 0.0
    route_reason: str = ""
    safety_net_triggered: bool = False
    token_count: int = 0
    cloud_mock: bool = False


class AgentEngine:
    """Agent 编排引擎"""

    def __init__(
        self,
        config: PlatformConfig,
        modelhub: ModelHub,
        router: SmartRouter,
        cloud_client: CloudModelClient,
    ):
        self.config = config
        self.modelhub = modelhub
        self.router = router
        self.cloud_client = cloud_client
        self._response_history: list[AgentResponse] = []

    async def execute(self, request: AgentRequest) -> AgentResponse:
        """执行 Agent 请求"""
        import time

        start = time.time()

        # 1. 匹配 SKILL
        skill = get_skill(request.skill_name)
        if not skill:
            return AgentResponse(
                request_id=request.request_id,
                status="failed",
                error=f"SKILL {request.skill_name} 未注册",
            )

        # 2. 构建推理任务
        task = skill.to_task(request.inputs)
        # 覆盖请求中指定的属性
        if request.task_type:
            task.task_type = request.task_type
        if request.complexity:
            task.complexity = request.complexity
        if request.data_sensitivity:
            task.data_sensitivity = request.data_sensitivity
        if request.priority:
            task.priority = request.priority

        # 3. 路由决策
        decision = self.router.decide(task)

        # 4. 执行
        if decision.target in ("LOCAL", "LOCAL_DEGRADE"):
            task = await self.modelhub.dispatch_local(task)
        elif decision.target == "CLOUD":
            task = await self._dispatch_cloud(task, skill)
        else:
            task = await self.modelhub.dispatch_local(task)

        # 5. 记录路由记忆
        self.router.record_result(task)

        # 6. 构建响应
        latency = (time.time() - start) * 1000
        response = AgentResponse(
            request_id=request.request_id,
            skill_name=request.skill_name,
            status=task.status.value if task.status == TaskStatus.SUCCESS else "failed",
            routed_to=task.routed_to or decision.target,
            result=task.result,
            error=task.error or "",
            latency_ms=round(latency, 1),
            route_reason=decision.reason,
            safety_net_triggered=decision.safety_net_triggered,
            token_count=task.token_count,
            cloud_mock=task.result.get("mock", False) if isinstance(task.result, dict) else False,
        )

        self._response_history.append(response)
        return response

    async def _dispatch_cloud(self, task: InferenceTask, skill: SkillSpec) -> InferenceTask:
        """调度到云端"""
        import time

        task.started_at = time.time()
        task.routed_to = "CLOUD"

        # 构建 OpenAI 消息
        user_content = f"[工业SKILL: {skill.scenario}]\n输入参数: {task.inputs}\n请基于工艺知识给出分析与建议。"
        messages = [
            {"role": "system", "content": "你是一个工业智能体，专注于流程制造工艺优化。"},
            {"role": "user", "content": user_content},
        ]

        model = self.config.get_model(task.model_name)
        provider_name = skill.cloud_provider or "deepseek"
        model_id = model.cloud_fallback if model else None

        result = await self.cloud_client.chat_completion(
            provider_name=provider_name,
            messages=messages,
            model_id=model_id,
        )

        if "error" in result:
            task.status = TaskStatus.FAILED
            task.error = result["error"]
        else:
            task.status = TaskStatus.SUCCESS
            task.result = result

        task.finished_at = time.time()
        logger.info("任务 %s 云端推理完成，耗时 %.0fms", task.task_id, task.duration_ms)
        return task

    def get_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """获取执行历史"""
        return [
            {
                "request_id": r.request_id,
                "skill_name": r.skill_name,
                "status": r.status,
                "routed_to": r.routed_to,
                "latency_ms": r.latency_ms,
                "route_reason": r.route_reason,
                "safety_net_triggered": r.safety_net_triggered,
                "token_count": r.token_count,
                "cloud_mock": r.cloud_mock,
            }
            for r in self._response_history[-limit:]
        ]

    def get_skills(self) -> list[dict[str, Any]]:
        return list_skills()
