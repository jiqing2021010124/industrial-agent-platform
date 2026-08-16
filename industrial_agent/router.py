"""端云智能路由 + 资源安全网

参考 ClawChips 的 Local Router + Context Router Proxy（Token 安全网）设计：
- 第一层：任务复杂度判定（规则路由 + 记忆路由）
- 第二层：资源容量兜底（Token / 内存 / 延迟安全网）
- 第三层：网络状态兜底（断网降级）
- 第四层：工艺安全兜底（安全等级强制本地 + 人工确认）
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .config import PlatformConfig
from .modelhub import InferenceTask, TaskStatus

logger = logging.getLogger(__name__)


@dataclass
class RouteDecision:
    """路由决策结果"""

    target: str  # LOCAL / CLOUD
    reason: str = ""
    safety_net_triggered: bool = False
    safety_net_reason: str = ""
    memory_used: bool = False
    confidence: float = 1.0


@dataclass
class RouteMemory:
    """路由记忆 - 持续优化决策"""

    entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    # key: f"{task_type}:{model_name}", value: {local_success, local_fail, cloud_success, last_decision}

    def record(self, task: InferenceTask, success: bool) -> None:
        key = f"{task.task_type}:{task.model_name}"
        if key not in self.entries:
            self.entries[key] = {"local_success": 0, "local_fail": 0, "cloud_success": 0, "cloud_fail": 0}

        if task.routed_to == "LOCAL":
            if success:
                self.entries[key]["local_success"] += 1
            else:
                self.entries[key]["local_fail"] += 1
        elif "CLOUD" in task.routed_to:
            if success:
                self.entries[key]["cloud_success"] += 1
            else:
                self.entries[key]["cloud_fail"] += 1

    def get_stats(self, task_type: str, model_name: str) -> dict[str, Any] | None:
        key = f"{task_type}:{model_name}"
        return self.entries.get(key)

    def should_prefer_cloud(self, task_type: str, model_name: str) -> bool:
        """基于记忆判断是否应优先走云端"""
        stats = self.get_stats(task_type, model_name)
        if not stats:
            return False
        local_total = stats["local_success"] + stats["local_fail"]
        if local_total < 3:
            return False
        local_success_rate = stats["local_success"] / local_total if local_total > 0 else 0
        return local_success_rate < 0.5  # 本地成功率低于 50% 则转云端


class SafetyNet:
    """资源安全网 - 参考 ClawChips 的 Context Router Proxy

    多层兜底：
    1. Token 容量检查（模拟 Context Router Proxy 的真实 token 统计）
    2. 内存容量检查
    3. 延迟要求检查
    4. 网络连通性检查
    5. 工艺安全等级检查
    """

    def __init__(self, config: PlatformConfig):
        self.config = config
        self.router_cfg = config.router

    def check(self, task: InferenceTask) -> tuple[bool, str]:
        """检查任务是否可以安全地在本地执行

        Returns:
            (passed, reason) - passed=True 表示可本地执行，False 表示需转云端
        """
        model = self.config.get_model(task.model_name)
        if not model:
            return False, f"模型 {task.model_name} 未注册"

        # 1. Token 容量检查
        estimated_tokens = self._estimate_tokens(task)
        task.token_count = estimated_tokens
        if estimated_tokens > self.router_cfg.token_limit:
            return False, f"Token 超限（{estimated_tokens} > {self.router_cfg.token_limit}）"

        if estimated_tokens > model.token_limit:
            return False, f"模型 Token 限制（{estimated_tokens} > {model.token_limit}）"

        # 2. 内存容量检查
        device = self.config.find_device_for_model(task.model_name)
        if device and device.available_vram_mb < model.vram_mb:
            return False, f"设备 VRAM 不足（{device.available_vram_mb} < {model.vram_mb}MB）"

        # 3. 延迟要求检查（如果模型本身延迟要求高于任务要求，可本地执行）
        # 此处简化：默认可本地执行

        # 4. 工艺安全检查 - 核心工艺参数强制本地（不转云端）
        if task.data_sensitivity == "core_process_params":
            return True, "核心工艺参数，强制本地（安全约束）"

        return True, "OK"

    def _estimate_tokens(self, task: InferenceTask) -> int:
        """估算任务的 Token 数

        真实场景中由 Context Router Proxy 统计完整请求体；
        Mock 模式基于任务复杂度与输入数据量估算。
        """
        base = 500  # 基础 system prompt
        complexity_map = {"low": 500, "medium": 2000, "high": 5000}
        base += complexity_map.get(task.complexity, 1000)

        # 输入数据量影响
        for v in task.inputs.values():
            if isinstance(v, list):
                base += len(v) * 10
            elif isinstance(v, str):
                base += len(v) // 3
            else:
                base += 20

        return base


class SmartRouter:
    """端云智能路由器 - 参考 ClawChips Local Router

    两层决策架构：
    第一层：任务复杂度判定（规则 + 记忆）
    第二层：资源安全网兜底
    """

    def __init__(self, config: PlatformConfig):
        self.config = config
        self.safety_net = SafetyNet(config)
        self.memory = RouteMemory() if config.router.memory_enable else None
        self._cloud_available = True  # 模拟网络状态

    def set_cloud_available(self, available: bool) -> None:
        """模拟网络状态变化"""
        self._cloud_available = available
        logger.info("云端可用性变更: %s", available)

    def decide(self, task: InferenceTask) -> RouteDecision:
        """路由决策"""
        if not self.config.router.enable:
            return RouteDecision(target=self.config.router.default_target, reason="路由未启用")

        # ── 第一层：规则路由 ──────────────────────────────
        decision = self._rule_based_route(task)

        # 记忆路由优化
        if self.memory and decision.target == "LOCAL":
            if self.memory.should_prefer_cloud(task.task_type, task.model_name):
                decision.target = "CLOUD"
                decision.reason = "记忆路由：历史本地成功率低，转云端"
                decision.memory_used = True

        # ── 第二层：安全网兜底 ────────────────────────────
        if decision.target == "LOCAL":
            passed, reason = self.safety_net.check(task)
            if not passed:
                # 安全网触发，转云端
                decision.safety_net_triggered = True
                decision.safety_net_reason = reason
                if task.data_sensitivity == "core_process_params":
                    # 核心工艺参数即使超限也不能转云端 → 降级为轻量模型
                    decision.target = "LOCAL_DEGRADE"
                    decision.reason = f"安全网触发但数据敏感，降级本地轻量模型（{reason}）"
                else:
                    decision.target = "CLOUD"
                    decision.reason = f"安全网触发，转云端（{reason}）"

        # ── 第三层：网络状态兜底 ──────────────────────────
        if decision.target == "CLOUD" and not self._cloud_available:
            decision.target = "LOCAL_DEGRADE"
            decision.reason = "云端不可用（断网），降级本地轻量模型"

        # ── 第四层：工艺安全兜底 ──────────────────────────
        if task.data_sensitivity == "core_process_params" and decision.target == "CLOUD":
            decision.target = "LOCAL_DEGRADE"
            decision.reason = "工艺安全约束：核心参数禁止出境，降级本地"

        logger.info("路由决策: 任务=%s → %s（%s）",
                    task.task_id, decision.target, decision.reason)
        return decision

    def _rule_based_route(self, task: InferenceTask) -> RouteDecision:
        """基于规则的路由决策"""
        # 安全约束最高优先级
        if task.data_sensitivity == "core_process_params":
            return RouteDecision(target="LOCAL", reason="核心工艺参数强制本地", confidence=1.0)

        # 实时控制类强制本地
        if task.priority == "P0":
            return RouteDecision(target="LOCAL", reason="P0 实时控制，强制本地")

        # 匹配路由规则
        for rule in self.config.router.rules:
            if self._match_rule(rule.match, task):
                return RouteDecision(target=rule.target, reason=f"规则匹配: {rule.match}")

        # 默认路由
        return RouteDecision(
            target=self.config.router.default_target,
            reason="默认路由",
        )

    def _match_rule(self, match: dict[str, Any], task: InferenceTask) -> bool:
        """检查任务是否匹配规则"""
        for key, value in match.items():
            if key == "task_type" and task.task_type != value:
                return False
            elif key == "complexity" and task.complexity != value:
                return False
            elif key == "data_sensitivity" and task.data_sensitivity != value:
                return False
        return True

    def record_result(self, task: InferenceTask) -> None:
        """记录路由结果到记忆"""
        if self.memory:
            success = task.status == TaskStatus.SUCCESS
            self.memory.record(task, success)

    def get_memory_stats(self) -> dict[str, Any]:
        """获取路由记忆统计"""
        if not self.memory:
            return {"enabled": False}
        return {
            "enabled": True,
            "entries": self.memory.entries,
        }
