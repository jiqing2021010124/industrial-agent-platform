"""ModelHub - 边缘模型调度网关

参考 ClawChips 的 model_hub_py 设计：
- 设备感知的资源调度
- 模型生命周期管理（启动/停止/健康检查）
- OpenAI 兼容 API
- 任务队列与并发控制
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .config import DeviceConfig, ModelConfig, PlatformConfig

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    FALLBACK_CLOUD = "fallback_cloud"


@dataclass
class InferenceTask:
    """推理任务"""

    task_id: str
    model_name: str
    inputs: dict[str, Any]
    task_type: str = "general"
    complexity: str = "low"  # low / medium / high
    data_sensitivity: str = "normal"  # normal / core_process_params
    latency_requirement_ms: int = 1000
    priority: str = "P2"  # P0/P1/P2/P3
    # 运行态
    status: TaskStatus = TaskStatus.QUEUED
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    routed_to: str = ""  # LOCAL / CLOUD
    device_name: str = ""
    token_count: int = 0

    @property
    def duration_ms(self) -> float:
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at) * 1000
        return 0.0


@dataclass
class MockModelBackend:
    """Mock 模型后端 - 模拟边缘 NPU 推理

    在真实环境中替换为 RKLLM / RKNN 调用。
    Mock 模式根据模型类型返回模拟推理结果，便于无硬件环境运行。
    """

    model: ModelConfig

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """模拟推理延迟 + 返回行业相关结果"""
        # 模拟 NPU 推理延迟
        latency = self._simulate_latency()
        await asyncio.sleep(latency / 1000)

        return {
            "model": self.model.name,
            "type": self.model.type,
            "latency_ms": latency,
            "result": self._generate_result(inputs),
            "backend": "edge-npu-mock",
        }

    def _simulate_latency(self) -> float:
        """根据模型类型与延迟要求模拟推理耗时"""
        base = min(self.model.latency_requirement_ms * 0.4, 800)
        import random

        return base + random.uniform(10, 80)

    def _generate_result(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """根据模型类型生成行业相关的模拟结果"""
        name = self.model.name

        if name == "chlor-alkali-voltage-forecast":
            history = inputs.get("voltage_history", [3.5, 3.48, 3.52])
            avg = sum(history) / len(history) if history else 3.5
            return {
                "forecast_24h": [round(avg + 0.01 * i, 3) for i in range(24)],
                "abnormal_warning": avg > 3.6,
                "suggestion": "碳酸钠加入量建议调整为 2.3 m³/h",
            }

        elif name == "yellow-phosphorus-batching":
            return {
                "optimal_ratio": {"磷矿": 0.62, "硅石": 0.22, "焦炭": 0.16},
                "estimated_energySaving": "15.3%",
                "confidence": 0.92,
            }

        elif name == "phosphogypsum-quality":
            grade = inputs.get("grade", "medium")
            return {
                "classification": grade,
                "recycling_path": "建材原料" if grade == "high" else "土壤改良剂",
                "environmental_compliance": True,
            }

        elif name == "aluminum-fluoride-dosing":
            temp = inputs.get("cell_temperature", 955)
            current = inputs.get("current", 420)
            dosage = round(18 + (temp - 950) * 0.3 + (current - 420) * 0.05, 2)
            return {
                "fluoride_dosage_kg": dosage,
                "confidence": 0.94,
                "error_margin": 0.18,
            }

        elif name == "cell-condition-diagnosis":
            return {
                "health_score": 87,
                "status": "正常",
                "warnings": [],
                "anomaly_patterns": [],
            }

        elif name == "evaporation-optimization":
            return {
                "steam_consumption_reduction": "3.2%",
                "annual_saving": "410万元",
                "param_adjustments": {"效一温度": "+2℃", "效三压力": "-0.5kPa"},
            }

        elif name == "industrial-vlm":
            return {
                "detected_objects": [
                    {"label": "未佩戴安全帽人员", "confidence": 0.96, "bbox": [120, 80, 200, 240]},
                ],
                "alert_level": "high",
                "description": "检测到 1 名未佩戴安全帽人员",
            }

        elif name == "industrial-rag":
            query = inputs.get("query", "")
            return {
                "answer": f"根据工艺手册，关于「{query}」的规范操作为：...",
                "source_docs": ["SOP-2024-001.pdf", "设备手册-v3.pdf"],
                "confidence": 0.85,
            }

        return {"message": "推理完成", "model": name}


class ModelHub:
    """模型调度网关 - 参考 ClawChips ModelHub 设计

    职责：
    1. 设备感知的资源调度（避免多模型资源争抢）
    2. 模型生命周期管理（健康检查）
    3. OpenAI 兼容的统一 API
    4. 任务队列与并发控制
    """

    def __init__(self, config: PlatformConfig):
        self.config = config
        self._backends: dict[str, MockModelBackend] = {}
        self._task_history: list[InferenceTask] = []
        self._init_backends()
        logger.info("ModelHub 初始化完成，注册模型 %d 个，设备 %d 个",
                    len(config.models), len(config.devices))

    def _init_backends(self) -> None:
        for m in self.config.models:
            self._backends[m.name] = MockModelBackend(m)

    def list_models(self) -> list[dict[str, Any]]:
        """列出所有已注册模型"""
        result = []
        for m in self.config.models:
            dev = self.config.get_device(m.deployed_on)
            result.append({
                "name": m.name,
                "type": m.type,
                "framework": m.framework,
                "deployed_on": m.deployed_on,
                "device_location": dev.location if dev else "",
                "vram_mb": m.vram_mb,
                "latency_requirement_ms": m.latency_requirement_ms,
                "cloud_fallback": m.cloud_fallback,
            })
        return result

    def list_devices(self) -> list[dict[str, Any]]:
        """列出所有边缘设备及其资源状态"""
        return [
            {
                "name": d.name,
                "hardware": d.hardware,
                "location": d.location,
                "npu_cores": d.npu_cores,
                "memory_mb": d.memory_mb,
                "available_vram_mb": d.available_vram_mb,
                "current_concurrency": d.current_concurrency,
                "max_concurrency": d.max_concurrency,
            }
            for d in self.config.devices
        ]

    async def dispatch_local(self, task: InferenceTask) -> InferenceTask:
        """调度任务到本地边缘设备"""
        model = self.config.get_model(task.model_name)
        if not model:
            task.status = TaskStatus.FAILED
            task.error = f"模型 {task.model_name} 未注册"
            return task

        device = self.config.find_device_for_model(task.model_name)
        if not device:
            task.status = TaskStatus.FAILED
            task.error = f"模型 {task.model_name} 未部署到任何设备"
            return task

        # 资源检查
        if not device.acquire(model.vram_mb):
            task.status = TaskStatus.FAILED
            task.error = f"设备 {device.name} 资源不足（VRAM={device.available_vram_mb}MB, 需 {model.vram_mb}MB）"
            logger.warning("设备 %s 资源不足，任务 %s 将转云端", device.name, task.task_id)
            task.routed_to = "CLOUD_FALLBACK"
            return task

        # 执行推理
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        task.device_name = device.name
        task.routed_to = "LOCAL"

        try:
            backend = self._backends.get(task.model_name)
            if not backend:
                raise RuntimeError(f"后端 {task.model_name} 未初始化")

            result = await backend.infer(task.inputs)
            task.result = result
            task.status = TaskStatus.SUCCESS
            task.finished_at = time.time()
            logger.info("任务 %s 在 %s 本地推理完成，耗时 %.0fms",
                        task.task_id, device.name, task.duration_ms)

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.finished_at = time.time()
            logger.error("任务 %s 本地推理失败: %s", task.task_id, e)

        finally:
            device.release(model.vram_mb)

        self._task_history.append(task)
        return task

    def get_stats(self) -> dict[str, Any]:
        """获取 ModelHub 运行统计"""
        total = len(self._task_history)
        success = sum(1 for t in self._task_history if t.status == TaskStatus.SUCCESS)
        failed = sum(1 for t in self._task_history if t.status == TaskStatus.FAILED)
        local = sum(1 for t in self._task_history if t.routed_to == "LOCAL")
        cloud = sum(1 for t in self._task_history if "CLOUD" in t.routed_to)

        avg_latency = 0.0
        if success > 0:
            avg_latency = sum(t.duration_ms for t in self._task_history if t.status == TaskStatus.SUCCESS) / success

        return {
            "total_tasks": total,
            "success": success,
            "failed": failed,
            "local_dispatched": local,
            "cloud_dispatched": cloud,
            "local_ratio": f"{local}/{total}" if total > 0 else "0/0",
            "avg_latency_ms": round(avg_latency, 1),
            "devices": self.list_devices(),
            "models": len(self.config.models),
        }
