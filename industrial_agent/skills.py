"""工业 SKILL 技能集

参考 ClawChips 的 skills/ 目录设计：
- 每个技能封装一个工业场景的完整调用链
- 技能可组合（多智能体协作）
- 技能声明路由优先级与安全约束
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from .modelhub import InferenceTask

logger = logging.getLogger(__name__)


@dataclass
class SkillSpec:
    """SKILL 规范定义 - 参考 ClawChips SKILL.yaml"""

    name: str
    version: str = "1.0.0"
    industry: str = ""
    scenario: str = ""
    description: str = ""
    model_name: str = ""
    task_type: str = "general"
    complexity: str = "low"
    data_sensitivity: str = "normal"
    latency_requirement_ms: int = 1000
    priority: str = "P2"  # P0实时控制 / P1实时监测 / P2分析决策 / P3离线分析
    cloud_provider: str = "deepseek"
    write_mode: str = "suggestion"  # suggestion / auto / human_confirm
    inputs_example: dict[str, Any] = field(default_factory=dict)

    def to_task(self, inputs: dict[str, Any]) -> InferenceTask:
        """将技能调用转换为推理任务"""
        return InferenceTask(
            task_id=f"task-{uuid.uuid4().hex[:8]}",
            model_name=self.model_name,
            inputs=inputs,
            task_type=self.task_type,
            complexity=self.complexity,
            data_sensitivity=self.data_sensitivity,
            latency_requirement_ms=self.latency_requirement_ms,
            priority=self.priority,
        )


# ── SKILL 注册表 ──────────────────────────────────────────

SKILL_REGISTRY: dict[str, SkillSpec] = {}


def register_skill(spec: SkillSpec) -> None:
    SKILL_REGISTRY[spec.name] = spec
    logger.info("注册 SKILL: %s v%s（%s）", spec.name, spec.version, spec.scenario)


def get_skill(name: str) -> SkillSpec | None:
    return SKILL_REGISTRY.get(name)


def list_skills() -> list[dict[str, Any]]:
    return [
        {
            "name": s.name,
            "version": s.version,
            "industry": s.industry,
            "scenario": s.scenario,
            "description": s.description,
            "model": s.model_name,
            "task_type": s.task_type,
            "complexity": s.complexity,
            "priority": s.priority,
            "data_sensitivity": s.data_sensitivity,
            "cloud_provider": s.cloud_provider,
            "write_mode": s.write_mode,
        }
        for s in SKILL_REGISTRY.values()
    ]


# ── 磷化工技能集 ──────────────────────────────────────────

register_skill(SkillSpec(
    name="phosphorus.chlor-alkali-voltage-forecast",
    version="1.0.0",
    industry="磷化工",
    scenario="氯碱装置总槽电压长周期预测",
    description="基于历史电压序列预测未来24h趋势，优化碳酸钠加入量，年省百万级",
    model_name="chlor-alkali-voltage-forecast",
    task_type="process_optimization",
    complexity="medium",
    data_sensitivity="core_process_params",
    latency_requirement_ms=1000,
    priority="P2",
    cloud_provider="deepseek",
    write_mode="suggestion",
    inputs_example={"voltage_history": [3.48, 3.50, 3.49, 3.51, 3.52]},
))

register_skill(SkillSpec(
    name="phosphorus.yellow-phosphorus-batching",
    version="1.0.0",
    industry="磷化工",
    scenario="黄磷生产智能配料",
    description="算法动态调整原料配比，能耗降低15%+",
    model_name="yellow-phosphorus-batching",
    task_type="process_optimization",
    complexity="high",
    data_sensitivity="core_process_params",
    latency_requirement_ms=500,
    priority="P2",
    cloud_provider="deepseek",
    write_mode="human_confirm",
    inputs_example={"phosphate_ore_grade": 0.28, "furnace_temp": 1450},
))

register_skill(SkillSpec(
    name="phosphorus.phosphogypsum-recycling",
    version="1.0.0",
    industry="磷化工",
    scenario="磷石膏资源化路径推荐",
    description="基于成分分析推荐资源化路径，环保达标率99%+",
    model_name="phosphogypsum-quality",
    task_type="classification",
    complexity="low",
    data_sensitivity="normal",
    priority="P2",
    cloud_provider="qwen",
    write_mode="suggestion",
    inputs_example={"grade": "medium", "p2o5_content": 0.8},
))

register_skill(SkillSpec(
    name="phosphorus.hazard-source-detection",
    version="1.0.0",
    industry="磷化工",
    scenario="重大危险源AI风险识别",
    description="基于VLM的自然语言目标检测，覆盖9处重大危险源",
    model_name="industrial-vlm",
    task_type="safety_monitoring",
    complexity="medium",
    data_sensitivity="normal",
    latency_requirement_ms=500,
    priority="P1",
    cloud_provider="qwen",
    write_mode="auto",
    inputs_example={"image_url": "camera://hazard-zone-1", "query": "检测未佩戴防护装备人员"},
))

# ── 电解铝技能集 ──────────────────────────────────────────

register_skill(SkillSpec(
    name="aluminum.fluoride-dosing",
    version="1.0.0",
    industry="电解铝",
    scenario="氟化铝精准下料",
    description="基于槽温+电流+电解质成分预测最优下料量，误差<0.2，吨铝节电100+度",
    model_name="aluminum-fluoride-dosing",
    task_type="real_time_control",
    complexity="low",
    data_sensitivity="core_process_params",
    latency_requirement_ms=500,
    priority="P0",
    cloud_provider="deepseek",
    write_mode="suggestion",
    inputs_example={"cell_temperature": 955, "current": 420, "electrolyte_ratio": 2.7},
))

register_skill(SkillSpec(
    name="aluminum.cell-diagnosis",
    version="1.0.0",
    industry="电解铝",
    scenario="电解槽况诊断",
    description="实时计算槽况健康度，异常模式秒级预警",
    model_name="cell-condition-diagnosis",
    task_type="real_time_control",
    complexity="medium",
    data_sensitivity="core_process_params",
    latency_requirement_ms=500,
    priority="P1",
    cloud_provider="qwen",
    write_mode="suggestion",
    inputs_example={"cell_id": "A-101", "voltage": 4.15, "temperature": 955},
))

register_skill(SkillSpec(
    name="aluminum.tapout-planning",
    version="1.0.0",
    industry="电解铝",
    scenario="出铝计划智能排产",
    description="多目标优化：产量最大化+能耗最小化+槽况均衡",
    model_name="industrial-rag",
    task_type="long_term_analysis",
    complexity="high",
    data_sensitivity="normal",
    priority="P3",
    cloud_provider="deepseek",
    write_mode="human_confirm",
    inputs_example={"cell_count": 500, "shift": "day"},
))

register_skill(SkillSpec(
    name="aluminum.evaporation-optimization",
    version="1.0.0",
    industry="电解铝",
    scenario="蒸发工序协同优化",
    description="8个小模型协同优化，蒸气消耗降低3%+，年创效400万",
    model_name="evaporation-optimization",
    task_type="process_optimization",
    complexity="high",
    data_sensitivity="normal",
    priority="P2",
    cloud_provider="deepseek",
    write_mode="suggestion",
    inputs_example={"effect1_temp": 120, "effect3_pressure": -15},
))

# ── 通用工业技能 ──────────────────────────────────────────

register_skill(SkillSpec(
    name="common.predictive-maintenance",
    version="1.0.0",
    industry="通用",
    scenario="设备预测性维护",
    description="基于振动/温度时序数据预测设备故障",
    model_name="cell-condition-diagnosis",
    task_type="real_time_control",
    complexity="medium",
    data_sensitivity="normal",
    priority="P1",
    cloud_provider="qwen",
    write_mode="suggestion",
    inputs_example={"device_id": "pump-01", "vibration": [0.1, 0.12, 0.15]},
))

register_skill(SkillSpec(
    name="common.safety-monitoring",
    version="1.0.0",
    industry="通用",
    scenario="安全监控与应急响应",
    description="VLM持续监测危险区域，秒级告警",
    model_name="industrial-vlm",
    task_type="safety_monitoring",
    complexity="medium",
    data_sensitivity="normal",
    latency_requirement_ms=500,
    priority="P1",
    cloud_provider="qwen",
    write_mode="auto",
    inputs_example={"image_url": "camera://zone-A", "query": "检测未戴安全帽人员"},
))

register_skill(SkillSpec(
    name="common.process-doc-qa",
    version="1.0.0",
    industry="通用",
    scenario="工艺文档智能问答",
    description="本地RAG检索工艺手册/SOP，数据不出端",
    model_name="industrial-rag",
    task_type="knowledge_qa",
    complexity="low",
    data_sensitivity="core_process_params",
    priority="P2",
    cloud_provider="deepseek",
    write_mode="suggestion",
    inputs_example={"query": "氟化铝添加量的标准操作流程是什么？"},
))
