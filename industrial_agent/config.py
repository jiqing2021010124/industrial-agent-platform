"""核心配置模块 - YAML 配置加载与设备/模型注册"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class DeviceConfig(BaseModel):
    """边缘设备配置"""

    name: str
    hardware: str = "RK3588+RK1828"
    location: str = ""
    npu_cores: int = 3
    memory_mb: int = 16384
    max_concurrency: int = 8
    # 运行态
    current_concurrency: int = 0
    available_vram_mb: int = 16384

    def acquire(self, vram_mb: int) -> bool:
        """申请资源"""
        if self.current_concurrency >= self.max_concurrency:
            return False
        if self.available_vram_mb < vram_mb:
            return False
        self.current_concurrency += 1
        self.available_vram_mb -= vram_mb
        return True

    def release(self, vram_mb: int) -> None:
        """释放资源"""
        self.current_concurrency = max(0, self.current_concurrency - 1)
        self.available_vram_mb = min(self.memory_mb, self.available_vram_mb + vram_mb)


class ModelConfig(BaseModel):
    """模型服务配置"""

    name: str
    type: str  # timeseries-forecast / optimization / regression / vision-language ...
    framework: str = "RKNN"
    input_desc: str = ""
    output_desc: str = ""
    vram_mb: int = 512
    deployed_on: str = ""  # 边缘设备名
    latency_requirement_ms: int = 1000
    token_limit: int = 4096
    cloud_fallback: str = ""  # 云端回退模型


class RouteRule(BaseModel):
    """路由规则"""

    match: dict[str, Any] = Field(default_factory=dict)
    target: str  # LOCAL / CLOUD
    priority: int = 0


class RouterConfig(BaseModel):
    """路由器配置"""

    strategy: str = "hybrid"  # rules / memory / hybrid
    enable: bool = True
    default_target: str = "CLOUD"
    rules: list[RouteRule] = Field(default_factory=list)
    memory_enable: bool = True
    safety_net_enable: bool = True
    token_limit: int = 4096
    memory_limit_mb: int = 2048
    latency_limit_ms: int = 2000
    network_check: bool = True
    degrade_strategy: str = "lightweight_model"


class CloudProviderConfig(BaseModel):
    """云端模型 Provider 配置"""

    name: str
    base_url: str
    api_key: str = ""
    model_id: str = ""
    timeout: int = 30


class PlatformConfig(BaseModel):
    """平台总配置"""

    devices: list[DeviceConfig] = Field(default_factory=list)
    models: list[ModelConfig] = Field(default_factory=list)
    router: RouterConfig = Field(default_factory=RouterConfig)
    cloud_providers: list[CloudProviderConfig] = Field(default_factory=list)

    # 索引
    _device_map: dict[str, DeviceConfig] = {}
    _model_map: dict[str, ModelConfig] = {}

    def index(self) -> None:
        """构建索引"""
        self._device_map = {d.name: d for d in self.devices}
        self._model_map = {m.name: m for m in self.models}

    def get_device(self, name: str) -> DeviceConfig | None:
        return self._device_map.get(name)

    def get_model(self, name: str) -> ModelConfig | None:
        return self._model_map.get(name)

    def find_model_by_type(self, model_type: str) -> ModelConfig | None:
        for m in self.models:
            if m.type == model_type:
                return m
        return None

    def find_device_for_model(self, model_name: str) -> DeviceConfig | None:
        m = self.get_model(model_name)
        if not m:
            return None
        return self.get_device(m.deployed_on)


# ── 默认配置 ──────────────────────────────────────────────

DEFAULT_CONFIG_YAML = """
# 云边端工业智能体平台配置
# 参考 airockchip/clawchips 的 clawchips.yaml 格式

devices:
  - name: edge-phosphorus-01
    hardware: RK3588+RK1828
    location: 瓮福江山氯碱车间
    npu_cores: 3
    memory_mb: 16384
    max_concurrency: 8

  - name: edge-phosphorus-02
    hardware: RK3588+RK1828
    location: 瓮福江山黄磷车间
    npu_cores: 3
    memory_mb: 16384
    max_concurrency: 8

  - name: edge-aluminum-01
    hardware: RK3588+RK1828
    location: 遵义铝业电解一车间
    npu_cores: 3
    memory_mb: 16384
    max_concurrency: 8

  - name: edge-aluminum-02
    hardware: RK3588+RK1828
    location: 遵义铝业蒸发工序
    npu_cores: 3
    memory_mb: 16384
    max_concurrency: 8

models:
  # —— 磷化工模型 ——
  - name: chlor-alkali-voltage-forecast
    type: timeseries-forecast
    framework: RKNN
    input_desc: 总槽电压历史序列
    output_desc: 未来24h电压预测
    vram_mb: 512
    deployed_on: edge-phosphorus-01
    latency_requirement_ms: 1000
    token_limit: 2048
    cloud_fallback: deepseek-chat

  - name: yellow-phosphorus-batching
    type: optimization
    framework: RKNN
    input_desc: 原料成分+反应釜参数
    output_desc: 最优配料比
    vram_mb: 1024
    deployed_on: edge-phosphorus-02
    latency_requirement_ms: 500
    cloud_fallback: deepseek-chat

  - name: phosphogypsum-quality
    type: classification
    framework: RKNN
    input_desc: 磷石膏成分数据
    output_desc: 资源化路径推荐
    vram_mb: 256
    deployed_on: edge-phosphorus-01
    cloud_fallback: qwen-plus

  # —— 电解铝模型 ——
  - name: aluminum-fluoride-dosing
    type: regression
    framework: RKNN
    input_desc: 槽温+电流+电解质成分
    output_desc: 氟化铝精准下料量
    vram_mb: 512
    deployed_on: edge-aluminum-01
    latency_requirement_ms: 500
    cloud_fallback: deepseek-chat

  - name: cell-condition-diagnosis
    type: anomaly-detection
    framework: RKNN
    input_desc: 电解槽多参数时序
    output_desc: 槽况健康度+异常预警
    vram_mb: 768
    deployed_on: edge-aluminum-01
    cloud_fallback: qwen-plus

  - name: evaporation-optimization
    type: optimization
    framework: RKNN
    input_desc: 蒸发工序各效参数
    output_desc: 蒸气消耗最小化建议
    vram_mb: 1024
    deployed_on: edge-aluminum-02
    cloud_fallback: deepseek-chat

  # —— 通用模型 ——
  - name: industrial-vlm
    type: vision-language
    framework: RKNN
    input_desc: 摄像头图像+自然语言描述
    output_desc: 目标检测结果
    vram_mb: 2048
    deployed_on: edge-phosphorus-01
    cloud_fallback: qwen-vl-plus

  - name: industrial-rag
    type: retrieval-augmented
    framework: RKNN
    input_desc: 工艺问题
    output_desc: 文档检索+答案
    vram_mb: 1024
    deployed_on: edge-aluminum-02
    cloud_fallback: deepseek-chat

router:
  strategy: hybrid
  enable: true
  default_target: CLOUD
  memory_enable: true
  safety_net_enable: true
  token_limit: 4096
  memory_limit_mb: 2048
  latency_limit_ms: 2000
  network_check: true
  degrade_strategy: lightweight_model
  rules:
    - match: {task_type: real_time_control}
      target: LOCAL
      priority: 100
    - match: {task_type: quality_inspection}
      target: LOCAL
      priority: 90
    - match: {task_type: process_optimization, complexity: low}
      target: LOCAL
      priority: 80
    - match: {task_type: process_optimization, complexity: high}
      target: CLOUD
      priority: 70
    - match: {task_type: long_term_analysis}
      target: CLOUD
      priority: 60
    - match: {task_type: knowledge_qa}
      target: LOCAL
      priority: 50
    - match: {data_sensitivity: core_process_params}
      target: LOCAL
      priority: 200  # 安全约束最高优先级

cloud_providers:
  - name: deepseek
    base_url: https://api.deepseek.com/v1
    api_key: ""
    model_id: deepseek-chat
    timeout: 30
  - name: qwen
    base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
    api_key: ""
    model_id: qwen-plus
    timeout: 30
"""


def load_config(config_path: str | Path | None = None) -> PlatformConfig:
    """加载配置文件，未指定则使用内置默认配置"""
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    else:
        data = yaml.safe_load(DEFAULT_CONFIG_YAML)

    cfg = PlatformConfig(**data)
    cfg.index()
    return cfg
