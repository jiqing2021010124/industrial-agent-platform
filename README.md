# 云边端工业智能体平台

> 基于 [airockchip/clawchips](https://github.com/airockchip/clawchips) 架构的流程制造工业智能体平台
> 面向贵州磷化（磷化工）、遵义铝业（电解铝）等流程制造场景

## 项目简介

本项目参考 ClawChips 的核心架构设计，构建了一个可高效运行的云边端协同工业智能体平台。平台实现了 ClawChips 的三大核心机制——**ModelHub 模型调度网关**、**端云智能路由**、**资源安全网**，并针对流程制造场景进行了工业化适配。

### 核心特性

- **ModelHub 边缘模型调度**：设备感知的资源调度，模型生命周期管理，OpenAI 兼容 API
- **端云智能路由**：规则路由 + 记忆路由，基于任务复杂度自动在本地/云端间分流
- **多层资源安全网**：Token 容量 + 内存 + 网络状态 + 工艺安全四级兜底
- **工业 SKILL 技能集**：磷化工（氯碱电压预测、黄磷配料、磷石膏资源化、危险源识别）+ 电解铝（氟化铝下料、槽况诊断、出铝排产、蒸发优化）+ 通用（预测性维护、安全监控、工艺问答）
- **Mock 模式**：无需 RK3588 硬件和云端 API Key 即可完整运行，模拟边缘 NPU 推理

### 与 ClawChips 的对应关系

| ClawChips 原始模块 | 本项目对应实现 | 文件 |
|--------------------|----------------|------|
| `model_hub_py/` ModelHub | `ModelHub` 边缘模型调度网关 | `industrial_agent/modelhub.py` |
| `clawchips-plugin/` Local Router | `SmartRouter` 端云智能路由 | `industrial_agent/router.py` |
| Context Router Proxy（Token 安全网） | `SafetyNet` 多层资源安全网 | `industrial_agent/router.py` |
| `skills/` RK SKILLs | 工业技能集（磷化工+电解铝） | `industrial_agent/skills.py` |
| OpenClaw Agent 网关 | `AgentEngine` 编排引擎 | `industrial_agent/agent.py` |
| 云端模型接入 | `CloudModelClient` OpenAI 兼容客户端 | `industrial_agent/cloud_client.py` |
| `clawchips.yaml` 配置 | YAML 配置 + 内置默认配置 | `industrial_agent/config.py` |
| Dashboard | FastAPI REST API | `industrial_agent/api.py` |

## 快速开始

### 环境要求

- Python 3.10+
- 无需任何硬件（Mock 模式模拟边缘 NPU 推理）

### 安装

```bash
pip install -r requirements.txt
```

### 方式一：运行演示脚本（推荐首次体验）

```bash
python demo.py
```

演示脚本展示 6 个核心场景：
1. **氟化铝精准下料**（P0 实时控制，强制本地边缘推理）
2. **氯碱装置电压预测**（核心工艺参数，安全网触发降级本地）
3. **出铝计划排产**（高复杂度，路由至云端大模型）
4. **安全监控**（VLM 目标检测）
5. **断网降级测试**（模拟云端不可用，自动降级本地）
6. **安全网触发**（高复杂度核心参数，降级本地轻量模型）

### 方式二：启动 API 服务

```bash
python -m industrial_agent.main
```

服务启动后访问：
- 欢迎页：http://localhost:8000
- API 文档：http://localhost:8000/docs
- ReDoc：http://localhost:8000/redoc

### API 调用示例

```bash
# 1. 查看所有 SKILL
curl http://localhost:8000/api/skills

# 2. 查看边缘设备状态
curl http://localhost:8000/api/devices

# 3. 执行推理 - 氟化铝精准下料（本地边缘推理）
curl -X POST http://localhost:8000/api/infer \
  -H "Content-Type: application/json" \
  -d '{"skill_name": "aluminum.fluoride-dosing", "inputs": {"cell_temperature": 955, "current": 420}}'

# 4. 执行推理 - 出铝计划（云端大模型）
curl -X POST http://localhost:8000/api/infer \
  -H "Content-Type: application/json" \
  -d '{"skill_name": "aluminum.tapout-planning", "inputs": {"cell_count": 500, "shift": "day"}}'

# 5. 查看运行统计
curl http://localhost:8000/api/stats

# 6. 查看路由记忆
curl http://localhost:8000/api/router/memory

# 7. 模拟断网（测试降级策略）
curl -X POST http://localhost:8000/api/router/cloud \
  -H "Content-Type: application/json" \
  -d '{"available": false}'
```

## 项目结构

```
20260816205846/
├── industrial_agent/           # 主包
│   ├── __init__.py
│   ├── config.py               # YAML 配置加载（设备/模型/路由）
│   ├── modelhub.py             # ModelHub 边缘模型调度网关
│   ├── router.py               # 端云智能路由 + 资源安全网
│   ├── cloud_client.py         # 云端模型客户端（OpenAI 兼容）
│   ├── skills.py               # 工业 SKILL 技能集
│   ├── agent.py                # Agent 编排引擎
│   ├── api.py                  # FastAPI Dashboard API
│   └── main.py                 # 启动入口
├── demo.py                     # 演示脚本
├── requirements.txt            # 依赖
├── pyproject.toml              # 项目元数据
└── README.md                   # 本文件
```

## 架构设计

### 两层路由决策架构（参考 ClawChips）

```
┌──────────────────────────────────────────────────────┐
│  第一层：任务复杂度判定（SmartRouter 智能路由）         │
│  - 规则路由：task_type / complexity / data_sensitivity │
│  - 记忆路由：基于历史成功率优化决策                     │
│  - 安全约束：核心工艺参数强制本地                      │
└────────────────────┬─────────────────────────────────┘
                     ↓ 本地
┌──────────────────────────────────────────────────────┐
│  第二层：资源容量兜底（SafetyNet 多层安全网）           │
│  - Token 容量检查（参考 Context Router Proxy）         │
│  - 内存容量检查                                        │
│  - 网络状态检查（断网降级）                            │
│  - 工艺安全检查（核心参数不转云端）                    │
└──────────────────────────────────────────────────────┘
```

### 路由决策因子

| 决策因子 | 说明 | 路由影响 |
|----------|------|----------|
| 任务复杂度 | low / medium / high | 简单→本地，复杂→云端 |
| 实时性要求 | P0/P1/P2/P3 优先级 | P0 强制本地 |
| 数据敏感性 | normal / core_process_params | 核心参数强制本地 |
| Token 容量 | 估算 token 数 | 超限→安全网转云端 |
| 设备负载 | VRAM + 并发 | 过载→转云端 |
| 网络状态 | 云端连通性 | 断网→降级本地 |
| 路由记忆 | 历史成功率 | 低成功率→转云端 |

## 工业 SKILL 列表

### 磷化工场景

| SKILL | 场景 | 模型 | 优先级 | 路由 |
|-------|------|------|--------|------|
| `phosphorus.chlor-alkali-voltage-forecast` | 氯碱装置电压预测 | timeseries-forecast | P2 | 本地（核心参数） |
| `phosphorus.yellow-phosphorus-batching` | 黄磷智能配料 | optimization | P2 | 本地（核心参数） |
| `phosphorus.phosphogypsum-recycling` | 磷石膏资源化 | classification | P2 | 本地 |
| `phosphorus.hazard-source-detection` | 重大危险源识别 | vision-language | P1 | 本地 |

### 电解铝场景

| SKILL | 场景 | 模型 | 优先级 | 路由 |
|-------|------|------|--------|------|
| `aluminum.fluoride-dosing` | 氟化铝精准下料 | regression | P0 | 本地（实时控制） |
| `aluminum.cell-diagnosis` | 电解槽况诊断 | anomaly-detection | P1 | 本地 |
| `aluminum.tapout-planning` | 出铝计划排产 | retrieval-augmented | P3 | 云端（高复杂度） |
| `aluminum.evaporation-optimization` | 蒸发工序协同优化 | optimization | P2 | 本地/云端 |

### 通用场景

| SKILL | 场景 | 模型 | 优先级 |
|-------|------|------|--------|
| `common.predictive-maintenance` | 设备预测性维护 | anomaly-detection | P1 |
| `common.safety-monitoring` | 安全监控 | vision-language | P1 |
| `common.process-doc-qa` | 工艺文档问答 | retrieval-augmented | P2 |

## 配置真实云端 API（可选）

编辑 `industrial_agent/config.py` 中的 `DEFAULT_CONFIG_YAML`，或在项目根目录创建 `config.yaml`：

```yaml
cloud_providers:
  - name: deepseek
    base_url: https://api.deepseek.com/v1
    api_key: sk-your-real-api-key
    model_id: deepseek-chat
    timeout: 30
  - name: qwen
    base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
    api_key: sk-your-real-api-key
    model_id: qwen-plus
    timeout: 30
```

配置真实 API Key 后，云端路由将调用真实大模型；未配置时自动使用 Mock 模式。

## 对接真实边缘硬件（RK3588+RK1828）

当前项目使用 `MockModelBackend` 模拟边缘 NPU 推理。对接真实硬件时：

1. 替换 `industrial_agent/modelhub.py` 中的 `MockModelBackend.infer()` 方法
2. 调用 RKNN-Toolkit2 或 RKLLM Server 的 API
3. 参考 ClawChips 的 `model_hub_py/` 实现真实的模型加载与推理

```python
# 真实硬件对接示例
class RKNNModelBackend:
    async def infer(self, inputs):
        # 调用本地 RKLLM Server（OpenAI 兼容 API）
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "http://127.0.0.1:8080/v1/chat/completions",
                json={"model": "Qwen3-4B", "messages": [...]}
            )
            return resp.json()
```

## 参考项目

- [airockchip/clawchips](https://github.com/airockchip/clawchips) — 瑞芯微开源边缘 AI Agent 部署方案
- [ClawChips 架构与原理](https://forum.shimetapi.cn/wiki/zh/ai-model/RK1828/application-development/ch01-ClawChips架构与原理-RK1828.html)
- [ClawChips 智能路由 + Token 安全网](https://www.bearkey.com.cn/article/ClawChips智能路由Token安全网在RK3588RK1828实现OpenClaw端云协同的稳定对话.html)

## License

MIT
