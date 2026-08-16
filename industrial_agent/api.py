"""FastAPI 应用 - Dashboard API + 推理接口

提供 REST API：
- GET  /             欢迎页
- GET  /api/skills   列出所有 SKILL
- GET  /api/devices  列出边缘设备状态
- GET  /api/models   列出已注册模型
- POST /api/infer    执行推理（核心接口）
- GET  /api/stats    获取运行统计
- GET  /api/history  获取执行历史
- GET  /api/router/memory  获取路由记忆
- POST /api/router/cloud  模拟云端可用性切换
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .agent import AgentEngine, AgentRequest
from .cloud_client import CloudModelClient
from .config import load_config
from .modelhub import ModelHub
from .router import SmartRouter
from .skills import list_skills

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── 全局组件 ──────────────────────────────────────────────

_platform_config = None
_modelhub: ModelHub | None = None
_router: SmartRouter | None = None
_cloud_client: CloudModelClient | None = None
_agent: AgentEngine | None = None


def init_platform(config_path: str | Path | None = None) -> None:
    """初始化平台组件"""
    global _platform_config, _modelhub, _router, _cloud_client, _agent

    _platform_config = load_config(config_path)
    _modelhub = ModelHub(_platform_config)
    _router = SmartRouter(_platform_config)
    _cloud_client = CloudModelClient(_platform_config.cloud_providers)
    _agent = AgentEngine(_platform_config, _modelhub, _router, _cloud_client)

    logger.info("平台初始化完成")
    logger.info("  模型数: %d", len(_platform_config.models))
    logger.info("  设备数: %d", len(_platform_config.devices))
    logger.info("  云端可用: %s (mock=%s)",
                _cloud_client.available, not _cloud_client._has_real_keys())


# ── API 模型 ──────────────────────────────────────────────

class InferRequest(BaseModel):
    """推理请求"""

    skill_name: str = Field(..., description="SKILL 名称，如 aluminum.fluoride-dosing")
    inputs: dict[str, Any] = Field(default_factory=dict, description="输入参数")
    # 可选覆盖
    task_type: str = ""
    complexity: str = ""
    data_sensitivity: str = ""
    priority: str = ""


class CloudToggleRequest(BaseModel):
    available: bool


# ── FastAPI 应用 ──────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_platform()
    yield


app = FastAPI(
    title="云边端工业智能体平台",
    description="基于 ClawChips 架构的流程制造工业智能体平台 - 磷化工/电解铝场景",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
async def index():
    """欢迎页"""
    return """
    <!DOCTYPE html>
    <html><head><meta charset="utf-8"><title>云边端工业智能体平台</title></head>
    <body style="font-family: sans-serif; max-width: 900px; margin: 40px auto; padding: 20px;">
      <h1>🏭 云边端工业智能体平台</h1>
      <p>基于 <a href="https://github.com/airockchip/clawchips">ClawChips</a> 架构的流程制造工业智能体平台</p>
      <h2>核心模块</h2>
      <ul>
        <li><b>ModelHub</b> — 边缘模型调度网关（设备感知 + 资源调度）</li>
        <li><b>SmartRouter</b> — 端云智能路由（规则 + 记忆 + 安全网）</li>
        <li><b>AgentEngine</b> — Agent 编排引擎（SKILL 匹配 + 多模型协同）</li>
        <li><b>SKILL 商店</b> — 磷化工 + 电解铝场景化技能集</li>
      </ul>
      <h2>快速开始</h2>
      <pre style="background: #f4f4f4; padding: 15px; border-radius: 5px;">
# 1. 查看可用 SKILL
curl http://localhost:8000/api/skills

# 2. 执行氟化铝精准下料（本地边缘推理）
curl -X POST http://localhost:8000/api/infer \\
  -H "Content-Type: application/json" \\
  -d '{"skill_name": "aluminum.fluoride-dosing", "inputs": {"cell_temperature": 955, "current": 420}}'

# 3. 查看运行统计
curl http://localhost:8000/api/stats

# 4. 查看路由记忆
curl http://localhost:8000/api/router/memory
      </pre>
      <h2>API 文档</h2>
      <p><a href="/docs">Swagger UI</a> | <a href="/redoc">ReDoc</a></p>
    </body></html>
    """


@app.get("/api/skills")
async def api_list_skills():
    """列出所有 SKILL"""
    return {"skills": list_skills(), "total": len(list_skills())}


@app.get("/api/devices")
async def api_list_devices():
    """列出边缘设备状态"""
    if not _modelhub:
        raise HTTPException(500, "平台未初始化")
    return {"devices": _modelhub.list_devices()}


@app.get("/api/models")
async def api_list_models():
    """列出已注册模型"""
    if not _modelhub:
        raise HTTPException(500, "平台未初始化")
    return {"models": _modelhub.list_models()}


@app.post("/api/infer")
async def api_infer(req: InferRequest):
    """执行推理（核心接口）"""
    if not _agent:
        raise HTTPException(500, "平台未初始化")

    agent_req = AgentRequest(
        skill_name=req.skill_name,
        inputs=req.inputs,
        task_type=req.task_type,
        complexity=req.complexity,
        data_sensitivity=req.data_sensitivity,
        priority=req.priority,
    )
    resp = await _agent.execute(agent_req)
    return {
        "request_id": resp.request_id,
        "skill_name": resp.skill_name,
        "status": resp.status,
        "routed_to": resp.routed_to,
        "route_reason": resp.route_reason,
        "safety_net_triggered": resp.safety_net_triggered,
        "token_count": resp.token_count,
        "latency_ms": resp.latency_ms,
        "cloud_mock": resp.cloud_mock,
        "result": resp.result,
        "error": resp.error,
    }


@app.get("/api/stats")
async def api_stats():
    """获取运行统计"""
    if not _modelhub:
        raise HTTPException(500, "平台未初始化")
    return {
        "modelhub": _modelhub.get_stats(),
        "router_memory": _router.get_memory_stats() if _router else {},
        "cloud_available": _cloud_client.available if _cloud_client else False,
        "cloud_mock_mode": not _cloud_client._has_real_keys() if _cloud_client else True,
    }


@app.get("/api/history")
async def api_history(limit: int = 20):
    """获取执行历史"""
    if not _agent:
        raise HTTPException(500, "平台未初始化")
    return {"history": _agent.get_history(limit)}


@app.get("/api/router/memory")
async def api_router_memory():
    """获取路由记忆"""
    if not _router:
        raise HTTPException(500, "平台未初始化")
    return _router.get_memory_stats()


@app.post("/api/router/cloud")
async def api_toggle_cloud(req: CloudToggleRequest):
    """模拟云端可用性切换（用于测试断网降级）"""
    if not _router:
        raise HTTPException(500, "平台未初始化")
    _router.set_cloud_available(req.available)
    return {"cloud_available": req.available, "message": "云端可用性已切换"}
