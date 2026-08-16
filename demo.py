#!/usr/bin/env python
"""演示脚本 - 展示云边端智能体平台的核心能力

运行方式: python demo.py
无需启动 API 服务，直接在终端展示路由决策与推理效果。
"""

import asyncio
import json
import sys

from industrial_agent.agent import AgentEngine, AgentRequest
from industrial_agent.cloud_client import CloudModelClient
from industrial_agent.config import load_config
from industrial_agent.modelhub import ModelHub
from industrial_agent.router import SmartRouter
from industrial_agent.skills import list_skills


def print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(label: str, data: dict) -> None:
    print(f"\n  [{label}]")
    print(f"    路由目标: {data.get('routed_to', 'N/A')}")
    print(f"    路由原因: {data.get('route_reason', 'N/A')}")
    print(f"    安全网触发: {data.get('safety_net_triggered', False)}")
    print(f"    Token 估算: {data.get('token_count', 0)}")
    print(f"    耗时: {data.get('latency_ms', 0):.1f} ms")
    print(f"    状态: {data.get('status', 'N/A')}")

    result = data.get("result", {})
    if isinstance(result, dict):
        for k, v in result.items():
            if k in ("choices", "usage", "object", "id", "model"):
                continue
            vstr = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
            if len(vstr) > 120:
                vstr = vstr[:120] + "..."
            print(f"    {k}: {vstr}")


async def run_demo():
    print_header("云边端工业智能体平台 - 演示")

    config = load_config()
    modelhub = ModelHub(config)
    router = SmartRouter(config)
    cloud_client = CloudModelClient(config.cloud_providers)
    agent = AgentEngine(config, modelhub, router, cloud_client)

    print(f"\n  已加载 {len(config.models)} 个模型, {len(config.devices)} 个边缘设备")
    print(f"  已注册 {len(list_skills())} 个 SKILL")
    mock = cloud_client._has_real_keys()
    print(f"  云端模式: {'Mock' if not mock else '真实 API'}")

    # 场景 1: 氟化铝精准下料（P0 实时控制, 强制本地）
    print_header("场景 1: 电解铝 - 氟化铝精准下料 (P0 实时控制)")
    resp = await agent.execute(AgentRequest(
        skill_name="aluminum.fluoride-dosing",
        inputs={"cell_temperature": 955, "current": 420, "electrolyte_ratio": 2.7},
    ))
    result = resp.result.get("result", resp.result) if isinstance(resp.result, dict) else resp.result
    print_result("氟化铝下料", {
        "routed_to": resp.routed_to, "route_reason": resp.route_reason,
        "safety_net_triggered": resp.safety_net_triggered,
        "token_count": resp.token_count, "latency_ms": resp.latency_ms,
        "status": resp.status, "result": result,
    })

    # 场景 2: 氯碱电压预测（核心工艺参数, 强制本地）
    print_header("场景 2: 磷化工 - 氯碱装置总槽电压预测")
    resp = await agent.execute(AgentRequest(
        skill_name="phosphorus.chlor-alkali-voltage-forecast",
        inputs={"voltage_history": [3.48, 3.50, 3.49, 3.51, 3.52, 3.50, 3.49, 3.51]},
    ))
    result = resp.result.get("result", resp.result) if isinstance(resp.result, dict) else resp.result
    print_result("氯碱电压预测", {
        "routed_to": resp.routed_to, "route_reason": resp.route_reason,
        "safety_net_triggered": resp.safety_net_triggered,
        "token_count": resp.token_count, "latency_ms": resp.latency_ms,
        "status": resp.status, "result": result,
    })

    # 场景 3: 出铝计划（高复杂度, 转云端）
    print_header("场景 3: 电解铝 - 出铝计划智能排产 (高复杂度 -> 云端)")
    resp = await agent.execute(AgentRequest(
        skill_name="aluminum.tapout-planning",
        inputs={"cell_count": 500, "shift": "day", "target_output": 1200},
    ))
    print_result("出铝排产", {
        "routed_to": resp.routed_to, "route_reason": resp.route_reason,
        "safety_net_triggered": resp.safety_net_triggered,
        "token_count": resp.token_count, "latency_ms": resp.latency_ms,
        "status": resp.status, "cloud_mock": resp.cloud_mock,
        "result": resp.result,
    })

    # 场景 4: 安全监控（VLM 检测, P1 优先级）
    print_header("场景 4: 通用 - 安全监控 (VLM 目标检测)")
    resp = await agent.execute(AgentRequest(
        skill_name="common.safety-monitoring",
        inputs={"image_url": "camera://zone-A", "query": "检测未戴安全帽人员"},
    ))
    result = resp.result.get("result", resp.result) if isinstance(resp.result, dict) else resp.result
    print_result("安全监控", {
        "routed_to": resp.routed_to, "route_reason": resp.route_reason,
        "safety_net_triggered": resp.safety_net_triggered,
        "token_count": resp.token_count, "latency_ms": resp.latency_ms,
        "status": resp.status, "result": result,
    })

    # 场景 5: 断网降级测试
    print_header("场景 5: 断网降级测试 (模拟云端不可用)")
    router.set_cloud_available(False)
    print("  已切换: 云端不可用")
    resp = await agent.execute(AgentRequest(
        skill_name="aluminum.tapout-planning",
        inputs={"cell_count": 500, "shift": "day"},
    ))
    print_result("断网降级", {
        "routed_to": resp.routed_to, "route_reason": resp.route_reason,
        "safety_net_triggered": resp.safety_net_triggered,
        "latency_ms": resp.latency_ms, "status": resp.status,
    })
    router.set_cloud_available(True)

    # 场景 6: 安全网触发（高复杂度核心参数 -> 降级本地）
    print_header("场景 6: 安全网触发 (高复杂度核心参数 -> 降级本地)")
    resp = await agent.execute(AgentRequest(
        skill_name="phosphorus.yellow-phosphorus-batching",
        inputs={"phosphate_ore_grade": 0.28, "furnace_temp": 1450},
    ))
    result = resp.result.get("result", resp.result) if isinstance(resp.result, dict) else resp.result
    print_result("黄磷配料", {
        "routed_to": resp.routed_to, "route_reason": resp.route_reason,
        "safety_net_triggered": resp.safety_net_triggered,
        "token_count": resp.token_count, "latency_ms": resp.latency_ms,
        "status": resp.status, "result": result,
    })

    # 统计
    print_header("运行统计")
    stats = modelhub.get_stats()
    print(f"\n  总任务数: {stats['total_tasks']}")
    print(f"  成功: {stats['success']}, 失败: {stats['failed']}")
    print(f"  本地分发: {stats['local_dispatched']}, 云端分发: {stats['cloud_dispatched']}")
    print(f"  平均耗时: {stats['avg_latency_ms']} ms")
    print(f"\n  路由记忆:")
    memory = router.get_memory_stats()
    for key, val in memory.get("entries", {}).items():
        print(f"    {key}: {val}")

    print_header("演示完成")
    print("  启动 API 服务: python -m industrial_agent.main")
    print("  API 文档: http://localhost:8000/docs")


if __name__ == "__main__":
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        print("\n已中断")
        sys.exit(0)
