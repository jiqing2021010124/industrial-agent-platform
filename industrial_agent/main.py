"""平台启动入口"""

import sys

import uvicorn

from .api import app


def main():
    """启动云边端工业智能体平台"""
    host = "0.0.0.0"
    port = 8000
    print("=" * 60)
    print("  🏭 云边端工业智能体平台")
    print("  基于 ClawChips 架构 | 磷化工 + 电解铝场景")
    print("=" * 60)
    print(f"  服务地址: http://localhost:{port}")
    print(f"  API 文档: http://localhost:{port}/docs")
    print(f"  欢迎页:  http://localhost:{port}")
    print("=" * 60)
    print("  提示: 未配置云端 API Key 时自动启用 Mock 模式")
    print("  提示: 边缘推理使用 Mock NPU 后端模拟")
    print("=" * 60)

    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    except KeyboardInterrupt:
        print("\n平台已停止")
        sys.exit(0)


if __name__ == "__main__":
    main()
