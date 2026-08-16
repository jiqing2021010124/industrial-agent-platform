"""云端模型客户端 - 兼容 OpenAI API 格式

当边缘模型不可用或任务复杂度超出本地能力时，
通过 OpenAI 兼容 API 调用云端大模型（DeepSeek/Qwen/GLM 等）。
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from .config import CloudProviderConfig

logger = logging.getLogger(__name__)


class CloudModelClient:
    """云端模型客户端 - OpenAI 兼容 API"""

    def __init__(self, providers: list[CloudProviderConfig]):
        self.providers = {p.name: p for p in providers}
        self._available = bool(providers)
        self._mock_mode = not self._has_real_keys()

    def _has_real_keys(self) -> bool:
        """检查是否配置了真实的 API Key"""
        return any(p.api_key for p in self.providers.values())

    @property
    def available(self) -> bool:
        return self._available

    async def chat_completion(
        self,
        provider_name: str,
        messages: list[dict[str, str]],
        model_id: str | None = None,
    ) -> dict[str, Any]:
        """调用云端 chat completion API"""
        provider = self.providers.get(provider_name)
        if not provider:
            return {"error": f"Provider {provider_name} 未配置"}

        if self._mock_mode or not provider.api_key:
            return await self._mock_completion(provider, messages, model_id)

        # 真实调用
        try:
            async with httpx.AsyncClient(timeout=provider.timeout) as client:
                resp = await client.post(
                    f"{provider.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {provider.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model_id or provider.model_id,
                        "messages": messages,
                    },
                )
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.error("云端调用失败: %s", e)
            return {"error": str(e)}

    async def _mock_completion(
        self,
        provider: CloudProviderConfig,
        messages: list[dict[str, str]],
        model_id: str | None,
    ) -> dict[str, Any]:
        """Mock 云端响应（无 API Key 时使用）"""
        import asyncio
        import random

        await asyncio.sleep(random.uniform(0.3, 0.8))  # 模拟网络延迟

        user_msg = ""
        for m in messages:
            if m.get("role") == "user":
                user_msg = m.get("content", "")
                break

        return {
            "id": f"chatcmpl-mock-{random.randint(10000, 99999)}",
            "object": "chat.completion",
            "model": model_id or provider.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"[云端Mock-{provider.name}] 针对您的问题「{user_msg[:50]}」，"
                        f"这是来自 {provider.model_id} 的复杂推理结果。"
                        f"边缘模型因资源/复杂度限制已将此请求路由至云端处理。",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 150, "completion_tokens": 80, "total_tokens": 230},
            "mock": True,
        }
