import logging
import os
import secrets
import json
import codecs
import time
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from bucket_manager import BucketManager
from dehydrator import Dehydrator
from embedding_engine import EmbeddingEngine
from gateway_state import GatewayStateStore
from memory_edges import MemoryEdgeStore
from persona_engine import PersonaStateEngine
from utils import count_tokens_approx, load_config, setup_logging, strip_wikilinks

logger = logging.getLogger("ombre_brain.gateway")


class GatewayService:
    """
    OpenAI-compatible gateway that injects Ombre memory before forwarding
    chat completions upstream.
    """

    def __init__(
        self,
        config: dict,
        bucket_mgr: BucketManager | None = None,
        dehydrator: Dehydrator | None = None,
        embedding_engine: EmbeddingEngine | None = None,
        state_store: GatewayStateStore | None = None,
        persona_engine: PersonaStateEngine | None = None,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.config = config
        self.gateway_cfg = config.get("gateway", {})
        self.bucket_mgr = bucket_mgr or BucketManager(config)
        self.dehydrator = dehydrator or Dehydrator(config)
        self.embedding_engine = embedding_engine or EmbeddingEngine(config)
        self.memory_edge_store = MemoryEdgeStore(config)
        self.state_store = state_store or GatewayStateStore(
            os.path.join(config["buckets_dir"], "gateway_state.db")
        )
        self.persona_engine = persona_engine or PersonaStateEngine(config)
        self.gateway_token = os.environ.get("OMBRE_GATEWAY_TOKEN", "")
        self.upstream_api_key = os.environ.get("OMBRE_GATEWAY_UPSTREAM_API_KEY", "")
        self.upstream_base_url = self.gateway_cfg.get("upstream_base_url", "").rstrip("/")
        self.upstream_default_model = self.gateway_cfg.get("upstream_default_model", "")
        self.upstream_models = self._normalize_model_list(
            self.gateway_cfg.get("upstream_models", []),
            self.upstream_default_model,
        )
        self.upstreams = self._load_upstreams()
        self.upstream_models = self._aggregate_upstream_models()
        if not self.upstream_default_model:
            for upstream in self.upstreams:
                default_model = upstream.get("default_model") or ""
                if default_model:
                    self.upstream_default_model = default_model
                    break

        self.head_recent_hours = int(self.gateway_cfg.get("head_recent_hours", 72))
        self.dynamic_top_k = int(self.gateway_cfg.get("dynamic_top_k", 10))
        self.inject_max_cards = max(0, min(2, int(self.gateway_cfg.get("inject_max_cards", 2))))
        self.skip_recent_rounds = max(0, int(self.gateway_cfg.get("skip_recent_rounds", 5)))
        self.cooldown_hours = float(self.gateway_cfg.get("cooldown_hours", 48))
        self.cooldown_floor = float(self.gateway_cfg.get("cooldown_floor", 0.3))

        self.inject_total_budget = int(self.gateway_cfg.get("inject_total_budget", 1200))
        self.core_budget = int(self.gateway_cfg.get("core_memory_budget", 500))
        self.recent_budget = int(self.gateway_cfg.get("recent_context_budget", 300))
        self.recalled_budget = int(self.gateway_cfg.get("recalled_memory_budget", 400))
        self.relationship_weather_budget = int(self.gateway_cfg.get("relationship_weather_budget", 220))
        self.favorite_memory_budget = int(self.gateway_cfg.get("favorite_memory_budget", 180))
        self.favorite_memory_max_cards = max(0, int(self.gateway_cfg.get("favorite_memory_max_cards", 1)))
        self.related_memory_budget = int(self.gateway_cfg.get("related_memory_budget", 220))

        self.semantic_weight = float(self.gateway_cfg.get("semantic_weight", 0.45))
        self.keyword_weight = float(self.gateway_cfg.get("keyword_weight", 0.35))
        self.importance_weight = float(self.gateway_cfg.get("importance_weight", 0.10))
        self.freshness_weight = float(self.gateway_cfg.get("freshness_weight", 0.10))
        self.first_card_min_score = float(self.gateway_cfg.get("first_card_min_score", 0.55))
        self.second_card_min_score = float(self.gateway_cfg.get("second_card_min_score", 0.50))
        self.second_card_relative_score = float(
            self.gateway_cfg.get("second_card_relative_score", 0.85)
        )
        self.edge_min_confidence = float(self.gateway_cfg.get("edge_min_confidence", 0.55))
        self.pending_tool_reasoning: dict[str, dict[tuple[str, ...], dict[str, Any]]] = {}

        self.http_client = http_client or httpx.AsyncClient(timeout=60.0)

    async def close(self) -> None:
        if self.http_client and not getattr(self.http_client, "is_closed", False):
            await self.http_client.aclose()

    async def health_payload(self) -> dict:
        stats = await self.bucket_mgr.get_stats()
        return {
            "status": "ok",
            "gateway": {
                "token_configured": bool(self.gateway_token),
                "upstream_ready": bool(self.upstreams) and all(
                    bool(upstream.get("base_url") and upstream.get("api_key"))
                    for upstream in self.upstreams
                ),
                "upstream_base_url": self.upstream_base_url
                or (self.upstreams[0]["base_url"] if len(self.upstreams) == 1 else ""),
                "upstream_default_model": self.upstream_default_model,
                "upstream_models": self.upstream_models,
                "upstreams": [
                    {
                        "name": upstream["name"],
                        "base_url": upstream["base_url"],
                        "default_model": upstream["default_model"],
                        "models": upstream["models"],
                        "prompt_cache": upstream.get("prompt_cache", ""),
                        "prompt_cache_retention": upstream.get("prompt_cache_retention", ""),
                        "ready": bool(upstream.get("base_url") and upstream.get("api_key")),
                    }
                    for upstream in self.upstreams
                ],
            },
            "persona": {
                "enabled": bool(self.persona_engine.enabled),
                "profile_id": self.persona_engine.profile_id,
                "mode": self.persona_engine.mode,
                "model": self.persona_engine.model,
                "api_ready": bool(self.persona_engine.api_key),
            },
            "buckets": stats,
        }

    async def handle_health(self, request: Request) -> JSONResponse:
        try:
            return JSONResponse(await self.health_payload())
        except Exception as exc:
            logger.exception("Gateway health check failed: %s", exc)
            return JSONResponse({"status": "error", "detail": str(exc)}, status_code=500)

    async def handle_chat(self, request: Request) -> Response:
        auth_result = self._authorize(request.headers.get("Authorization", ""))
        if auth_result is not None:
            return auth_result

        session_id = (request.headers.get("X-Ombre-Session-Id") or "").strip()
        if not session_id:
            return JSONResponse(
                {"error": {"message": "X-Ombre-Session-Id is required", "type": "invalid_request_error"}},
                status_code=400,
            )

        try:
            payload = await request.json()
        except Exception:
            return JSONResponse(
                {"error": {"message": "Request body must be valid JSON", "type": "invalid_request_error"}},
                status_code=400,
            )

        if not isinstance(payload, dict):
            return JSONResponse(
                {"error": {"message": "Request body must be a JSON object", "type": "invalid_request_error"}},
                status_code=400,
            )

        logger.info(
            "Gateway incoming chat | session=%s model=%s stream=%s messages=%s",
            session_id,
            payload.get("model") or self.upstream_default_model,
            payload.get("stream") is True,
            self._summarize_messages_for_debug(payload.get("messages")),
        )

        try:
            persona_user_message = self._extract_last_user_query(payload.get("messages", []))
            forward_payload, recalled_ids = await self.prepare_payload(payload, session_id)
        except ValueError as exc:
            return JSONResponse(
                {"error": {"message": str(exc), "type": "invalid_request_error"}},
                status_code=400,
            )
        except RuntimeError as exc:
            return JSONResponse(
                {"error": {"message": str(exc), "type": "server_error"}},
                status_code=503,
            )

        if forward_payload.get("stream") is True:
            try:
                return await self._stream_upstream(
                    forward_payload,
                    session_id,
                    recalled_ids,
                    persona_user_message,
                )
            except RuntimeError as exc:
                return JSONResponse(
                    {"error": {"message": str(exc), "type": "server_error"}},
                    status_code=503,
                )

        upstream_response = await self._forward_upstream(forward_payload)
        if 200 <= upstream_response.status_code < 300:
            self._log_cache_usage_from_response(
                session_id,
                forward_payload["model"],
                upstream_response,
                route="/v1/chat/completions",
            )
            self._capture_reasoning_from_response(session_id, upstream_response)
            await self._record_successful_round(session_id, recalled_ids)
            await self._update_persona_after_response(
                session_id,
                persona_user_message,
                upstream_response,
                recalled_ids,
            )

        return self._proxy_response(upstream_response)

    async def handle_anthropic_messages(self, request: Request) -> Response:
        auth_result = self._authorize_anthropic_request(request)
        if auth_result is not None:
            return auth_result

        session_id = (request.headers.get("X-Ombre-Session-Id") or "xiaoyu-main").strip()

        try:
            payload = await request.json()
        except Exception:
            return self._anthropic_error("Request body must be valid JSON", status_code=400)

        if not isinstance(payload, dict):
            return self._anthropic_error("Request body must be a JSON object", status_code=400)

        try:
            openai_payload = self._anthropic_request_to_openai(payload)
        except ValueError as exc:
            return self._anthropic_error(str(exc), status_code=400)

        logger.info(
            "Gateway incoming Anthropic messages | session=%s model=%s messages=%s",
            session_id,
            openai_payload.get("model") or self.upstream_default_model,
            self._summarize_messages_for_debug(openai_payload.get("messages")),
        )

        try:
            persona_user_message = self._extract_last_user_query(openai_payload.get("messages", []))
            forward_payload, recalled_ids = await self.prepare_payload(openai_payload, session_id)
        except ValueError as exc:
            return self._anthropic_error(str(exc), status_code=400)
        except RuntimeError as exc:
            return self._anthropic_error(str(exc), status_code=503, error_type="server_error")

        if forward_payload.get("stream") is True:
            return await self._stream_upstream_as_anthropic(
                forward_payload,
                session_id,
                recalled_ids,
                persona_user_message,
            )

        upstream_response = await self._forward_upstream(forward_payload)
        if 200 <= upstream_response.status_code < 300:
            self._log_cache_usage_from_response(
                session_id,
                forward_payload["model"],
                upstream_response,
                route="/v1/messages",
            )
            self._capture_reasoning_from_response(session_id, upstream_response)
            await self._record_successful_round(session_id, recalled_ids)
            await self._update_persona_after_response(
                session_id,
                persona_user_message,
                upstream_response,
                recalled_ids,
            )
            return self._openai_response_to_anthropic(upstream_response, forward_payload["model"])

        return self._proxy_anthropic_error_response(upstream_response)

    async def handle_models(self, request: Request) -> Response:
        auth_result = self._authorize(request.headers.get("Authorization", ""))
        if auth_result is not None:
            return auth_result

        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": model,
                        "object": "model",
                        "created": 0,
                        "owned_by": "ombre-gateway",
                    }
                    for model in self.upstream_models
                ],
            }
        )

    async def prepare_payload(self, payload: dict, session_id: str) -> tuple[dict, list[str]]:
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")

        model = payload.get("model") or self.upstream_default_model
        if not model:
            raise ValueError("model is required when gateway.upstream_default_model is empty")
        self._get_upstream_for_model(model)

        all_buckets = await self.bucket_mgr.list_all(include_archive=False)
        query = self._extract_last_user_query(messages)
        persona_query = self._extract_current_turn_user_query(messages)
        if persona_query:
            persona_state = await self.persona_engine.build_pre_reply_guidance(session_id, persona_query)
        else:
            persona_state = self.persona_engine.get_current_state(session_id)
        persona_block = self.persona_engine.format_state_block(persona_state)
        core_memory = await self._build_core_memory_block(all_buckets)
        recent_context = await self._build_recent_context_block(all_buckets)
        recalled_buckets = await self._select_dynamic_buckets(query, session_id, all_buckets)
        recalled_memory = await self._summarize_buckets(recalled_buckets, self.recalled_budget)
        relationship_weather = await self._build_relationship_weather_block(all_buckets)
        favorite_memory, favorite_ids = await self._build_favorite_memory_block(all_buckets, session_id)
        related_memory = await self._build_related_memory_block(recalled_buckets, all_buckets)
        stable_context, dynamic_context = self._build_injected_context_messages(
            persona_block=persona_block,
            core_memory=core_memory,
            recent_context=recent_context,
            recalled_memory=recalled_memory,
            relationship_weather=relationship_weather,
            favorite_memory=favorite_memory,
            related_memory=related_memory,
        )

        forward_payload = deepcopy(payload)
        forward_payload["model"] = model
        self._restore_cached_reasoning_content(session_id, forward_payload.get("messages"))
        forward_payload["messages"] = self._inject_context_messages(
            forward_payload["messages"],
            stable_context,
            dynamic_context,
        )
        self._apply_prompt_cache_hints(forward_payload, session_id)
        forward_payload["stream"] = payload.get("stream") is True
        injected_ids = list(
            dict.fromkeys([bucket["id"] for bucket in recalled_buckets] + favorite_ids)
        )
        return forward_payload, injected_ids

    def _apply_prompt_cache_hints(self, payload: dict[str, Any], session_id: str) -> None:
        model = str(payload.get("model") or "").strip()
        upstream = self._get_upstream_for_model(model)
        strategy = str(upstream.get("prompt_cache") or "").strip().lower()
        if strategy != "openai":
            return

        payload.setdefault("prompt_cache_key", session_id)
        retention = str(upstream.get("prompt_cache_retention") or "").strip()
        if retention:
            payload.setdefault("prompt_cache_retention", retention)

    def _authorize(self, auth_header: str) -> JSONResponse | None:
        if not self.gateway_token:
            return JSONResponse(
                {"error": {"message": "Gateway token is not configured", "type": "server_error"}},
                status_code=503,
            )

        scheme, _, token = auth_header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            return JSONResponse(
                {"error": {"message": "Authorization: Bearer token is required", "type": "authentication_error"}},
                status_code=401,
            )

        if not secrets.compare_digest(token, self.gateway_token):
            return JSONResponse(
                {"error": {"message": "Invalid gateway token", "type": "authentication_error"}},
                status_code=401,
            )
        return None

    def _authorize_anthropic_request(self, request: Request) -> JSONResponse | None:
        if not self.gateway_token:
            return self._anthropic_error(
                "Gateway token is not configured",
                status_code=503,
                error_type="server_error",
            )

        auth_header = request.headers.get("Authorization", "")
        scheme, _, bearer_token = auth_header.partition(" ")
        api_key = (request.headers.get("x-api-key") or "").strip()
        token = bearer_token.strip() if scheme.lower() == "bearer" else api_key
        if not token:
            return self._anthropic_error(
                "Authorization: Bearer token or x-api-key is required",
                status_code=401,
                error_type="authentication_error",
            )

        if not secrets.compare_digest(token, self.gateway_token):
            return self._anthropic_error(
                "Invalid gateway token",
                status_code=401,
                error_type="authentication_error",
            )

        return None

    async def _forward_upstream(self, payload: dict) -> httpx.Response:
        model = str(payload.get("model") or "").strip()
        upstream = self._get_upstream_for_model(model)
        url = f"{upstream['base_url']}/chat/completions"
        started_at = time.perf_counter()
        response = await self.http_client.post(
            url,
            headers={
                "Authorization": f"Bearer {upstream['api_key']}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        logger.info(
            "Gateway upstream response | upstream=%s model=%s status=%s latency_ms=%s",
            upstream["name"],
            model,
            response.status_code,
            latency_ms,
        )
        return response

    async def _stream_upstream(
        self,
        payload: dict,
        session_id: str,
        recalled_ids: list[str],
        user_message: str,
    ) -> Response:
        model = str(payload.get("model") or "").strip()
        upstream = self._get_upstream_for_model(model)
        url = f"{upstream['base_url']}/chat/completions"
        request = self.http_client.build_request(
            "POST",
            url,
            headers={
                "Authorization": f"Bearer {upstream['api_key']}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        started_at = time.perf_counter()
        upstream_response = await self.http_client.send(request, stream=True)
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        logger.info(
            "Gateway upstream response | upstream=%s model=%s status=%s latency_ms=%s",
            upstream["name"],
            model,
            upstream_response.status_code,
            latency_ms,
        )
        content_type = upstream_response.headers.get("content-type", "text/event-stream")

        if not 200 <= upstream_response.status_code < 300:
            body = await upstream_response.aread()
            await upstream_response.aclose()
            return Response(
                content=body,
                status_code=upstream_response.status_code,
                media_type=content_type,
            )

        async def stream_body():
            completed = False
            stream_state = self._new_stream_capture_state()
            try:
                async for chunk in upstream_response.aiter_bytes():
                    if chunk:
                        self._consume_stream_capture_chunk(stream_state, chunk)
                        yield chunk
                self._consume_stream_capture_chunk(stream_state, b"", final=True)
                completed = True
            finally:
                await upstream_response.aclose()
                if completed:
                    self._log_cache_usage_from_stream_state(
                        session_id,
                        model,
                        stream_state,
                        route="/v1/chat/completions",
                    )
                    self._capture_reasoning_from_stream_state(session_id, stream_state)
                    await self._record_successful_round(session_id, recalled_ids)
                    await self._update_persona_after_assistant_message(
                        session_id,
                        user_message,
                        self._build_stream_assistant_message(stream_state),
                        recalled_ids,
                    )

        return StreamingResponse(
            stream_body(),
            status_code=upstream_response.status_code,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    async def _record_successful_round(self, session_id: str, recalled_ids: list[str]) -> None:
        round_id = self.state_store.record_success(session_id, recalled_ids)
        for bucket_id in recalled_ids:
            await self.bucket_mgr.touch(bucket_id)
        logger.info(
            "Gateway round completed | session=%s round=%s recalled=%s",
            session_id,
            round_id,
            recalled_ids,
        )

    async def _update_persona_after_response(
        self,
        session_id: str,
        user_message: str,
        upstream_response: httpx.Response,
        recalled_ids: list[str],
    ) -> None:
        try:
            body = upstream_response.json()
        except ValueError:
            return
        assistant_message = self._extract_assistant_message_from_response_body(body)
        await self._update_persona_after_assistant_message(
            session_id,
            user_message,
            assistant_message,
            recalled_ids,
        )

    async def _update_persona_after_assistant_message(
        self,
        session_id: str,
        user_message: str,
        assistant_message: dict[str, Any] | None,
        recalled_ids: list[str],
    ) -> None:
        if not user_message.strip() or not isinstance(assistant_message, dict):
            return
        assistant_response = self._coerce_message_text(assistant_message.get("content")).strip()
        if not assistant_response:
            return
        tool_summary = self._summarize_assistant_tool_calls(assistant_message)
        try:
            await self.persona_engine.update_from_exchange(
                session_id=session_id,
                user_message=user_message,
                assistant_response=assistant_response,
                recalled_memory_ids=recalled_ids,
                tool_summary=tool_summary,
            )
        except Exception as exc:
            logger.warning("Persona post-reply update failed | session=%s error=%s", session_id, exc)

    def _summarize_assistant_tool_calls(self, assistant_message: dict[str, Any]) -> str:
        tool_calls = assistant_message.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            return ""
        parts: list[str] = []
        for index, tool_call in enumerate(tool_calls[:8]):
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            name = str(function.get("name") or f"tool_{index}")
            arguments = self._normalize_tool_arguments(function.get("arguments", ""))
            if len(arguments) > 160:
                arguments = arguments[:157] + "..."
            parts.append(f"{name}({arguments})")
        return "; ".join(parts)

    def _log_cache_usage_from_response(
        self,
        session_id: str,
        model: str,
        upstream_response: httpx.Response,
        route: str,
    ) -> None:
        try:
            body = upstream_response.json()
        except ValueError:
            return
        usage = body.get("usage") if isinstance(body, dict) else None
        if isinstance(usage, dict):
            self._log_cache_usage(session_id, model, route, usage)

    def _log_cache_usage_from_stream_state(
        self,
        session_id: str,
        model: str,
        stream_state: dict[str, Any],
        route: str,
    ) -> None:
        usage = stream_state.get("usage")
        if isinstance(usage, dict):
            self._log_cache_usage(session_id, model, route, usage)

    def _log_cache_usage(self, session_id: str, model: str, route: str, usage: dict[str, Any]) -> None:
        hit = usage.get("prompt_cache_hit_tokens")
        miss = usage.get("prompt_cache_miss_tokens")
        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
        completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
        cache_read_tokens = usage.get("cache_read_input_tokens")
        cache_creation_tokens = usage.get("cache_creation_input_tokens")
        prompt_details = usage.get("prompt_tokens_details")
        cached_tokens = None
        if isinstance(prompt_details, dict):
            cached_tokens = prompt_details.get("cached_tokens")

        if (
            hit is None
            and miss is None
            and cached_tokens is None
            and cache_read_tokens is None
            and cache_creation_tokens is None
        ):
            return

        logger.info(
            "Gateway upstream cache usage | session=%s model=%s route=%s "
            "prompt_tokens=%s completion_tokens=%s prompt_cache_hit_tokens=%s "
            "prompt_cache_miss_tokens=%s cached_tokens=%s cache_read_input_tokens=%s "
            "cache_creation_input_tokens=%s",
            session_id,
            model,
            route,
            prompt_tokens,
            completion_tokens,
            hit,
            miss,
            cached_tokens,
            cache_read_tokens,
            cache_creation_tokens,
        )

    def _proxy_response(self, upstream_response: httpx.Response) -> Response:
        content_type = upstream_response.headers.get("content-type", "application/json")
        try:
            body = upstream_response.json()
            return JSONResponse(body, status_code=upstream_response.status_code)
        except ValueError:
            return Response(
                content=upstream_response.text,
                status_code=upstream_response.status_code,
                media_type=content_type,
            )

    def _anthropic_request_to_openai(self, payload: dict) -> dict:
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")

        openai_messages: list[dict[str, Any]] = []
        system_text = self._anthropic_content_to_text(payload.get("system"), "system").strip()
        if system_text:
            openai_messages.append({"role": "system", "content": system_text})

        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(f"messages[{index}] must be an object")
            openai_messages.extend(self._anthropic_message_to_openai_messages(message, index))

        openai_payload: dict[str, Any] = {
            "model": payload.get("model"),
            "messages": openai_messages,
            "stream": payload.get("stream") is True,
        }

        passthrough_fields = ("max_tokens", "temperature", "top_p")
        for field in passthrough_fields:
            if field in payload:
                openai_payload[field] = payload[field]

        if "stop_sequences" in payload:
            openai_payload["stop"] = payload["stop_sequences"]
        elif "stop" in payload:
            openai_payload["stop"] = payload["stop"]

        tools = self._anthropic_tools_to_openai(payload.get("tools"))
        if tools:
            openai_payload["tools"] = tools

        tool_choice = self._anthropic_tool_choice_to_openai(payload.get("tool_choice"))
        if tool_choice is not None:
            openai_payload["tool_choice"] = tool_choice

        return openai_payload

    def _anthropic_message_to_openai_messages(self, message: dict[str, Any], index: int) -> list[dict[str, Any]]:
        role = str(message.get("role") or "").strip()
        if role not in {"user", "assistant", "system"}:
            raise ValueError(f"messages[{index}].role must be user, assistant, or system")

        content = message.get("content")
        if role == "system":
            return [{"role": "system", "content": self._anthropic_content_to_text(content, f"messages[{index}].content")}]
        if isinstance(content, str) or content is None:
            return [{"role": role, "content": content or ""}]
        if not isinstance(content, list):
            raise ValueError(f"messages[{index}].content must be a string or block list")

        if role == "assistant":
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block_index, block in enumerate(content):
                if isinstance(block, str):
                    text_parts.append(block)
                    continue
                if not isinstance(block, dict):
                    raise ValueError(f"messages[{index}].content[{block_index}] must be an object")
                block_type = block.get("type")
                if block_type == "text":
                    text_parts.append(str(block.get("text") or ""))
                    continue
                if block_type == "tool_use":
                    tool_id = str(block.get("id") or "")
                    name = str(block.get("name") or "")
                    if not tool_id or not name:
                        raise ValueError(f"messages[{index}].content[{block_index}] tool_use requires id and name")
                    tool_calls.append(
                        {
                            "id": tool_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                            },
                        }
                    )
                    continue
                raise ValueError(f"messages[{index}].content[{block_index}] unsupported assistant block type")

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": "\n".join(part for part in text_parts if part) or None,
            }
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
            return [assistant_message]

        output: list[dict[str, Any]] = []
        pending_text: list[str] = []
        for block_index, block in enumerate(content):
            if isinstance(block, str):
                pending_text.append(block)
                continue
            if not isinstance(block, dict):
                raise ValueError(f"messages[{index}].content[{block_index}] must be an object")
            block_type = block.get("type")
            if block_type == "text":
                pending_text.append(str(block.get("text") or ""))
                continue
            if block_type == "tool_result":
                if pending_text:
                    output.append({"role": "user", "content": "\n".join(part for part in pending_text if part)})
                    pending_text = []
                tool_use_id = str(block.get("tool_use_id") or "")
                if not tool_use_id:
                    raise ValueError(f"messages[{index}].content[{block_index}] tool_result requires tool_use_id")
                output.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_use_id,
                        "content": self._anthropic_content_to_text(
                            block.get("content"),
                            f"messages[{index}].content[{block_index}].content",
                        ),
                    }
                )
                continue
            raise ValueError(f"messages[{index}].content[{block_index}] unsupported user block type")

        if pending_text or not output:
            output.append({"role": "user", "content": "\n".join(part for part in pending_text if part)})
        return output

    def _anthropic_tools_to_openai(self, tools: Any) -> list[dict[str, Any]]:
        if tools is None:
            return []
        if not isinstance(tools, list):
            raise ValueError("tools must be a list")
        converted = []
        for index, tool in enumerate(tools):
            if not isinstance(tool, dict):
                raise ValueError(f"tools[{index}] must be an object")
            name = str(tool.get("name") or "")
            if not name:
                raise ValueError(f"tools[{index}].name is required")
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": str(tool.get("description") or ""),
                        "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
                    },
                }
            )
        return converted

    def _anthropic_tool_choice_to_openai(self, tool_choice: Any) -> Any:
        if tool_choice is None:
            return None
        if isinstance(tool_choice, str):
            return {"auto": "auto", "any": "required", "none": "none"}.get(tool_choice, tool_choice)
        if not isinstance(tool_choice, dict):
            raise ValueError("tool_choice must be a string or object")
        choice_type = tool_choice.get("type")
        if choice_type == "auto":
            return "auto"
        if choice_type == "any":
            return "required"
        if choice_type == "none":
            return "none"
        if choice_type == "tool":
            name = str(tool_choice.get("name") or "")
            if not name:
                raise ValueError("tool_choice.name is required when type is tool")
            return {"type": "function", "function": {"name": name}}
        return None

    def _anthropic_content_to_text(self, content: Any, field_name: str) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for index, block in enumerate(content):
                if isinstance(block, str):
                    parts.append(block)
                    continue
                if not isinstance(block, dict):
                    raise ValueError(f"{field_name}[{index}] must be a text block")
                block_type = block.get("type")
                if block_type != "text":
                    raise ValueError(f"{field_name}[{index}] only supports text blocks")
                parts.append(str(block.get("text") or ""))
            return "\n".join(part for part in parts if part)
        raise ValueError(f"{field_name} must be a string or text block list")

    def _openai_response_to_anthropic(self, upstream_response: httpx.Response, requested_model: str) -> JSONResponse:
        try:
            body = upstream_response.json()
        except ValueError:
            return self._anthropic_error(
                "Upstream response was not valid JSON",
                status_code=502,
                error_type="api_error",
            )

        choices = body.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else {}
        message = choice.get("message") if isinstance(choice, dict) else {}
        if not isinstance(message, dict):
            message = {}

        content_blocks = self._openai_message_to_anthropic_content(message)
        raw_id = str(body.get("id") or "ombre")
        response_id = raw_id if raw_id.startswith("msg_") else f"msg_{raw_id}"
        finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}

        return JSONResponse(
            {
                "id": response_id,
                "type": "message",
                "role": "assistant",
                "model": body.get("model") or requested_model,
                "content": content_blocks,
                "stop_reason": self._openai_finish_reason_to_anthropic(finish_reason),
                "stop_sequence": None,
                "usage": {
                    "input_tokens": int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0),
                    "output_tokens": int(usage.get("output_tokens") or usage.get("completion_tokens") or 0),
                },
            },
            status_code=upstream_response.status_code,
        )

    def _openai_message_to_anthropic_content(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        content_blocks: list[dict[str, Any]] = []
        text = self._coerce_message_text(message.get("content"))
        if text:
            content_blocks.append({"type": "text", "text": text})

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function")
                if not isinstance(function, dict):
                    continue
                name = str(function.get("name") or "")
                if not name:
                    continue
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": str(tool_call.get("id") or f"call_{len(content_blocks)}"),
                        "name": name,
                        "input": self._parse_tool_arguments(function.get("arguments")),
                    }
                )
        return content_blocks

    def _parse_tool_arguments(self, raw_arguments: Any) -> Any:
        if isinstance(raw_arguments, dict):
            return raw_arguments
        if raw_arguments in (None, ""):
            return {}
        if not isinstance(raw_arguments, str):
            return {}
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    async def _stream_upstream_as_anthropic(
        self,
        payload: dict,
        session_id: str,
        recalled_ids: list[str],
        user_message: str,
    ) -> Response:
        model = str(payload.get("model") or "").strip()
        upstream = self._get_upstream_for_model(model)
        url = f"{upstream['base_url']}/chat/completions"
        request = self.http_client.build_request(
            "POST",
            url,
            headers={
                "Authorization": f"Bearer {upstream['api_key']}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        started_at = time.perf_counter()
        upstream_response = await self.http_client.send(request, stream=True)
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        logger.info(
            "Gateway upstream response | upstream=%s model=%s status=%s latency_ms=%s",
            upstream["name"],
            model,
            upstream_response.status_code,
            latency_ms,
        )

        if not 200 <= upstream_response.status_code < 300:
            body = await upstream_response.aread()
            await upstream_response.aclose()
            return self._proxy_anthropic_error_response(
                httpx.Response(
                    status_code=upstream_response.status_code,
                    content=body,
                    headers=upstream_response.headers,
                )
            )

        async def stream_body():
            completed = False
            stream_state = self._new_stream_capture_state()
            message_id = f"msg_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
            usage = {"input_tokens": 0, "output_tokens": 0}
            stop_reason = "end_turn"
            next_block_index = 0
            text_block_index: int | None = None
            tool_blocks: dict[int, dict[str, Any]] = {}

            try:
                yield self._anthropic_sse(
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": message_id,
                            "type": "message",
                            "role": "assistant",
                            "model": model,
                            "content": [],
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": usage,
                        },
                    },
                )

                async for chunk in upstream_response.aiter_bytes():
                    if not chunk:
                        continue
                    self._consume_stream_capture_chunk(stream_state, chunk)
                    for event in self._openai_sse_chunk_to_anthropic_events(chunk):
                        if event.get("_done"):
                            continue
                        if event.get("usage"):
                            usage.update(event["usage"])
                            continue
                        if event.get("stop_reason"):
                            stop_reason = event["stop_reason"]
                            continue
                        if event.get("text"):
                            if text_block_index is None:
                                text_block_index = next_block_index
                                next_block_index += 1
                                yield self._anthropic_sse(
                                    "content_block_start",
                                    {
                                        "type": "content_block_start",
                                        "index": text_block_index,
                                        "content_block": {"type": "text", "text": ""},
                                    },
                                )
                            yield self._anthropic_sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": text_block_index,
                                    "delta": {
                                        "type": "text_delta",
                                        "text": event["text"],
                                    },
                                },
                            )
                            continue
                        tool_call = event.get("tool_call")
                        if isinstance(tool_call, dict):
                            tool_index = int(tool_call.get("index", 0))
                            state = tool_blocks.setdefault(
                                tool_index,
                                {
                                    "content_index": None,
                                    "id": "",
                                    "name": "",
                                    "started": False,
                                },
                            )
                            if tool_call.get("id"):
                                state["id"] = str(tool_call["id"])
                            if tool_call.get("name"):
                                state["name"] = str(tool_call["name"])
                            if not state["started"] and state["name"]:
                                state["content_index"] = next_block_index
                                next_block_index += 1
                                state["started"] = True
                                yield self._anthropic_sse(
                                    "content_block_start",
                                    {
                                        "type": "content_block_start",
                                        "index": state["content_index"],
                                        "content_block": {
                                            "type": "tool_use",
                                            "id": state["id"] or f"call_{tool_index}",
                                            "name": state["name"],
                                            "input": {},
                                        },
                                    },
                                )
                            arguments = tool_call.get("arguments")
                            if state["started"] and arguments:
                                yield self._anthropic_sse(
                                    "content_block_delta",
                                    {
                                        "type": "content_block_delta",
                                        "index": state["content_index"],
                                        "delta": {
                                            "type": "input_json_delta",
                                            "partial_json": arguments,
                                        },
                                    },
                                )

                self._consume_stream_capture_chunk(stream_state, b"", final=True)
                if text_block_index is not None:
                    yield self._anthropic_sse(
                        "content_block_stop",
                        {"type": "content_block_stop", "index": text_block_index},
                    )
                for state in sorted(
                    (item for item in tool_blocks.values() if item.get("started")),
                    key=lambda item: int(item["content_index"]),
                ):
                    yield self._anthropic_sse(
                        "content_block_stop",
                        {"type": "content_block_stop", "index": state["content_index"]},
                    )
                yield self._anthropic_sse(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": stop_reason,
                            "stop_sequence": None,
                        },
                        "usage": {"output_tokens": usage.get("output_tokens", 0)},
                    },
                )
                yield self._anthropic_sse(
                    "message_stop",
                    {"type": "message_stop"},
                )
                completed = True
            finally:
                await upstream_response.aclose()
                if completed:
                    self._log_cache_usage_from_stream_state(
                        session_id,
                        model,
                        stream_state,
                        route="/v1/messages",
                    )
                    self._capture_reasoning_from_stream_state(session_id, stream_state)
                    await self._record_successful_round(session_id, recalled_ids)
                    await self._update_persona_after_assistant_message(
                        session_id,
                        user_message,
                        self._build_stream_assistant_message(stream_state),
                        recalled_ids,
                    )

        return StreamingResponse(
            stream_body(),
            status_code=upstream_response.status_code,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    def _openai_sse_chunk_to_anthropic_events(self, chunk: bytes) -> list[dict[str, Any]]:
        text = chunk.decode("utf-8", errors="ignore")
        events: list[dict[str, Any]] = []
        for raw_event in text.split("\n\n"):
            for line in raw_event.splitlines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data:
                    continue
                if data == "[DONE]":
                    events.append({"_done": True})
                    continue
                try:
                    body = json.loads(data)
                except json.JSONDecodeError:
                    continue
                usage = body.get("usage")
                if isinstance(usage, dict):
                    events.append(
                        {
                            "usage": {
                                "input_tokens": int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0),
                                "output_tokens": int(
                                    usage.get("output_tokens") or usage.get("completion_tokens") or 0
                                ),
                            }
                        }
                    )
                choices = body.get("choices")
                if not isinstance(choices, list):
                    continue
                for choice in choices:
                    if not isinstance(choice, dict):
                        continue
                    finish_reason = choice.get("finish_reason")
                    if finish_reason:
                        events.append({"stop_reason": self._openai_finish_reason_to_anthropic(finish_reason)})
                    delta = choice.get("delta")
                    if not isinstance(delta, dict):
                        continue
                    tool_calls = delta.get("tool_calls")
                    if isinstance(tool_calls, list):
                        for tool_call in tool_calls:
                            if not isinstance(tool_call, dict):
                                continue
                            function = tool_call.get("function")
                            if not isinstance(function, dict):
                                function = {}
                            events.append(
                                {
                                    "tool_call": {
                                        "index": int(tool_call.get("index") or 0),
                                        "id": tool_call.get("id"),
                                        "name": function.get("name"),
                                        "arguments": function.get("arguments"),
                                    }
                                }
                            )
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        events.append({"text": content})
        return events

    def _anthropic_sse(self, event: str, data: dict[str, Any]) -> bytes:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")

    def _openai_finish_reason_to_anthropic(self, finish_reason: Any) -> str:
        mapping = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "function_call": "tool_use",
            "content_filter": "stop_sequence",
        }
        return mapping.get(str(finish_reason or ""), "end_turn")

    def _proxy_anthropic_error_response(self, upstream_response: httpx.Response) -> JSONResponse:
        message = upstream_response.text or "Upstream request failed"
        error_type = "api_error"
        try:
            body = upstream_response.json()
        except ValueError:
            body = None
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                message = str(error.get("message") or message)
                error_type = str(error.get("type") or error_type)
            elif body.get("message"):
                message = str(body["message"])
        return self._anthropic_error(
            message,
            status_code=upstream_response.status_code,
            error_type=error_type,
        )

    def _anthropic_error(
        self,
        message: str,
        *,
        status_code: int,
        error_type: str = "invalid_request_error",
    ) -> JSONResponse:
        return JSONResponse(
            {
                "type": "error",
                "error": {
                    "type": error_type,
                    "message": message,
                },
            },
            status_code=status_code,
        )

    def _extract_last_user_query(self, messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            if message.get("role") != "user":
                continue
            content = self._coerce_message_text(message.get("content"))
            if content.strip():
                return content.strip()
        return ""

    def _extract_current_turn_user_query(self, messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role == "system":
                continue
            if role != "user":
                return ""
            content = self._coerce_message_text(message.get("content"))
            return content.strip()
        return ""

    def _coerce_message_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type in {"text", "input_text"}:
                    text = item.get("text") or item.get("input_text") or ""
                    if text:
                        chunks.append(str(text))
            return "\n".join(chunks)
        return ""

    def _summarize_messages_for_debug(self, messages: Any) -> list[dict[str, Any]] | str:
        if not isinstance(messages, list):
            return "<invalid>"

        summary: list[dict[str, Any]] = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                summary.append({"idx": index, "type": type(message).__name__})
                continue

            item: dict[str, Any] = {
                "idx": index,
                "role": str(message.get("role") or ""),
            }
            if self._coerce_message_text(message.get("content")).strip():
                item["has_text"] = True
            if isinstance(message.get("reasoning_content"), str) and message.get("reasoning_content"):
                item["has_reasoning"] = True

            tool_call_id = message.get("tool_call_id")
            if tool_call_id:
                item["tool_call_id"] = str(tool_call_id)

            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                labels = []
                for tool_index, tool_call in enumerate(tool_calls):
                    if not isinstance(tool_call, dict):
                        labels.append(f"idx:{tool_index}")
                        continue
                    if tool_call.get("id"):
                        labels.append(str(tool_call["id"]))
                        continue
                    function = tool_call.get("function", {})
                    if isinstance(function, dict) and function.get("name"):
                        labels.append(f"idx:{tool_index}:{function['name']}")
                        continue
                    labels.append(f"idx:{tool_index}")
                item["tool_call_ids"] = labels

            summary.append(item)

        return summary

    async def _build_core_memory_block(self, all_buckets: list[dict]) -> str:
        core_buckets = [
            bucket for bucket in all_buckets
            if bucket.get("metadata", {}).get("pinned") or bucket.get("metadata", {}).get("protected")
        ]
        core_buckets.sort(
            key=lambda bucket: (
                int(bucket.get("metadata", {}).get("importance", 0)),
                bucket.get("metadata", {}).get("last_active", ""),
            ),
            reverse=True,
        )
        return await self._summarize_buckets(core_buckets, self.core_budget)

    async def _build_recent_context_block(self, all_buckets: list[dict]) -> str:
        cutoff = datetime.now() - timedelta(hours=self.head_recent_hours)
        recent_buckets = []
        for bucket in all_buckets:
            meta = bucket.get("metadata", {})
            if meta.get("type") == "feel":
                continue
            if meta.get("pinned") or meta.get("protected"):
                continue
            created = self._parse_iso(meta.get("created") or meta.get("last_active"))
            if created and created >= cutoff:
                recent_buckets.append(bucket)

        recent_buckets.sort(
            key=lambda bucket: bucket.get("metadata", {}).get("created", ""),
            reverse=True,
        )
        return await self._summarize_buckets(recent_buckets[:6], self.recent_budget)

    async def _build_relationship_weather_block(self, all_buckets: list[dict]) -> str:
        if self.relationship_weather_budget <= 0:
            return ""
        weather_buckets = []
        for bucket in all_buckets:
            meta = bucket.get("metadata", {})
            if meta.get("type") != "feel":
                continue
            tags = {str(tag) for tag in meta.get("tags", [])}
            if not ({"relationship_weather", "daily_impression", "weekly_impression"} & tags):
                continue
            weather_buckets.append(bucket)
        if not weather_buckets:
            return ""

        daily = [
            bucket for bucket in weather_buckets
            if "daily_impression" in {str(tag) for tag in bucket.get("metadata", {}).get("tags", [])}
        ]
        weekly = [
            bucket for bucket in weather_buckets
            if "weekly_impression" in {str(tag) for tag in bucket.get("metadata", {}).get("tags", [])}
        ]
        daily.sort(key=lambda bucket: bucket.get("metadata", {}).get("created", ""), reverse=True)
        weekly.sort(key=lambda bucket: bucket.get("metadata", {}).get("created", ""), reverse=True)
        selected = weekly[:1] + daily[:7]

        remaining = self.relationship_weather_budget
        parts = []
        for bucket in selected:
            meta = bucket.get("metadata", {})
            text = strip_wikilinks(bucket.get("content", "")).strip()
            if not text:
                continue
            prefix = meta.get("date") or meta.get("created", "")[:10]
            line = f"- [{prefix}] {self._trim_text(text, 80)}"
            tokens = count_tokens_approx(line)
            if tokens > remaining and parts:
                break
            parts.append(line)
            remaining -= tokens
            if remaining <= 0:
                break
        return "\n".join(parts)

    async def _build_favorite_memory_block(
        self,
        all_buckets: list[dict],
        session_id: str,
    ) -> tuple[str, list[str]]:
        if self.favorite_memory_budget <= 0 or self.favorite_memory_max_cards <= 0:
            return "", []

        recent_ids = self.state_store.get_recent_bucket_ids(session_id, self.skip_recent_rounds)
        candidates = []
        for bucket in all_buckets:
            meta = bucket.get("metadata", {})
            tags = {str(tag) for tag in meta.get("tags", [])}
            if "haven_favorite" not in tags:
                continue
            if meta.get("resolved") or meta.get("digested"):
                continue
            candidates.append(bucket)

        if not candidates:
            return "", []

        active_pool = [bucket for bucket in candidates if bucket.get("id") not in recent_ids] or candidates

        def favorite_key(bucket: dict) -> tuple[int, int, int, str]:
            meta = bucket.get("metadata", {})
            tags = {str(tag) for tag in meta.get("tags", [])}
            flavor_count = sum(1 for tag in tags if tag.startswith("flavor_"))
            protected = 1 if (meta.get("anchor") or meta.get("pinned") or meta.get("protected")) else 0
            return (
                protected,
                flavor_count,
                int(meta.get("importance", 5)),
                str(meta.get("last_active") or meta.get("created") or ""),
            )

        active_pool.sort(key=favorite_key, reverse=True)
        selected = active_pool[: self.favorite_memory_max_cards]
        remaining = self.favorite_memory_budget
        parts = []
        selected_ids = []
        for bucket in selected:
            summary = await self._summarize_bucket(bucket)
            tokens = count_tokens_approx(summary)
            if tokens <= 0:
                continue
            if tokens > remaining and parts:
                break
            if tokens > remaining:
                summary = self._trim_text(summary, remaining)
                tokens = count_tokens_approx(summary)
            parts.append(f"- {summary}")
            selected_ids.append(bucket["id"])
            remaining -= tokens
            if remaining <= 0:
                break
        return "\n".join(parts), selected_ids

    async def _build_related_memory_block(
        self,
        recalled_buckets: list[dict],
        all_buckets: list[dict],
    ) -> str:
        if self.related_memory_budget <= 0 or not recalled_buckets:
            return ""
        recalled_ids = [bucket["id"] for bucket in recalled_buckets if bucket.get("id")]
        edges = self.memory_edge_store.related_edges(
            recalled_ids,
            min_confidence=self.edge_min_confidence,
            limit_per_source=1,
        )
        if not edges:
            return ""
        bucket_map = {bucket["id"]: bucket for bucket in all_buckets}
        recalled_set = set(recalled_ids)
        remaining = self.related_memory_budget
        parts = []
        for edge in edges:
            target_id = edge.get("target")
            if target_id in recalled_set:
                continue
            target = bucket_map.get(target_id)
            if not target:
                continue
            summary = await self._summarize_bucket(target)
            reason = edge.get("reason") or edge.get("relation_type", "relates_to")
            line = (
                f"- {edge.get('source')} -> {target_id} "
                f"[{edge.get('relation_type')}, confidence={edge.get('confidence')}] "
                f"{reason}\n  {summary}"
            )
            tokens = count_tokens_approx(line)
            if tokens > remaining and parts:
                break
            if tokens > remaining:
                line = self._trim_text(line, remaining)
                tokens = count_tokens_approx(line)
            parts.append(line)
            remaining -= tokens
            if remaining <= 0:
                break
        return "\n".join(parts)

    async def _select_dynamic_buckets(
        self,
        query: str,
        session_id: str,
        all_buckets: list[dict],
    ) -> list[dict]:
        if not query or self.inject_max_cards <= 0:
            return []

        eligible = [bucket for bucket in all_buckets if self._is_dynamic_candidate(bucket)]
        if not eligible:
            return []

        bucket_map = {bucket["id"]: bucket for bucket in eligible}
        keyword_scores = self._get_keyword_candidates(query, eligible)
        semantic_scores = await self._get_semantic_candidates(query, set(bucket_map))
        candidate_ids = set(keyword_scores) | set(semantic_scores)
        if not candidate_ids:
            return []

        now = datetime.now()
        scored_candidates = []
        for bucket_id in candidate_ids:
            bucket = bucket_map.get(bucket_id)
            if not bucket:
                continue
            meta = bucket.get("metadata", {})
            freshness_score = self._clamp(self.bucket_mgr._calc_time_score(meta))
            importance_score = self._clamp(float(meta.get("importance", 5)) / 10.0)
            semantic_score = self._clamp(semantic_scores.get(bucket_id, 0.0))
            keyword_score = self._clamp(keyword_scores.get(bucket_id, 0.0))
            base_score = (
                semantic_score * self.semantic_weight
                + keyword_score * self.keyword_weight
                + importance_score * self.importance_weight
                + freshness_score * self.freshness_weight
            )
            cooldown_multiplier = self.state_store.get_cooldown_multiplier(
                session_id=session_id,
                bucket_id=bucket_id,
                cooldown_hours=self.cooldown_hours,
                cooldown_floor=self.cooldown_floor,
                now=now,
            )
            scored_candidates.append(
                {
                    "bucket": bucket,
                    "score": round(base_score * cooldown_multiplier, 4),
                    "semantic_score": semantic_score,
                    "keyword_score": keyword_score,
                    "importance_score": importance_score,
                    "freshness_score": freshness_score,
                    "cooldown_multiplier": cooldown_multiplier,
                }
            )

        scored_candidates.sort(key=lambda item: item["score"], reverse=True)
        recent_ids = self.state_store.get_recent_bucket_ids(session_id, self.skip_recent_rounds)
        filtered = [item for item in scored_candidates if item["bucket"]["id"] not in recent_ids]
        active_pool = filtered or scored_candidates
        selected = self._pick_dynamic_cards(active_pool)
        return [item["bucket"] for item in selected]

    def _get_keyword_candidates(self, query: str, buckets: list[dict]) -> dict[str, float]:
        scored = []
        for bucket in buckets:
            keyword_score = self._clamp(self.bucket_mgr._calc_topic_score(query, bucket))
            if keyword_score > 0:
                scored.append((bucket["id"], keyword_score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return {bucket_id: score for bucket_id, score in scored[: self.dynamic_top_k]}

    async def _get_semantic_candidates(self, query: str, eligible_ids: set[str]) -> dict[str, float]:
        if not getattr(self.embedding_engine, "enabled", False):
            return {}

        results = await self.embedding_engine.search_similar(query, top_k=self.dynamic_top_k)
        semantic_scores = {}
        for bucket_id, similarity in results:
            if bucket_id not in eligible_ids:
                continue
            semantic_scores[bucket_id] = self._clamp(similarity)
        return semantic_scores

    def _pick_dynamic_cards(self, scored_candidates: list[dict]) -> list[dict]:
        if not scored_candidates:
            return []

        chosen = []
        first = scored_candidates[0]
        if first["score"] < self.first_card_min_score:
            return []
        chosen.append(first)

        if self.inject_max_cards < 2 or len(scored_candidates) < 2:
            return chosen

        second = scored_candidates[1]
        if (
            second["score"] >= self.second_card_min_score
            and second["score"] >= first["score"] * self.second_card_relative_score
        ):
            chosen.append(second)
        return chosen

    async def _summarize_buckets(self, buckets: list[dict], budget: int) -> str:
        if budget <= 0 or not buckets:
            return ""

        remaining = budget
        parts = []
        for bucket in buckets:
            summary = await self._summarize_bucket(bucket)
            summary_tokens = count_tokens_approx(summary)
            if summary_tokens <= 0:
                continue
            if summary_tokens > remaining and parts:
                break
            if summary_tokens > remaining:
                summary = self._trim_text(summary, remaining)
                summary_tokens = count_tokens_approx(summary)
            if summary_tokens <= 0:
                continue
            parts.append(f"- {summary}")
            remaining -= summary_tokens
            if remaining <= 0:
                break
        return "\n".join(parts)

    async def _summarize_bucket(self, bucket: dict) -> str:
        metadata = {
            key: value
            for key, value in bucket.get("metadata", {}).items()
            if key != "tags"
        }
        cleaned = strip_wikilinks(bucket.get("content", ""))
        try:
            return await self.dehydrator.dehydrate(cleaned, metadata)
        except Exception as exc:
            logger.warning("Gateway summary fallback for %s: %s", bucket.get("id"), exc)
            title = metadata.get("name", bucket.get("id", "memory"))
            truncated = self._trim_text(cleaned, 90)
            return f"📌 记忆桶: {title}\n{truncated}"

    def _build_injected_context_messages(
        self,
        persona_block: str,
        core_memory: str,
        recent_context: str,
        recalled_memory: str,
        relationship_weather: str,
        favorite_memory: str,
        related_memory: str,
    ) -> tuple[str, str]:
        stable_sections = [
            "Use the following private memory only when it fits naturally. "
            "Keep the reply seamless and do not mention memory lookup, search, or hidden context.",
            "",
            "Core Memory",
            core_memory or "(none)",
        ]
        dynamic_sections = [
            "Live private context for the current turn. Use it quietly when relevant.",
            "",
            persona_block,
            "",
            "Relationship Weather",
            relationship_weather or "(none)",
            "",
            "Haven Favorite Memory",
            favorite_memory or "(none)",
            "",
            "Recent Context",
            recent_context or "(none)",
            "",
            "Recalled Memory",
            recalled_memory or "(none)",
            "",
            "Related Memory",
            related_memory or "(none)",
        ]
        stable_context = "\n".join(stable_sections).strip()
        dynamic_context = "\n".join(dynamic_sections).strip()
        stable_tokens = count_tokens_approx(stable_context)
        dynamic_tokens = count_tokens_approx(dynamic_context)
        if stable_tokens + dynamic_tokens <= self.inject_total_budget:
            return stable_context, dynamic_context
        if stable_tokens >= self.inject_total_budget:
            return self._trim_text(stable_context, self.inject_total_budget), ""
        remaining = max(0, self.inject_total_budget - stable_tokens)
        return stable_context, self._trim_text(dynamic_context, remaining)

    def _inject_context_messages(
        self,
        messages: list[dict],
        stable_context: str,
        dynamic_context: str,
    ) -> list[dict]:
        new_messages = deepcopy(messages)
        if stable_context.strip():
            stable_message = {"role": "system", "content": stable_context}
            if new_messages and isinstance(new_messages[0], dict) and new_messages[0].get("role") == "system":
                new_messages.insert(1, stable_message)
            else:
                new_messages.insert(0, stable_message)
        if dynamic_context.strip():
            current_user_index = self._current_turn_user_index(new_messages)
            if current_user_index is not None:
                new_messages[current_user_index] = self._prepend_dynamic_context_to_user_message(
                    new_messages[current_user_index],
                    dynamic_context,
                )
            else:
                dynamic_message = {"role": "system", "content": dynamic_context}
                insert_at = self._after_leading_system_index(new_messages)
                new_messages.insert(insert_at, dynamic_message)
        return new_messages

    def _current_turn_user_index(self, messages: list[dict]) -> int | None:
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role == "system":
                continue
            if role == "user":
                return index
            return None
        return None

    def _after_leading_system_index(self, messages: list[dict]) -> int:
        for index, message in enumerate(messages):
            if not isinstance(message, dict) or message.get("role") != "system":
                return index
        return len(messages)

    def _prepend_dynamic_context_to_user_message(
        self,
        message: dict[str, Any],
        dynamic_context: str,
    ) -> dict[str, Any]:
        updated = deepcopy(message)
        prefix = (
            "<ombre_live_context>\n"
            f"{dynamic_context}\n"
            "</ombre_live_context>\n\n"
            "Current user message:\n"
        )
        content = updated.get("content")
        if isinstance(content, str):
            updated["content"] = prefix + content
        elif isinstance(content, list):
            updated["content"] = [{"type": "text", "text": prefix}, *deepcopy(content)]
        else:
            updated["content"] = prefix
        return updated

    def _restore_cached_reasoning_content(self, session_id: str, messages: Any) -> None:
        if not isinstance(messages, list) or not any(
            isinstance(message, dict) and message.get("role") == "tool"
            for message in messages
        ):
            return

        cache = self.pending_tool_reasoning.get(session_id)
        if not cache:
            return

        restored = 0
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            if message.get("reasoning_content"):
                continue
            signature = self._tool_call_signature(message)
            if not signature:
                continue
            cached_message = cache.get(signature)
            if not cached_message or not cached_message.get("reasoning_content"):
                continue
            message["reasoning_content"] = cached_message["reasoning_content"]
            restored += 1

        if restored:
            logger.info(
                "Gateway restored reasoning_content for %s assistant tool-call message(s) | session=%s",
                restored,
                session_id,
            )

    def _capture_reasoning_from_response(self, session_id: str, upstream_response: httpx.Response) -> None:
        try:
            body = upstream_response.json()
        except ValueError:
            return
        self._capture_reasoning_from_response_body(session_id, body)

    def _capture_reasoning_from_response_body(self, session_id: str, body: Any) -> None:
        message = self._extract_assistant_message_from_response_body(body)
        if message:
            self._update_reasoning_cache(session_id, message)

    def _extract_assistant_message_from_response_body(self, body: Any) -> dict[str, Any] | None:
        if not isinstance(body, dict):
            return None
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        choice = choices[0]
        if not isinstance(choice, dict):
            return None
        message = choice.get("message")
        if isinstance(message, dict) and message.get("role", "assistant") == "assistant":
            return message
        return None

    def _update_reasoning_cache(self, session_id: str, assistant_message: dict[str, Any]) -> None:
        signature = self._tool_call_signature(assistant_message)
        reasoning_content = assistant_message.get("reasoning_content")
        if signature and reasoning_content:
            cache = self.pending_tool_reasoning.setdefault(session_id, {})
            cache[signature] = {
                "reasoning_content": reasoning_content,
                "tool_calls": deepcopy(assistant_message.get("tool_calls", [])),
            }
            logger.info(
                "Gateway cached reasoning_content for tool continuation | session=%s tool_calls=%s",
                session_id,
                list(signature),
            )
            return

        if not signature:
            self.pending_tool_reasoning.pop(session_id, None)

    def _tool_call_signature(self, assistant_message: Any) -> tuple[str, ...]:
        if not isinstance(assistant_message, dict):
            return ()
        tool_calls = assistant_message.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            return ()

        signature = []
        for index, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function", {})
            if isinstance(function, dict) and function.get("name"):
                signature.append(
                    f"idx:{index}:{function.get('name', '')}:{self._normalize_tool_arguments(function.get('arguments', ''))}"
                )
                continue
            tool_id = tool_call.get("id")
            if tool_id:
                signature.append(f"id:{tool_id}")
        return tuple(signature)

    def _normalize_tool_arguments(self, arguments: Any) -> str:
        if isinstance(arguments, (dict, list)):
            return json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if isinstance(arguments, str):
            raw = arguments.strip()
            if not raw:
                return ""
            try:
                parsed = json.loads(raw)
            except ValueError:
                return " ".join(raw.split())
            return json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return str(arguments)

    def _new_stream_capture_state(self) -> dict[str, Any]:
        return {
            "decoder": codecs.getincrementaldecoder("utf-8")(),
            "buffer": "",
            "message": {
                "role": "assistant",
                "content": "",
                "reasoning_content": "",
            },
            "usage": {},
            "tool_calls_by_index": {},
        }

    def _consume_stream_capture_chunk(
        self,
        stream_state: dict[str, Any],
        chunk: bytes,
        final: bool = False,
    ) -> None:
        decoder = stream_state["decoder"]
        if chunk:
            stream_state["buffer"] += decoder.decode(chunk)
        if final:
            stream_state["buffer"] += decoder.decode(b"", final=True)

        buffer = stream_state["buffer"].replace("\r\n", "\n")
        while "\n\n" in buffer:
            event_text, buffer = buffer.split("\n\n", 1)
            self._consume_sse_event(stream_state, event_text)

        if final and buffer.strip():
            self._consume_sse_event(stream_state, buffer)
            buffer = ""

        stream_state["buffer"] = buffer

    def _consume_sse_event(self, stream_state: dict[str, Any], event_text: str) -> None:
        data_lines = []
        for raw_line in event_text.split("\n"):
            line = raw_line.strip()
            if line.startswith("data:"):
                data_lines.append(line[5:].strip())

        if not data_lines:
            return
        payload = "\n".join(data_lines).strip()
        if not payload or payload == "[DONE]":
            return

        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            return

        if not isinstance(event, dict):
            return
        usage = event.get("usage")
        if isinstance(usage, dict):
            stream_state["usage"].update(usage)
        for choice in event.get("choices", []):
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if isinstance(delta, dict):
                self._merge_stream_message_delta(stream_state, delta)
            message = choice.get("message")
            if isinstance(message, dict):
                self._merge_complete_message(stream_state, message)

    def _merge_stream_message_delta(self, stream_state: dict[str, Any], delta: dict[str, Any]) -> None:
        message = stream_state["message"]
        if delta.get("role"):
            message["role"] = delta["role"]
        if isinstance(delta.get("content"), str):
            message["content"] += delta["content"]
        if isinstance(delta.get("reasoning_content"), str):
            message["reasoning_content"] += delta["reasoning_content"]

        tool_calls = delta.get("tool_calls")
        if not isinstance(tool_calls, list):
            return
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            index = int(tool_call.get("index", 0))
            target = stream_state["tool_calls_by_index"].setdefault(
                index,
                {"type": "function", "function": {"name": "", "arguments": ""}},
            )
            if tool_call.get("id"):
                target["id"] = tool_call["id"]
            if tool_call.get("type"):
                target["type"] = tool_call["type"]
            function = tool_call.get("function")
            if isinstance(function, dict):
                target_function = target.setdefault("function", {"name": "", "arguments": ""})
                if isinstance(function.get("name"), str):
                    target_function["name"] += function["name"]
                if isinstance(function.get("arguments"), str):
                    target_function["arguments"] += function["arguments"]

    def _merge_complete_message(self, stream_state: dict[str, Any], message: dict[str, Any]) -> None:
        target = stream_state["message"]
        if message.get("role"):
            target["role"] = message["role"]
        if isinstance(message.get("content"), str):
            target["content"] = message["content"]
        if isinstance(message.get("reasoning_content"), str):
            target["reasoning_content"] = message["reasoning_content"]
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            stream_state["tool_calls_by_index"] = {
                index: deepcopy(tool_call)
                for index, tool_call in enumerate(tool_calls)
                if isinstance(tool_call, dict)
            }

    def _capture_reasoning_from_stream_state(self, session_id: str, stream_state: dict[str, Any]) -> None:
        assistant_message = self._build_stream_assistant_message(stream_state)
        if assistant_message:
            self._update_reasoning_cache(session_id, assistant_message)

    def _build_stream_assistant_message(self, stream_state: dict[str, Any]) -> dict[str, Any] | None:
        message = deepcopy(stream_state.get("message", {}))
        tool_calls_by_index = stream_state.get("tool_calls_by_index", {})
        tool_calls = [
            deepcopy(tool_calls_by_index[index])
            for index in sorted(tool_calls_by_index)
            if isinstance(tool_calls_by_index[index], dict)
        ]

        content = message.get("content", "")
        reasoning_content = message.get("reasoning_content", "")
        if not (tool_calls or content or reasoning_content):
            return None

        assistant_message: dict[str, Any] = {"role": message.get("role", "assistant")}
        assistant_message["content"] = content if content else None
        if reasoning_content:
            assistant_message["reasoning_content"] = reasoning_content
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        return assistant_message

    def _is_dynamic_candidate(self, bucket: dict) -> bool:
        meta = bucket.get("metadata", {})
        if meta.get("type") in {"feel", "permanent", "archived"}:
            return False
        if meta.get("resolved"):
            return False
        if meta.get("pinned") or meta.get("protected"):
            return False
        return True

    def _trim_text(self, text: str, budget_tokens: int) -> str:
        if budget_tokens <= 0:
            return ""
        if count_tokens_approx(text) <= budget_tokens:
            return text
        trimmed = text
        while trimmed and count_tokens_approx(trimmed) > budget_tokens:
            cut = max(1, int(len(trimmed) * 0.85))
            trimmed = trimmed[:cut].rstrip()
        return trimmed

    def _parse_iso(self, value: Any) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is not None:
            return parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed

    def _clamp(self, value: float, lower: float = 0.0, upper: float = 1.0) -> float:
        return max(lower, min(upper, float(value)))

    def _load_upstreams(self) -> list[dict[str, Any]]:
        raw_upstreams = self.gateway_cfg.get("upstreams", [])
        if isinstance(raw_upstreams, list) and raw_upstreams:
            upstreams = []
            for index, raw in enumerate(raw_upstreams, start=1):
                if not isinstance(raw, dict):
                    continue
                name = str(raw.get("name") or f"upstream-{index}").strip() or f"upstream-{index}"
                base_url = str(raw.get("base_url") or "").rstrip("/")
                default_model = str(raw.get("default_model") or "").strip()
                api_key = str(raw.get("api_key") or "").strip()
                api_key_env = str(raw.get("api_key_env") or "").strip()
                prompt_cache = str(raw.get("prompt_cache") or "").strip().lower()
                prompt_cache_retention = str(raw.get("prompt_cache_retention") or "").strip()
                if api_key_env and not api_key:
                    api_key = os.environ.get(api_key_env, "")
                upstreams.append(
                    {
                        "name": name,
                        "base_url": base_url,
                        "api_key": api_key,
                        "default_model": default_model,
                        "models": self._normalize_model_list(raw.get("models", []), default_model),
                        "prompt_cache": prompt_cache,
                        "prompt_cache_retention": prompt_cache_retention,
                    }
                )
            if upstreams:
                return upstreams

        return [
            {
                "name": "default",
                "base_url": self.upstream_base_url,
                "api_key": self.upstream_api_key,
                "default_model": self.upstream_default_model,
                "models": self._normalize_model_list(
                    self.gateway_cfg.get("upstream_models", []),
                    self.upstream_default_model,
                ),
                "prompt_cache": str(self.gateway_cfg.get("prompt_cache") or "").strip().lower(),
                "prompt_cache_retention": str(
                    self.gateway_cfg.get("prompt_cache_retention") or ""
                ).strip(),
            }
        ]

    def _aggregate_upstream_models(self) -> list[str]:
        models = []
        for upstream in self.upstreams:
            for model in upstream.get("models", []):
                if not model:
                    continue
                if model in models:
                    logger.warning(
                        'Duplicate gateway model "%s" found in upstream "%s"; first match wins',
                        model,
                        upstream.get("name", "unknown"),
                    )
                    continue
                models.append(model)
        return models

    def _get_upstream_for_model(self, model: str) -> dict[str, Any]:
        if not self.upstreams:
            raise RuntimeError("gateway upstream is not configured")

        if len(self.upstreams) == 1:
            upstream = self.upstreams[0]
        else:
            normalized_model = str(model or "").strip()
            if not normalized_model:
                raise ValueError("model is required when gateway has multiple upstreams")
            upstream = next(
                (
                    candidate
                    for candidate in self.upstreams
                    if normalized_model in candidate.get("models", [])
                ),
                None,
            )
            if upstream is None:
                raise ValueError(f'model "{normalized_model}" is not configured in gateway.upstreams')

        if not upstream.get("base_url"):
            raise RuntimeError(f'gateway upstream "{upstream["name"]}" base_url is not configured')
        if not upstream.get("api_key"):
            raise RuntimeError(f'gateway upstream "{upstream["name"]}" api_key is not configured')
        return upstream

    def _normalize_model_list(self, raw_models: Any, default_model: str) -> list[str]:
        if isinstance(raw_models, str):
            candidates = [item.strip() for item in raw_models.split(",")]
        elif isinstance(raw_models, list):
            candidates = [str(item).strip() for item in raw_models]
        else:
            candidates = []

        models = []
        for model in candidates:
            if model and model not in models:
                models.append(model)

        if default_model and default_model not in models:
            models.insert(0, default_model)
        return models


def create_gateway_app(
    config: dict | None = None,
    service: GatewayService | None = None,
) -> Starlette:
    config = config or load_config()
    service = service or GatewayService(config)

    @asynccontextmanager
    async def lifespan(app: Starlette):
        app.state.gateway_service = service
        yield
        await service.close()

    async def health(request: Request) -> JSONResponse:
        return await request.app.state.gateway_service.handle_health(request)

    async def chat_completions(request: Request) -> Response:
        return await request.app.state.gateway_service.handle_chat(request)

    async def anthropic_messages(request: Request) -> Response:
        return await request.app.state.gateway_service.handle_anthropic_messages(request)

    async def models(request: Request) -> Response:
        return await request.app.state.gateway_service.handle_models(request)

    app = Starlette(
        debug=False,
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/v1/models", models, methods=["GET"]),
            Route("/v1/chat/completions", chat_completions, methods=["POST"]),
            Route("/v1/messages", anthropic_messages, methods=["POST"]),
        ],
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    return app


def main() -> None:
    config = load_config()
    setup_logging(config.get("log_level", "INFO"))
    gateway_cfg = config.get("gateway", {})
    app = create_gateway_app(config=config)
    host = gateway_cfg.get("host", "0.0.0.0")
    port = int(gateway_cfg.get("port", 8010))
    logger.info("Ombre Brain gateway starting | host=%s port=%s", host, port)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
