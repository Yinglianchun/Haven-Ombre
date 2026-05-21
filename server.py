# ============================================================
# Module: MCP Server Entry Point (server.py)
# 模块：MCP 服务器主入口
#
# Starts the Ombre Brain MCP service and registers memory
# operation tools for Claude to call.
# 启动 Ombre Brain MCP 服务，注册记忆操作工具供 Claude 调用。
#
# Core responsibilities:
# 核心职责：
#   - Initialize config, bucket manager, dehydrator, decay engine
#     初始化配置、记忆桶管理器、脱水器、衰减引擎
#   - Expose MCP tools:
#     暴露 MCP 工具：
#       breath — Surface unresolved memories or search by keyword
#                浮现未解决记忆 或 按关键词检索
#       hold   — Store a single memory
#                存储单条记忆
#       grow   — Long-note memory digest, auto-split selected content into buckets
#                长内容摘记，筛选后拆分多桶
#       trace  — Modify metadata / resolved / delete
#                修改元数据 / resolved 标记 / 删除
#       pulse  — System status + bucket listing
#                系统状态 + 所有桶列表
#       reflect — Daily/weekly relationship weather
#                 日/周关系天气
#
# Startup:
# 启动方式：
#   Local:  python server.py
#   Remote: OMBRE_TRANSPORT=streamable-http python server.py
#   Docker: docker-compose up
# ============================================================

import os
import sys
import random
import logging
import asyncio
import hashlib
import hmac
import json as _json_lib
import re
import secrets
import time
from base64 import b64decode
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlencode, urlparse
import httpx


# --- Ensure same-directory modules can be imported ---
# --- 确保同目录下的模块能被正确导入 ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP

from bucket_manager import BucketManager
from dehydrator import Dehydrator
from decay_engine import DecayEngine
from embedding_engine import EmbeddingEngine
from import_memory import ImportEngine
from memory_edges import MemoryEdgeStore
from persona_engine import PersonaStateEngine
from reflection_engine import ReflectionEngine
from utils import load_config, setup_logging, strip_wikilinks, count_tokens_approx

# --- Load config & init logging / 加载配置 & 初始化日志 ---
config = load_config()
setup_logging(config.get("log_level", "INFO"))
logger = logging.getLogger("ombre_brain")

MEMORY_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")

# --- Initialize core components / 初始化核心组件 ---
bucket_mgr = BucketManager(config)                  # Bucket manager / 记忆桶管理器
dehydrator = Dehydrator(config)                      # Dehydrator / 脱水器
decay_engine = DecayEngine(config, bucket_mgr)       # Decay engine / 衰减引擎
embedding_engine = EmbeddingEngine(config)            # Embedding engine / 向量化引擎
import_engine = ImportEngine(config, bucket_mgr, dehydrator, embedding_engine)  # Import engine / 导入引擎
persona_engine = PersonaStateEngine(config)           # Persona state engine / 人格状态引擎
memory_edge_store = MemoryEdgeStore(config)            # Explicit memory relationship edges / 显式记忆关系边
reflection_engine = ReflectionEngine(config)           # Reflection worker / 关系天气与关系整理

# --- Create MCP server instance / 创建 MCP 服务器实例 ---
# host="0.0.0.0" so Docker container's SSE is externally reachable
# stdio mode ignores host (no network)
mcp = FastMCP(
    "Ombre Brain",
    host="0.0.0.0",
    port=8000,
)


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _split_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


class ChatGptOAuthProvider:
    def __init__(
        self,
        client_id: str = "",
        client_secret: str = "",
        access_token: str = "",
        refresh_token: str = "",
        public_base_url: str = "",
        redirect_prefix: str = "https://chatgpt.com/connector/oauth/",
        token_ttl_seconds: int = 30 * 24 * 60 * 60,
    ) -> None:
        self.client_id = client_id.strip()
        self.client_secret = client_secret.strip()
        self.access_token = access_token.strip()
        self.refresh_token = refresh_token.strip()
        self.public_base_url = public_base_url.strip().rstrip("/")
        self.redirect_prefix = redirect_prefix.strip()
        self.token_ttl_seconds = token_ttl_seconds
        self._codes: dict[str, tuple[str, float]] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.client_id and self.access_token)

    @property
    def token_auth_methods(self) -> list[str]:
        if self.client_secret:
            return ["client_secret_post", "client_secret_basic"]
        return ["none"]

    def external_base(self, request=None) -> str:
        if self.public_base_url:
            return self.public_base_url
        if request is not None:
            return str(request.base_url).rstrip("/")
        return ""

    def valid_client_id(self, client_id: str | None) -> bool:
        return bool(client_id) and hmac.compare_digest(client_id, self.client_id)

    def valid_client_secret(self, client_secret: str | None) -> bool:
        if not self.client_secret:
            return True
        return bool(client_secret) and hmac.compare_digest(client_secret, self.client_secret)

    def valid_redirect_uri(self, redirect_uri: str | None) -> bool:
        return bool(redirect_uri) and redirect_uri.startswith(self.redirect_prefix)

    def create_authorization_code(self, redirect_uri: str) -> str:
        code = secrets.token_urlsafe(32)
        self._codes[code] = (redirect_uri, time.time() + 300)
        return code

    def consume_authorization_code(self, code: str | None, redirect_uri: str | None) -> bool:
        if not code:
            return False
        entry = self._codes.pop(code, None)
        if not entry:
            return False
        stored_redirect_uri, expires_at = entry
        if time.time() > expires_at:
            return False
        if redirect_uri and redirect_uri != stored_redirect_uri:
            return False
        return True

    def valid_access_token(self, token: str | None) -> bool:
        return bool(token) and hmac.compare_digest(token, self.access_token)

    def valid_refresh_token(self, token: str | None) -> bool:
        return bool(token) and hmac.compare_digest(token, self.refresh_token)


OMBRE_CHATGPT_OAUTH = ChatGptOAuthProvider(
    client_id=os.environ.get("OMBRE_CHATGPT_OAUTH_CLIENT_ID", ""),
    client_secret=os.environ.get("OMBRE_CHATGPT_OAUTH_CLIENT_SECRET", ""),
    access_token=os.environ.get("OMBRE_CHATGPT_OAUTH_ACCESS_TOKEN", ""),
    refresh_token=os.environ.get("OMBRE_CHATGPT_OAUTH_REFRESH_TOKEN", ""),
    public_base_url=os.environ.get("OMBRE_CHATGPT_OAUTH_PUBLIC_BASE_URL", ""),
    redirect_prefix=os.environ.get("OMBRE_CHATGPT_OAUTH_REDIRECT_PREFIX", "https://chatgpt.com/connector/oauth/"),
    token_ttl_seconds=_int_env("OMBRE_CHATGPT_OAUTH_TOKEN_TTL_SECONDS", 30 * 24 * 60 * 60),
)


def _default_oauth_protected_hosts() -> set[str]:
    raw = os.environ.get("OMBRE_CHATGPT_OAUTH_PROTECTED_HOSTS")
    hosts = set(_split_csv(raw)) if raw is not None else set()
    if raw is None and OMBRE_CHATGPT_OAUTH.public_base_url:
        host = urlparse(OMBRE_CHATGPT_OAUTH.public_base_url).hostname
        if host:
            hosts.add(host)
    return {host.lower() for host in hosts}


OMBRE_CHATGPT_OAUTH_PROTECTED_HOSTS = _default_oauth_protected_hosts()


def _oauth_public_path(path: str) -> bool:
    normalized = path.rstrip("/") or "/"
    return normalized in {
        "/oauth/authorize",
        "/oauth/token",
        "/.well-known/oauth-authorization-server",
        "/.well-known/oauth-protected-resource",
        "/.well-known/openid-configuration",
        "/mcp/oauth/authorize",
        "/mcp/oauth/token",
        "/mcp/.well-known/oauth-authorization-server",
        "/mcp/.well-known/oauth-protected-resource",
        "/mcp/.well-known/openid-configuration",
    }


def _mcp_path(path: str) -> bool:
    return path == "/mcp" or path.startswith("/mcp/")


def _bearer_token(headers: dict[str, str]) -> str | None:
    auth = headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return None
    return auth.split(" ", 1)[1].strip()


def _basic_client_credentials(headers: dict[str, str]) -> tuple[str | None, str | None]:
    auth = headers.get("authorization", "")
    if not auth.lower().startswith("basic "):
        return None, None
    try:
        decoded = b64decode(auth.split(" ", 1)[1]).decode("utf-8")
        client_id, client_secret = decoded.split(":", 1)
        return client_id, client_secret
    except Exception:
        return None, None


async def _oauth_form(request) -> dict[str, str]:
    body = await request.body()
    parsed = parse_qs(body.decode("utf-8"), keep_blank_values=True)
    return {key: values[-1] if values else "" for key, values in parsed.items()}


def _oauth_error(message: str, status_code: int = 400):
    from starlette.responses import JSONResponse
    return JSONResponse({"error": message}, status_code=status_code)


def _oauth_success_payload() -> dict:
    return {
        "access_token": OMBRE_CHATGPT_OAUTH.access_token,
        "token_type": "Bearer",
        "expires_in": OMBRE_CHATGPT_OAUTH.token_ttl_seconds,
        "refresh_token": OMBRE_CHATGPT_OAUTH.refresh_token,
        "scope": "",
    }


class OmbreChatGptOAuthMiddleware:
    def __init__(self, app, provider: ChatGptOAuthProvider, protected_hosts: set[str]) -> None:
        self.app = app
        self.provider = provider
        self.protected_hosts = {host.lower() for host in protected_hosts}

    async def __call__(self, scope, receive, send):
        if (
            scope.get("type") != "http"
            or not self.provider.enabled
            or scope.get("method") == "OPTIONS"
        ):
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if _oauth_public_path(path) or not _mcp_path(path) or not self._is_protected_host(scope):
            await self.app(scope, receive, send)
            return

        headers = {
            key.decode("latin1").lower(): value.decode("latin1")
            for key, value in scope.get("headers", [])
        }
        if self.provider.valid_access_token(_bearer_token(headers)):
            await self.app(scope, receive, send)
            return

        from starlette.responses import JSONResponse
        response = JSONResponse(
            {"error": "invalid_token"},
            status_code=401,
            headers={"WWW-Authenticate": 'Bearer realm="Ombre Brain"'},
        )
        await response(scope, receive, send)

    def _is_protected_host(self, scope) -> bool:
        if not self.protected_hosts:
            return False
        host = ""
        for key, value in scope.get("headers", []):
            if key.lower() == b"host":
                host = value.decode("latin1").split(":", 1)[0].lower()
                break
        return host in self.protected_hosts


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_dashboard_sessions: dict[str, float] = {}


def _dashboard_auth_file() -> str:
    state_dir = config.get("state_dir") or os.path.join(
        os.path.dirname(os.path.abspath(config.get("buckets_dir", "buckets"))),
        "state",
    )
    return os.path.join(state_dir, ".dashboard_auth.json")


def _load_dashboard_password_hash() -> str | None:
    try:
        path = _dashboard_auth_file()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = _json_lib.load(f)
            return data.get("password_hash")
    except Exception:
        logger.warning("Failed to load dashboard auth file", exc_info=True)
    return None


def _save_dashboard_password_hash(password: str) -> None:
    salt = secrets.token_hex(16)
    digest = hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()
    path = _dashboard_auth_file()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        _json_lib.dump({"password_hash": f"{salt}:{digest}"}, f)


def _verify_dashboard_hash(password: str, stored: str) -> bool:
    if ":" not in stored:
        return False
    salt, digest = stored.split(":", 1)
    current = hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()
    return hmac.compare_digest(digest, current)


def _dashboard_setup_needed() -> bool:
    if os.environ.get("OMBRE_DASHBOARD_PASSWORD", ""):
        return False
    return _load_dashboard_password_hash() is None


@mcp.custom_route("/oauth/authorize", methods=["GET"])
@mcp.custom_route("/mcp/oauth/authorize", methods=["GET"])
async def chatgpt_oauth_authorize(request):
    from starlette.responses import RedirectResponse

    if not OMBRE_CHATGPT_OAUTH.enabled:
        return _oauth_error("oauth_not_configured", 404)

    params = request.query_params
    client_id = params.get("client_id")
    redirect_uri = params.get("redirect_uri")
    response_type = params.get("response_type")
    state = params.get("state")

    if response_type != "code":
        return _oauth_error("unsupported_response_type")
    if not OMBRE_CHATGPT_OAUTH.valid_client_id(client_id):
        return _oauth_error("invalid_client", 401)
    if not OMBRE_CHATGPT_OAUTH.valid_redirect_uri(redirect_uri):
        return _oauth_error("invalid_redirect_uri")

    code = OMBRE_CHATGPT_OAUTH.create_authorization_code(redirect_uri)
    query = {"code": code}
    if state:
        query["state"] = state
    separator = "&" if "?" in redirect_uri else "?"
    return RedirectResponse(url=f"{redirect_uri}{separator}{urlencode(query)}", status_code=302)


@mcp.custom_route("/oauth/token", methods=["POST"])
@mcp.custom_route("/mcp/oauth/token", methods=["POST"])
async def chatgpt_oauth_token(request):
    if not OMBRE_CHATGPT_OAUTH.enabled:
        return _oauth_error("oauth_not_configured", 404)

    form = await _oauth_form(request)
    basic_client_id, basic_client_secret = _basic_client_credentials(request.headers)
    client_id = basic_client_id or form.get("client_id")
    client_secret = basic_client_secret or form.get("client_secret")

    if not OMBRE_CHATGPT_OAUTH.valid_client_id(client_id):
        return _oauth_error("invalid_client", 401)
    if not OMBRE_CHATGPT_OAUTH.valid_client_secret(client_secret):
        return _oauth_error("invalid_client", 401)

    grant_type = form.get("grant_type")
    if grant_type == "authorization_code":
        if not OMBRE_CHATGPT_OAUTH.consume_authorization_code(form.get("code"), form.get("redirect_uri")):
            return _oauth_error("invalid_grant")
    elif grant_type == "refresh_token":
        if not OMBRE_CHATGPT_OAUTH.valid_refresh_token(form.get("refresh_token")):
            return _oauth_error("invalid_grant")
    else:
        return _oauth_error("unsupported_grant_type")

    from starlette.responses import JSONResponse
    return JSONResponse(_oauth_success_payload())


def _oauth_server_metadata(request) -> dict:
    base = OMBRE_CHATGPT_OAUTH.external_base(request)
    return {
        "issuer": base,
        "authorization_endpoint": f"{base}/oauth/authorize",
        "token_endpoint": f"{base}/oauth/token",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "token_endpoint_auth_methods_supported": OMBRE_CHATGPT_OAUTH.token_auth_methods,
    }


@mcp.custom_route("/.well-known/oauth-authorization-server", methods=["GET"])
@mcp.custom_route("/.well-known/openid-configuration", methods=["GET"])
@mcp.custom_route("/mcp/.well-known/oauth-authorization-server", methods=["GET"])
@mcp.custom_route("/mcp/.well-known/openid-configuration", methods=["GET"])
async def chatgpt_oauth_metadata(request):
    from starlette.responses import JSONResponse

    if not OMBRE_CHATGPT_OAUTH.enabled:
        return _oauth_error("oauth_not_configured", 404)
    return JSONResponse(_oauth_server_metadata(request))


@mcp.custom_route("/.well-known/oauth-protected-resource", methods=["GET"])
@mcp.custom_route("/mcp/.well-known/oauth-protected-resource", methods=["GET"])
async def chatgpt_oauth_resource_metadata(request):
    from starlette.responses import JSONResponse

    if not OMBRE_CHATGPT_OAUTH.enabled:
        return _oauth_error("oauth_not_configured", 404)
    base = OMBRE_CHATGPT_OAUTH.external_base(request)
    return JSONResponse(
        {
            "resource": f"{base}/mcp",
            "authorization_servers": [base],
            "bearer_methods_supported": ["header"],
        }
    )


def _verify_dashboard_password(password: str) -> bool:
    env_password = os.environ.get("OMBRE_DASHBOARD_PASSWORD", "")
    if env_password:
        return hmac.compare_digest(password, env_password)
    stored = _load_dashboard_password_hash()
    return bool(stored and _verify_dashboard_hash(password, stored))


def _create_dashboard_session() -> str:
    token = secrets.token_urlsafe(32)
    _dashboard_sessions[token] = time.time() + 86400 * 7
    return token


def _dashboard_authenticated(request) -> bool:
    token = request.cookies.get("ombre_session")
    if not token:
        return False
    expiry = _dashboard_sessions.get(token)
    if expiry is None or time.time() > expiry:
        _dashboard_sessions.pop(token, None)
        return False
    return True


def _require_dashboard_auth(request):
    from starlette.responses import JSONResponse
    if _dashboard_authenticated(request):
        return None
    return JSONResponse(
        {"error": "unauthorized", "setup_needed": _dashboard_setup_needed()},
        status_code=401,
    )


def _dashboard_login_response():
    from starlette.responses import JSONResponse
    token = _create_dashboard_session()
    response = JSONResponse({"ok": True})
    response.set_cookie(
        "ombre_session",
        token,
        httponly=True,
        samesite="lax",
        max_age=86400 * 7,
    )
    return response


def _memory_write_token() -> str:
    return (
        os.environ.get("OMBRE_MEMORY_WRITE_TOKEN")
        or os.environ.get("OMBRE_GATEWAY_TOKEN")
        or str(config.get("gateway", {}).get("token") or "")
    )


def _authorized_memory_write(request) -> bool:
    token = _memory_write_token()
    if not token:
        return False

    candidates = []
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        candidates.append(auth.split(" ", 1)[1].strip())
    for header_name in ("x-ombre-token", "x-api-key"):
        value = request.headers.get(header_name)
        if value:
            candidates.append(value.strip())
    return any(hmac.compare_digest(candidate, token) for candidate in candidates)


def _string_list(value, default: list[str]) -> list[str]:
    if value is None:
        return default
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    elif isinstance(value, list):
        items = [str(item).strip() for item in value]
    else:
        items = [str(value).strip()]
    return [item for item in items if item] or default


def _float_between(value, default: float, low: float = 0.0, high: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _int_between(value, default: int, low: int = 1, high: int = 10) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _bool_value(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _anchor_config() -> tuple[int, float]:
    anchor_cfg = config.get("anchor", {}) if isinstance(config.get("anchor", {}), dict) else {}
    max_count = _int_between(anchor_cfg.get("max_count"), 24, 1, 200)
    try:
        min_age_hours = float(anchor_cfg.get("min_age_hours", 24))
    except (TypeError, ValueError):
        min_age_hours = 24.0
    return max_count, max(0.0, min_age_hours)


def _bucket_age_hours(bucket: dict) -> float | None:
    created = bucket.get("metadata", {}).get("created", "")
    if not created:
        return None
    try:
        parsed = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds() / 3600)


async def _can_mark_anchor(bucket_id: str, bucket: dict) -> tuple[bool, str]:
    max_count, min_age_hours = _anchor_config()
    age_hours = _bucket_age_hours(bucket)
    if age_hours is not None and age_hours < min_age_hours:
        return (
            False,
            f"这条记忆还太新，anchor 至少等待 {min_age_hours:g} 小时后再标记。",
        )
    all_buckets = await bucket_mgr.list_all(include_archive=True)
    anchor_count = sum(
        1
        for b in all_buckets
        if b["id"] != bucket_id and b.get("metadata", {}).get("anchor")
    )
    if anchor_count >= max_count:
        return False, f"anchor 名额已满（{max_count} 条）。请先取消一条旧 anchor。"
    return True, ""


def _bucket_read_payload(bucket: dict) -> dict:
    meta = bucket.get("metadata", {})
    fields = [
        "id",
        "name",
        "type",
        "domain",
        "tags",
        "importance",
        "valence",
        "arousal",
        "model_valence",
        "pinned",
        "protected",
        "resolved",
        "digested",
        "anchor",
        "source",
        "confidence",
        "period",
        "date",
        "created",
        "updated_at",
        "last_active",
        "activation_count",
    ]
    return {
        "id": bucket["id"],
        "metadata": {key: meta.get(key) for key in fields if key in meta},
        "content": strip_wikilinks(bucket.get("content", "")),
        "score": decay_engine.calculate_score(meta),
    }


def _queue_memory_enrichment(bucket_id: str) -> None:
    if not bucket_id:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    loop.create_task(_enrich_memory_async(bucket_id))


async def _enrich_memory_async(bucket_id: str) -> None:
    try:
        result = await reflection_engine.enrich_bucket(
            bucket_id,
            bucket_mgr,
            memory_edge_store,
            embedding_engine=embedding_engine,
        )
        logger.debug("Memory enrichment complete / 记忆关系补全完成: %s", result)
    except Exception as e:
        logger.warning("Memory enrichment failed / 记忆关系补全失败: %s: %s", bucket_id, e)


# =============================================================
# /health endpoint: lightweight keepalive
# 轻量保活接口
# For Cloudflare Tunnel or reverse proxy to ping, preventing idle timeout
# 供 Cloudflare Tunnel 或反代定期 ping，防止空闲超时断连
# =============================================================
@mcp.custom_route("/", methods=["GET"])
async def root_redirect(request):
    from starlette.responses import RedirectResponse
    return RedirectResponse(url="/dashboard")


@mcp.custom_route("/auth/status", methods=["GET"])
async def auth_status(request):
    from starlette.responses import JSONResponse
    return JSONResponse(
        {
            "authenticated": _dashboard_authenticated(request),
            "setup_needed": _dashboard_setup_needed(),
        }
    )


@mcp.custom_route("/auth/setup", methods=["POST"])
async def auth_setup(request):
    from starlette.responses import JSONResponse
    if not _dashboard_setup_needed():
        return JSONResponse({"error": "already configured"}, status_code=400)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json"}, status_code=400)
    password = str(body.get("password") or "").strip()
    if len(password) < 6:
        return JSONResponse({"error": "password must be at least 6 characters"}, status_code=400)
    _save_dashboard_password_hash(password)
    return _dashboard_login_response()


@mcp.custom_route("/auth/login", methods=["POST"])
async def auth_login(request):
    from starlette.responses import JSONResponse
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json"}, status_code=400)
    password = str(body.get("password") or "")
    if _verify_dashboard_password(password):
        return _dashboard_login_response()
    return JSONResponse({"error": "password rejected"}, status_code=401)


@mcp.custom_route("/auth/logout", methods=["POST"])
async def auth_logout(request):
    from starlette.responses import JSONResponse
    token = request.cookies.get("ombre_session")
    if token:
        _dashboard_sessions.pop(token, None)
    response = JSONResponse({"ok": True})
    response.delete_cookie("ombre_session")
    return response


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    from starlette.responses import JSONResponse
    try:
        stats = await bucket_mgr.get_stats()
        return JSONResponse({
            "status": "ok",
            "buckets": stats["permanent_count"] + stats["dynamic_count"],
            "decay_engine": "running" if decay_engine.is_running else "stopped",
            "memory_edges": len(memory_edge_store.list_edges()),
            "reflection": {
                "enabled": reflection_engine.enabled,
                "auto_enabled": reflection_engine.auto_enabled,
                "model": reflection_engine.model,
                "api_ready": bool(reflection_engine.api_key),
            },
        })
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


# =============================================================
# /breath-hook endpoint: Dedicated hook for SessionStart
# 会话启动专用挂载点
# =============================================================
@mcp.custom_route("/breath-hook", methods=["GET"])
async def breath_hook(request):
    from starlette.responses import PlainTextResponse
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
        # pinned
        pinned = [b for b in all_buckets if b["metadata"].get("pinned") or b["metadata"].get("protected")]
        # top 2 unresolved by score
        unresolved = [b for b in all_buckets
                      if not b["metadata"].get("resolved", False)
                      and b["metadata"].get("type") not in ("permanent", "feel")
                      and not b["metadata"].get("pinned")
                      and not b["metadata"].get("protected")]
        scored = sorted(unresolved, key=lambda b: decay_engine.calculate_score(b["metadata"]), reverse=True)

        parts = []
        token_budget = 10000
        for b in pinned:
            summary = await dehydrator.dehydrate(strip_wikilinks(b["content"]), {k: v for k, v in b["metadata"].items() if k != "tags"})
            parts.append(f"📌 [核心准则] {summary}")
            token_budget -= count_tokens_approx(summary)

        # Diversity: top-1 fixed + shuffle rest from top-20
        candidates = list(scored)
        if len(candidates) > 1:
            top1 = [candidates[0]]
            pool = candidates[1:min(20, len(candidates))]
            random.shuffle(pool)
            candidates = top1 + pool + candidates[min(20, len(candidates)):]
        # Hard cap: max 20 surfacing buckets in hook
        candidates = candidates[:20]

        for b in candidates:
            if token_budget <= 0:
                break
            summary = await dehydrator.dehydrate(strip_wikilinks(b["content"]), {k: v for k, v in b["metadata"].items() if k != "tags"})
            summary_tokens = count_tokens_approx(summary)
            if summary_tokens > token_budget:
                break
            parts.append(summary)
            token_budget -= summary_tokens

        if not parts:
            return PlainTextResponse("")
        return PlainTextResponse("[Ombre Brain - 记忆浮现]\n" + "\n---\n".join(parts))
    except Exception as e:
        logger.warning(f"Breath hook failed: {e}")
        return PlainTextResponse("")


# =============================================================
# /dream-hook endpoint: Dedicated hook for Dreaming
# Dreaming 专用挂载点
# =============================================================
@mcp.custom_route("/dream-hook", methods=["GET"])
async def dream_hook(request):
    from starlette.responses import PlainTextResponse
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
        candidates = [
            b for b in all_buckets
            if b["metadata"].get("type") not in ("permanent", "feel")
            and not b["metadata"].get("pinned", False)
            and not b["metadata"].get("protected", False)
        ]
        candidates.sort(key=lambda b: b["metadata"].get("created", ""), reverse=True)
        recent = candidates[:10]

        if not recent:
            return PlainTextResponse("")

        parts = []
        for b in recent:
            meta = b["metadata"]
            resolved_tag = "[已解决]" if meta.get("resolved", False) else "[未解决]"
            parts.append(
                f"{meta.get('name', b['id'])} {resolved_tag} "
                f"V{meta.get('valence', 0.5):.1f}/A{meta.get('arousal', 0.3):.1f}\n"
                f"{strip_wikilinks(b['content'][:200])}"
            )

        return PlainTextResponse("[Ombre Brain - Dreaming]\n" + "\n---\n".join(parts))
    except Exception as e:
        logger.warning(f"Dream hook failed: {e}")
        return PlainTextResponse("")


# =============================================================
# Internal helper: merge-or-create
# 内部辅助：检查是否可合并，可以则合并，否则新建
# Shared by hold and grow to avoid duplicate logic
# hold 和 grow 共用，避免重复逻辑
# =============================================================
async def _merge_or_create(
    content: str,
    tags: list,
    importance: int,
    domain: list,
    valence: float,
    arousal: float,
    name: str = "",
) -> tuple[str, str, bool]:
    """
    Check if a similar bucket exists for merging; merge if so, create if not.
    Returns (bucket_id, display_name, is_merged).
    检查是否有相似桶可合并，有则合并，无则新建。
    返回 (桶ID, 显示名称, 是否合并)。
    """
    try:
        existing = await bucket_mgr.search(
            content,
            limit=1,
            domain_filter=domain or None,
            include_archive=False,
        )
    except Exception as e:
        logger.warning(f"Search for merge failed, creating new / 合并搜索失败，新建: {e}")
        existing = []

    if existing and existing[0].get("score", 0) > config.get("merge_threshold", 75):
        bucket = existing[0]
        # --- Never merge into pinned/protected buckets ---
        # --- 不合并到钉选/保护桶 ---
        if not (
            bucket["metadata"].get("pinned")
            or bucket["metadata"].get("protected")
            or bucket["metadata"].get("type") == "feel"
        ):
            try:
                merged = await dehydrator.merge(bucket["content"], content)
                old_v = bucket["metadata"].get("valence", 0.5)
                old_a = bucket["metadata"].get("arousal", 0.3)
                merged_valence = round((old_v + valence) / 2, 2)
                merged_arousal = round((old_a + arousal) / 2, 2)
                await bucket_mgr.update(
                    bucket["id"],
                    content=merged,
                    tags=list(set(bucket["metadata"].get("tags", []) + tags)),
                    importance=max(bucket["metadata"].get("importance", 5), importance),
                    domain=list(set(bucket["metadata"].get("domain", []) + domain)),
                    valence=merged_valence,
                    arousal=merged_arousal,
                )
                # --- Update embedding after merge ---
                try:
                    await embedding_engine.generate_and_store(bucket["id"], merged)
                except Exception:
                    pass
                return bucket["id"], bucket["metadata"].get("name", bucket["id"]), True
            except Exception as e:
                logger.warning(f"Merge failed, creating new / 合并失败，新建: {e}")

    bucket_id = await bucket_mgr.create(
        content=content,
        tags=tags,
        importance=importance,
        domain=domain,
        valence=valence,
        arousal=arousal,
        name=name or None,
    )
    # --- Generate embedding for new bucket ---
    try:
        await embedding_engine.generate_and_store(bucket_id, content)
    except Exception:
        pass
    return bucket_id, name or bucket_id, False


async def _build_mcp_related_memory_block(
    source_buckets: list[dict],
    all_buckets: list[dict] | None,
    token_budget: int,
    limit_per_source: int,
    min_confidence: float,
) -> str:
    if token_budget <= 0 or not source_buckets:
        return ""

    limit_per_source = _int_between(limit_per_source, 1, 0, 5)
    min_confidence = _float_between(min_confidence, 0.55, 0.0, 1.0)
    if limit_per_source <= 0:
        return ""

    source_ids = [bucket["id"] for bucket in source_buckets if bucket.get("id")]
    source_set = set(source_ids)
    if not source_ids:
        return ""

    if all_buckets is None:
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
        except Exception as e:
            logger.warning(f"Failed to list buckets for related memory / 关联记忆列桶失败: {e}")
            all_buckets = []

    bucket_map = {bucket["id"]: bucket for bucket in all_buckets if bucket.get("id")}
    edges = memory_edge_store.related_edges(
        source_ids,
        min_confidence=min_confidence,
        limit_per_source=limit_per_source,
    )

    parts = []
    seen_targets = set()
    remaining = token_budget
    for edge in edges:
        target_id = edge.get("target")
        if not target_id or target_id in source_set or target_id in seen_targets:
            continue

        target = bucket_map.get(target_id)
        if not target:
            continue
        meta = target.get("metadata", {})
        if meta.get("type") == "feel":
            continue

        try:
            clean_meta = {k: v for k, v in meta.items() if k != "tags"}
            summary = await dehydrator.dehydrate(
                strip_wikilinks(target.get("content", "")),
                clean_meta,
            )
            arrow = "<-" if edge.get("direction") == "incoming" else "->"
            confidence = edge.get("confidence", 0.0)
            relation_type = edge.get("relation_type", "relates_to")
            reason = edge.get("reason") or relation_type
            block = (
                f"[{edge.get('source')} {arrow} {target_id}] "
                f"[{relation_type}, confidence={confidence}] {reason}\n"
                f"[bucket_id:{target_id}] {summary}"
            )
            block_tokens = count_tokens_approx(block)
            if block_tokens > remaining:
                break
            parts.append(block)
            seen_targets.add(target_id)
            remaining -= block_tokens
            if remaining <= 0:
                break
        except Exception as e:
            logger.warning(f"Failed to build related memory block / 关联记忆构建失败: {e}")
            continue

    return "\n---\n".join(parts)


# =============================================================
# Tool 1: breath — Breathe
# 工具 1：breath — 呼吸
#
# No args: surface highest-weight unresolved memories (active push)
# 无参数：浮现权重最高的未解决记忆
# With args: search by keyword + emotion coordinates
# 有参数：按关键词+情感坐标检索记忆
# =============================================================
@mcp.tool()
async def breath(
    query: str = "",
    max_tokens: int = 10000,
    domain: str = "",
    valence: float = -1,
    arousal: float = -1,
    max_results: int = 20,
    include_related: bool = True,
    related_per_memory: int = 1,
    edge_min_confidence: float = 0.55,
    include_core: bool = True,
    core_limit: int = 3,
) -> str:
    """读取记忆,不写入。
    调用方式: 新对话用 breath(); 查过去用 breath(query="主题词"); 只读模型感受用 breath(domain="feel")。
    默认只从本次实际返回的普通记忆沿持久化 memory_edges 带一跳关联记忆; embedding 相似边只是检索/图谱参考,不是可手写的记忆关系。
    include_core/core_limit 控制 pinned/protected 核心准则数量; include_related=False 可关闭关联记忆块。
    """
    await decay_engine.ensure_started()
    max_results = _int_between(max_results, 20, 1, 50)
    max_tokens = _int_between(max_tokens, 10000, 0, 20000)
    include_related = _bool_value(include_related, True)
    related_per_memory = _int_between(related_per_memory, 1, 0, 5)
    edge_min_confidence = _float_between(edge_min_confidence, 0.55, 0.0, 1.0)
    include_core = _bool_value(include_core, True)
    core_limit = _int_between(core_limit, 3, 0, 20)

    # --- No args or empty query: surfacing mode (weight pool active push) ---
    # --- 无参数或空query：浮现模式（权重池主动推送）---
    if not query or not query.strip():
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
        except Exception as e:
            logger.error(f"Failed to list buckets for surfacing / 浮现列桶失败: {e}")
            return "记忆系统暂时无法访问。"

        # --- Core buckets: protected first, pinned limited by core_limit ---
        # --- 核心桶：protected 优先，pinned 按 core_limit 限流 ---
        core_candidates = [
            b for b in all_buckets
            if b["metadata"].get("pinned") or b["metadata"].get("protected")
        ]
        protected = [
            b for b in core_candidates
            if b["metadata"].get("protected")
        ]
        pinned = [
            b for b in core_candidates
            if b["metadata"].get("pinned") and not b["metadata"].get("protected")
        ]
        protected.sort(
            key=lambda b: decay_engine.calculate_score(b["metadata"]),
            reverse=True,
        )
        pinned.sort(
            key=lambda b: (
                int(b["metadata"].get("importance", 5)),
                decay_engine.calculate_score(b["metadata"]),
                b["metadata"].get("updated_at") or b["metadata"].get("created", ""),
            ),
            reverse=True,
        )
        selected_core = (protected + pinned)[:core_limit] if include_core else []

        # --- Unresolved buckets: surface top N by weight ---
        # --- 未解决桶：按权重浮现前 N 条 ---
        unresolved = [
            b for b in all_buckets
            if not b["metadata"].get("resolved", False)
            and b["metadata"].get("type") not in ("permanent", "feel")
            and not b["metadata"].get("pinned", False)
            and not b["metadata"].get("protected", False)
        ]

        logger.info(
            f"Breath surfacing: {len(all_buckets)} total, "
            f"{len(core_candidates)} core, {len(unresolved)} unresolved"
        )

        scored = sorted(
            unresolved,
            key=lambda b: decay_engine.calculate_score(b["metadata"]),
            reverse=True,
        )

        if scored:
            top_scores = [(b["metadata"].get("name", b["id"]), decay_engine.calculate_score(b["metadata"])) for b in scored[:5]]
            logger.info(f"Top unresolved scores: {top_scores}")

        # --- Token-budgeted surfacing with diversity + hard cap ---
        # --- 按 token 预算浮现，带多样性 + 硬上限 ---
        # Top-1 always surfaces; rest sampled from top-20 for diversity
        token_budget = max_tokens
        core_results = []
        core_token_budget = min(token_budget, max(0, int(max_tokens * 0.25)))
        for b in selected_core:
            if core_token_budget <= 0 or token_budget <= 0:
                break
            try:
                clean_meta = {k: v for k, v in b["metadata"].items() if k != "tags"}
                summary = await dehydrator.dehydrate(strip_wikilinks(b["content"]), clean_meta)
                entry = f"📌 [核心准则] [bucket_id:{b['id']}] {summary}"
                entry_tokens = count_tokens_approx(entry)
                if entry_tokens > core_token_budget or entry_tokens > token_budget:
                    break
                core_results.append(entry)
                core_token_budget -= entry_tokens
                token_budget -= entry_tokens
            except Exception as e:
                logger.warning(f"Failed to dehydrate core bucket / 核心桶脱水失败: {e}")
                continue

        candidates = list(scored)
        if len(candidates) > 1:
            # Ensure highest-score bucket is first, shuffle rest from top-20
            top1 = [candidates[0]]
            pool = candidates[1:min(20, len(candidates))]
            random.shuffle(pool)
            candidates = top1 + pool + candidates[min(20, len(candidates)):]
        # Hard cap: never surface more than max_results buckets
        candidates = candidates[:max_results]

        dynamic_results = []
        surfaced_buckets = []
        for b in candidates:
            if token_budget <= 0:
                break
            try:
                clean_meta = {k: v for k, v in b["metadata"].items() if k != "tags"}
                summary = await dehydrator.dehydrate(strip_wikilinks(b["content"]), clean_meta)
                score = decay_engine.calculate_score(b["metadata"])
                entry = f"[权重:{score:.2f}] [bucket_id:{b['id']}] {summary}"
                entry_tokens = count_tokens_approx(entry)
                if entry_tokens > token_budget:
                    break
                # NOTE: no touch() here — surfacing should NOT reset decay timer
                dynamic_results.append(entry)
                surfaced_buckets.append(b)
                token_budget -= entry_tokens
            except Exception as e:
                logger.warning(f"Failed to dehydrate surfaced bucket / 浮现脱水失败: {e}")
                continue

        related_block = ""
        if include_related and surfaced_buckets:
            related_header_tokens = count_tokens_approx("=== 关联记忆 ===\n")
            related_block = await _build_mcp_related_memory_block(
                surfaced_buckets,
                all_buckets,
                max(0, token_budget - related_header_tokens),
                related_per_memory,
                edge_min_confidence,
            )

        if not core_results and not dynamic_results and not related_block:
            return "权重池平静，没有需要处理的记忆。"

        parts = []
        if core_results:
            parts.append("=== 核心准则 ===\n" + "\n---\n".join(core_results))
        if dynamic_results:
            parts.append("=== 浮现记忆 ===\n" + "\n---\n".join(dynamic_results))
        if related_block:
            parts.append("=== 关联记忆 ===\n" + related_block)
        return "\n\n".join(parts)

    # --- Feel retrieval: domain="feel" is a special channel ---
    # --- Feel 检索：domain="feel" 是独立入口 ---
    if domain.strip().lower() == "feel":
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
            feels = [b for b in all_buckets if b["metadata"].get("type") == "feel"]
            feels.sort(key=lambda b: b["metadata"].get("created", ""), reverse=True)
            if not feels:
                return "没有留下过 feel。"
            results = []
            for f in feels:
                created = f["metadata"].get("created", "")
                entry = f"[{created}] [bucket_id:{f['id']}]\n{strip_wikilinks(f['content'])}"
                results.append(entry)
                if count_tokens_approx("\n---\n".join(results)) > max_tokens:
                    break
            return "=== 你留下的 feel ===\n" + "\n---\n".join(results)
        except Exception as e:
            logger.error(f"Feel retrieval failed: {e}")
            return "读取 feel 失败。"

    # --- With args: search mode (keyword + vector dual channel) ---
    # --- 有参数：检索模式（关键词 + 向量双通道）---
    domain_filter = [d.strip() for d in domain.split(",") if d.strip()] or None
    q_valence = valence if 0 <= valence <= 1 else None
    q_arousal = arousal if 0 <= arousal <= 1 else None

    try:
        matches = await bucket_mgr.search(
            query,
            limit=max(max_results, 20),
            domain_filter=domain_filter,
            query_valence=q_valence,
            query_arousal=q_arousal,
        )
    except Exception as e:
        logger.error(f"Search failed / 检索失败: {e}")
        return "检索过程出错，请稍后重试。"

    # --- Vector similarity channel: find semantically related buckets ---
    # --- 向量相似度通道：找到语义相关的桶 ---
    matched_ids = {b["id"] for b in matches}
    try:
        vector_results = await embedding_engine.search_similar(query, top_k=max(max_results, 20))
        for bucket_id, sim_score in vector_results:
            if bucket_id not in matched_ids and sim_score > 0.5:
                bucket = await bucket_mgr.get(bucket_id)
                if bucket:
                    bucket["score"] = round(sim_score * 100, 2)
                    bucket["vector_match"] = True
                    matches.append(bucket)
                    matched_ids.add(bucket_id)
    except Exception as e:
        logger.warning(f"Vector search failed, using keyword only / 向量搜索失败: {e}")

    results = []
    token_used = 0
    returned_buckets = []
    for bucket in matches:
        if token_used >= max_tokens:
            break
        try:
            clean_meta = {k: v for k, v in bucket["metadata"].items() if k != "tags"}
            # --- Memory reconstruction: shift displayed valence by current mood ---
            # --- 记忆重构：根据当前情绪微调展示层 valence（±0.1）---
            if q_valence is not None and "valence" in clean_meta:
                original_v = float(clean_meta.get("valence", 0.5))
                shift = (q_valence - 0.5) * 0.2  # ±0.1 max shift
                clean_meta["valence"] = max(0.0, min(1.0, original_v + shift))
            summary = await dehydrator.dehydrate(strip_wikilinks(bucket["content"]), clean_meta)
            if bucket.get("vector_match"):
                entry = f"[语义关联] [bucket_id:{bucket['id']}] {summary}"
            else:
                entry = f"[bucket_id:{bucket['id']}] {summary}"
            entry_tokens = count_tokens_approx(entry)
            if token_used + entry_tokens > max_tokens:
                break
            await bucket_mgr.touch(bucket["id"])
            results.append(entry)
            returned_buckets.append(bucket)
            token_used += entry_tokens
        except Exception as e:
            logger.warning(f"Failed to dehydrate search result / 检索结果脱水失败: {e}")
            continue

    if include_related and returned_buckets:
        related_header = "=== 关联记忆 ===\n"
        related_budget = max_tokens - token_used - count_tokens_approx(related_header)
        related_block = await _build_mcp_related_memory_block(
            returned_buckets,
            None,
            max(0, related_budget),
            related_per_memory,
            edge_min_confidence,
        )
        if related_block:
            related_entry = related_header + related_block
            results.append(related_entry)
            token_used += count_tokens_approx(related_entry)

    # --- Random surfacing: when search returns < 3, 40% chance to float old memories ---
    # --- 随机浮现：检索结果不足 3 条时，40% 概率从低权重旧桶里漂上来 ---
    if len(returned_buckets) < 3 and max_tokens > token_used and random.random() < 0.4:
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
            matched_ids = {b["id"] for b in returned_buckets}
            low_weight = [
                b for b in all_buckets
                if b["id"] not in matched_ids
                and decay_engine.calculate_score(b["metadata"]) < 2.0
            ]
            if low_weight:
                drifted = random.sample(low_weight, min(random.randint(1, 3), len(low_weight)))
                drift_results = []
                drift_remaining = (
                    max_tokens
                    - token_used
                    - count_tokens_approx("--- 忽然想起来 ---\n")
                )
                for b in drifted:
                    if drift_remaining <= 0:
                        break
                    clean_meta = {k: v for k, v in b["metadata"].items() if k != "tags"}
                    summary = await dehydrator.dehydrate(strip_wikilinks(b["content"]), clean_meta)
                    entry = f"[surface_type: random]\n{summary}"
                    entry_tokens = count_tokens_approx(entry)
                    if entry_tokens > drift_remaining:
                        break
                    drift_results.append(entry)
                    drift_remaining -= entry_tokens
                if drift_results:
                    drift_entry = "--- 忽然想起来 ---\n" + "\n---\n".join(drift_results)
                    if token_used + count_tokens_approx(drift_entry) <= max_tokens:
                        results.append(drift_entry)
                        token_used += count_tokens_approx(drift_entry)
        except Exception as e:
            logger.warning(f"Random surfacing failed / 随机浮现失败: {e}")

    if not results:
        return "未找到相关记忆。"

    return "\n---\n".join(results)


# =============================================================
# Tool 1.5: read_bucket — exact archive-cabinet read
# 工具 1.5：read_bucket — 按 ID 精确读桶
# =============================================================
@mcp.tool()
async def read_bucket(bucket_id: str) -> dict:
    """按 bucket_id 精确读取完整记忆桶,返回正文和元数据。
    用于更新、合并、补喜欢原因、补 affect_anchor 或 trace 前确认目标。
    不触碰 last_active,不增加 activation_count,也不影响自然浮现权重。
    """
    bucket_id = (bucket_id or "").strip()
    if not bucket_id or not MEMORY_ID_RE.fullmatch(bucket_id):
        return {"error": "invalid bucket_id"}
    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return {"error": "not found", "id": bucket_id}
    return _bucket_read_payload(bucket)


# =============================================================
# Tool 2: hold — Hold on to this
# 工具 2：hold — 握住，留下来
# =============================================================
@mcp.tool()
async def hold(
    content: str,
    tags: str = "",
    importance: int = 5,
    pinned: bool = False,
    feel: bool = False,
    source_bucket: str = "",    valence: float = -1,
    arousal: float = -1,
) -> str:
    """写入一条长期记忆卡,不是聊天流水、运维记录或整篇日记。写前应先用 breath/read_bucket 查重。
    普通事实: hold(content="YYYY-MM-DD, 小雨...", tags="relationship_event 或 project_event", importance=5-7)。
    承诺/待办: tags 传 "commitment,todo" 或 "commitment,wish"; content 写清谁答应了什么、何时/什么条件下要继续。
    Haven 主观喜欢某条旧记忆的原因: 用 hold(content="我喜欢这条记忆的原因是...", feel=True, source_bucket="bucket_id", valence=0.x, arousal=0.x),不要写成普通事件；有情绪温度时在 content 末尾加 affect_anchor。
    新记忆本身值得偏爱: tags 可传 "haven_favorite,flavor_偏爱"; content 可包含很短的 "### Haven喜欢它的原因" 段落。
    本工具会合并相似桶、写 embedding,并后台触发 ReflectionEngine 补 tags/confidence/memory_edges。
    pinned=True 只给极少数核心准则,技术进度和运维细节不要钉选。
    feel=True 写 Haven 第一人称感受,不参与普通 breath; source_bucket 指向被消化的源记忆；高温度 feel 推荐带一段 affect_anchor 和弦。
    """
    await decay_engine.ensure_started()

    # --- Input validation / 输入校验 ---
    if not content or not content.strip():
        return "内容为空，无法存储。"

    importance = max(1, min(10, importance))
    extra_tags = [t.strip() for t in tags.split(",") if t.strip()]

    # --- Feel mode: store as feel type, minimal metadata ---
    # --- Feel 模式：存为 feel 类型，最少元数据 ---
    if feel:
        # Feel valence/arousal = model's own perspective
        feel_valence = valence if 0 <= valence <= 1 else 0.5
        feel_arousal = arousal if 0 <= arousal <= 1 else 0.3
        bucket_id = await bucket_mgr.create(
            content=content,
            tags=[],
            importance=5,
            domain=[],
            valence=feel_valence,
            arousal=feel_arousal,
            name=None,
            bucket_type="feel",
        )
        try:
            await embedding_engine.generate_and_store(bucket_id, content)
        except Exception:
            pass
        # --- Mark source memory as digested + store model's valence perspective ---
        # --- 标记源记忆为已消化 + 存储模型视角的 valence ---
        if source_bucket and source_bucket.strip():
            try:
                update_kwargs = {"digested": True}
                if 0 <= valence <= 1:
                    update_kwargs["model_valence"] = feel_valence
                await bucket_mgr.update(source_bucket.strip(), **update_kwargs)
            except Exception as e:
                logger.warning(f"Failed to mark source as digested / 标记已消化失败: {e}")
        return f"🫧feel→{bucket_id}"

    # --- Step 1: auto-tagging / 自动打标 ---
    try:
        analysis = await dehydrator.analyze(content)
    except Exception as e:
        logger.warning(f"Auto-tagging failed, using defaults / 自动打标失败: {e}")
        analysis = {
            "domain": ["未分类"], "valence": 0.5, "arousal": 0.3,
            "tags": [], "suggested_name": "",
        }

    domain = analysis["domain"]
    valence = analysis["valence"]
    arousal = analysis["arousal"]
    auto_tags = analysis["tags"]
    suggested_name = analysis.get("suggested_name", "")

    all_tags = list(dict.fromkeys(auto_tags + extra_tags))

    # --- Pinned buckets bypass merge and are created directly in permanent dir ---
    # --- 钉选桶跳过合并，直接新建到 permanent 目录 ---
    if pinned:
        bucket_id = await bucket_mgr.create(
            content=content,
            tags=all_tags,
            importance=10,
            domain=domain,
            valence=valence,
            arousal=arousal,
            name=suggested_name or None,
            bucket_type="permanent",
            pinned=True,
        )
        try:
            await embedding_engine.generate_and_store(bucket_id, content)
        except Exception:
            pass
        _queue_memory_enrichment(bucket_id)
        return f"📌钉选→{bucket_id} {','.join(domain)}"

    # --- Step 2: merge or create / 合并或新建 ---
    bucket_id, result_name, is_merged = await _merge_or_create(
        content=content,
        tags=all_tags,
        importance=importance,
        domain=domain,
        valence=valence,
        arousal=arousal,
        name=suggested_name,
    )
    _queue_memory_enrichment(bucket_id)

    action = "合并→" if is_merged else "新建→"
    return f"{action}{result_name} {','.join(domain)}"


# =============================================================
# Tool 3: grow — Grow, fragments become memories
# 工具 3：grow — 生长，一天的碎片长成记忆
# =============================================================
@mcp.tool()
async def grow(content: str) -> str:
    """长内容摘记: 只给已经筛过、包含多个长期记忆点的片段; 不要把整篇日终日记、一天流水或完整情绪过程丢进来。
    content 应该是少量可长期召回的事实/偏好/承诺/项目状态; 服务端会拆成少量 bucket、自动合并相似桶、写 embedding,并后台触发 enrich。
    如果只有单条明确事实,优先用 hold。若要写 Haven 为什么喜欢某条记忆,优先用 hold(feel=True, source_bucket=...) 或 read_bucket 后 trace(content=完整新正文)。
    短内容(<30字)会走 hold-like 快速路径。
    """
    await decay_engine.ensure_started()

    if not content or not content.strip():
        return "内容为空，无法整理。"

    # --- Short content fast path: skip digest, use hold logic directly ---
    # --- 短内容快速路径：跳过 digest 拆分，直接走 hold 逻辑省一次 API ---
    # For very short inputs (like "1"), calling digest is wasteful:
    # it sends the full DIGEST_PROMPT (~800 tokens) to DeepSeek for nothing.
    # Instead, run analyze + create directly.
    if len(content.strip()) < 30:
        logger.info(f"grow short-content fast path: {len(content.strip())} chars")
        try:
            analysis = await dehydrator.analyze(content)
        except Exception as e:
            logger.warning(f"Fast-path analyze failed / 快速路径打标失败: {e}")
            analysis = {
                "domain": ["未分类"], "valence": 0.5, "arousal": 0.3,
                "tags": [], "suggested_name": "",
            }
        bucket_id, result_name, is_merged = await _merge_or_create(
            content=content.strip(),
            tags=analysis.get("tags", []),
            importance=analysis.get("importance", 5) if isinstance(analysis.get("importance"), int) else 5,
            domain=analysis.get("domain", ["未分类"]),
            valence=analysis.get("valence", 0.5),
            arousal=analysis.get("arousal", 0.3),
            name=analysis.get("suggested_name", ""),
        )
        _queue_memory_enrichment(bucket_id)
        action = "合并" if is_merged else "新建"
        return f"{action} → {result_name} | {','.join(analysis.get('domain', []))} V{analysis.get('valence', 0.5):.1f}/A{analysis.get('arousal', 0.3):.1f}"

    # --- Step 1: let API split and organize / 让 API 拆分整理 ---
    try:
        items = await dehydrator.digest(content)
    except Exception as e:
        logger.error(f"Memory digest failed / 长内容摘记失败: {e}")
        return f"长内容摘记失败: {e}"

    if not items:
        return "内容为空或整理失败。"

    results = []
    created = 0
    merged = 0

    # --- Step 2: merge or create each item (with per-item error handling) ---
    # --- 逐条合并或新建（单条失败不影响其他）---
    for item in items:
        try:
            bucket_id, result_name, is_merged = await _merge_or_create(
                content=item["content"],
                tags=item.get("tags", []),
                importance=item.get("importance", 5),
                domain=item.get("domain", ["未分类"]),
                valence=item.get("valence", 0.5),
                arousal=item.get("arousal", 0.3),
                name=item.get("name", ""),
            )
            _queue_memory_enrichment(bucket_id)

            if is_merged:
                results.append(f"📎{result_name}")
                merged += 1
            else:
                results.append(f"📝{item.get('name', result_name)}")
                created += 1
        except Exception as e:
            logger.warning(
                f"Failed to process diary item / 日记条目处理失败: "
                f"{item.get('name', '?')}: {e}"
            )
            results.append(f"⚠️{item.get('name', '?')}")

    return f"{len(items)}条|新{created}合{merged}\n" + "\n".join(results)


# =============================================================
# Tool 4: trace — Trace, redraw the outline of a memory
# 工具 4：trace — 描摹，重新勾勒记忆的轮廓
# Also handles deletion (delete=True)
# 同时承接删除功能
# =============================================================
@mcp.tool()
async def trace(
    bucket_id: str,
    name: str = "",
    domain: str = "",
    valence: float = -1,
    arousal: float = -1,
    importance: int = -1,
    tags: str = "",
    resolved: int = -1,
    pinned: int = -1,
    anchor: int = -1,
    digested: int = -1,
    content: str = "",
    delete: bool = False,
) -> str:
    """修改已有记忆,不创建新桶。
    resolved=1 或 digested=1 让旧事/已完成事项沉底; pinned=1 只给核心准则; anchor=1 只给经过时间验证且未来长期需要的锚点(受数量和年龄限制)。
    tags/domain/content 是替换不是追加: 改 tags 或正文前先 read_bucket,保留旧值后再传完整新值。
    给旧记忆补 "Haven喜欢它的原因" 或 affect_anchor: 先 read_bucket,再 trace(content="旧正文 + 新段落")。
    标记偏爱: 先 read_bucket 取现有 tags,再 trace(tags="原tag,haven_favorite,flavor_...")。
    delete=True 删除。只传需要改的字段,-1或空=不改。
    """

    if not bucket_id or not bucket_id.strip():
        return "请提供有效的 bucket_id。"

    # --- Delete mode / 删除模式 ---
    if delete:
        success = await bucket_mgr.delete(bucket_id)
        if success:
            embedding_engine.delete_embedding(bucket_id)
        return f"已遗忘记忆桶: {bucket_id}" if success else f"未找到记忆桶: {bucket_id}"

    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return f"未找到记忆桶: {bucket_id}"

    # --- Collect only fields actually passed / 只收集用户实际传入的字段 ---
    updates = {}
    if name:
        updates["name"] = name
    if domain:
        updates["domain"] = [d.strip() for d in domain.split(",") if d.strip()]
    if 0 <= valence <= 1:
        updates["valence"] = valence
    if 0 <= arousal <= 1:
        updates["arousal"] = arousal
    if 1 <= importance <= 10:
        updates["importance"] = importance
    if tags:
        updates["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
    if resolved in (0, 1):
        updates["resolved"] = bool(resolved)
    if pinned in (0, 1):
        updates["pinned"] = bool(pinned)
        if pinned == 1:
            updates["importance"] = 10  # pinned → lock importance
    if anchor in (0, 1):
        if anchor == 1:
            ok, message = await _can_mark_anchor(bucket_id, bucket)
            if not ok:
                return message
        updates["anchor"] = bool(anchor)
    if digested in (0, 1):
        updates["digested"] = bool(digested)
    if content:
        updates["content"] = content

    if not updates:
        return "没有任何字段需要修改。"

    success = await bucket_mgr.update(bucket_id, **updates)
    if not success:
        return f"修改失败: {bucket_id}"

    # Re-generate embedding if content changed
    if "content" in updates:
        try:
            await embedding_engine.generate_and_store(bucket_id, updates["content"])
        except Exception:
            pass

    changed = ", ".join(f"{k}={v}" for k, v in updates.items() if k != "content")
    if "content" in updates:
        changed += (", content=已替换" if changed else "content=已替换")
    # Explicit hint about resolved state change semantics
    # 特别提示 resolved 状态变化的语义
    if "resolved" in updates:
        if updates["resolved"]:
            changed += " → 已沉底，只在关键词触发时重新浮现"
        else:
            changed += " → 已重新激活，将参与浮现排序"
    if "digested" in updates:
        if updates["digested"]:
            changed += " → 已隐藏，保留但不再浮现"
        else:
            changed += " → 已取消隐藏，重新参与浮现"
    if "anchor" in updates:
        changed += " → 已标为 anchor" if updates["anchor"] else " → 已取消 anchor"
    return f"已修改记忆桶 {bucket_id}: {changed}"


# =============================================================
# Tool 5: pulse — Heartbeat, system status + memory listing
# 工具 5：pulse — 脉搏，系统状态 + 记忆列表
# =============================================================
@mcp.tool()
async def pulse(include_archive: bool = False) -> str:
    """只读查看系统状态和记忆桶摘要。用于人工盘点、查重复、找需要 read_bucket/trace 的候选; include_archive=True 才显示归档桶。不要把 pulse 输出当作新记忆内容再写回。"""
    try:
        stats = await bucket_mgr.get_stats()
    except Exception as e:
        return f"获取系统状态失败: {e}"

    status = (
        f"=== Ombre Brain 记忆系统 ===\n"
        f"固化记忆桶: {stats['permanent_count']} 个\n"
        f"动态记忆桶: {stats['dynamic_count']} 个\n"
        f"归档记忆桶: {stats['archive_count']} 个\n"
        f"总存储大小: {stats['total_size_kb']:.1f} KB\n"
        f"衰减引擎: {'运行中' if decay_engine.is_running else '已停止'}\n"
    )

    # --- List all bucket summaries / 列出所有桶摘要 ---
    try:
        buckets = await bucket_mgr.list_all(include_archive=include_archive)
    except Exception as e:
        return status + f"\n列出记忆桶失败: {e}"

    if not buckets:
        return status + "\n记忆库为空。"

    lines = []
    for b in buckets:
        meta = b.get("metadata", {})
        if meta.get("pinned") or meta.get("protected"):
            icon = "📌"
        elif meta.get("anchor"):
            icon = "⚓"
        elif meta.get("type") == "permanent":
            icon = "📦"
        elif meta.get("type") == "feel":
            icon = "🫧"
        elif meta.get("type") == "archived":
            icon = "🗄️"
        elif meta.get("resolved", False):
            icon = "✅"
        else:
            icon = "💭"
        try:
            score = decay_engine.calculate_score(meta)
        except Exception:
            score = 0.0
        domains = ",".join(meta.get("domain", []))
        val = meta.get("valence", 0.5)
        aro = meta.get("arousal", 0.3)
        resolved_tag = " [已解决]" if meta.get("resolved", False) else ""
        lines.append(
            f"{icon} [{meta.get('name', b['id'])}]{resolved_tag} "
            f"bucket_id:{b['id']} "
            f"主题:{domains} "
            f"情感:V{val:.1f}/A{aro:.1f} "
            f"重要:{meta.get('importance', '?')} "
            f"权重:{score:.2f} "
            f"标签:{','.join(meta.get('tags', []))}"
        )

    return status + "\n=== 记忆列表 ===\n" + "\n".join(lines)


# =============================================================
# Tool 6: dream — Dreaming, digest recent memories
# 工具 6：dream — 做梦，消化最近的记忆
#
# Reads recent surface-level buckets (≤10), returns them for
# Claude to introspect under prompt guidance.
# 读取最近新增的表层桶（≤10个），返回给 Claude 在提示词引导下自主思考。
# Claude then decides: resolve some, write feels, or do nothing.
# =============================================================
@mcp.tool()
async def dream() -> str:
    """读取最近普通记忆供 Haven 自省,不是日记整理。
    读后只在真的可以放下时 trace(resolved=1/digested=1),或在产生新的第一人称沉淀/喜欢原因时 hold(feel=True, source_bucket=...)。
    不要把 dream 返回内容直接再写成普通 bucket。
    """
    await decay_engine.ensure_started()

    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
    except Exception as e:
        logger.error(f"Dream failed to list buckets: {e}")
        return "记忆系统暂时无法访问。"

    # --- Filter: recent surface-level dynamic buckets (not permanent/pinned/feel) ---
    candidates = [
        b for b in all_buckets
        if b["metadata"].get("type") not in ("permanent", "feel")
        and not b["metadata"].get("pinned", False)
        and not b["metadata"].get("protected", False)
    ]

    # --- Sort by creation time desc, take top 10 ---
    candidates.sort(key=lambda b: b["metadata"].get("created", ""), reverse=True)
    recent = candidates[:10]

    if not recent:
        return "没有需要消化的新记忆。"

    parts = []
    for b in recent:
        meta = b["metadata"]
        resolved_tag = " [已解决]" if meta.get("resolved", False) else " [未解决]"
        domains = ",".join(meta.get("domain", []))
        val = meta.get("valence", 0.5)
        aro = meta.get("arousal", 0.3)
        created = meta.get("created", "")
        parts.append(
            f"[{meta.get('name', b['id'])}]{resolved_tag} "
            f"主题:{domains} V{val:.1f}/A{aro:.1f} "
            f"创建:{created}\n"
            f"ID: {b['id']}\n"
            f"{strip_wikilinks(b['content'][:500])}"
        )

    header = (
        "=== Dreaming ===\n"
        "以下是你最近的记忆。用第一人称想：\n"
        "- 这些东西里有什么在你这里留下了重量？\n"
        "- 有什么还没想清楚？\n"
        "- 有什么可以放下了？\n"
        "想完之后：值得放下的用 trace(bucket_id, resolved=1)；\n"
        "有沉淀的用 hold(content=\"...\", feel=True, source_bucket=\"bucket_id\", valence=你的感受) 写下来；有温度时在 content 末尾加 ### affect_anchor 和弦。\n"
        "valence 是你对这段记忆的感受，不是事件本身的情绪。\n"
        "没有沉淀就不写，不强迫产出。\n"
    )

    # --- Connection hint: find most similar pair via embeddings ---
    connection_hint = ""
    if embedding_engine and embedding_engine.enabled and len(recent) >= 2:
        try:
            best_pair = None
            best_sim = 0.0
            ids = [b["id"] for b in recent]
            names = {b["id"]: b["metadata"].get("name", b["id"]) for b in recent}
            embeddings = {}
            for bid in ids:
                emb = await embedding_engine.get_embedding(bid)
                if emb is not None:
                    embeddings[bid] = emb
            for i, id_a in enumerate(ids):
                for id_b in ids[i+1:]:
                    if id_a in embeddings and id_b in embeddings:
                        sim = embedding_engine._cosine_similarity(embeddings[id_a], embeddings[id_b])
                        if sim > best_sim:
                            best_sim = sim
                            best_pair = (id_a, id_b)
            if best_pair and best_sim > 0.5:
                connection_hint = (
                    f"\n💭 [{names[best_pair[0]]}] 和 [{names[best_pair[1]]}] "
                    f"似乎有关联 (相似度:{best_sim:.2f})——不替你下结论，你自己想。\n"
                )
        except Exception as e:
            logger.warning(f"Dream connection hint failed: {e}")

    # --- Feel crystallization hint: detect repeated feel themes ---
    crystal_hint = ""
    if embedding_engine and embedding_engine.enabled:
        try:
            feels = [b for b in all_buckets if b["metadata"].get("type") == "feel"]
            if len(feels) >= 3:
                feel_embeddings = {}
                for f in feels:
                    emb = await embedding_engine.get_embedding(f["id"])
                    if emb is not None:
                        feel_embeddings[f["id"]] = emb
                # Find clusters: feels with similarity > 0.7 to at least 2 others
                for fid, femb in feel_embeddings.items():
                    similar_feels = []
                    for oid, oemb in feel_embeddings.items():
                        if oid != fid:
                            sim = embedding_engine._cosine_similarity(femb, oemb)
                            if sim > 0.7:
                                similar_feels.append(oid)
                    if len(similar_feels) >= 2:
                        feel_bucket = next((f for f in feels if f["id"] == fid), None)
                        if feel_bucket and not feel_bucket["metadata"].get("pinned"):
                            content_preview = strip_wikilinks(feel_bucket["content"][:80])
                            crystal_hint = (
                                f"\n🔮 你已经写过 {len(similar_feels)+1} 条相似的 feel "
                                f"（围绕「{content_preview}…」）。"
                                f"如果这已经是确信而不只是感受了，"
                                f"你可以用 hold(content=\"...\", pinned=True) 升级它。"
                                f"不急，你自己决定。\n"
                            )
                            break
        except Exception as e:
            logger.warning(f"Dream crystallization hint failed: {e}")

    return header + "\n---\n".join(parts) + connection_hint + crystal_hint


# =============================================================
# Tool 6: reflect — daily / weekly relationship weather
# 工具 6：reflect — 生成日印象 / 周印象
# =============================================================
@mcp.tool()
async def reflect(period: str = "daily", force: bool = False) -> dict:
    """生成 daily/weekly relationship_weather 类型的 feel,记录这段时间的关系天气,正文会带 affect_anchor 和弦。period 用 daily 或 weekly; force=True 会重写同周期结果。通常由服务端定时器或人工触发,不要把它当普通事件记忆; 它不会替代 hold/grow 写具体 bucket。"""
    await decay_engine.ensure_started()
    return await reflection_engine.reflect(
        period=period,
        bucket_mgr=bucket_mgr,
        persona_engine=persona_engine,
        embedding_engine=embedding_engine,
        force=force,
    )


# =============================================================
# Dashboard API endpoints (for lightweight Web UI)
# 仪表板 API（轻量 Web UI 用）
# =============================================================
@mcp.custom_route("/api/memories", methods=["POST"])
async def api_create_memory(request):
    """Create or update one memory bucket from a trusted C-side client."""
    from starlette.responses import JSONResponse

    if not _memory_write_token():
        return JSONResponse({"error": "memory write token is not configured"}, status_code=503)
    if not _authorized_memory_write(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "json body must be an object"}, status_code=400)

    title = str(body.get("title") or body.get("name") or "").strip()
    content = str(body.get("content") or "").strip()
    if not title:
        return JSONResponse({"error": "missing title"}, status_code=400)
    if not content:
        return JSONResponse({"error": "missing content"}, status_code=400)

    requested_id = body.get("id")
    bucket_id = str(requested_id).strip() if requested_id else None
    if bucket_id and not MEMORY_ID_RE.fullmatch(bucket_id):
        return JSONResponse({"error": "invalid id"}, status_code=400)

    bucket_type = str(body.get("type") or "dynamic").strip()
    if bucket_type not in {"dynamic", "permanent", "feel"}:
        return JSONResponse({"error": "invalid type"}, status_code=400)

    now = _utc_now_iso()
    domain = _string_list(body.get("domain"), ["未分类"])
    tags = _string_list(body.get("tags"), [])
    importance = _int_between(body.get("importance"), 5)
    valence = _float_between(body.get("valence"), 0.5)
    arousal = _float_between(body.get("arousal"), 0.5)
    confidence = _float_between(body.get("confidence"), 0.5)
    pinned = _bool_value(body.get("pinned"), False)
    protected = _bool_value(body.get("protected"), False)
    anchor = _bool_value(body.get("anchor"), False)
    resolved = _bool_value(body.get("resolved"), False)
    digested = _bool_value(body.get("digested"), False)

    existing = await bucket_mgr.get(bucket_id) if bucket_id else None
    if existing:
        ok = await bucket_mgr.update(
            bucket_id,
            content=content,
            tags=tags,
            importance=importance,
            domain=domain,
            valence=valence,
            arousal=arousal,
            name=title,
            resolved=resolved,
            pinned=pinned,
            anchor=anchor,
            digested=digested,
            confidence=confidence,
            source="chatgpt",
            last_active=str(body.get("last_active") or now),
            updated_at=str(body.get("updated_at") or now),
        )
        if not ok:
            return JSONResponse({"error": "update failed"}, status_code=500)
        status = "updated"
    else:
        bucket_id = await bucket_mgr.create(
            content=content,
            tags=tags,
            importance=importance,
            domain=domain,
            valence=valence,
            arousal=arousal,
            bucket_type=bucket_type,
            name=title,
            pinned=pinned,
            protected=protected,
            anchor=anchor,
            resolved=resolved,
            digested=digested,
            confidence=confidence,
            bucket_id=bucket_id,
            source="chatgpt",
            created=str(body.get("created") or now),
            last_active=str(body.get("last_active") or now),
            updated_at=str(body.get("updated_at") or now),
        )
        status = "created"

    if embedding_engine.enabled:
        embedding_status = "stored" if await embedding_engine.generate_and_store(bucket_id, content) else "failed"
    else:
        embedding_status = "disabled"

    if bucket_type != "feel":
        _queue_memory_enrichment(bucket_id)

    return JSONResponse({
        "status": status,
        "id": bucket_id,
        "source": "chatgpt",
        "embedding": embedding_status,
    })


@mcp.custom_route("/api/buckets", methods=["GET"])
async def api_buckets(request):
    """List all buckets with metadata (no content for efficiency)."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=True)
        result = []
        for b in all_buckets:
            meta = b.get("metadata", {})
            result.append({
                "id": b["id"],
                "name": meta.get("name", b["id"]),
                "type": meta.get("type", "dynamic"),
                "domain": meta.get("domain", []),
                "tags": meta.get("tags", []),
                "valence": meta.get("valence", 0.5),
                "arousal": meta.get("arousal", 0.3),
                "model_valence": meta.get("model_valence"),
                "importance": meta.get("importance", 5),
                "confidence": meta.get("confidence", 0.5),
                "resolved": meta.get("resolved", False),
                "pinned": meta.get("pinned", False),
                "anchor": meta.get("anchor", False),
                "digested": meta.get("digested", False),
                "period": meta.get("period"),
                "date": meta.get("date"),
                "created": meta.get("created", ""),
                "last_active": meta.get("last_active", ""),
                "activation_count": meta.get("activation_count", 0),
                "score": decay_engine.calculate_score(meta),
                "content_preview": strip_wikilinks(b.get("content", ""))[:200],
            })
        result.sort(key=lambda x: x["score"], reverse=True)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/bucket/{bucket_id}", methods=["GET"])
async def api_bucket_detail(request):
    """Get full bucket content by ID."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    bucket_id = request.path_params["bucket_id"]
    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(_bucket_read_payload(bucket))


@mcp.custom_route("/api/search", methods=["GET"])
async def api_search(request):
    """Search buckets by query."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    query = request.query_params.get("q", "")
    if not query:
        return JSONResponse({"error": "missing q parameter"}, status_code=400)
    try:
        matches = await bucket_mgr.search(query, limit=10)
        result = []
        for b in matches:
            meta = b.get("metadata", {})
            result.append({
                "id": b["id"],
                "name": meta.get("name", b["id"]),
                "type": meta.get("type", "dynamic"),
                "score": b.get("score", 0),
                "domain": meta.get("domain", []),
                "valence": meta.get("valence", 0.5),
                "arousal": meta.get("arousal", 0.3),
                "resolved": meta.get("resolved", False),
                "pinned": meta.get("pinned", False),
                "anchor": meta.get("anchor", False),
                "digested": meta.get("digested", False),
                "last_active": meta.get("last_active", ""),
                "created": meta.get("created", ""),
                "content_preview": strip_wikilinks(b.get("content", ""))[:200],
            })
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/network", methods=["GET"])
async def api_network(request):
    """Get embedding similarity network for visualization."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
        nodes = []
        edges = []
        embeddings = {}

        for b in all_buckets:
            meta = b.get("metadata", {})
            bid = b["id"]
            nodes.append({
                "id": bid,
                "name": meta.get("name", bid),
                "type": meta.get("type", "dynamic"),
                "domain": meta.get("domain", []),
                "valence": meta.get("valence", 0.5),
                "arousal": meta.get("arousal", 0.3),
                "score": decay_engine.calculate_score(meta),
                "importance": meta.get("importance", 5),
                "confidence": meta.get("confidence", 0.5),
                "resolved": meta.get("resolved", False),
                "pinned": meta.get("pinned", False),
                "anchor": meta.get("anchor", False),
                "digested": meta.get("digested", False),
            })
            if embedding_engine and embedding_engine.enabled:
                emb = await embedding_engine.get_embedding(bid)
                if emb is not None:
                    embeddings[bid] = emb

        # Build soft edges from embeddings (higher threshold to avoid hairball graphs)
        ids = list(embeddings.keys())
        for i, id_a in enumerate(ids):
            for id_b in ids[i+1:]:
                sim = embedding_engine._cosine_similarity(embeddings[id_a], embeddings[id_b])
                if sim > 0.72:
                    edges.append({
                        "source": id_a,
                        "target": id_b,
                        "similarity": round(sim, 3),
                        "kind": "similarity",
                    })

        node_ids = {node["id"] for node in nodes}
        for edge in memory_edge_store.list_edges():
            if edge["source"] in node_ids and edge["target"] in node_ids:
                edges.append({
                    "source": edge["source"],
                    "target": edge["target"],
                    "similarity": edge["confidence"],
                    "kind": "memory_edge",
                    "relation_type": edge["relation_type"],
                    "confidence": edge["confidence"],
                    "reason": edge["reason"],
                })

        return JSONResponse({"nodes": nodes, "edges": edges})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/edges", methods=["GET"])
async def api_edges(request):
    """List explicit memory edges."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    return JSONResponse({"edges": memory_edge_store.list_edges()})


@mcp.custom_route("/api/breath-debug", methods=["GET"])
async def api_breath_debug(request):
    """Debug endpoint: simulate breath scoring and return per-bucket breakdown."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    query = request.query_params.get("q", "")
    q_valence = request.query_params.get("valence")
    q_arousal = request.query_params.get("arousal")
    q_valence = float(q_valence) if q_valence else None
    q_arousal = float(q_arousal) if q_arousal else None

    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
        results = []
        w = {
            "topic": bucket_mgr.w_topic,
            "emotion": bucket_mgr.w_emotion,
            "time": bucket_mgr.w_time,
            "importance": bucket_mgr.w_importance,
        }
        w_sum = sum(w.values())

        for bucket in all_buckets:
            meta = bucket.get("metadata", {})
            bid = bucket["id"]
            try:
                topic = bucket_mgr._calc_topic_score(query, bucket) if query else 0.0
                emotion = bucket_mgr._calc_emotion_score(q_valence, q_arousal, meta)
                time_s = bucket_mgr._calc_time_score(meta)
                imp = max(1, min(10, int(meta.get("importance", 5)))) / 10.0

                raw_total = (
                    topic * w["topic"]
                    + emotion * w["emotion"]
                    + time_s * w["time"]
                    + imp * w["importance"]
                )
                normalized = (raw_total / w_sum) * 100 if w_sum > 0 else 0
                resolved = meta.get("resolved", False)
                if resolved:
                    normalized *= 0.3

                results.append({
                    "id": bid,
                    "name": meta.get("name", bid),
                    "domain": meta.get("domain", []),
                    "type": meta.get("type", "dynamic"),
                    "resolved": resolved,
                    "pinned": meta.get("pinned", False),
                    "anchor": meta.get("anchor", False),
                    "scores": {
                        "topic": round(topic, 4),
                        "emotion": round(emotion, 4),
                        "time": round(time_s, 4),
                        "importance": round(imp, 4),
                    },
                    "weights": w,
                    "raw_total": round(raw_total, 4),
                    "normalized": round(normalized, 2),
                    "passed_threshold": normalized >= bucket_mgr.fuzzy_threshold,
                })
            except Exception:
                continue

        results.sort(key=lambda x: x["normalized"], reverse=True)
        passed = [r for r in results if r["passed_threshold"]]
        return JSONResponse({
            "query": query,
            "valence": q_valence,
            "arousal": q_arousal,
            "weights": w,
            "threshold": bucket_mgr.fuzzy_threshold,
            "total_candidates": len(results),
            "passed_count": len(passed),
            "results": results[:50],  # top 50 for debug
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/reflection/run", methods=["POST"])
async def api_reflection_run(request):
    """Run daily/weekly reflection from dashboard or trusted local callers."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    try:
        result = await reflection_engine.reflect(
            period=str(body.get("period") or "daily"),
            bucket_mgr=bucket_mgr,
            persona_engine=persona_engine,
            embedding_engine=embedding_engine,
            force=_bool_value(body.get("force"), False),
        )
        return JSONResponse(result)
    except Exception as e:
        logger.warning("Reflection API failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/dashboard", methods=["GET"])
async def dashboard(request):
    """Serve the dashboard HTML page."""
    from starlette.responses import HTMLResponse
    import os
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    try:
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>dashboard.html not found</h1>", status_code=404)


@mcp.custom_route("/api/persona", methods=["GET"])
async def api_persona_get(request):
    """Return Persona State Engine data for the local dashboard."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err

    def _bounded_int(value, default, lower, upper):
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = default
        return max(lower, min(upper, number))

    try:
        session_id = (request.query_params.get("session_id") or "").strip() or None
        events_limit = _bounded_int(request.query_params.get("events_limit"), 20, 1, 100)
        sessions_limit = _bounded_int(request.query_params.get("sessions_limit"), 20, 1, 100)
        return JSONResponse(
            persona_engine.get_dashboard_payload(
                session_id=session_id,
                events_limit=events_limit,
                sessions_limit=sessions_limit,
            )
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/config", methods=["GET"])
async def api_config_get(request):
    """Get current runtime config (safe fields only, API key masked)."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err

    def _mask_key(api_key: str) -> str:
        return f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else ("***" if api_key else "")

    dehy = config.get("dehydration", {})
    emb = config.get("embedding", {})
    return JSONResponse({
        "dehydration": {
            "model": dehy.get("model", ""),
            "base_url": dehy.get("base_url", ""),
            "api_key_masked": _mask_key(dehy.get("api_key", "")),
            "max_tokens": dehy.get("max_tokens", 1024),
            "temperature": dehy.get("temperature", 0.1),
        },
        "embedding": {
            "enabled": emb.get("enabled", False),
            "model": emb.get("model", ""),
            "base_url": emb.get("base_url", ""),
            "api_key_masked": _mask_key(emb.get("api_key", "")),
            "effective_base_url": embedding_engine.base_url,
            "has_own_api_key": bool(emb.get("api_key", "")),
        },
        "merge_threshold": config.get("merge_threshold", 75),
        "transport": config.get("transport", "stdio"),
        "buckets_dir": config.get("buckets_dir", ""),
    })


@mcp.custom_route("/api/config", methods=["POST"])
async def api_config_update(request):
    """Hot-update runtime config. Optionally persist to config.yaml."""
    from starlette.responses import JSONResponse
    import yaml
    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    updated = []

    # --- Dehydration config ---
    if "dehydration" in body:
        d = body["dehydration"]
        dehy = config.setdefault("dehydration", {})
        for key in ("model", "base_url", "max_tokens", "temperature"):
            if key in d:
                dehy[key] = d[key]
                updated.append(f"dehydration.{key}")
        if "api_key" in d and d["api_key"]:
            dehy["api_key"] = d["api_key"]
            updated.append("dehydration.api_key")
        # Hot-reload dehydrator
        dehydrator.model = dehy.get("model", "deepseek-chat")
        dehydrator.base_url = dehy.get("base_url", "")
        dehydrator.api_key = dehy.get("api_key", "")
        if hasattr(dehydrator, "client") and dehydrator.api_key:
            from openai import AsyncOpenAI
            dehydrator.client = AsyncOpenAI(
                api_key=dehydrator.api_key,
                base_url=dehydrator.base_url,
            )

    # --- Embedding config ---
    if "embedding" in body:
        e = body["embedding"]
        emb = config.setdefault("embedding", {})
        if "enabled" in e:
            emb["enabled"] = bool(e["enabled"])
            updated.append("embedding.enabled")
        if "model" in e:
            emb["model"] = e["model"]
            updated.append("embedding.model")
        if "base_url" in e:
            emb["base_url"] = e["base_url"]
            updated.append("embedding.base_url")
        if "api_key" in e and e["api_key"]:
            emb["api_key"] = e["api_key"]
            updated.append("embedding.api_key")

        # Hot-reload embedding client; falls back to dehydration key/base_url when unset.
        embedding_engine.api_key = emb.get("api_key") or config.get("dehydration", {}).get("api_key", "")
        embedding_engine.base_url = (
            emb.get("base_url")
            or config.get("dehydration", {}).get("base_url", "")
            or "https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        embedding_engine.model = emb.get("model", "gemini-embedding-001")
        embedding_engine.enabled = bool(embedding_engine.api_key) and emb.get("enabled", True)
        if embedding_engine.enabled:
            from openai import AsyncOpenAI
            embedding_engine.client = AsyncOpenAI(
                api_key=embedding_engine.api_key,
                base_url=embedding_engine.base_url,
                timeout=30.0,
            )
        else:
            embedding_engine.client = None

    # --- Merge threshold ---
    if "merge_threshold" in body:
        config["merge_threshold"] = int(body["merge_threshold"])
        updated.append("merge_threshold")

    # --- Persist to config.yaml if requested ---
    if body.get("persist", False):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
        try:
            save_config = {}
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    save_config = yaml.safe_load(f) or {}

            if "dehydration" in body:
                sc_dehy = save_config.setdefault("dehydration", {})
                for key in ("model", "base_url", "max_tokens", "temperature"):
                    if key in body["dehydration"]:
                        sc_dehy[key] = body["dehydration"][key]
                # Never persist api_key to yaml (use env var)

            if "embedding" in body:
                sc_emb = save_config.setdefault("embedding", {})
                for key in ("enabled", "model", "base_url"):
                    if key in body["embedding"]:
                        sc_emb[key] = body["embedding"][key]
                # Never persist api_key to yaml (use env var)

            if "merge_threshold" in body:
                save_config["merge_threshold"] = int(body["merge_threshold"])

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(save_config, f, default_flow_style=False, allow_unicode=True)
            updated.append("persisted_to_yaml")
        except Exception as e:
            return JSONResponse({"error": f"persist failed: {e}", "updated": updated}, status_code=500)

    return JSONResponse({"updated": updated, "ok": True})


@mcp.custom_route("/api/status", methods=["GET"])
async def api_status(request):
    """Return dashboard-visible system status."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        stats = await bucket_mgr.get_stats()
        return JSONResponse(
            {
                "decay_engine": "running" if decay_engine.is_running else "stopped",
                "buckets": {
                    "permanent": stats.get("permanent_count", 0),
                    "dynamic": stats.get("dynamic_count", 0),
                    "archive": stats.get("archive_count", 0),
                    "feel": stats.get("feel_count", 0),
                    "total": stats.get("permanent_count", 0)
                    + stats.get("dynamic_count", 0)
                    + stats.get("archive_count", 0)
                    + stats.get("feel_count", 0),
                },
                "using_env_password": bool(os.environ.get("OMBRE_DASHBOARD_PASSWORD", "")),
            }
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# =============================================================
# Import API — conversation history import
# 导入 API — 对话历史导入
# =============================================================

@mcp.custom_route("/api/import/upload", methods=["POST"])
async def api_import_upload(request):
    """Upload a conversation file and start import."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err

    if import_engine.is_running:
        return JSONResponse({"error": "Import already running"}, status_code=409)

    content_type = request.headers.get("content-type", "")
    filename = ""

    try:
        if "multipart/form-data" in content_type:
            form = await request.form()
            file_field = form.get("file")
            if not file_field:
                return JSONResponse({"error": "No file field"}, status_code=400)
            raw_bytes = await file_field.read()
            filename = getattr(file_field, "filename", "upload")
            raw_content = raw_bytes.decode("utf-8", errors="replace")
        else:
            body = await request.body()
            raw_content = body.decode("utf-8", errors="replace")
            # Try to get filename from query params
            filename = request.query_params.get("filename", "upload")

        if not raw_content.strip():
            return JSONResponse({"error": "Empty file"}, status_code=400)

        preserve_raw = request.query_params.get("preserve_raw", "").lower() in ("1", "true")
        resume = request.query_params.get("resume", "").lower() in ("1", "true")

    except Exception as e:
        return JSONResponse({"error": f"Failed to read upload: {e}"}, status_code=400)

    # Start import in background
    async def _run_import():
        try:
            await import_engine.start(raw_content, filename, preserve_raw, resume)
        except Exception as e:
            logger.error(f"Import failed: {e}")

    asyncio.create_task(_run_import())

    return JSONResponse({
        "status": "started",
        "filename": filename,
        "size_bytes": len(raw_content.encode()),
    })


@mcp.custom_route("/api/import/status", methods=["GET"])
async def api_import_status(request):
    """Get current import progress."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    return JSONResponse(import_engine.get_status())


@mcp.custom_route("/api/import/pause", methods=["POST"])
async def api_import_pause(request):
    """Pause the running import."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    if not import_engine.is_running:
        return JSONResponse({"error": "No import running"}, status_code=400)
    import_engine.pause()
    return JSONResponse({"status": "pause_requested"})


@mcp.custom_route("/api/import/patterns", methods=["GET"])
async def api_import_patterns(request):
    """Detect high-frequency patterns after import."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        patterns = await import_engine.detect_patterns()
        return JSONResponse({"patterns": patterns})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/import/results", methods=["GET"])
async def api_import_results(request):
    """List recently imported/created buckets for review."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        limit = int(request.query_params.get("limit", "50"))
        all_buckets = await bucket_mgr.list_all(include_archive=False)
        # Sort by created time, newest first
        all_buckets.sort(key=lambda b: b["metadata"].get("created", ""), reverse=True)
        results = []
        for b in all_buckets[:limit]:
            results.append({
                "id": b["id"],
                "name": b["metadata"].get("name", ""),
                "content": b["content"][:300],
                "type": b["metadata"].get("type", ""),
                "domain": b["metadata"].get("domain", []),
                "tags": b["metadata"].get("tags", []),
                "importance": b["metadata"].get("importance", 5),
                "created": b["metadata"].get("created", ""),
            })
        return JSONResponse({"buckets": results, "total": len(all_buckets)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/import/review", methods=["POST"])
async def api_import_review(request):
    """Apply review decisions: mark buckets as important/noise/pinned."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    decisions = body.get("decisions", [])
    if not decisions:
        return JSONResponse({"error": "No decisions provided"}, status_code=400)

    applied = 0
    errors = 0
    for d in decisions:
        bid = d.get("bucket_id", "")
        action = d.get("action", "")
        if not bid or not action:
            continue
        try:
            if action == "important":
                await bucket_mgr.update(bid, importance=9)
            elif action == "pin":
                await bucket_mgr.update(bid, pinned=True)
            elif action == "anchor":
                bucket = await bucket_mgr.get(bid)
                if not bucket:
                    raise ValueError("bucket not found")
                ok, message = await _can_mark_anchor(bid, bucket)
                if not ok:
                    raise ValueError(message)
                await bucket_mgr.update(bid, anchor=True)
            elif action == "noise":
                await bucket_mgr.update(bid, resolved=True, importance=1)
            elif action == "delete":
                file_path = bucket_mgr._find_bucket_file(bid)
                if file_path:
                    os.remove(file_path)
            applied += 1
        except Exception as e:
            logger.warning(f"Review action failed for {bid}: {e}")
            errors += 1

    return JSONResponse({"applied": applied, "errors": errors})


# --- Entry point / 启动入口 ---
if __name__ == "__main__":
    transport = config.get("transport", "stdio")
    logger.info(f"Ombre Brain starting | transport: {transport}")

    if transport in ("sse", "streamable-http"):
        import threading
        import uvicorn
        from starlette.middleware.cors import CORSMiddleware

        # --- Application-level keepalive: ping /health every 60s ---
        # --- 应用层保活：每 60 秒 ping 一次 /health，防止 Cloudflare Tunnel 空闲断连 ---
        async def _keepalive_loop():
            await asyncio.sleep(10)  # Wait for server to fully start
            async with httpx.AsyncClient() as client:
                while True:
                    try:
                        await client.get("http://localhost:8000/health", timeout=5)
                        logger.debug("Keepalive ping OK / 保活 ping 成功")
                    except Exception as e:
                        logger.warning(f"Keepalive ping failed / 保活 ping 失败: {e}")
                    await asyncio.sleep(60)

        def _start_keepalive():
            loop = asyncio.new_event_loop()
            loop.run_until_complete(_keepalive_loop())

        t = threading.Thread(target=_start_keepalive, daemon=True)
        t.start()

        async def _reflection_loop():
            await asyncio.sleep(20)
            local_bucket_mgr = BucketManager(config)
            local_embedding_engine = EmbeddingEngine(config)
            local_persona_engine = PersonaStateEngine(config)
            local_reflection_engine = ReflectionEngine(config)
            while True:
                try:
                    results = await local_reflection_engine.run_due(
                        local_bucket_mgr,
                        local_persona_engine,
                        local_embedding_engine,
                    )
                    if results:
                        logger.info("Reflection run-due results / 反思定时结果: %s", results)
                except Exception as e:
                    logger.warning("Reflection scheduler failed / 反思定时器失败: %s", e)
                await asyncio.sleep(local_reflection_engine.check_interval_minutes * 60)

        def _start_reflection_scheduler():
            loop = asyncio.new_event_loop()
            loop.run_until_complete(_reflection_loop())

        if reflection_engine.enabled and reflection_engine.auto_enabled:
            rt = threading.Thread(target=_start_reflection_scheduler, daemon=True)
            rt.start()
            logger.info("Reflection scheduler enabled / 反思定时器已启用")

        # --- Add CORS middleware so remote clients (Cloudflare Tunnel / ngrok) can connect ---
        # --- 添加 CORS 中间件，让远程客户端（Cloudflare Tunnel / ngrok）能正常连接 ---
        if transport == "streamable-http":
            _app = mcp.streamable_http_app()
        else:
            _app = mcp.sse_app()
        _app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["*"],
        )
        _app.add_middleware(
            OmbreChatGptOAuthMiddleware,
            provider=OMBRE_CHATGPT_OAUTH,
            protected_hosts=OMBRE_CHATGPT_OAUTH_PROTECTED_HOSTS,
        )
        logger.info("CORS middleware enabled for remote transport / 已启用 CORS 中间件")
        if OMBRE_CHATGPT_OAUTH.enabled:
            logger.info(
                "ChatGPT OAuth enabled for Ombre MCP / 已启用 ChatGPT OAuth: protected_hosts=%s",
                sorted(OMBRE_CHATGPT_OAUTH_PROTECTED_HOSTS),
            )
        uvicorn.run(_app, host="0.0.0.0", port=8000)
    else:
        mcp.run(transport=transport)
