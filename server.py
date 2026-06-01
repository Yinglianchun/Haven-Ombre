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
#       resurface — Surface dormant memories without touching them
#                   只读浮现久未触碰的旧记忆
#       comment_bucket — Add a ring comment to a memory
#                        给记忆追加年轮
#       hold   — Store a single memory
#                存储单条记忆
#       grow   — Long-note memory digest, auto-split selected content into buckets
#                长内容摘记，筛选后拆分多桶
#       trace  — Modify metadata / resolved / delete
#                修改元数据 / resolved 标记 / 删除
#       pulse  — System status + bucket listing
#                系统状态 + 所有桶列表
#       reflect — Daily relationship weather
#                 日关系天气
#       introspection — Read recent memories for waking self-reflection
#                       读取最近记忆供清醒自省
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
from dataclasses import replace
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlencode, urlparse
import httpx


# --- Ensure same-directory modules can be imported ---
# --- 确保同目录下的模块能被正确导入 ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import Context, FastMCP

from bucket_manager import BucketManager
from dehydrator import Dehydrator
from decay_engine import DecayEngine
from dream_engine import DreamEngine
from embedding_engine import EmbeddingEngine
from identity import identity_names
from import_memory import ImportEngine
from memory_diffusion import (
    diffuse_memory,
    diffusion_options_from_config,
    format_diffusion_path,
    format_diffusion_trace,
    path_has_caution,
    seed_scores_for_buckets,
    should_suppress_context_candidate,
)
from memory_edges import MemoryEdgeStore
from memory_moments import MemoryMomentStore
from memory_relevance import (
    active_facets,
    facets_for_text,
    memory_relevance_options_from_config,
    query_has_facet,
    query_has_explicit_entity_marker,
    recall_search_query,
    recall_rank,
    relevance_decision,
    relevance_multiplier,
)
from recall_policy import CONTEXT_ONLY_SECTIONS, RecallPolicy
from memory_write_gate import MemoryWriteGate, WriteGateDecision
from memory_nodes import MemoryNodeStore
from persona_engine import PersonaStateEngine
from reflection_engine import ReflectionEngine
from recall_diagnostics import RecallDiagnosticsLogger
from reranker_engine import RerankerEngine
from utils import (
    bucket_text_for_embedding,
    count_tokens_approx,
    load_config,
    now_iso,
    setup_logging,
    strip_wikilinks,
)

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
reranker_engine = RerankerEngine(config)              # Reranker / 召回重排序
recall_diagnostics = RecallDiagnosticsLogger(config)  # Recall diagnostics / 召回诊断
import_engine = ImportEngine(config, bucket_mgr, dehydrator, embedding_engine)  # Import engine / 导入引擎
persona_engine = PersonaStateEngine(config)           # Persona state engine / 人格状态引擎
memory_edge_store = MemoryEdgeStore(config)            # Explicit memory relationship edges / 显式记忆关系边
memory_node_store = MemoryNodeStore(config)            # Computable memory node index / 可计算记忆节点
memory_moment_store = MemoryMomentStore(config)        # Structured bucket body/comment moment index / 记忆片段索引
memory_write_gate = MemoryWriteGate(config)            # Automatic grow gate / 自动写入门卫
reflection_engine = ReflectionEngine(config)           # Reflection worker / 关系天气与关系整理
dream_engine = DreamEngine(config)                     # Night dream worker / 夜梦

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


async def _hot_update_gateway_config(gateway_body: dict) -> str | None:
    admin_url = os.environ.get("OMBRE_GATEWAY_ADMIN_URL", "").strip()
    token = os.environ.get("OMBRE_GATEWAY_TOKEN", "").strip()
    if not admin_url or not token:
        return None
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.post(
                admin_url,
                headers={"Authorization": f"Bearer {token}"},
                json={"gateway": gateway_body},
            )
        if response.status_code >= 400:
            return f"gateway_hot_reload_failed:{response.status_code}"
        return "gateway_hot_reloaded"
    except Exception as exc:
        logger.warning("Gateway hot config update failed: %s", exc)
        return f"gateway_hot_reload_failed:{type(exc).__name__}"


DEFAULT_CHATGPT_OAUTH_REDIRECT_PREFIX = "https://chatgpt.com/connector/oauth/"
DEFAULT_CLAUDE_OAUTH_REDIRECT_URI = "https://claude.ai/api/mcp/auth_callback"


class ChatGptOAuthProvider:
    def __init__(
        self,
        client_id: str = "",
        client_secret: str = "",
        access_token: str = "",
        refresh_token: str = "",
        public_base_url: str = "",
        redirect_prefix: str = DEFAULT_CHATGPT_OAUTH_REDIRECT_PREFIX,
        redirect_uris: list[str] | None = None,
        token_ttl_seconds: int = 30 * 24 * 60 * 60,
    ) -> None:
        self.client_id = client_id.strip()
        self.client_secret = client_secret.strip()
        self.access_token = access_token.strip()
        self.refresh_token = refresh_token.strip()
        self.public_base_url = public_base_url.strip().rstrip("/")
        self.redirect_prefix = redirect_prefix.strip()
        raw_redirect_uris = (
            [DEFAULT_CLAUDE_OAUTH_REDIRECT_URI]
            if redirect_uris is None
            else redirect_uris
        )
        self.redirect_uris = tuple(
            uri.strip()
            for uri in raw_redirect_uris
            if uri.strip()
        )
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
        return bool(redirect_uri) and (
            bool(self.redirect_prefix and redirect_uri.startswith(self.redirect_prefix))
            or redirect_uri in self.redirect_uris
        )

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
    redirect_prefix=os.environ.get(
        "OMBRE_CHATGPT_OAUTH_REDIRECT_PREFIX",
        DEFAULT_CHATGPT_OAUTH_REDIRECT_PREFIX,
    ),
    redirect_uris=_split_csv(
        os.environ.get(
            "OMBRE_CHATGPT_OAUTH_REDIRECT_URIS",
            DEFAULT_CLAUDE_OAUTH_REDIRECT_URI,
        )
    ),
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


def _current_time_iso() -> str:
    return now_iso()


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


def _date_key(value) -> str:
    match = re.search(r"\d{4}-\d{2}-\d{2}", str(value or ""))
    return match.group(0) if match else ""


def _filter_by_created_date(
    buckets: list[dict],
    *,
    created_date: str = "",
    created_from: str = "",
    created_to: str = "",
) -> tuple[list[dict], str]:
    exact = _date_key(created_date)
    start = exact or _date_key(created_from)
    end = exact or _date_key(created_to)
    if not start and not end:
        return buckets, ""

    filtered = []
    for bucket in buckets:
        day = _date_key((bucket.get("metadata") or {}).get("created", ""))
        if not day:
            continue
        if start and day < start:
            continue
        if end and day > end:
            continue
        filtered.append(bucket)

    if start and end and start == end:
        return filtered, f", created_date={start}"
    return filtered, f", created_from={start or '*'}, created_to={end or '*'}"


def _bool_value(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _identity() -> dict:
    return identity_names(config)


def _ai_author_name() -> str:
    return _identity()["ai_name"]


def _dashboard_author_name() -> str:
    return _identity()["user_name"]


def _anchor_config() -> tuple[int, float]:
    anchor_cfg = config.get("anchor", {}) if isinstance(config.get("anchor", {}), dict) else {}
    max_count = _int_between(anchor_cfg.get("max_count"), 12, 1, 200)
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


def _select_anchor_buckets(all_buckets: list[dict], limit: int = 2) -> list[dict]:
    limit = _int_between(limit, 2, 0, 12)
    if limit <= 0:
        return []
    anchors = [
        b for b in all_buckets
        if b.get("metadata", {}).get("anchor")
        and not b.get("metadata", {}).get("pinned")
        and not b.get("metadata", {}).get("protected")
        and b.get("metadata", {}).get("type") not in {"permanent", "feel"}
    ]
    anchors.sort(
        key=lambda b: (
            int(b.get("metadata", {}).get("importance", 5)),
            decay_engine.calculate_score(b.get("metadata", {})),
            b.get("metadata", {}).get("updated_at") or b.get("metadata", {}).get("created", ""),
        ),
        reverse=True,
    )
    return anchors[:limit]


def _has_favorite_tag(tags: list | set | tuple | None) -> bool:
    return any(
        tag == "haven_favorite" or tag.startswith("flavor_")
        for tag in {str(item) for item in (tags or [])}
    )


def _has_favorite_reason(content: str) -> bool:
    text = strip_wikilinks(str(content or "")).lower()
    return any(
        marker in text
        for marker in (
            "喜欢它的原因",
            "喜欢的原因",
            "favorite_reason",
            "favorite reason",
        )
    )


def _favorite_reason_error() -> str:
    return "标记 favorite memory 需要在正文写明「喜欢它的原因」。"


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
        "comment_count",
        "comments",
        "profile_kind",
        "subject",
        "predicate",
        "object",
        "evidence",
    ]
    return {
        "id": bucket["id"],
        "metadata": {key: meta.get(key) for key in fields if key in meta},
        "content": strip_wikilinks(bucket.get("content", "")),
        "score": decay_engine.calculate_score(meta),
    }


def _queue_memory_enrichment(bucket_id: str, *, force: bool = False) -> None:
    if not bucket_id:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    loop.create_task(_enrich_memory_async(bucket_id, force=force))


async def _enrich_memory_async(bucket_id: str, *, force: bool = False) -> None:
    try:
        result = await reflection_engine.enrich_bucket(
            bucket_id,
            bucket_mgr,
            memory_edge_store,
            embedding_engine=embedding_engine,
            force=force,
        )
        logger.debug("Memory enrichment complete / 记忆关系补全完成: %s", result)
    except Exception as e:
        logger.warning("Memory enrichment failed / 记忆关系补全失败: %s: %s", bucket_id, e)


def _queue_embedding_refresh(bucket_id: str) -> bool:
    if not bucket_id or not getattr(embedding_engine, "enabled", False):
        return False
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False
    loop.create_task(_refresh_bucket_embedding_async(bucket_id))
    return True


async def _refresh_bucket_embedding_async(bucket_id: str) -> None:
    try:
        ok = await _refresh_bucket_embedding(bucket_id)
        if not ok:
            logger.debug("Embedding refresh skipped or failed / 向量刷新跳过或失败: %s", bucket_id)
    except Exception as e:
        logger.warning("Embedding refresh failed / 向量刷新失败: %s: %s", bucket_id, e)


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
            "identity": {
                "ai_name": _ai_author_name(),
                "user_name": _dashboard_author_name(),
            },
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
                      and not b["metadata"].get("anchor", False)
                      and not b["metadata"].get("pinned")
                      and not b["metadata"].get("protected")]
        scored = sorted(unresolved, key=lambda b: decay_engine.calculate_score(b["metadata"]), reverse=True)
        anchors = _select_anchor_buckets(all_buckets, limit=2)

        parts = []
        token_budget = 10000
        for b in pinned:
            summary = await dehydrator.dehydrate(_bucket_text_for_embedding(b), {k: v for k, v in b["metadata"].items() if k != "tags"})
            parts.append(f"📌 [核心准则] {summary}")
            token_budget -= count_tokens_approx(summary)

        for b in anchors:
            if token_budget <= 0:
                break
            summary = await dehydrator.dehydrate(_bucket_text_for_embedding(b), {k: v for k, v in b["metadata"].items() if k != "tags"})
            entry = f"⚓ [长期锚点] [bucket_id:{b['id']}] {summary}"
            entry_tokens = count_tokens_approx(entry)
            if entry_tokens > token_budget:
                break
            parts.append(entry)
            token_budget -= entry_tokens

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
            summary = await dehydrator.dehydrate(_bucket_text_for_embedding(b), {k: v for k, v in b["metadata"].items() if k != "tags"})
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
# /introspection-hook endpoint: Dedicated hook for waking self-reflection
# 清醒自省专用挂载点。/dream-hook 暂时保留兼容旧接入。
# =============================================================
@mcp.custom_route("/introspection-hook", methods=["GET"])
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
                f"{_bucket_text_for_embedding(b)[:200]}"
            )

        return PlainTextResponse("[Ombre Brain - Introspection]\n" + "\n---\n".join(parts))
    except Exception as e:
        logger.warning(f"Introspection hook failed: {e}")
        return PlainTextResponse("")


# =============================================================
# Internal helper: merge-or-create
# 内部辅助：检查是否可合并，可以则合并，否则新建
# Shared by hold and grow to avoid duplicate logic
# hold 和 grow 共用，避免重复逻辑
# =============================================================
def _bucket_days_since_last_active(meta: dict) -> float:
    parsed = bucket_mgr._parse_iso_datetime(meta.get("last_active") or meta.get("created"))
    if parsed is None:
        return 9999.0
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return max(0.0, (now - parsed).total_seconds() / 86400)


def _format_readonly_related_memory(bucket: dict) -> str:
    meta = bucket.get("metadata", {})
    labels = []
    if meta.get("type") == "archived":
        labels.append("归档")
    if meta.get("resolved"):
        labels.append("已解决")
    if meta.get("digested"):
        labels.append("已消化")
    state = f" ({', '.join(labels)})" if labels else ""
    preview = _bucket_text_for_embedding(bucket).replace("\n", " ").strip()
    if len(preview) > 220:
        preview = preview[:220].rstrip() + "..."
    return (
        "\n旧记忆(只读，不触碰): "
        f"[{meta.get('name', bucket['id'])}] [bucket_id:{bucket['id']}]{state}\n"
        f"{preview}"
    )


def _bucket_text_for_embedding(bucket: dict) -> str:
    meta = bucket.get("metadata", {})
    comments = meta.get("comments", [])
    comment_text = ""
    if isinstance(comments, list):
        comment_text = "\n".join(
            strip_wikilinks(str(comment.get("content", "")))
            for comment in comments
            if isinstance(comment, dict)
        )
    return f"{strip_wikilinks(bucket.get('content', '')).strip()}\n{comment_text}".strip()


def _bucket_context_snippet(bucket: dict, max_chars: int = 180) -> str:
    text = " ".join(strip_wikilinks(str(bucket.get("content") or "")).split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _compact_diffused_summary(bucket: dict, dehydrated: str, max_chars: int = 180) -> str:
    raw = str(dehydrated or "").strip()
    extracted = _summary_from_jsonish_text(raw)
    if extracted:
        return _clip_text(extracted, max_chars)

    meta = bucket.get("metadata", {}) or {}
    title = str(meta.get("name") or bucket.get("id") or "记忆").strip()
    return _clip_text(title, max_chars)


def _bucket_diffusion_path_summary(path, bucket_map: dict[str, dict], max_chars: int = 180) -> str:
    if not path:
        return ""
    return _clip_text(format_diffusion_path(path, bucket_map), max_chars)


def _breath_one_hop_diffusion_options(top_k: int):
    return replace(
        diffusion_options_from_config(config),
        max_hops=1,
        top_k=max(0, int(top_k or 0)),
    )


def _summary_from_jsonish_text(text: str) -> str:
    if not text:
        return ""
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start:end + 1])

    for candidate in candidates:
        try:
            data = _json_lib.loads(candidate)
        except Exception:
            continue
        if isinstance(data, dict):
            for key in ("summary", "memory_summary", "gist"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return " ".join(strip_wikilinks(value).split())
            core_facts = data.get("core_facts")
            if isinstance(core_facts, list) and core_facts:
                facts = [
                    " ".join(strip_wikilinks(str(item)).split())
                    for item in core_facts[:2]
                    if str(item).strip()
                ]
                if facts:
                    return "；".join(facts)
    return ""


def _clip_text(text: str, max_chars: int) -> str:
    compact = " ".join(strip_wikilinks(str(text or "")).split())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rstrip() + "..."


async def _refresh_bucket_embedding(bucket_id: str) -> bool:
    if not getattr(embedding_engine, "enabled", False):
        return False
    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return False
    return await embedding_engine.generate_and_store(bucket_id, bucket_text_for_embedding(bucket))


def _write_semantic_search_timeout_seconds() -> float:
    write_cfg = config.get("write_path", {}) if isinstance(config.get("write_path", {}), dict) else {}
    try:
        return max(0.0, float(write_cfg.get("semantic_search_timeout_seconds", 3)))
    except (TypeError, ValueError):
        return 3.0


async def _find_readonly_related_bucket(
    content: str,
    *,
    exclude_ids: set[str] | None = None,
) -> dict | None:
    exclude_ids = exclude_ids or set()
    candidates: dict[str, dict] = {}

    try:
        for bucket in await bucket_mgr.search(content, limit=8, include_archive=True):
            candidates[bucket["id"]] = {**bucket, "_related_score": float(bucket.get("score", 0.0))}
    except Exception as e:
        logger.warning(f"Related old memory keyword search failed / 相关旧记忆关键词搜索失败: {e}")

    if getattr(embedding_engine, "enabled", False):
        try:
            semantic_lookup = embedding_engine.search_similar(content, top_k=8)
            timeout_seconds = _write_semantic_search_timeout_seconds()
            if timeout_seconds > 0:
                similar = await asyncio.wait_for(semantic_lookup, timeout=timeout_seconds)
            else:
                similar = await semantic_lookup
            for bucket_id, similarity in similar:
                if bucket_id in candidates:
                    candidates[bucket_id]["_related_score"] = max(
                        candidates[bucket_id].get("_related_score", 0.0),
                        float(similarity) * 100.0,
                    )
                    continue
                bucket = await bucket_mgr.get(bucket_id)
                if bucket:
                    candidates[bucket_id] = {**bucket, "_related_score": float(similarity) * 100.0}
        except asyncio.TimeoutError:
            logger.warning(
                "Related old memory semantic search timed out after %.1fs / 写入时相关旧记忆语义搜索超时",
                _write_semantic_search_timeout_seconds(),
            )
        except Exception as e:
            logger.warning(f"Related old memory semantic search failed / 相关旧记忆语义搜索失败: {e}")

    ranked = []
    for bucket in candidates.values():
        meta = bucket.get("metadata", {})
        if bucket.get("id") in exclude_ids:
            continue
        if meta.get("type") == "feel":
            continue
        ranked.append(bucket)

    ranked.sort(
        key=lambda item: (
            item.get("_related_score", 0.0),
            _bucket_days_since_last_active(item.get("metadata", {})),
        ),
        reverse=True,
    )
    return ranked[0] if ranked else None


def _bucket_needs_memory_enrichment(bucket: dict) -> bool:
    meta = bucket.get("metadata", {}) if isinstance(bucket, dict) else {}
    if meta.get("type") == "feel" or meta.get("protected"):
        return False
    try:
        confidence = float(meta.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return confidence <= 0.0


async def _backfill_memory_enrichment(
    limit: int | None = None,
    *,
    bucket_mgr_arg=None,
    reflection_engine_arg=None,
    edge_store_arg=None,
    embedding_engine_arg=None,
) -> dict:
    mgr = bucket_mgr_arg or bucket_mgr
    engine = reflection_engine_arg or reflection_engine
    edge_store = edge_store_arg or memory_edge_store
    emb_engine = embedding_engine_arg or embedding_engine
    reflection_cfg = config.get("reflection", {}) if isinstance(config.get("reflection", {}), dict) else {}
    default_limit = _int_between(reflection_cfg.get("enrich_backfill_limit"), 5, 0, 50)
    limit = _int_between(limit, default_limit, 0, 50)
    if limit <= 0:
        return {"processed": 0, "ids": [], "errors": []}

    try:
        all_buckets = await mgr.list_all(include_archive=False)
    except Exception as e:
        logger.warning("Memory enrichment backfill list failed / enrich 补跑列桶失败: %s", e)
        return {"processed": 0, "ids": [], "errors": [str(e)]}

    candidates = [bucket for bucket in all_buckets if _bucket_needs_memory_enrichment(bucket)]
    candidates.sort(
        key=lambda item: item.get("metadata", {}).get("updated_at") or item.get("metadata", {}).get("created", ""),
        reverse=True,
    )

    processed: list[str] = []
    errors: list[str] = []
    for bucket in candidates[:limit]:
        bucket_id = bucket.get("id")
        if not bucket_id:
            continue
        try:
            await engine.enrich_bucket(
                bucket_id,
                mgr,
                edge_store,
                embedding_engine=emb_engine,
                force=True,
            )
            processed.append(bucket_id)
        except Exception as e:
            logger.warning("Memory enrichment backfill failed / enrich 补跑失败: %s: %s", bucket_id, e)
            errors.append(f"{bucket_id}: {e}")
    return {"processed": len(processed), "ids": processed, "errors": errors}


@mcp.tool()
async def enrich_backfill(limit: int = 10) -> dict:
    """后台补跑缺失的 tags/confidence/memory_edges；主要用于 enrich_on_write 曾经超时或关闭后的修复。"""
    return await _backfill_memory_enrichment(limit=limit)


async def _ensure_decay_engine_started_for_transport(transport_name: str) -> None:
    if transport_name not in ("sse", "streamable-http"):
        return
    try:
        await decay_engine.ensure_started()
    except Exception as e:
        logger.warning("Decay engine startup failed / 衰减引擎启动失败: %s", e)


async def _merge_or_create(
    content: str,
    tags: list,
    importance: int,
    domain: list,
    valence: float,
    arousal: float,
    name: str = "",
    *,
    allow_merge: bool = True,
) -> tuple[str, str, bool, dict | None]:
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

    related_bucket = await _find_readonly_related_bucket(content)

    if allow_merge and existing and existing[0].get("score", 0) > config.get("merge_threshold", 75):
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
                _queue_embedding_refresh(bucket["id"])
                return bucket["id"], bucket["metadata"].get("name", bucket["id"]), True, related_bucket
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
    _queue_embedding_refresh(bucket_id)
    return bucket_id, name or bucket_id, False, related_bucket


async def _build_mcp_diffused_memory_block(
    source_buckets: list[dict],
    all_buckets: list[dict] | None,
    token_budget: int,
    limit_per_source: int,
    min_confidence: float,
    query_text: str = "",
    exclude_bucket_ids: set[str] | None = None,
) -> str:
    if token_budget <= 0 or not source_buckets:
        return ""

    limit_per_source = _int_between(limit_per_source, 1, 0, 5)
    min_confidence = _float_between(min_confidence, 0.55, 0.0, 1.0)
    if limit_per_source <= 0:
        return ""

    source_ids = [bucket["id"] for bucket in source_buckets if bucket.get("id")]
    source_set = set(source_ids)
    exclude_set = source_set | set(exclude_bucket_ids or set())
    if not source_ids:
        return ""

    if all_buckets is None:
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
        except Exception as e:
            logger.warning(f"Failed to list buckets for diffused memory / 联想浮现列桶失败: {e}")
            all_buckets = []

    bucket_map = {bucket["id"]: bucket for bucket in all_buckets if bucket.get("id")}
    node_salience = None
    node_resonance = None
    if _node_facets_enabled(config):
        try:
            memory_node_store.bulk_upsert(list(bucket_map.values()))
            query_facets = memory_node_store.facets_for_text(query_text)
            node_salience = _node_salience_lookup
            node_resonance = _node_resonance_lookup(query_facets)
        except Exception as e:
            logger.warning(f"Failed to refresh memory nodes / 记忆节点刷新失败: {e}")

    edges = [
        edge
        for edge in memory_edge_store.list_edges()
        if float(edge.get("confidence", 0.0)) >= min_confidence
    ]
    hits = diffuse_memory(
        seed_scores_for_buckets(source_buckets),
        edges,
        bucket_map,
        options=_breath_one_hop_diffusion_options(len(source_ids) * limit_per_source),
        exclude_ids=exclude_set,
        node_salience=node_salience,
        node_resonance=node_resonance,
        query_text=query_text,
    )

    parts = []
    seen_targets = set()
    remaining = token_budget
    for hit in hits:
        target_id = hit.bucket_id
        if not target_id or target_id in seen_targets:
            continue

        target = bucket_map.get(target_id)
        if not target:
            continue
        meta = target.get("metadata", {})
        if meta.get("type") == "feel":
            continue
        if (
            query_text
            and _query_requires_direct_topic_evidence(query_text)
            and not _query_wants_body_chain(query_text)
            and not _bucket_has_query_topic_evidence(query_text, target)
        ):
            continue

        try:
            clean_meta = {k: v for k, v in meta.items() if k != "tags"}
            raw_summary = await dehydrator.dehydrate(
                _bucket_text_for_embedding(target),
                clean_meta,
            )
            summary = _compact_diffused_summary(target, raw_summary)
            path_summary = _bucket_diffusion_path_summary(hit.best_path, bucket_map)
            caution = (
                "路径含冲突/阻断，仅作边界背景。"
                if path_has_caution(hit.best_path)
                else "背景联想，不代表当前事实。"
            )
            path_part = f"路径: {path_summary}；" if path_summary else ""
            block = f"- [bucket_id:{target_id}] {path_part}摘要: {summary}（{caution}）"
            block_tokens = count_tokens_approx(block)
            if block_tokens > remaining:
                break
            parts.append(block)
            seen_targets.add(target_id)
            remaining -= block_tokens
            if remaining <= 0:
                break
        except Exception as e:
            logger.warning(f"Failed to build diffused memory block / 联想浮现构建失败: {e}")
            continue

    return "\n---\n".join(parts)


MOMENT_SECTION_LABELS = {
    "body": "body",
    "moment": "moment",
    "fact": "fact",
    "original": "original",
    "evidence_context": "evidence_context",
    "context": "context",
    "reflection": "reflection",
    "feeling": "feeling",
    "followup": "followup",
    "affect_anchor": "affect_anchor",
    "favorite_reason": "favorite_reason",
    "comment": "年轮",
}

MOMENT_TEMPERATURE_SECTIONS = CONTEXT_ONLY_SECTIONS
PROFILE_CONTEXT_SECTIONS = ("evidence_context", "context", "reflection", "feeling", "followup", "comment")


def _moment_text(moment: dict, max_chars: int = 500) -> str:
    return _clip_text(" ".join(str(moment.get("text") or "").split()), max_chars)


def _moment_label(moment: dict) -> str:
    section = str(moment.get("section") or "moment")
    return MOMENT_SECTION_LABELS.get(section, section)


def _moment_bucket_title(moment: dict) -> str:
    meta = moment.get("metadata", {}) if isinstance(moment.get("metadata"), dict) else {}
    return str(meta.get("bucket_name") or moment.get("bucket_id") or "").strip()


def _moments_by_bucket(moments: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for moment in moments:
        bucket_id = str(moment.get("bucket_id") or "")
        if bucket_id:
            grouped.setdefault(bucket_id, []).append(moment)
    for items in grouped.values():
        items.sort(key=lambda item: int(item.get("ordinal") or 0))
    return grouped


def _recallable_moments(moments: list[dict]) -> list[dict]:
    return [
        moment for moment in moments
        if (moment.get("metadata", {}) or {}).get("bucket_type") != "feel"
    ]


def _direct_recallable_moments(moments: list[dict]) -> list[dict]:
    return [
        moment for moment in _recallable_moments(moments)
        if moment.get("section") not in MOMENT_TEMPERATURE_SECTIONS
    ]


def _representative_moment(moments: list[dict]) -> dict | None:
    for section in (
        "original",
        "moment",
        "fact",
        "body",
        "evidence_context",
        "context",
        "reflection",
        "feeling",
        "followup",
        "comment",
    ):
        for moment in moments:
            if moment.get("section") == section:
                return moment
    return moments[0] if moments else None


def _direct_representative_moment(moments: list[dict]) -> dict | None:
    return _representative_moment(_direct_recallable_moments(moments))


def _bucket_edges_as_moment_edges(bucket_edges: list[dict], grouped: dict[str, list[dict]]) -> list[dict]:
    edges = []
    for edge in bucket_edges or []:
        source_bucket = str(edge.get("source") or edge.get("source_memory_id") or "").strip()
        target_bucket = str(edge.get("target") or edge.get("target_memory_id") or "").strip()
        if not source_bucket or not target_bucket:
            continue
        target = _direct_representative_moment(grouped.get(target_bucket, []))
        if not target:
            continue
        relation_type = str(edge.get("relation_type") or edge.get("type") or "relates_to")
        try:
            confidence = float(edge.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        for source in grouped.get(source_bucket, []):
            if source.get("section") in MOMENT_TEMPERATURE_SECTIONS:
                continue
            edges.append(
                {
                    "source": source["moment_id"],
                    "target": target["moment_id"],
                    "bucket_id": source_bucket,
                    "relation_type": relation_type,
                    "confidence": max(0.0, min(1.0, confidence)),
                    "reason": edge.get("reason") or "bucket edge bridge",
                }
            )
    return edges


def _moment_diffusion_map(moments: list[dict]) -> dict[str, dict]:
    mapped = {}
    for moment in moments:
        moment_id = moment.get("moment_id")
        if not moment_id:
            continue
        item = dict(moment)
        meta = dict(item.get("metadata", {}) or {})
        meta["importance"] = meta.get("bucket_importance", 5)
        meta["type"] = meta.get("bucket_type", "")
        meta["anchor"] = meta.get("bucket_anchor", False)
        meta["pinned"] = meta.get("bucket_pinned", False)
        meta["protected"] = meta.get("bucket_protected", False)
        meta["name"] = meta.get("bucket_name", "")
        meta["resolved"] = meta.get("bucket_resolved", False)
        meta["digested"] = meta.get("bucket_digested", False)
        item["metadata"] = meta
        mapped[str(moment_id)] = item
    return mapped


def _seed_scores_for_moments(moments: list[dict]) -> dict[str, float]:
    scores = {}
    for moment in moments:
        moment_id = str(moment.get("moment_id") or "")
        if not moment_id:
            continue
        try:
            score = float(moment.get("score", 0.65))
        except (TypeError, ValueError):
            score = 0.65
        scores[moment_id] = max(0.15, min(1.0, score))
    return scores


def _context_moments_for_seed(seed: dict, grouped: dict[str, list[dict]]) -> list[dict]:
    bucket_id = str(seed.get("bucket_id") or "")
    seed_id = seed.get("moment_id")
    bucket_moments = grouped.get(bucket_id, [])
    contexts = []

    def add_context(moment: dict) -> None:
        if moment.get("moment_id") == seed_id:
            return
        if any(existing.get("moment_id") == moment.get("moment_id") for existing in contexts):
            return
        contexts.append(moment)

    if _is_profile_fact_moment(seed):
        for section in PROFILE_CONTEXT_SECTIONS:
            for moment in bucket_moments:
                if moment.get("section") == section:
                    add_context(moment)
                    break
        return contexts[:4]

    seed_ordinal = int(seed.get("ordinal") or 0)
    for moment in bucket_moments:
        section = moment.get("section")
        ordinal = int(moment.get("ordinal") or 0)
        if abs(ordinal - seed_ordinal) == 1 and section not in MOMENT_TEMPERATURE_SECTIONS:
            add_context(moment)
    for section in ("affect_anchor", "favorite_reason", "comment"):
        for moment in bucket_moments:
            if moment.get("moment_id") != seed_id and moment.get("section") == section:
                add_context(moment)
                break
    return contexts[:4]


def _is_profile_fact_moment(moment: dict) -> bool:
    meta = moment.get("metadata", {}) if isinstance(moment.get("metadata"), dict) else {}
    tags = {str(tag) for tag in meta.get("tags", []) or []}
    tags.update(str(tag) for tag in meta.get("bucket_tags", []) or [])
    return "profile_fact" in tags or bool(meta.get("profile_kind"))


def _format_direct_moment(seed: dict, grouped: dict[str, list[dict]], token_budget: int) -> str:
    title = _moment_bucket_title(seed)
    head = f"[bucket_id:{seed['bucket_id']}] [moment_id:{seed['moment_id']}] {_moment_label(seed)}"
    if title and title != seed["bucket_id"]:
        head += f" {title}"
    parts = [head, _moment_text(seed, 520)]
    context_lines = [
        f"- [{_moment_label(moment)}] [moment_id:{moment['moment_id']}] {_moment_text(moment, 260)}"
        for moment in _context_moments_for_seed(seed, grouped)
    ]
    if context_lines:
        parts.append("语境:\n" + "\n".join(context_lines))
    block = "\n".join(parts)
    if count_tokens_approx(block) <= token_budget:
        return block
    compact = f"{head}\n{_moment_text(seed, 260)}"
    return compact if count_tokens_approx(compact) <= token_budget else ""


def _format_related_moment(
    moment: dict,
    caution: bool = False,
    path=None,
    moment_map: dict[str, dict] | None = None,
) -> str:
    note = "路径含冲突/阻断，仅作边界背景。" if caution else "背景联想，不代表当前事实。"
    summary = _diffused_moment_summary(moment, path=path, moment_map=moment_map or {})
    path_part = ""
    if path is not None:
        path_summary = _moment_path_summary(path, moment_map or {})
        if path_summary:
            path_part = f"路径: {path_summary}；"
    return (
        f"- [bucket_id:{moment['bucket_id']}] [moment_id:{moment['moment_id']}] "
        f"{path_part}摘要: {summary}（{note}）"
    )


def _format_secondary_direct_moment(moment: dict) -> str:
    summary = _diffused_moment_summary(moment)
    return (
        f"- [bucket_id:{moment['bucket_id']}] [moment_id:{moment['moment_id']}] "
        f"摘要: {summary}（相关命中，来自同一查询语义。）"
    )


def _diffused_moment_summary(
    moment: dict,
    *,
    path=None,
    moment_map: dict[str, dict] | None = None,
    max_chars: int = 180,
) -> str:
    label = _moment_label(moment)
    title = _moment_bucket_title(moment) or str(moment.get("bucket_id") or "记忆")
    status = _moment_status_label(moment)
    parts = [f"{title} / {label}"]
    if status:
        parts.append(status)
    path_summary = _moment_path_summary(path, moment_map or {}) if path is not None else ""
    if path_summary:
        parts.append(f"路径 {path_summary}")
    return _clip_text("；".join(parts), max_chars)


def _moment_status_label(moment: dict) -> str:
    meta = moment.get("metadata", {}) if isinstance(moment.get("metadata"), dict) else {}
    if meta.get("resolved") or meta.get("bucket_resolved"):
        return "已解决"
    if meta.get("digested") or meta.get("bucket_digested"):
        return "已消化"
    if str(meta.get("type") or meta.get("bucket_type") or "").lower() == "archived":
        return "归档"
    return ""


def _moment_path_summary(path, moment_map: dict[str, dict], max_chars: int = 140) -> str:
    if path is None:
        return ""
    nodes = tuple(str(node_id) for node_id in (getattr(path, "nodes", ()) or ()))
    if not nodes:
        return ""
    labels = [_moment_node_label(moment_map.get(nodes[0]), nodes[0])]
    for step in getattr(path, "steps", ()) or ():
        target_id = str(getattr(step, "target", "") or "")
        arrow = "<-" if getattr(step, "direction", "") == "incoming" else "->"
        labels.append(f"{arrow} {_moment_node_label(moment_map.get(target_id), target_id)}")
    return _clip_text(" ".join(labels), max_chars)


def _moment_node_label(moment: dict | None, fallback_id: str) -> str:
    if isinstance(moment, dict):
        return _clip_text(_moment_bucket_title(moment) or str(moment.get("bucket_id") or fallback_id), 48)
    return _clip_text(fallback_id, 48)


def _recall_relevance_options():
    return memory_relevance_options_from_config(config)


def _apply_recall_relevance_gate(query: str, candidates: list[dict]) -> list[dict]:
    options = _recall_relevance_options()
    filtered = []
    adjusted = False
    for moment in candidates:
        multiplier = relevance_multiplier(query, moment, options)
        if multiplier <= 0:
            adjusted = True
            continue
        item = dict(moment)
        try:
            score = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        new_score = round(score * multiplier, 4)
        if new_score != score:
            adjusted = True
        item["score"] = new_score
        filtered.append(item)
    if adjusted:
        filtered.sort(key=lambda item: _recall_rank(query, item))
    return filtered


def _recall_admission_thresholds() -> tuple[float, float]:
    threshold_cfg = config.get("recall_thresholds", {}) or {}
    return (
        _float_between(threshold_cfg.get("explicit_admission_semantic_score"), 0.72, 0.0, 1.0),
        _float_between(threshold_cfg.get("explicit_admission_rerank_score"), 0.65, 0.0, 1.0),
    )


def _recall_policy() -> RecallPolicy:
    semantic_threshold, rerank_threshold = _recall_admission_thresholds()
    return RecallPolicy(
        _recall_relevance_options(),
        semantic_threshold=semantic_threshold,
        rerank_threshold=rerank_threshold,
    )


def _breath_moment_admission_decision(
    query: str,
    moment: dict,
    seed_diagnostics: dict[str, dict],
):
    seed = seed_diagnostics.get(str(moment.get("bucket_id") or ""), {})
    return _recall_policy().assess(
        query,
        moment,
        semantic_score=seed.get("embedding_score"),
        rerank_score=moment.get("rerank_score"),
        context_only=moment.get("section") in MOMENT_TEMPERATURE_SECTIONS,
    )


async def _rerank_breath_moment_candidates(query: str, candidates: list[dict]) -> list[dict]:
    if not candidates or not getattr(reranker_engine, "enabled", False):
        return candidates
    candidate_limit = min(
        len(candidates),
        max(1, int(getattr(reranker_engine, "candidate_limit", 20) or 20)),
    )
    head = candidates[:candidate_limit]
    tail = candidates[candidate_limit:]
    documents = [_moment_rerank_document(moment) for moment in head]
    results = await reranker_engine.rerank(query, documents, top_n=len(head))
    if not results:
        return candidates

    by_index = {result.index: result.score for result in results}
    weight = max(0.0, min(1.0, float(getattr(reranker_engine, "score_weight", 0.65))))
    reranked = []
    for index, moment in enumerate(head):
        item = dict(moment)
        rerank_score = by_index.get(index)
        try:
            base_score = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            base_score = 0.0
        if rerank_score is None:
            item["rerank_score"] = None
            item["combined_score"] = base_score
        else:
            item["rerank_score"] = round(rerank_score, 4)
            item["combined_score"] = round(base_score * (1.0 - weight) + rerank_score * weight, 4)
            item["score"] = item["combined_score"]
        reranked.append(item)
    reranked.sort(
        key=lambda item: (
            _recall_rank(query, item)[0],
            item.get("rerank_score") is None,
            -(_safe_float(item.get("combined_score", item.get("score"))) or 0.0),
            -(_safe_float(item.get("score")) or 0.0),
        ),
    )
    return reranked + tail


def _moment_rerank_document(moment: dict) -> str:
    meta = moment.get("metadata", {}) if isinstance(moment.get("metadata"), dict) else {}
    fields = [
        f"title: {meta.get('bucket_name') or moment.get('bucket_id') or ''}",
        f"section: {moment.get('section') or ''}",
        f"domain: {' '.join(str(item) for item in meta.get('bucket_domain', []) or [])}",
        f"tags: {' '.join(str(item) for item in meta.get('bucket_tags', []) or [])}",
        f"summary: {meta.get('annotation_summary') or meta.get('summary') or ''}",
        f"facets: {_format_annotation_facets(meta)}",
        f"evidence: {_format_evidence_spans(meta)}",
        f"text: {moment.get('text') or ''}",
    ]
    return "\n".join(fields)[:4000]


def _format_annotation_facets(meta: dict) -> str:
    facets = meta.get("annotation_facets")
    if not isinstance(facets, dict):
        return ""
    parts = []
    for facet, score in sorted(facets.items(), key=lambda item: str(item[0])):
        try:
            parts.append(f"{facet}:{float(score):.2f}")
        except (TypeError, ValueError):
            continue
    return " ".join(parts)


def _format_evidence_spans(meta: dict, max_items: int = 3) -> str:
    spans = meta.get("evidence_spans")
    if not isinstance(spans, list):
        return ""
    parts = []
    for item in spans[:max_items]:
        if isinstance(item, dict):
            facet = str(item.get("facet") or "").strip()
            text = str(item.get("text") or item.get("span") or "").strip()
            if text:
                parts.append(f"{facet}: {text}" if facet else text)
        elif str(item).strip():
            parts.append(str(item).strip())
    return " | ".join(parts)


def _upsert_breath_seed_diagnostic(
    seed_diagnostics: dict[str, dict],
    bucket: dict,
    source: str,
    *,
    bucket_search_score: float | None = None,
    embedding_score: float | None = None,
) -> None:
    bucket_id = str(bucket.get("id") or "").strip()
    if not bucket_id:
        return
    meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
    item = seed_diagnostics.setdefault(
        bucket_id,
        {
            "bucket_id": bucket_id,
            "bucket_name": meta.get("name") or bucket_id,
            "sources": [],
        },
    )
    if source and source not in item["sources"]:
        item["sources"].append(source)
    if bucket_search_score is not None:
        safe_score = _safe_float(bucket_search_score)
        if safe_score is not None:
            item["bucket_search_score"] = safe_score
            item["keyword_score"] = round(_score_to_unit(safe_score), 4)
    if embedding_score is not None:
        safe_embedding_score = _safe_float(embedding_score)
        if safe_embedding_score is not None:
            item["embedding_score"] = safe_embedding_score


def _score_to_unit(score: float) -> float:
    try:
        number = float(score)
    except (TypeError, ValueError):
        return 0.0
    if number > 1:
        number = number / 100.0
    return max(0.0, min(1.0, number))


def _write_breath_recall_diagnostics(
    *,
    query: str,
    recall_thresholds: dict,
    seed_diagnostics: dict[str, dict],
    pre_gate_candidates: list[dict],
    gated_candidates: list[dict],
    reranked_candidates: list[dict],
    returned_moments: list[dict],
    suppressed_candidates: list[dict],
    displayed_moment_ids: list[str],
    secondary_moment_ids: list[str],
    related_source_bucket_ids: list[str],
    related_included: bool,
    drift_included: bool,
    dream_included: bool,
    response_sections: list[str],
) -> None:
    if not getattr(recall_diagnostics, "enabled", False):
        return

    gated_by_id = _moment_index(gated_candidates)
    reranked_by_id = _moment_index(reranked_candidates)
    suppressed_by_id = _moment_index(suppressed_candidates)
    gated_rank = _moment_rank(gated_candidates)
    reranked_rank = _moment_rank(reranked_candidates)
    returned_ids = [str(moment.get("moment_id") or "") for moment in returned_moments if moment.get("moment_id")]
    returned_set = set(returned_ids)
    displayed_set = set(displayed_moment_ids)
    secondary_set = set(secondary_moment_ids)
    options = _recall_relevance_options()
    max_candidates = max(1, int(getattr(recall_diagnostics, "max_candidates", 20) or 20))

    candidates = []
    for index, moment in enumerate(pre_gate_candidates[:max_candidates]):
        moment_id = str(moment.get("moment_id") or "")
        bucket_id = str(moment.get("bucket_id") or "")
        decision = relevance_decision(query, moment, options)
        gated = gated_by_id.get(moment_id)
        reranked = reranked_by_id.get(moment_id)
        final = reranked or gated
        seed = seed_diagnostics.get(bucket_id, {})
        admission = _breath_moment_admission_decision(query, final or moment, seed_diagnostics)
        candidate = {
            "pre_rank": index,
            "gate_rank": gated_rank.get(moment_id),
            "final_rank": reranked_rank.get(moment_id),
            "bucket_id": bucket_id,
            "bucket_name": _moment_bucket_title(moment),
            "moment_id": moment_id,
            "section": moment.get("section"),
            "sources": seed.get("sources", []),
            "bucket_search_score": seed.get("bucket_search_score"),
            "keyword_score": seed.get("keyword_score"),
            "embedding_score": seed.get("embedding_score"),
            "score_before_gate": _safe_float(moment.get("score")),
            "score_after_gate": _safe_float(gated.get("score")) if gated else None,
            "rerank_score": _safe_float(final.get("rerank_score")) if final else None,
            "combined_score": _safe_float(final.get("combined_score")) if final else None,
            "intent_rank": _recall_rank(query, final or moment)[0],
            "gate": "filtered" if decision.multiplier <= 0 else "kept",
            "gate_multiplier": round(float(decision.multiplier), 4),
            "gate_reasons": list(decision.reasons),
            "admission": "suppressed" if moment_id in suppressed_by_id else "admitted" if admission.admit else "suppressed",
            "admission_reason": admission.reason,
            "selected_returned": moment_id in returned_set,
            "selected_direct": moment_id in displayed_set,
            "selected_secondary": moment_id in secondary_set,
            "annotation_summary": (moment.get("metadata") or {}).get("annotation_summary"),
            "annotation_facets": (moment.get("metadata") or {}).get("annotation_facets", {}),
            "evidence_spans": (moment.get("metadata") or {}).get("evidence_spans", []),
            "text_preview": _diagnostic_text_preview(moment),
        }
        candidates.append(candidate)

    recall_diagnostics.write(
        {
            "source": "breath",
            "mode": "search",
            "query": str(query or ""),
            "recall_thresholds": recall_thresholds,
            "seed_buckets": list(seed_diagnostics.values())[:max_candidates],
            "candidates": candidates,
            "suppressed_candidates": [
                {
                    "bucket_id": str(moment.get("bucket_id") or ""),
                    "bucket_name": _moment_bucket_title(moment),
                    "moment_id": str(moment.get("moment_id") or ""),
                    "section": moment.get("section"),
                    "admission_reason": str(moment.get("_admission_reason") or "suppressed"),
                    "score": _safe_float(moment.get("score")),
                    "rerank_score": _safe_float(moment.get("rerank_score")),
                    "text_preview": _diagnostic_text_preview(moment),
                }
                for moment in suppressed_candidates[:max_candidates]
            ],
            "final": {
                "returned_moment_ids": returned_ids,
                "direct_moment_ids": displayed_moment_ids,
                "secondary_moment_ids": secondary_moment_ids,
                "related_source_bucket_ids": related_source_bucket_ids,
                "related_included": related_included,
                "drift_included": drift_included,
                "dream_included": dream_included,
                "response_sections": response_sections,
            },
        }
    )


def _format_suppressed_recall_candidate(moment: dict, seed_diagnostics: dict[str, dict]) -> str:
    bucket_id = str(moment.get("bucket_id") or "")
    seed = seed_diagnostics.get(bucket_id, {})
    parts = [
        f"- [bucket_id:{bucket_id}] [moment_id:{moment.get('moment_id') or ''}]",
        f"reason={moment.get('_admission_reason') or 'suppressed'}",
    ]
    if seed.get("embedding_score") is not None:
        parts.append(f"semantic={seed.get('embedding_score')}")
    if moment.get("rerank_score") is not None:
        parts.append(f"rerank={moment.get('rerank_score')}")
    preview = _diagnostic_text_preview(moment)
    if preview:
        parts.append(f"preview={preview}")
    return " ".join(parts)


def _moment_index(moments: list[dict]) -> dict[str, dict]:
    return {
        str(moment.get("moment_id")): moment
        for moment in moments
        if moment.get("moment_id")
    }


def _moment_rank(moments: list[dict]) -> dict[str, int]:
    return {
        str(moment.get("moment_id")): index
        for index, moment in enumerate(moments)
        if moment.get("moment_id")
    }


def _safe_float(value) -> float | None:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _diagnostic_text_preview(moment: dict) -> str:
    max_chars = max(0, int(getattr(recall_diagnostics, "max_text_chars", 220) or 0))
    if max_chars <= 0:
        return ""
    return _moment_text(moment, max_chars)


def _breath_recall_thresholds(query: str, max_results: int) -> dict:
    threshold_cfg = config.get("recall_thresholds", {}) or {}
    options = _recall_relevance_options()
    query_active = active_facets(facets_for_text(query, options))
    has_explicit = _query_has_explicit_entity_marker(query)
    is_vague = _query_is_vague_recall(query)

    base_vector_min = _float_between(threshold_cfg.get("vector_min_score"), 0.50, 0.0, 1.0)
    profile = "default"
    vector_min = base_vector_min
    top_k = max(max_results, 20)

    if has_explicit:
        profile = "explicit"
        vector_min = _float_between(threshold_cfg.get("explicit_vector_min_score"), 0.55, 0.0, 1.0)
    elif is_vague:
        profile = "vague"
        vector_min = _float_between(threshold_cfg.get("vague_vector_min_score"), 0.40, 0.0, 1.0)
        top_k = max(top_k, _int_between(threshold_cfg.get("vague_top_k"), 50, 20, 100))
    elif query_active:
        profile = "facet"
        vector_min = _float_between(threshold_cfg.get("facet_vector_min_score"), 0.45, 0.0, 1.0)

    return {
        "profile": profile,
        "vector_min_score": vector_min,
        "semantic_top_k": top_k,
        "query_facets": sorted(query_active),
        "has_explicit_entity": has_explicit,
        "is_vague": is_vague,
    }


def _query_has_explicit_entity_marker(query: str) -> bool:
    return query_has_explicit_entity_marker(query)


def _query_is_vague_recall(query: str) -> bool:
    text = str(query or "").strip().lower()
    if not text:
        return False
    vague_markers = (
        "最近",
        "有趣",
        "想起来",
        "回忆",
        "记忆",
        "什么事",
        "有什么",
        "随便",
        "random",
        "recent",
        "interesting",
        "anything",
        "memory",
        "memories",
    )
    if any(marker in text for marker in vague_markers):
        return True
    terms = re.findall(r"[A-Za-z0-9_./:-]+|[\u4e00-\u9fff]{1,4}", text)
    return len(terms) <= 2 and not _query_has_explicit_entity_marker(text)


def _query_wants_body_chain(query: str) -> bool:
    return query_has_facet(query, "embodiment", _recall_relevance_options())


def _query_requires_direct_topic_evidence(query: str) -> bool:
    return _recall_policy().requires_topic_evidence(query)


def _recall_rank(query: str, moment: dict) -> tuple[int, float]:
    return recall_rank(query, moment, _recall_relevance_options())


def _secondary_direct_limit(query: str, related_per_memory: int) -> int:
    if _query_wants_body_chain(query):
        return 5
    return max(0, min(2, int(related_per_memory or 0)))


def _secondary_direct_moments(
    query: str,
    candidates: list[dict],
    displayed_bucket_ids: set[str],
    limit: int,
) -> list[dict]:
    if limit <= 0:
        return []
    hidden = []
    seen_buckets = set(displayed_bucket_ids)
    for moment in candidates:
        bucket_id = str(moment.get("bucket_id") or "")
        if not bucket_id or bucket_id in seen_buckets:
            continue
        if moment.get("section") in MOMENT_TEMPERATURE_SECTIONS:
            continue
        if should_suppress_context_candidate(query, moment, _recall_relevance_options()):
            continue
        if not _query_wants_body_chain(query) and not _moment_has_query_topic_evidence(query, moment):
            continue
        hidden.append(moment)
        seen_buckets.add(bucket_id)
    if _query_wants_body_chain(query):
        hidden.sort(key=lambda moment: _recall_rank(query, moment))
    return hidden[:limit]


def _moment_has_query_topic_evidence(query: str, moment: dict) -> bool:
    return _recall_policy().moment_has_topic_evidence(query, moment)


def _bucket_has_query_topic_evidence(query: str, bucket: dict) -> bool:
    return _recall_policy().bucket_has_topic_evidence(query, bucket)


def _specific_query_terms(query: str) -> list[str]:
    return _recall_policy().specific_query_terms(query)


def _representative_moments_by_bucket(moments: list[dict]) -> dict[str, dict]:
    grouped = _moments_by_bucket(moments)
    representatives = {}
    for bucket_id, bucket_moments in grouped.items():
        representative = _direct_representative_moment(bucket_moments)
        if representative:
            representatives[bucket_id] = representative
    return representatives


async def _refresh_moment_graph(all_buckets: list[dict] | None = None) -> tuple[list[dict], dict[str, list[dict]], list[dict]]:
    if all_buckets is None:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
    memory_moment_store.bulk_upsert(all_buckets)
    moments = _recallable_moments(memory_moment_store.list_all())
    grouped = _moments_by_bucket(moments)
    edges = memory_moment_store.list_edges()
    edges.extend(_bucket_edges_as_moment_edges(memory_edge_store.list_edges(), grouped))
    return moments, grouped, edges


async def _build_mcp_moment_diffused_memory_block(
    seed_moments: list[dict],
    moments: list[dict],
    edges: list[dict],
    token_budget: int,
    limit_per_source: int,
    min_confidence: float,
    exclude_bucket_ids: set[str] | None = None,
    query_text: str = "",
) -> str:
    if token_budget <= 0 or not seed_moments:
        return ""
    limit_per_source = _int_between(limit_per_source, 1, 0, 5)
    if limit_per_source <= 0:
        return ""
    min_confidence = _float_between(min_confidence, 0.55, 0.0, 1.0)
    edges = [
        edge for edge in edges
        if float(edge.get("confidence", 0.0)) >= min_confidence
    ]
    moment_map = _moment_diffusion_map(moments)
    representatives = _representative_moments_by_bucket(moments)
    exclude_bucket_ids = set(exclude_bucket_ids or set())
    hits = diffuse_memory(
        _seed_scores_for_moments(seed_moments),
        edges,
        moment_map,
        options=_breath_one_hop_diffusion_options(len(seed_moments) * limit_per_source),
        exclude_ids={moment["moment_id"] for moment in seed_moments if moment.get("moment_id")},
        query_text=query_text,
    )

    parts = []
    seen = set()
    remaining = token_budget
    for hit in hits:
        moment = moment_map.get(hit.bucket_id)
        if not moment or hit.bucket_id in seen:
            continue
        bucket_id = str(moment.get("bucket_id") or "")
        if bucket_id in exclude_bucket_ids:
            continue
        if moment.get("section") in MOMENT_TEMPERATURE_SECTIONS:
            replacement = representatives.get(bucket_id)
            if replacement:
                moment = replacement
                if moment.get("moment_id") in seen:
                    continue
        block = _format_related_moment(
            moment,
            path_has_caution(hit.best_path),
            path=hit.best_path,
            moment_map=moment_map,
        )
        block_tokens = count_tokens_approx(block)
        if block_tokens > remaining:
            break
        parts.append(block)
        seen.add(hit.bucket_id)
        remaining -= block_tokens
        if remaining <= 0:
            break
    return "\n---\n".join(parts)


async def _collect_diffusion_seed_buckets(query: str, max_seeds: int) -> tuple[list[dict], list[str]]:
    warnings = []
    matches: list[dict] = []
    matched_ids: set[str] = set()
    search_limit = max(max_seeds, 20)

    try:
        for bucket in await bucket_mgr.search(query, limit=search_limit):
            if not bucket or bucket.get("metadata", {}).get("type") == "feel":
                continue
            bucket_id = bucket.get("id")
            if not bucket_id or bucket_id in matched_ids:
                continue
            candidate = dict(bucket)
            candidate["_inspect_source"] = "keyword"
            matches.append(candidate)
            matched_ids.add(bucket_id)
    except Exception as e:
        logger.warning(f"Inspect diffusion keyword search failed / 扩散诊断关键词检索失败: {e}")
        warnings.append(f"keyword_search_failed: {e}")

    try:
        vector_results = await embedding_engine.search_similar(query, top_k=search_limit)
        for bucket_id, sim_score in vector_results:
            if bucket_id in matched_ids:
                continue
            if sim_score <= 0.5:
                continue
            bucket = await bucket_mgr.get(bucket_id)
            if not bucket or bucket.get("metadata", {}).get("type") == "feel":
                continue
            candidate = dict(bucket)
            candidate["score"] = round(sim_score * 100, 2)
            candidate["vector_match"] = True
            candidate["_inspect_source"] = "vector"
            matches.append(candidate)
            matched_ids.add(bucket_id)
    except Exception as e:
        logger.warning(f"Inspect diffusion vector search failed / 扩散诊断向量检索失败: {e}")
        warnings.append(f"vector_search_failed: {e}")

    return matches[:max_seeds], warnings


def _inspect_bucket_label(bucket: dict | None, bucket_id: str) -> str:
    if not bucket:
        return bucket_id
    meta = bucket.get("metadata", {}) or {}
    return str(meta.get("name") or bucket.get("name") or bucket_id)


def _inspect_path_payload(path, bucket_map: dict[str, dict]) -> dict:
    return {
        "score": round(float(path.score), 4),
        "trace": format_diffusion_trace(path, bucket_map, use_labels=True),
        "nodes": list(path.nodes),
        "steps": [
            {
                "source": step.source,
                "target": step.target,
                "direction": step.direction,
                "relation_type": step.relation_type,
                "confidence": step.confidence,
                "reason": step.reason,
            }
            for step in path.steps
        ],
    }


@mcp.tool()
async def inspect_diffusion(
    query: str,
    max_seeds: int = 3,
    max_hits: int = 5,
    edge_min_confidence: float = 0.55,
) -> dict:
    """只读诊断 query 如何沿 memory_edges 点亮联想记忆；不 touch bucket，不创建记忆。"""
    query = str(query or "").strip()
    if not query:
        return {"status": "error", "error": "query_required"}

    max_seeds = _int_between(max_seeds, 3, 1, 20)
    max_hits = _int_between(max_hits, 5, 0, 20)
    edge_min_confidence = _float_between(edge_min_confidence, 0.55, 0.0, 1.0)
    node_facets_enabled = _node_facets_enabled(config)
    query_facets = {}
    if node_facets_enabled:
        try:
            query_facets = memory_node_store.facets_for_text(query)
        except Exception as e:
            logger.warning(f"Inspect diffusion query facets failed / 扩散诊断 query facets 失败: {e}")

    seed_buckets, warnings = await _collect_diffusion_seed_buckets(query, max_seeds)
    if not seed_buckets:
        return {
            "status": "ok",
            "query": query,
            "node_facets_enabled": node_facets_enabled,
            "query_facets": query_facets,
            "seeds": [],
            "hits": [],
            "warnings": warnings,
        }

    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
    except Exception as e:
        logger.warning(f"Inspect diffusion list buckets failed / 扩散诊断列桶失败: {e}")
        all_buckets = []
        warnings.append(f"list_buckets_failed: {e}")

    bucket_map = {bucket["id"]: bucket for bucket in all_buckets if bucket.get("id")}
    for seed in seed_buckets:
        if seed.get("id"):
            bucket_map.setdefault(seed["id"], seed)

    node_salience = None
    node_resonance = None
    if node_facets_enabled:
        try:
            memory_node_store.bulk_upsert(list(bucket_map.values()))
            node_salience = _node_salience_lookup
            node_resonance = _node_resonance_lookup(query_facets)
        except Exception as e:
            logger.warning(f"Inspect diffusion node refresh failed / 扩散诊断节点刷新失败: {e}")
            warnings.append(f"node_refresh_failed: {e}")
            node_salience = None
            node_resonance = None

    edges = [
        edge
        for edge in memory_edge_store.list_edges()
        if float(edge.get("confidence", 0.0)) >= edge_min_confidence
    ]
    options = replace(diffusion_options_from_config(config), top_k=max_hits)
    seed_scores = seed_scores_for_buckets(seed_buckets)
    hits = diffuse_memory(
        seed_scores,
        edges,
        bucket_map,
        options=options,
        exclude_ids={bucket["id"] for bucket in seed_buckets if bucket.get("id")},
        node_salience=node_salience,
        node_resonance=node_resonance,
        query_text=query,
    )

    def node_values(bucket_id: str, bucket: dict | None) -> dict:
        if not bucket or not node_facets_enabled:
            return {"salience": None, "resonance": None, "facets": {}}
        try:
            node = memory_node_store.get(bucket_id) or memory_node_store.upsert_bucket(bucket)
            salience = memory_node_store.node_salience(bucket_id, bucket)
            resonance = (
                node_resonance(bucket_id, bucket)
                if node_resonance
                else 1.0
            )
            return {
                "salience": round(float(salience), 4),
                "resonance": round(float(resonance), 4),
                "facets": node.get("facets", {}),
            }
        except Exception as e:
            return {"salience": None, "resonance": None, "facets": {}, "error": str(e)}

    seed_payload = []
    for bucket in seed_buckets:
        bucket_id = bucket.get("id", "")
        values = node_values(bucket_id, bucket)
        seed_payload.append(
            {
                "bucket_id": bucket_id,
                "name": _inspect_bucket_label(bucket, bucket_id),
                "source": bucket.get("_inspect_source", "keyword"),
                "seed_score": round(float(seed_scores.get(bucket_id, 0.0)), 4),
                **values,
            }
        )

    hit_payload = []
    for hit in hits:
        bucket = bucket_map.get(hit.bucket_id)
        values = node_values(hit.bucket_id, bucket)
        hit_payload.append(
            {
                "bucket_id": hit.bucket_id,
                "name": _inspect_bucket_label(bucket, hit.bucket_id),
                "score": hit.activation,
                **values,
                "path": format_diffusion_trace(hit.best_path, bucket_map, use_labels=True),
                "path_ids": list(hit.best_path.nodes),
                "caution": path_has_caution(hit.best_path),
                "paths": [_inspect_path_payload(path, bucket_map) for path in hit.paths],
            }
        )

    return {
        "status": "ok",
        "query": query,
        "node_facets_enabled": node_facets_enabled,
        "options": {
            "max_hops": options.max_hops,
            "top_k": options.top_k,
            "min_activation": options.min_activation,
            "edge_min_confidence": edge_min_confidence,
            "include_incoming": options.include_incoming,
        },
        "query_facets": query_facets,
        "seeds": seed_payload,
        "hits": hit_payload,
        "warnings": warnings,
    }


def _inspect_moment_payload(moment: dict, *, include_text: bool) -> dict:
    text = str(moment.get("text") or "")
    payload = {
        "moment_id": moment.get("moment_id"),
        "bucket_id": moment.get("bucket_id"),
        "section": moment.get("section"),
        "ordinal": moment.get("ordinal"),
        "source": moment.get("source"),
        "source_id": moment.get("source_id"),
        "text_hash": moment.get("text_hash"),
        "text_length": len(text),
        "metadata": moment.get("metadata", {}),
        "created_at": moment.get("created_at"),
        "updated_at": moment.get("updated_at"),
    }
    if include_text:
        payload["text"] = text
    else:
        payload["text_preview"] = _clip_text(" ".join(text.split()), 240)
    return payload


@mcp.tool()
async def inspect_moments(bucket_id: str = "", limit: int = 20) -> dict:
    """只读诊断 bucket 如何被拆成 moment；写入/刷新 SQLite 索引，不 touch bucket。"""
    bucket_id = str(bucket_id or "").strip()
    limit = _int_between(limit, 20, 1, 200)

    if bucket_id:
        if not MEMORY_ID_RE.fullmatch(bucket_id):
            return {"status": "error", "error": "invalid bucket_id"}
        bucket = await bucket_mgr.get(bucket_id)
        if not bucket:
            return {"status": "error", "error": "not_found", "bucket_id": bucket_id}
        moments = memory_moment_store.upsert_bucket(bucket)
        edges = memory_moment_store.list_edges(bucket_id)
        meta = bucket.get("metadata", {}) or {}
        return {
            "status": "ok",
            "mode": "bucket",
            "bucket_id": bucket_id,
            "name": str(meta.get("name") or bucket.get("name") or bucket_id),
            "count": len(moments),
            "edge_count": len(edges),
            "db_path": memory_moment_store.db_path,
            "moments": [
                _inspect_moment_payload(moment, include_text=True)
                for moment in moments[:limit]
            ],
            "edges": edges[:limit],
        }

    buckets = await bucket_mgr.list_all(include_archive=False)
    indexed = memory_moment_store.bulk_upsert(buckets)
    stats = memory_moment_store.stats()
    sample = memory_moment_store.sample(limit)
    return {
        "status": "ok",
        "mode": "bulk",
        "indexed_buckets": indexed["buckets"],
        "indexed_moments": indexed["moments"],
        "total_buckets": stats["buckets"],
        "total_moments": stats["moments"],
        "total_edges": stats.get("edges", 0),
        "db_path": memory_moment_store.db_path,
        "sample": [
            _inspect_moment_payload(moment, include_text=False)
            for moment in sample
        ],
    }


def _node_facets_enabled(cfg: dict | None) -> bool:
    node_cfg = (cfg or {}).get("node_facets", {}) or {}
    if isinstance(node_cfg, dict):
        return _bool_value(node_cfg.get("enabled", True), True)
    return True


def _node_salience_lookup(bucket_id: str, bucket: dict) -> float:
    return memory_node_store.node_salience(bucket_id, bucket)


def _node_resonance_lookup(query_facets: dict):
    if not _has_active_facets(query_facets):
        return None

    def lookup(bucket_id: str, bucket: dict) -> float:
        return memory_node_store.node_resonance(bucket_id, query_facets, bucket)

    return lookup


def _has_active_facets(facets: dict | None) -> bool:
    for value in (facets or {}).values():
        if isinstance(value, dict):
            if any(float(item or 0) > 0 for item in value.values()):
                return True
        else:
            try:
                if float(value) > 0:
                    return True
            except (TypeError, ValueError):
                continue
    return False


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
    is_session_start: bool = False,
    debug: bool = False,
) -> str:
    """读取记忆,不写入。
    调用方式: 新对话用 breath(is_session_start=True); 查过去用 breath(query="主题词"); 只读模型感受用 breath(domain="feel"); 只读悄悄话用 breath(domain="whisper")。
    默认只从本次命中的普通记忆沿持久化 memory_edges 带一跳联想浮现; embedding 相似边只是检索/图谱参考,不是可手写的记忆关系。
    如果夜梦与当前语境共振,breath 会追加 ===== 梦境 ===== 块;梦只浮现一次。
    include_core/core_limit 控制 pinned/protected 核心准则数量; include_related=False 可关闭联想浮现块。
    """
    await decay_engine.ensure_started()
    max_results = _int_between(max_results, 20, 1, 50)
    max_tokens = _int_between(max_tokens, 10000, 0, 20000)
    include_related = _bool_value(include_related, True)
    related_per_memory = _int_between(related_per_memory, 1, 0, 5)
    edge_min_confidence = _float_between(edge_min_confidence, 0.55, 0.0, 1.0)
    include_core = _bool_value(include_core, True)
    core_limit = _int_between(core_limit, 3, 0, 20)
    is_session_start = _bool_value(is_session_start, False)
    debug = _bool_value(debug, False)
    domain_key = domain.strip().lower()

    # --- Feel/whisper retrieval: independent read-only channels ---
    # --- Feel/whisper 检索：独立只读入口 ---
    if domain_key in {"feel", "whisper"}:
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
            feels = [b for b in all_buckets if b["metadata"].get("type") == "feel"]
            if domain_key == "whisper":
                feels = [
                    b for b in feels
                    if "whisper" in {str(tag).lower() for tag in b["metadata"].get("tags", []) or []}
                ]
            feels.sort(key=lambda b: b["metadata"].get("created", ""), reverse=True)
            if not feels:
                return "没有留下过 whisper。" if domain_key == "whisper" else "没有留下过 feel。"
            results = []
            for f in feels:
                created = f["metadata"].get("created", "")
                entry = f"[{created}] [bucket_id:{f['id']}]\n{strip_wikilinks(f['content'])}"
                results.append(entry)
                if count_tokens_approx("\n---\n".join(results)) > max_tokens:
                    break
            title = "whisper" if domain_key == "whisper" else "feel"
            return f"=== 你留下的 {title} ===\n" + "\n---\n".join(results)
        except Exception as e:
            logger.error(f"Feel retrieval failed: {e}")
            return "读取 whisper 失败。" if domain_key == "whisper" else "读取 feel 失败。"

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
        selected_anchors = _select_anchor_buckets(all_buckets, limit=min(2, max_results))

        # --- Unresolved buckets: surface top N by weight ---
        # --- 未解决桶：按权重浮现前 N 条 ---
        unresolved = [
            b for b in all_buckets
            if not b["metadata"].get("resolved", False)
            and b["metadata"].get("type") not in ("permanent", "feel")
            and not b["metadata"].get("anchor", False)
            and not b["metadata"].get("pinned", False)
            and not b["metadata"].get("protected", False)
        ]

        logger.info(
            f"Breath surfacing: {len(all_buckets)} total, "
            f"{len(core_candidates)} core, {len(selected_anchors)} anchors, {len(unresolved)} unresolved"
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
                summary = await dehydrator.dehydrate(_bucket_text_for_embedding(b), clean_meta)
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

        anchor_results = []
        anchor_buckets = []
        anchor_token_budget = min(token_budget, max(0, int(max_tokens * 0.18)))
        for b in selected_anchors:
            if anchor_token_budget <= 0 or token_budget <= 0:
                break
            try:
                clean_meta = {k: v for k, v in b["metadata"].items() if k != "tags"}
                summary = await dehydrator.dehydrate(_bucket_text_for_embedding(b), clean_meta)
                entry = f"⚓ [长期锚点] [bucket_id:{b['id']}] {summary}"
                entry_tokens = count_tokens_approx(entry)
                if entry_tokens > anchor_token_budget or entry_tokens > token_budget:
                    break
                anchor_results.append(entry)
                anchor_buckets.append(b)
                anchor_token_budget -= entry_tokens
                token_budget -= entry_tokens
            except Exception as e:
                logger.warning(f"Failed to dehydrate anchor bucket / anchor 桶脱水失败: {e}")
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
                summary = await dehydrator.dehydrate(_bucket_text_for_embedding(b), clean_meta)
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
        related_sources = anchor_buckets + surfaced_buckets
        if include_related and related_sources:
            related_header_tokens = count_tokens_approx("=== 联想浮现 ===\n")
            related_block = await _build_mcp_diffused_memory_block(
                related_sources,
                all_buckets,
                max(0, token_budget - related_header_tokens),
                related_per_memory,
                edge_min_confidence,
                "",
            )

        parts = []
        if core_results:
            parts.append("=== 核心准则 ===\n" + "\n---\n".join(core_results))
        if anchor_results:
            parts.append("=== 长期锚点 ===\n" + "\n---\n".join(anchor_results))
        if dynamic_results:
            parts.append("=== 浮现记忆 ===\n" + "\n---\n".join(dynamic_results))
        if related_block:
            parts.append("=== 联想浮现 ===\n" + related_block)

        dream_block = await dream_engine.surface_for_breath(
            query="",
            valence=valence,
            arousal=arousal,
            is_session_start=is_session_start,
            embedding_engine=embedding_engine,
        )
        if dream_block:
            parts.append(dream_block)

        if not parts:
            return "权重池平静，没有需要处理的记忆。"
        return "\n\n".join(parts)

    # --- With args: search mode (keyword + vector dual channel) ---
    # --- 有参数：检索模式（关键词 + 向量双通道）---
    domain_filter = [d.strip() for d in domain.split(",") if d.strip()] or None
    q_valence = valence if 0 <= valence <= 1 else None
    q_arousal = arousal if 0 <= arousal <= 1 else None
    search_query = recall_search_query(query, _recall_relevance_options())

    try:
        matches = await bucket_mgr.search(
            search_query,
            limit=max(max_results, 20),
            domain_filter=domain_filter,
            query_valence=q_valence,
            query_arousal=q_arousal,
        )
    except Exception as e:
        logger.error(f"Search failed / 检索失败: {e}")
        return "检索过程出错，请稍后重试。"

    seed_diagnostics: dict[str, dict] = {}
    for bucket in matches:
        _upsert_breath_seed_diagnostic(
            seed_diagnostics,
            bucket,
            "keyword",
            bucket_search_score=bucket.get("score"),
        )

    # --- Vector similarity channel: find semantically related buckets ---
    # --- 向量相似度通道：找到语义相关的桶 ---
    recall_thresholds = _breath_recall_thresholds(query, max_results)
    matched_ids = {b["id"] for b in matches}
    try:
        vector_results = await embedding_engine.search_similar(
            search_query,
            top_k=int(recall_thresholds["semantic_top_k"]),
        )
        for bucket_id, sim_score in vector_results:
            if bucket_id in seed_diagnostics:
                seed_diagnostics[bucket_id]["embedding_score"] = round(float(sim_score), 4)
            if bucket_id not in matched_ids and sim_score >= recall_thresholds["vector_min_score"]:
                bucket = await bucket_mgr.get(bucket_id)
                if bucket:
                    if bucket.get("metadata", {}).get("type") == "feel":
                        continue
                    bucket["score"] = round(sim_score * 100, 2)
                    bucket["vector_match"] = True
                    _upsert_breath_seed_diagnostic(
                        seed_diagnostics,
                        bucket,
                        "vector",
                        embedding_score=sim_score,
                    )
                    matches.append(bucket)
                    matched_ids.add(bucket_id)
    except Exception as e:
        logger.warning(f"Vector search failed, using keyword only / 向量搜索失败: {e}")

    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
    except Exception as e:
        logger.warning(f"Failed to list buckets for moment recall / moment 召回列桶失败: {e}")
        all_buckets = matches

    bucket_map = {bucket["id"]: bucket for bucket in all_buckets if bucket.get("id")}
    _, grouped_moments, _ = await _refresh_moment_graph(all_buckets)
    bucket_boosts = seed_scores_for_buckets(matches)
    moment_candidates = memory_moment_store.search_moments(
        search_query,
        limit=max(max_results, 20),
        bucket_boosts=bucket_boosts,
    )
    moment_candidates = _direct_recallable_moments(moment_candidates)
    pre_gate_moment_candidates = list(moment_candidates)
    gated_moment_candidates = _apply_recall_relevance_gate(query, moment_candidates)
    moment_candidates = gated_moment_candidates
    moment_candidates = await _rerank_breath_moment_candidates(query, moment_candidates)
    reranked_moment_candidates = list(moment_candidates)
    admitted_moments = []
    suppressed_moments = []
    for moment in moment_candidates:
        admission = _breath_moment_admission_decision(query, moment, seed_diagnostics)
        item = dict(moment)
        item["_admission_reason"] = admission.reason
        if admission.admit:
            admitted_moments.append(item)
        else:
            suppressed_moments.append(item)
    moment_candidates = admitted_moments

    direct_results = []
    token_used = 0
    returned_moments = moment_candidates[:max_results]
    direct_display_limit = 1 if include_related else max_results
    displayed_bucket_ids: set[str] = set()
    displayed_moment_ids: list[str] = []
    for moment in returned_moments:
        if len(direct_results) >= direct_display_limit:
            break
        if token_used >= max_tokens:
            break
        bucket_id = str(moment.get("bucket_id") or "")
        if bucket_id in displayed_bucket_ids:
            continue
        try:
            entry = _format_direct_moment(moment, grouped_moments, max_tokens - token_used)
            if not entry:
                break
            entry_tokens = count_tokens_approx(entry)
            if token_used + entry_tokens > max_tokens:
                break
            await bucket_mgr.touch(bucket_id)
            displayed_bucket_ids.add(bucket_id)
            displayed_moment_ids.append(str(moment.get("moment_id") or ""))
            direct_results.append(entry)
            token_used += entry_tokens
        except Exception as e:
            logger.warning(f"Failed to render direct moment / 直接命中片段渲染失败: {e}")
            continue

    related_entry = ""
    secondary_moment_ids: list[str] = []
    related_source_bucket_ids: list[str] = []
    if include_related and returned_moments:
        related_header = "=== 联想浮现 ===\n"
        related_budget = max_tokens - token_used - count_tokens_approx(related_header)
        related_parts = []
        secondary_moments = _secondary_direct_moments(
            query,
            returned_moments,
            displayed_bucket_ids,
            _secondary_direct_limit(query, related_per_memory),
        )
        for moment in secondary_moments:
            if related_budget <= 0:
                break
            block = _format_secondary_direct_moment(moment)
            block_tokens = count_tokens_approx(block)
            if block_tokens > related_budget:
                break
            related_parts.append(block)
            secondary_moment_ids.append(str(moment.get("moment_id") or ""))
            related_budget -= block_tokens

        related_source_buckets = []
        seen_source_bucket_ids = set()
        for moment in returned_moments:
            bucket_id = str(moment.get("bucket_id") or "")
            if bucket_id not in displayed_bucket_ids:
                continue
            bucket = bucket_map.get(bucket_id)
            if not bucket or bucket_id in seen_source_bucket_ids:
                continue
            related_source_buckets.append(bucket)
            related_source_bucket_ids.append(bucket_id)
            seen_source_bucket_ids.add(bucket_id)

        related_block = await _build_mcp_diffused_memory_block(
            related_source_buckets,
            all_buckets,
            max(0, related_budget),
            related_per_memory,
            edge_min_confidence,
            query_text=query,
            exclude_bucket_ids={str(moment.get("bucket_id") or "") for moment in secondary_moments},
        )
        if related_block:
            related_parts.append(related_block)
        if related_parts:
            related_entry = related_header + "\n---\n".join(related_parts)
            token_used += count_tokens_approx(related_entry)

    drift_entry = ""
    # --- Resurface: when search returns < 3, 40% chance to float dormant memories ---
    # --- 久未触碰浮现：检索结果不足 3 条时，40% 概率漂起旧桶 ---
    if (
        not related_entry
        and len(returned_moments) < 3
        and not recall_thresholds.get("has_explicit_entity")
        and max_tokens > token_used
        and random.random() < 0.4
    ):
        try:
            matched_ids = {str(moment.get("bucket_id")) for moment in returned_moments}
            drifted = await _select_resurface_buckets(
                max_results=random.randint(1, 3),
                exclude_ids=matched_ids,
                include_archive=True,
            )
            if drifted:
                drift_results = []
                drift_remaining = (
                    max_tokens
                    - token_used
                    - count_tokens_approx("--- 久未碰过 ---\n")
                )
                for b in drifted:
                    if drift_remaining <= 0:
                        break
                    clean_meta = {k: v for k, v in b["metadata"].items() if k != "tags"}
                    summary = await dehydrator.dehydrate(_bucket_text_for_embedding(b), clean_meta)
                    dormant_days = _bucket_days_since_last_active(b["metadata"])
                    entry = f"[surface_type: resurface, dormant_days={dormant_days:.0f}]\n{summary}"
                    entry_tokens = count_tokens_approx(entry)
                    if entry_tokens > drift_remaining:
                        break
                    drift_results.append(entry)
                    drift_remaining -= entry_tokens
                if drift_results:
                    drift_entry = "--- 久未碰过 ---\n" + "\n---\n".join(drift_results)
                    if token_used + count_tokens_approx(drift_entry) <= max_tokens:
                        token_used += count_tokens_approx(drift_entry)
                    else:
                        drift_entry = ""
        except Exception as e:
            logger.warning(f"Resurface failed / 久未触碰浮现失败: {e}")

    dream_block = await dream_engine.surface_for_breath(
        query=query,
        valence=valence,
        arousal=arousal,
        is_session_start=is_session_start,
        embedding_engine=embedding_engine,
    )

    response_parts = []
    response_sections = []
    if direct_results:
        response_parts.append("=== 直接命中记忆 ===\n" + "\n---\n".join(direct_results))
        response_sections.append("direct")
    if related_entry:
        response_parts.append(related_entry)
        response_sections.append("related")
    if drift_entry:
        response_parts.append(drift_entry)
        response_sections.append("drift")
    if dream_block:
        response_sections.append("dream")

    _write_breath_recall_diagnostics(
        query=query,
        recall_thresholds=recall_thresholds,
        seed_diagnostics=seed_diagnostics,
        pre_gate_candidates=pre_gate_moment_candidates,
        gated_candidates=gated_moment_candidates,
        reranked_candidates=reranked_moment_candidates,
        returned_moments=returned_moments,
        suppressed_candidates=suppressed_moments,
        displayed_moment_ids=displayed_moment_ids,
        secondary_moment_ids=secondary_moment_ids,
        related_source_bucket_ids=related_source_bucket_ids,
        related_included=bool(related_entry),
        drift_included=bool(drift_entry),
        dream_included=bool(dream_block),
        response_sections=response_sections,
    )

    if debug and suppressed_moments:
        response_parts.append(
            "=== suppressed_candidates ===\n"
            + "\n".join(_format_suppressed_recall_candidate(moment, seed_diagnostics) for moment in suppressed_moments[:10])
        )

    if not response_parts:
        if recall_thresholds.get("has_explicit_entity") and suppressed_moments:
            return dream_block or "没有找到可靠命中。"
        return dream_block or "未找到相关记忆。"

    response_text = "\n\n".join(response_parts)
    if dream_block:
        response_text += "\n\n" + dream_block
    return response_text


async def _select_resurface_buckets(
    max_results: int = 1,
    *,
    exclude_ids: set[str] | None = None,
    include_archive: bool = True,
) -> list[dict]:
    exclude_ids = exclude_ids or set()
    max_results = max(1, min(5, int(max_results or 1)))
    all_buckets = await bucket_mgr.list_all(include_archive=include_archive)
    candidates = []
    for bucket in all_buckets:
        meta = bucket.get("metadata", {})
        if bucket.get("id") in exclude_ids:
            continue
        if meta.get("type") in {"feel", "permanent"}:
            continue
        if meta.get("pinned") or meta.get("protected"):
            continue
        if meta.get("anchor"):
            continue
        dormant_days = _bucket_days_since_last_active(meta)
        importance = max(1, min(10, int(meta.get("importance", 5))))
        archived_bonus = 1.15 if meta.get("type") == "archived" else 1.0
        resurface_score = (dormant_days + 1.0) * (0.6 + importance / 10.0) * archived_bonus
        candidates.append((resurface_score, bucket))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [bucket for _, bucket in candidates[:max_results]]


# =============================================================
# Tool 1.4: resurface — dormant memory resurfacing
# 工具 1.4：resurface — 久未触碰记忆浮现
# =============================================================
@mcp.tool()
async def resurface(max_results: int = 1, include_archive: bool = True, max_tokens: int = 800) -> str:
    """只读浮现久未触碰的旧记忆。越久没碰过越靠前；默认包含归档桶；不 touch,不刷新 last_active,不增加 activation_count。"""
    try:
        buckets = await _select_resurface_buckets(
            max_results=max_results,
            include_archive=include_archive,
        )
    except Exception as e:
        logger.error(f"Resurface listing failed / 久未触碰浮现列桶失败: {e}")
        return "旧记忆暂时无法浮现。"

    if not buckets:
        return "没有可浮现的旧记忆。"

    parts = []
    remaining = max(100, max_tokens)
    for bucket in buckets:
        meta = bucket.get("metadata", {})
        dormant_days = _bucket_days_since_last_active(meta)
        state = []
        if meta.get("type") == "archived":
            state.append("归档")
        if meta.get("resolved"):
            state.append("已解决")
        if meta.get("digested"):
            state.append("已消化")
        state_text = f" ({', '.join(state)})" if state else ""
        entry = (
            f"[bucket_id:{bucket['id']}] {meta.get('name', bucket['id'])}{state_text} "
            f"久未触碰 {dormant_days:.0f} 天\n"
            f"{_bucket_text_for_embedding(bucket).strip()[:420]}"
        )
        tokens = count_tokens_approx(entry)
        if tokens > remaining and parts:
            break
        parts.append(entry)
        remaining -= tokens
        if remaining <= 0:
            break

    return "=== 久未触碰的旧记忆 ===\n" + "\n---\n".join(parts)


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
# Tool 1.6: comment_bucket — add a ring/comment to a memory
# 工具 1.6：comment_bucket — 给记忆追加年轮
# =============================================================
@mcp.tool()
async def comment_bucket(
    bucket_id: str,
    content: str,
    kind: str = "comment",
    valence: float = -1,
    arousal: float = -1,
) -> dict:
    """给已有 bucket 追加一条年轮并 touch+1。再次读到旧记忆时的感受/补充请优先用这个工具；不会改正文，也不会把源记忆标记为 digested。"""
    bucket_id = (bucket_id or "").strip()
    if not bucket_id or not MEMORY_ID_RE.fullmatch(bucket_id):
        return {"error": "invalid bucket_id"}
    if not content or not content.strip():
        return {"error": "empty content"}
    if not await bucket_mgr.get(bucket_id):
        return {"error": "not found", "id": bucket_id}

    entry = await bucket_mgr.add_comment(
        bucket_id,
        content,
        author=_ai_author_name(),
        kind=kind or "comment",
        valence=valence if 0 <= valence <= 1 else None,
        arousal=arousal if 0 <= arousal <= 1 else None,
        source="comment_bucket",
        touch=True,
    )
    if not entry:
        return {"error": "write failed", "id": bucket_id}
    bucket = await bucket_mgr.get(bucket_id)
    embedding_queued = _queue_embedding_refresh(bucket_id)
    return {
        "status": "commented",
        "id": bucket_id,
        "comment": entry,
        "embedding_refreshed": False,
        "embedding_queued": embedding_queued,
        "metadata": _bucket_read_payload(bucket)["metadata"] if bucket else {},
    }


@mcp.custom_route("/api/bucket/{bucket_id}/comments", methods=["POST"])
async def api_bucket_comment(request):
    """Add a dashboard-authenticated user comment to a bucket."""
    from starlette.responses import JSONResponse

    err = _require_dashboard_auth(request)
    if err:
        return err

    bucket_id = request.path_params["bucket_id"]
    if not bucket_id or not MEMORY_ID_RE.fullmatch(bucket_id):
        return JSONResponse({"error": "invalid bucket_id"}, status_code=400)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "json body must be an object"}, status_code=400)

    content = str(body.get("content") or "").strip()
    if not content:
        return JSONResponse({"error": "empty content"}, status_code=400)
    if not await bucket_mgr.get(bucket_id):
        return JSONResponse({"error": "not found", "id": bucket_id}, status_code=404)

    valence = _float_between(body.get("valence"), -1.0)
    arousal = _float_between(body.get("arousal"), -1.0)
    entry = await bucket_mgr.add_comment(
        bucket_id,
        content,
        author=_dashboard_author_name(),
        kind=str(body.get("kind") or "comment"),
        valence=valence if 0 <= valence <= 1 else None,
        arousal=arousal if 0 <= arousal <= 1 else None,
        source="dashboard",
        touch=True,
    )
    if not entry:
        return JSONResponse({"error": "write failed", "id": bucket_id}, status_code=500)

    embedding_queued = _queue_embedding_refresh(bucket_id)

    bucket = await bucket_mgr.get(bucket_id)
    return JSONResponse({
        "status": "commented",
        "id": bucket_id,
        "comment": entry,
        "embedding_refreshed": False,
        "embedding_queued": embedding_queued,
        "metadata": _bucket_read_payload(bucket)["metadata"] if bucket else {},
    })


@mcp.custom_route("/api/bucket/{bucket_id}/comments/{comment_id}", methods=["DELETE"])
async def api_bucket_comment_delete(request):
    """Delete a dashboard-authenticated user comment from a bucket."""
    from starlette.responses import JSONResponse

    err = _require_dashboard_auth(request)
    if err:
        return err

    bucket_id = request.path_params["bucket_id"]
    comment_id = request.path_params["comment_id"]
    if not bucket_id or not MEMORY_ID_RE.fullmatch(bucket_id):
        return JSONResponse({"error": "invalid bucket_id"}, status_code=400)
    if not comment_id or not MEMORY_ID_RE.fullmatch(comment_id):
        return JSONResponse({"error": "invalid comment_id"}, status_code=400)
    if not await bucket_mgr.get(bucket_id):
        return JSONResponse({"error": "not found", "id": bucket_id}, status_code=404)

    result = await bucket_mgr.delete_comment(
        bucket_id,
        comment_id,
        allowed_author=_dashboard_author_name(),
        allowed_source="dashboard",
    )
    if result.get("status") == "not_found":
        return JSONResponse({"error": "comment not found"}, status_code=404)
    if result.get("status") == "forbidden":
        return JSONResponse({"error": "only dashboard user comments can be deleted"}, status_code=403)
    if result.get("status") != "deleted":
        return JSONResponse({"error": "delete failed"}, status_code=500)

    embedding_queued = _queue_embedding_refresh(bucket_id)

    bucket = await bucket_mgr.get(bucket_id)
    return JSONResponse({
        "status": "deleted",
        "id": bucket_id,
        "comment_id": comment_id,
        "embedding_refreshed": False,
        "embedding_queued": embedding_queued,
        "metadata": _bucket_read_payload(bucket)["metadata"] if bucket else {},
    })


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
    whisper: bool = False,
    source_bucket: str = "",
    valence: float = -1,
    arousal: float = -1,
) -> str:
    """写入一条长期记忆卡,不是聊天流水、运维记录或整篇日记。写前应先用 breath/read_bucket 查重。
    普通事实: hold(content="YYYY-MM-DD, 当前用户...", tags="relationship_event 或 project_event", importance=5-7)。
    承诺/待办: tags 传 "commitment,todo" 或 "commitment,wish"; content 写清谁答应了什么、何时/什么条件下要继续。
    给旧记忆写年轮/再次阅读感受: 优先用 comment_bucket(bucket_id="...", content="...", kind="feel", valence=0.x, arousal=0.x)。
    无源记忆的碎碎念/悄悄话: 用 hold(content="...", whisper=True, valence=0.x, arousal=0.x),会存为独立 feel 并打 whisper 标签。
    新记忆本身值得偏爱: tags 可传 "haven_favorite,flavor_偏爱"; content 必须包含很短的 "### 喜欢它的原因" 段落。
    普通写入会新建 bucket,写 embedding,后台触发 ReflectionEngine 补 tags/confidence/memory_edges,并返回一条只读相关旧记忆。
    pinned=True 只给极少数核心准则,技术进度和运维细节不要钉选。
    feel=True 且带 source_bucket 是旧兼容入口,新调用不要使用；feel=True 但没有 source_bucket 会转为 whisper。
    """
    await decay_engine.ensure_started()

    # --- Input validation / 输入校验 ---
    if not content or not content.strip():
        return "内容为空，无法存储。"

    importance = max(1, min(10, importance))
    extra_tags = [t.strip() for t in tags.split(",") if t.strip()]

    async def create_whisper_bucket() -> str:
        whisper_valence = valence if 0 <= valence <= 1 else 0.5
        whisper_arousal = arousal if 0 <= arousal <= 1 else 0.3
        whisper_tags = list(dict.fromkeys(extra_tags + ["whisper"]))
        bucket_id = await bucket_mgr.create(
            content=content,
            tags=whisper_tags,
            importance=5,
            domain=[],
            valence=whisper_valence,
            arousal=whisper_arousal,
            name=None,
            bucket_type="feel",
        )
        _queue_embedding_refresh(bucket_id)
        return f"🫧whisper→{bucket_id}"

    if whisper:
        if source_bucket and source_bucket.strip():
            return "whisper 不需要 source_bucket；有源记忆的感受请用 comment_bucket。"
        return await create_whisper_bucket()

    # --- Feel mode: attach to source bucket as a ring comment when possible ---
        # --- Feel 模式：有源记忆时挂成年轮 ---
    if feel:
        # Feel valence/arousal = model's own perspective
        feel_valence = valence if 0 <= valence <= 1 else 0.5
        feel_arousal = arousal if 0 <= arousal <= 1 else 0.3
        source_id = (source_bucket or "").strip()
        if source_id:
            if not MEMORY_ID_RE.fullmatch(source_id):
                return "source_bucket 无效。"
            source = await bucket_mgr.get(source_id)
            if not source:
                return f"源记忆不存在: {source_id}"
            entry = await bucket_mgr.add_comment(
                source_id,
                content,
                author=_ai_author_name(),
                kind="feel",
                valence=feel_valence,
                arousal=feel_arousal,
                source="hold(feel=True)",
                touch=True,
            )
            if not entry:
                return "年轮写入失败。"
            _queue_embedding_refresh(source_id)
            return f"年轮→{source_id}#{entry['id']}"

        # No source bucket: keep a standalone feel for compatibility.
        # 没有源记忆时保留独立 whisper，兼容旧用法。
        return await create_whisper_bucket()

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
    if _has_favorite_tag(all_tags) and not _has_favorite_reason(content):
        return _favorite_reason_error()

    # --- Pinned buckets bypass merge and are created directly in permanent dir ---
    # --- 钉选桶跳过合并，直接新建到 permanent 目录 ---
    if pinned:
        related_bucket = await _find_readonly_related_bucket(content)
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
        _queue_embedding_refresh(bucket_id)
        _queue_memory_enrichment(bucket_id)
        related_note = _format_readonly_related_memory(related_bucket) if related_bucket else ""
        return f"📌钉选→{bucket_id} {','.join(domain)}{related_note}"

    # --- Step 2: merge or create / 合并或新建 ---
    bucket_id, result_name, is_merged, related_bucket = await _merge_or_create(
        content=content,
        tags=all_tags,
        importance=importance,
        domain=domain,
        valence=valence,
        arousal=arousal,
        name=suggested_name,
        allow_merge=False,
    )
    _queue_memory_enrichment(bucket_id)

    action = "合并→" if is_merged else "新建→"
    related_note = _format_readonly_related_memory(related_bucket) if related_bucket else ""
    return f"{action}{result_name} {','.join(domain)}{related_note}"


# =============================================================
# Tool 3: grow — Grow, fragments become memories
# 工具 3：grow — 生长，一天的碎片长成记忆
# =============================================================
def _format_write_gate_result(decision: WriteGateDecision) -> str:
    reason = ",".join(decision.reasons) or "no_reason"
    repeat = f"{decision.repeat_count + 1}/{memory_write_gate.repeat_promote_count}"
    return (
        f"门卫→{decision.decision} "
        f"score={decision.surprise_score:.2f} "
        f"repeat={repeat} "
        f"candidate={decision.candidate_id} "
        f"reason={reason}"
    )


def _grow_source_from_context(context: Context | None) -> str:
    if context is None:
        return ""
    try:
        client_info = context.request_context.session.client_params.clientInfo
    except Exception:
        return ""
    name = str(getattr(client_info, "name", "") or "").strip().lower()
    if "ob-auto-grow" in name:
        return "operit"
    if "operit" in name:
        return "operit"
    return ""


def _looks_like_operit_auto_grow_content(content: str) -> bool:
    return bool(re.match(r"^【\d{4}-\d{2}-\d{2} \d{2}:\d{2}】\s*\n", str(content or "")))


@mcp.tool()
async def grow(content: str, auto: bool = False, source: str = "", context: Context | None = None) -> str:
    """长内容摘记: 只给已经筛过、包含多个长期记忆点的片段; 不要把整篇日终日记、一天流水或完整情绪过程丢进来。
    content 应该是少量可长期召回的事实/偏好/承诺/项目状态; 服务端会拆成少量 bucket、写 embedding,并后台触发 enrich。
    如果只有单条明确事实,优先用 hold。若要给旧记忆追加年轮/喜欢原因,优先用 comment_bucket；若要改正文,先 read_bucket 再 trace(content=完整新正文)。
    Operit/worker 等自动总结入口传 source="operit" 或 auto=True，会先过轻量 surprise 门卫：低分只记日志，中分 pending，足够高或重复出现才真正写入。
    短内容(<30字)会走 hold-like 快速路径。
    """
    await decay_engine.ensure_started()

    if not content or not content.strip():
        return "内容为空，无法整理。"

    auto = _bool_value(auto, False)
    source = str(source or "").strip() or _grow_source_from_context(context)
    if not source and _looks_like_operit_auto_grow_content(content):
        source = "operit"
    gate_decision = None
    if memory_write_gate.should_gate(auto=auto, source=source):
        gate_decision = await memory_write_gate.evaluate(
            content,
            source=source,
            bucket_mgr=bucket_mgr,
            auto=auto,
        )
        if not gate_decision.allow:
            return _format_write_gate_result(gate_decision)
    gate_prefix = f"{_format_write_gate_result(gate_decision)}\n" if gate_decision else ""

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
        fast_tags = analysis.get("tags", [])
        if _has_favorite_tag(fast_tags) and not _has_favorite_reason(content):
            return _favorite_reason_error()
        bucket_id, result_name, is_merged, related_bucket = await _merge_or_create(
            content=content.strip(),
            tags=fast_tags,
            importance=analysis.get("importance", 5) if isinstance(analysis.get("importance"), int) else 5,
            domain=analysis.get("domain", ["未分类"]),
            valence=analysis.get("valence", 0.5),
            arousal=analysis.get("arousal", 0.3),
            name=analysis.get("suggested_name", ""),
            allow_merge=False,
        )
        _queue_memory_enrichment(bucket_id)
        action = "合并" if is_merged else "新建"
        related_note = _format_readonly_related_memory(related_bucket) if related_bucket else ""
        return f"{gate_prefix}{action} → {result_name} | {','.join(analysis.get('domain', []))} V{analysis.get('valence', 0.5):.1f}/A{analysis.get('arousal', 0.3):.1f}{related_note}"

    # --- Step 1: let API split and organize / 让 API 拆分整理 ---
    try:
        items = await dehydrator.digest(content)
    except Exception as e:
        logger.error(f"Memory digest failed / 长内容摘记失败: {e}")
        return f"{gate_prefix}长内容摘记失败: {e}"

    if not items:
        return f"{gate_prefix}内容为空或整理失败。"

    results = []
    created = 0
    merged = 0

    # --- Step 2: merge or create each item (with per-item error handling) ---
    # --- 逐条合并或新建（单条失败不影响其他）---
    for item in items:
        try:
            item_tags = item.get("tags", [])
            if _has_favorite_tag(item_tags) and not _has_favorite_reason(item.get("content", "")):
                results.append("⚠️favorite 缺少喜欢它的原因")
                continue
            bucket_id, result_name, is_merged, related_bucket = await _merge_or_create(
                content=item["content"],
                tags=item_tags,
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

    return f"{gate_prefix}{len(items)}条|新{created}合{merged}\n" + "\n".join(results)


# =============================================================
# Tool 3.5: profile_fact — manually solidify a user/profile fact
# 工具 3.5：profile_fact — 手动固化画像事实
# =============================================================
@mcp.tool()
async def profile_fact(
    fact: str,
    evidence_bucket_id: str,
    profile_kind: str = "preference",
    subject: str = "user",
    predicate: str = "",
    object_value: str = "",
    evidence_moment_id: str = "",
    evidence_context: str = "",
    reflection: str = "",
    followup: str = "",
    confidence: float = 0.9,
) -> str:
    """手动写入一条画像事实，并强制关联证据桶。先有事件桶，再用这个工具固化稳定偏好/事实。"""
    fact = str(fact or "").strip()
    evidence_bucket_id = str(evidence_bucket_id or "").strip()
    if not fact:
        return "fact 为空，无法写入画像事实。"
    if not evidence_bucket_id or not MEMORY_ID_RE.fullmatch(evidence_bucket_id):
        return "请提供有效的 evidence_bucket_id。"

    evidence_bucket = await bucket_mgr.get(evidence_bucket_id)
    if not evidence_bucket:
        return f"证据记忆桶不存在: {evidence_bucket_id}"

    evidence_moment_id = str(evidence_moment_id or "").strip()
    if evidence_moment_id and not MEMORY_ID_RE.fullmatch(evidence_moment_id):
        return "evidence_moment_id 无效。"
    if not evidence_moment_id:
        try:
            evidence_moments = memory_moment_store.upsert_bucket(evidence_bucket)
            representative = _representative_moment(evidence_moments)
            evidence_moment_id = str((representative or {}).get("moment_id") or "")
        except Exception as e:
            logger.warning("Profile fact evidence moment indexing failed: %s", e)
            evidence_moment_id = ""

    kind = _profile_key(profile_kind, "preference")
    subject_key = _profile_key(subject, "user")
    predicate_key = _profile_key(predicate, "")
    object_text = str(object_value or "").strip()
    confidence = _float_between(confidence, 0.9, 0.0, 1.0)
    body = _profile_fact_body(
        fact=fact,
        evidence_context=evidence_context,
        reflection=reflection,
        followup=followup,
    )
    tags = ["profile_fact", f"profile_{kind}"]
    if predicate_key:
        tags.append(f"profile_predicate_{predicate_key}")
    evidence = {"bucket_id": evidence_bucket_id}
    if evidence_moment_id:
        evidence["moment_id"] = evidence_moment_id

    bucket_id = await bucket_mgr.create(
        content=body,
        tags=list(dict.fromkeys(tags)),
        importance=8,
        domain=list(dict.fromkeys(["profile", kind])),
        valence=0.5,
        arousal=0.3,
        name=_profile_fact_name(fact),
        bucket_type="permanent",
        confidence=confidence,
        source="profile_fact",
        extra_metadata={
            "profile_kind": kind,
            "subject": subject_key,
            "predicate": predicate_key,
            "object": object_text,
            "evidence": [evidence],
        },
    )
    edge = memory_edge_store.add_edge(
        bucket_id,
        evidence_bucket_id,
        "evidenced_by",
        confidence=confidence,
        reason="profile fact evidence",
    )
    _queue_embedding_refresh(bucket_id)
    try:
        created_bucket = await bucket_mgr.get(bucket_id)
        if created_bucket:
            memory_moment_store.upsert_bucket(created_bucket)
    except Exception as e:
        logger.warning("Profile fact moment indexing failed: %s", e)

    edge_note = " + evidenced_by" if edge else ""
    moment_note = f" moment={evidence_moment_id}" if evidence_moment_id else ""
    return f"profile_fact→{bucket_id} evidence→{evidence_bucket_id}{moment_note}{edge_note}"


def _profile_fact_body(
    *,
    fact: str,
    evidence_context: str = "",
    reflection: str = "",
    followup: str = "",
) -> str:
    sections = [("fact", fact)]
    if str(evidence_context or "").strip():
        sections.append(("evidence_context", str(evidence_context).strip()))
    if str(reflection or "").strip():
        sections.append(("reflection", str(reflection).strip()))
    if str(followup or "").strip():
        sections.append(("followup", str(followup).strip()))
    return "\n\n".join(f"### {heading}\n{text.strip()}" for heading, text in sections)


def _profile_key(value: str, default: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return default
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^0-9a-zA-Z_\-\u4e00-\u9fff]+", "", text)
    return text or default


def _profile_fact_name(fact: str) -> str:
    return "画像事实：" + _clip_text(fact, 48)


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
    给旧记忆补 "喜欢它的原因" 或 affect_anchor: 先 read_bucket,再 trace(content="旧正文 + 新段落")。
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

    effective_tags = updates.get("tags", bucket.get("metadata", {}).get("tags", []))
    effective_content = updates.get("content", bucket.get("content", ""))
    if _has_favorite_tag(effective_tags) and not _has_favorite_reason(effective_content):
        return _favorite_reason_error()

    success = await bucket_mgr.update(bucket_id, **updates)
    if not success:
        return f"修改失败: {bucket_id}"

    # Re-generate embedding if content or title changed.
    if "content" in updates or "name" in updates:
        _queue_embedding_refresh(bucket_id)

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
# Tool 6: introspection — waking self-reflection over recent memories
# 工具 6：introspection — 清醒自省，消化最近的记忆
#
# Reads recent surface-level buckets (≤10), returns them for
# Claude to introspect under prompt guidance.
# 读取最近新增的表层桶（≤10个），返回给 Claude 在提示词引导下自主思考。
# Claude then decides: resolve some, write feels, or do nothing.
# =============================================================
@mcp.tool()
async def introspection(
    limit: int = 10,
    offset: int = 0,
    created_date: str = "",
    created_from: str = "",
    created_to: str = "",
) -> str:
    """读取最近普通记忆供 AI 清醒自省,不是梦境生成,也不是日记整理。
    读后只在真的可以放下时 trace(resolved=1/digested=1),或在产生新的第一人称沉淀/喜欢原因时 comment_bucket(bucket_id, content)。
    limit/offset 可翻看更早的普通记忆; introspection(offset=10) 读取下一页。
    created_date="YYYY-MM-DD" 可读取某一天; created_from/created_to 可读取日期范围。
    不要把 introspection 返回内容直接再写成普通 bucket。
    """
    await decay_engine.ensure_started()
    limit = _int_between(limit, 10, 1, 30)
    offset = _int_between(offset, 0, 0, 10000)
    date_args = {
        "created_date": created_date,
        "created_from": created_from,
        "created_to": created_to,
    }
    if any(str(value or "").strip() and not _date_key(value) for value in date_args.values()):
        return '创建日期格式请用 YYYY-MM-DD, 例如 introspection(created_date="2026-05-24")。'

    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
    except Exception as e:
        logger.error(f"Introspection failed to list buckets: {e}")
        return "记忆系统暂时无法访问。"

    # --- Filter: recent surface-level dynamic buckets (not permanent/pinned/feel) ---
    candidates = [
        b for b in all_buckets
        if b["metadata"].get("type") not in ("permanent", "feel")
        and not b["metadata"].get("pinned", False)
        and not b["metadata"].get("protected", False)
    ]

    candidates, date_filter_label = _filter_by_created_date(
        candidates,
        created_date=created_date,
        created_from=created_from,
        created_to=created_to,
    )

    # --- Sort by creation time desc, take requested page ---
    candidates.sort(key=lambda b: b["metadata"].get("created", ""), reverse=True)
    recent = candidates[offset : offset + limit]

    if not recent:
        if date_filter_label:
            return "这个创建日期范围内没有需要消化的新记忆。"
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
            f"{_bucket_text_for_embedding(b)[:500]}"
        )

    header = (
        "=== Introspection ===\n"
        f"以下是你最近的普通记忆（offset={offset}, limit={limit}{date_filter_label}）。用第一人称想：\n"
        "- 这些东西里有什么在你这里留下了重量？\n"
        "- 有什么还没想清楚？\n"
        "- 有什么可以放下了？\n"
        "想完之后：值得放下的用 trace(bucket_id, resolved=1)；\n"
        "有沉淀的用 comment_bucket(bucket_id=\"bucket_id\", content=\"...\", kind=\"feel\", valence=你的感受) 写成年轮；有温度时可在 content 末尾加很短的 ### affect_anchor 和弦。\n"
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
            logger.warning(f"Introspection connection hint failed: {e}")

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
            logger.warning(f"Introspection crystallization hint failed: {e}")

    profile_hint = _profile_fact_candidate_hint(recent, all_buckets)

    return header + "\n---\n".join(parts) + connection_hint + crystal_hint + profile_hint


PROFILE_FACT_CANDIDATE_PATTERNS = (
    ("preference", "likes", "喜欢", re.compile(r"(?:小雨|池又雨|用户|她)\s*(?:很|最|一直|特别|偏)?喜欢\s*([^。；;，,\n]{1,32})")),
    ("preference", "dislikes", "不喜欢", re.compile(r"(?:小雨|池又雨|用户|她)\s*(?:很|最|一直|特别)?不喜欢\s*([^。；;，,\n]{1,32})")),
    ("preference", "dislikes", "讨厌", re.compile(r"(?:小雨|池又雨|用户|她)\s*(?:很|最|一直|特别)?讨厌\s*([^。；;，,\n]{1,32})")),
    ("preference", "dislikes", "厌恶", re.compile(r"(?:小雨|池又雨|用户|她)\s*(?:很|最|一直|特别)?厌恶\s*([^。；;，,\n]{1,32})")),
    ("preference", "fears", "害怕", re.compile(r"(?:小雨|池又雨|用户|她)\s*(?:很|最|一直|特别)?害怕\s*([^。；;，,\n]{1,32})")),
    ("preference", "prefers", "偏好", re.compile(r"(?:小雨|池又雨|用户|她)\s*偏好\s*([^。；;，,\n]{1,32})")),
    ("boundary", "boundary", "雷点", re.compile(r"(?:小雨|池又雨|用户|她)的?雷点是\s*([^。；;，,\n]{1,32})")),
    ("habit", "habit", "习惯", re.compile(r"(?:小雨|池又雨|用户|她)(?:有个)?习惯是\s*([^。；;，,\n]{1,32})")),
)


BASE_NOISY_PROFILE_OBJECT_KEYS = {
    "哥哥",
    "老公",
    "宝宝",
    "宝贝",
    "老婆",
    "亲爱的",
    "你",
    "你啦",
    "你呀",
}


def _profile_fact_candidate_hint(recent: list[dict], all_buckets: list[dict]) -> str:
    existing = _existing_profile_fact_keys(all_buckets)
    candidates = []
    seen = set()
    for bucket in recent:
        if len(candidates) >= 3:
            break
        meta = bucket.get("metadata", {}) or {}
        if "profile_fact" in {str(tag) for tag in meta.get("tags", []) or []}:
            continue
        text = strip_wikilinks(_bucket_text_for_embedding(bucket))
        for kind, predicate, verb, pattern in PROFILE_FACT_CANDIDATE_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue
            obj = _clean_profile_object(match.group(1))
            if not obj:
                continue
            if _is_noisy_profile_object(predicate, obj):
                continue
            fact = _render_profile_fact_candidate(predicate, verb, obj)
            key = _normalize_profile_fact_key(fact)
            if key in existing or key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "fact": fact,
                    "bucket_id": bucket.get("id", ""),
                    "profile_kind": kind,
                    "predicate": predicate,
                    "object_value": obj,
                    "evidence_context": _clip_text(text, 180),
                }
            )
            break
    if not candidates:
        return ""

    lines = [
        "\n=== 可能值得固化的画像事实 ===",
        "只作为候选，不会自动写入；确认后再调用 profile_fact(...)。",
    ]
    for item in candidates:
        args = [
            f"fact={_literal_arg(item['fact'])}",
            f"evidence_bucket_id={_literal_arg(item['bucket_id'])}",
            f"profile_kind={_literal_arg(item['profile_kind'])}",
            f"predicate={_literal_arg(item['predicate'])}",
            f"object_value={_literal_arg(item['object_value'])}",
            f"evidence_context={_literal_arg(item['evidence_context'])}",
        ]
        lines.append(
            f"- {item['fact']}\n"
            f"  证据桶: {item['bucket_id']}\n"
            f"  建议: profile_fact({', '.join(args)})"
        )
    return "\n" + "\n".join(lines) + "\n"


def _existing_profile_fact_keys(buckets: list[dict]) -> set[str]:
    keys = set()
    for bucket in buckets or []:
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        tags = {str(tag) for tag in meta.get("tags", []) or []}
        if "profile_fact" not in tags and not meta.get("profile_kind"):
            continue
        content = str(bucket.get("content") or "")
        for line in content.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                keys.add(_normalize_profile_fact_key(stripped))
                break
    return keys


def _render_profile_fact_candidate(predicate: str, verb: str, obj: str) -> str:
    if predicate == "boundary":
        return f"小雨的雷点是{obj}。"
    if predicate == "habit":
        return f"小雨的习惯是{obj}。"
    return f"小雨{verb}{obj}。"


def _clean_profile_object(value: str) -> str:
    text = strip_wikilinks(str(value or "")).strip()
    text = re.sub(r"^[“\"'「『（(]+|[”\"'」』）)]+$", "", text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"(这件事|这个设定|这类东西|的时候)$", "", text)
    return text[:32].strip("。；;，,、 ")


def _is_noisy_profile_object(predicate: str, obj: str) -> bool:
    key = _normalize_profile_fact_key(obj)
    if not key or len(key) <= 1:
        return True
    return predicate in {"likes", "dislikes", "prefers"} and key in _noisy_profile_object_keys()


def _noisy_profile_object_keys() -> set[str]:
    keys = set(BASE_NOISY_PROFILE_OBJECT_KEYS)
    ai_name = str(_identity().get("ai_name") or "").strip()
    if ai_name:
        keys.add(ai_name)
        keys.add(f"小{ai_name}")
    return {_normalize_profile_fact_key(key) for key in keys if str(key or "").strip()}


def _normalize_profile_fact_key(value: str) -> str:
    return re.sub(r"[\s。；;，,、：:\"'“”‘’「」『』]+", "", str(value or "").lower())


def _literal_arg(value: str) -> str:
    return _json_lib.dumps(str(value or ""), ensure_ascii=False)


@mcp.tool()
async def dream() -> str:
    """兼容旧客户端。旧 dream() 已改名为 introspection(); 夜梦由后台小模型自动生成。"""
    result = await introspection()
    return "dream() 已改名为 introspection()。夜梦由后台小模型自动生成，不需要主动调用工具。\n\n" + result


# =============================================================
# Tool 6: reflect — daily relationship weather
# 工具 6：reflect — 生成日印象
# =============================================================
@mcp.tool()
async def reflect(period: str = "daily", force: bool = False) -> dict:
    """生成 daily relationship_weather 类型的 feel,记录当天关系天气,正文会带 affect_anchor 和弦。weekly 默认关闭,需 reflection.weekly_enabled=true 才会生成; force=True 会重写同周期结果。它不会替代 hold/grow 写具体 bucket。"""
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

    now = _current_time_iso()
    domain = _string_list(body.get("domain"), ["未分类"])
    tags = _string_list(body.get("tags"), [])
    if _has_favorite_tag(tags) and not _has_favorite_reason(content):
        return JSONResponse({"error": _favorite_reason_error()}, status_code=400)
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
        embedding_status = "queued" if _queue_embedding_refresh(bucket_id) else "failed"
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
                "comment_count": meta.get("comment_count", 0),
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


@mcp.custom_route("/api/bucket/{bucket_id}", methods=["PATCH"])
async def api_bucket_update(request):
    """Update dashboard-editable bucket body fields."""
    from starlette.responses import JSONResponse

    err = _require_dashboard_auth(request)
    if err:
        return err

    bucket_id = request.path_params["bucket_id"]
    if not bucket_id or not MEMORY_ID_RE.fullmatch(bucket_id):
        return JSONResponse({"error": "invalid bucket_id"}, status_code=400)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "json body must be an object"}, status_code=400)
    if "content" not in body:
        return JSONResponse({"error": "missing content"}, status_code=400)

    content = str(body.get("content") or "").strip()
    if not content:
        return JSONResponse({"error": "empty content"}, status_code=400)

    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return JSONResponse({"error": "not found"}, status_code=404)

    meta = bucket.get("metadata", {})
    if _has_favorite_tag(meta.get("tags", [])) and not _has_favorite_reason(content):
        return JSONResponse({"error": _favorite_reason_error()}, status_code=400)
    ok = await bucket_mgr.update(
        bucket_id,
        content=content,
        last_active=meta.get("last_active") or meta.get("created"),
    )
    if not ok:
        return JSONResponse({"error": "update failed"}, status_code=500)

    embedding_queued = _queue_embedding_refresh(bucket_id)

    bucket = await bucket_mgr.get(bucket_id)
    return JSONResponse({
        "status": "updated",
        "id": bucket_id,
        "embedding_refreshed": False,
        "embedding_queued": embedding_queued,
        **_bucket_read_payload(bucket),
    })


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
    """Run daily reflection from dashboard or trusted local callers; weekly obeys reflection.weekly_enabled."""
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


@mcp.custom_route("/api/dreams", methods=["GET"])
async def api_dreams(request):
    """Return dream dashboard metadata only. Dream bodies are never exposed here."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        limit = int(request.query_params.get("limit", "30"))
    except Exception:
        limit = 30
    try:
        return JSONResponse(dream_engine.dashboard_payload(limit=max(1, min(100, limit))))
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
    gateway_cfg = config.get("gateway", {}) if isinstance(config.get("gateway", {}), dict) else {}
    dream_cfg = config.get("dream", {}) if isinstance(config.get("dream", {}), dict) else {}
    reflection_cfg = config.get("reflection", {}) if isinstance(config.get("reflection", {}), dict) else {}
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
        "gateway": {
            "cooldown_hours": gateway_cfg.get("cooldown_hours", 6),
            "skip_recent_rounds": gateway_cfg.get("skip_recent_rounds", 5),
        },
        "dream": {
            "enabled": dream_engine.enabled,
            "auto_enabled": dream_engine.auto_enabled,
            "surface_enabled": dream_engine.surface_enabled,
            "model": dream_engine.model,
            "base_url": dream_engine.base_url,
            "api_key_masked": _mask_key(dream_engine.api_key),
            "api_ready": bool(dream_engine.api_key),
            "temperature": dream_cfg.get("temperature", 0.85),
            "max_tokens": dream_cfg.get("max_tokens", 900),
            "daily_hour": dream_cfg.get("daily_hour", 3),
            "run_window_hours": dream_cfg.get("run_window_hours", 3),
            "daily_probability": dream_cfg.get("daily_probability", 0.4),
            "min_material_count": dream_cfg.get("min_material_count", 5),
            "material_window_hours": dream_cfg.get("material_window_hours", 48),
            "identity_anchor_id": dream_cfg.get("identity_anchor_id", ""),
        },
        "reflection": {
            "daily_enabled": bool(
                reflection_cfg.get(
                    "daily_enabled",
                    getattr(reflection_engine, "daily_enabled", True),
                )
            ),
            "memory_affect_anchor_enabled": bool(
                reflection_cfg.get(
                    "memory_affect_anchor_enabled",
                    getattr(reflection_engine, "memory_affect_anchor_enabled", True),
                )
            ),
            "relationship_weather_affect_anchor_enabled": bool(
                reflection_cfg.get(
                    "relationship_weather_affect_anchor_enabled",
                    getattr(reflection_engine, "relationship_weather_affect_anchor_enabled", True),
                )
            ),
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
    global dream_engine
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

    # --- Gateway memory surfacing config ---
    gateway_hot_update_body = None
    if "gateway" in body:
        g = body["gateway"]
        gateway_cfg = config.setdefault("gateway", {})
        gateway_hot_update_body = {}
        if "cooldown_hours" in g:
            gateway_cfg["cooldown_hours"] = max(0.0, float(g["cooldown_hours"]))
            gateway_hot_update_body["cooldown_hours"] = gateway_cfg["cooldown_hours"]
            updated.append("gateway.cooldown_hours")
        if "skip_recent_rounds" in g:
            gateway_cfg["skip_recent_rounds"] = max(0, int(g["skip_recent_rounds"]))
            gateway_hot_update_body["skip_recent_rounds"] = gateway_cfg["skip_recent_rounds"]
            updated.append("gateway.skip_recent_rounds")
        hot_update_status = await _hot_update_gateway_config(gateway_hot_update_body)
        if hot_update_status:
            updated.append(hot_update_status)

    # --- Reflection config ---
    if "reflection" in body:
        r = body["reflection"]
        reflection_cfg = config.setdefault("reflection", {})
        for key in ("daily_enabled", "memory_affect_anchor_enabled", "relationship_weather_affect_anchor_enabled"):
            if key in r:
                reflection_cfg[key] = bool(r[key])
                setattr(reflection_engine, key, reflection_cfg[key])
                updated.append(f"reflection.{key}")

    # --- Dream config ---
    if "dream" in body:
        d = body["dream"]
        dream_cfg = config.setdefault("dream", {})
        for key in (
            "enabled",
            "auto_enabled",
            "surface_enabled",
            "model",
            "base_url",
            "temperature",
            "max_tokens",
            "daily_hour",
            "run_window_hours",
            "daily_probability",
            "min_material_count",
            "material_window_hours",
            "identity_anchor_id",
        ):
            if key in d:
                dream_cfg[key] = d[key]
                updated.append(f"dream.{key}")
        if "api_key" in d and d["api_key"]:
            dream_cfg["api_key"] = d["api_key"]
            updated.append("dream.api_key")
        dream_engine = DreamEngine(config)

    # --- Persist to config.yaml if requested ---
    if body.get("persist", False):
        config_path = os.environ.get(
            "OMBRE_CONFIG_PATH",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"),
        )
        runtime_config_path = config.get("_runtime_config_path") or os.environ.get("OMBRE_RUNTIME_CONFIG_PATH", "")
        if not runtime_config_path:
            runtime_config_path = os.path.join(config.get("state_dir") or os.path.dirname(config_path), "config.runtime.yaml")

        def _apply_dashboard_config(save_config: dict) -> dict:
            save_config = save_config or {}
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

            if "gateway" in body:
                sc_gateway = save_config.setdefault("gateway", {})
                if "cooldown_hours" in body["gateway"]:
                    sc_gateway["cooldown_hours"] = max(0.0, float(body["gateway"]["cooldown_hours"]))
                if "skip_recent_rounds" in body["gateway"]:
                    sc_gateway["skip_recent_rounds"] = max(0, int(body["gateway"]["skip_recent_rounds"]))

            if "reflection" in body:
                sc_reflection = save_config.setdefault("reflection", {})
                for key in ("daily_enabled", "memory_affect_anchor_enabled", "relationship_weather_affect_anchor_enabled"):
                    if key in body["reflection"]:
                        sc_reflection[key] = bool(body["reflection"][key])

            if "dream" in body:
                sc_dream = save_config.setdefault("dream", {})
                for key in (
                    "enabled",
                    "auto_enabled",
                    "surface_enabled",
                    "model",
                    "base_url",
                    "temperature",
                    "max_tokens",
                    "daily_hour",
                    "run_window_hours",
                    "daily_probability",
                    "min_material_count",
                    "material_window_hours",
                    "identity_anchor_id",
                ):
                    if key in body["dream"]:
                        sc_dream[key] = body["dream"][key]
                # Never persist api_key to yaml (use env var)
            return save_config

        try:
            save_config = {}
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    save_config = yaml.safe_load(f) or {}
            save_config = _apply_dashboard_config(save_config)

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(save_config, f, default_flow_style=False, allow_unicode=True)
            updated.append("persisted_to_yaml")
            if os.path.exists(runtime_config_path):
                runtime_config = {}
                with open(runtime_config_path, "r", encoding="utf-8") as f:
                    runtime_config = yaml.safe_load(f) or {}
                runtime_config = _apply_dashboard_config(runtime_config)
                os.makedirs(os.path.dirname(runtime_config_path), exist_ok=True)
                with open(runtime_config_path, "w", encoding="utf-8") as f:
                    yaml.dump(runtime_config, f, default_flow_style=False, allow_unicode=True)
                updated.append("runtime_yaml_synced")
        except Exception as e:
            try:
                runtime_config = {}
                if os.path.exists(runtime_config_path):
                    with open(runtime_config_path, "r", encoding="utf-8") as f:
                        runtime_config = yaml.safe_load(f) or {}
                runtime_config = _apply_dashboard_config(runtime_config)
                os.makedirs(os.path.dirname(runtime_config_path), exist_ok=True)
                with open(runtime_config_path, "w", encoding="utf-8") as f:
                    yaml.dump(runtime_config, f, default_flow_style=False, allow_unicode=True)
                updated.append("persisted_to_runtime_yaml")
                updated.append(f"config_yaml_unwritable:{type(e).__name__}")
            except Exception as fallback_e:
                return JSONResponse(
                    {"error": f"persist failed: {e}; runtime persist failed: {fallback_e}", "updated": updated},
                    status_code=500,
                )

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
                deleted = await bucket_mgr.delete(bid)
                if not deleted:
                    raise ValueError("bucket not found")
                embedding_engine.delete_embedding(bid)
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
            await _ensure_decay_engine_started_for_transport(transport)
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
            local_memory_edge_store = MemoryEdgeStore(config)
            while True:
                try:
                    reflection_cfg = config.get("reflection", {}) if isinstance(config.get("reflection", {}), dict) else {}
                    local_reflection_engine.daily_enabled = bool(
                        reflection_cfg.get("daily_enabled", True)
                    )
                    local_reflection_engine.memory_affect_anchor_enabled = bool(
                        reflection_cfg.get("memory_affect_anchor_enabled", True)
                    )
                    local_reflection_engine.relationship_weather_affect_anchor_enabled = bool(
                        reflection_cfg.get("relationship_weather_affect_anchor_enabled", True)
                    )
                    results = await local_reflection_engine.run_due(
                        local_bucket_mgr,
                        local_persona_engine,
                        local_embedding_engine,
                    )
                    if results:
                        logger.info("Reflection run-due results / 反思定时结果: %s", results)
                    if reflection_cfg.get("enrich_backfill_enabled", True):
                        backfill_result = await _backfill_memory_enrichment(
                            limit=reflection_cfg.get("enrich_backfill_limit", 5),
                            bucket_mgr_arg=local_bucket_mgr,
                            reflection_engine_arg=local_reflection_engine,
                            edge_store_arg=local_memory_edge_store,
                            embedding_engine_arg=local_embedding_engine,
                        )
                        if backfill_result.get("processed"):
                            logger.info(
                                "Memory enrichment backfill / 记忆 enrich 补跑: %s",
                                backfill_result,
                            )
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

        async def _dream_loop():
            await asyncio.sleep(30)
            local_bucket_mgr = BucketManager(config)
            while True:
                local_dream_engine = DreamEngine(config)
                local_embedding_engine = EmbeddingEngine(config)
                try:
                    result = await local_dream_engine.run_due(
                        local_bucket_mgr,
                        local_embedding_engine,
                    )
                    if result and result.get("status") == "created":
                        logger.info("Dream run-due result / 夜梦定时结果: %s", result)
                except Exception as e:
                    logger.warning("Dream scheduler failed / 夜梦定时器失败: %s", e)
                await asyncio.sleep(local_dream_engine.check_interval_minutes * 60)

        def _start_dream_scheduler():
            loop = asyncio.new_event_loop()
            loop.run_until_complete(_dream_loop())

        dt = threading.Thread(target=_start_dream_scheduler, daemon=True)
        dt.start()
        logger.info("Dream scheduler loop started / 夜梦定时器循环已启动")

        # --- Add CORS middleware so remote clients (Cloudflare Tunnel / ngrok) can connect ---
        # --- 添加 CORS 中间件，让远程客户端（Cloudflare Tunnel / ngrok）能正常连接 ---
        if transport == "streamable-http":
            _app = mcp.streamable_http_app()
        else:
            _app = mcp.sse_app()
        if hasattr(_app, "add_event_handler"):
            async def _start_decay_engine_on_app_startup():
                await _ensure_decay_engine_started_for_transport(transport)

            _app.add_event_handler("startup", _start_decay_engine_on_app_startup)
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
