from __future__ import annotations

import argparse
import asyncio
import copy
import json
import sys
import tempfile
import time
from pathlib import Path

from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scene_linker import SceneEdgeProposalStore, SceneLinker
from scripts.evaluate_scene_linker_model_pool import _windows_scoped_env, benchmark_cases
from utils import load_config


class SyntheticBucketStore:
    def __init__(self, scenes: list[dict]):
        self._scenes = {str(scene["id"]): scene for scene in scenes}

    async def get(self, scene_id: str) -> dict | None:
        return self._scenes.get(str(scene_id))

    async def list_all(self, include_archive: bool = False) -> list[dict]:
        return list(self._scenes.values())


def _base_url(value: str) -> str:
    url = str(value or "").strip().rstrip("/")
    suffix = "/chat/completions"
    if url.endswith(suffix):
        return url[: -len(suffix)]
    return url


async def run(spec_path: Path, output_path: Path, timeout_seconds: float) -> dict:
    specs = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(specs, list) or len(specs) < 2:
        raise ValueError("failover smoke spec must contain at least two providers")

    clients: dict[str, AsyncOpenAI] = {}
    models = []
    for raw in specs:
        name = str(raw.get("name") or raw.get("model") or "").strip()
        secret = _windows_scoped_env(str(raw.get("api_key_env") or ""))
        if not name or not secret:
            raise ValueError(f"provider {name or '<unnamed>'} is missing its environment key")
        clients[name] = AsyncOpenAI(
            api_key=secret,
            base_url=_base_url(str(raw.get("endpoint") or "")),
            timeout=float(raw.get("timeout_seconds", timeout_seconds)),
        )
        models.append(
            {
                "name": name,
                "model": str(raw.get("model") or ""),
                "base_url": _base_url(str(raw.get("endpoint") or "")),
                "max_tokens": int(raw.get("max_tokens", 1400)),
                "temperature": raw.get("temperature"),
            }
        )

    case = benchmark_cases()[0]
    bucket_store = SyntheticBucketStore([case["anchor"], *case["candidates"]])
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="ombre-scene-linker-failover-") as temp_dir:
            root = Path(temp_dir)
            config = {
                "buckets_dir": str(root / "buckets"),
                "state_dir": str(root / "state"),
                "scene_linker": {
                    "enabled": True,
                    "auto_enabled": True,
                    "semantic_candidates": 0,
                    "recent_candidates": len(case["candidates"]),
                    "max_candidates": len(case["candidates"]),
                    "max_proposals": 5,
                    "min_confidence": 0.78,
                    "timeout_seconds": timeout_seconds,
                    "models": models,
                },
            }
            store = SceneEdgeProposalStore(config)
            linker = SceneLinker(config, proposal_store=store, clients=clients)
            result = await linker.link_scene(case["anchor"]["id"], bucket_store)
            proposals = store.list_pending(anchor_scene_id=case["anchor"]["id"])
    finally:
        await asyncio.gather(*(client.close() for client in clients.values()))

    report = {
        "mode": "synthetic_scene_linker_live_failover_smoke",
        "source_memory_sent": False,
        "provider_order": [item["name"] for item in models],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "result": result,
        "proposals": proposals,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


async def run_local_config(config_path: Path, output_path: Path) -> dict:
    config = copy.deepcopy(load_config(str(config_path)))
    case = benchmark_cases()[0]
    bucket_store = SyntheticBucketStore([case["anchor"], *case["candidates"]])
    started = time.perf_counter()
    clients = []
    try:
        with tempfile.TemporaryDirectory(prefix="ombre-scene-linker-local-config-") as temp_dir:
            config["state_dir"] = str(Path(temp_dir) / "state")
            linker = SceneLinker(config)
            initial_status = linker.status()
            clients = [
                provider["client"]
                for provider in linker.providers
                if provider.get("client") is not None
            ]
            result = await linker.link_scene(case["anchor"]["id"], bucket_store)
            store = linker.proposal_store(create=False)
            proposals = (
                store.list_pending(anchor_scene_id=case["anchor"]["id"])
                if store is not None
                else []
            )
    finally:
        await asyncio.gather(*(client.close() for client in clients))

    report = {
        "mode": "synthetic_scene_linker_local_config_smoke",
        "source_memory_sent": False,
        "config_path": str(config_path.resolve()),
        "configured_models": initial_status["configured_models"],
        "ready_models": initial_status["ready_models"],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "result": result,
        "proposals": proposals,
        "temporary_sidecar_removed": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test real Scene linker provider failover with synthetic Scenes")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--spec")
    source.add_argument("--local-config")
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.local_config:
        report = asyncio.run(run_local_config(Path(args.local_config), Path(args.output)))
    else:
        report = asyncio.run(run(Path(args.spec), Path(args.output), args.timeout_seconds))
    print(
        json.dumps(
            {
                "status": report["result"].get("status"),
                "model": report["result"].get("model"),
                "attempts": report["result"].get("attempts"),
                "elapsed_seconds": report["elapsed_seconds"],
                "output": str(Path(args.output).resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
