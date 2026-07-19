#!/usr/bin/env python3
"""Run a read-only Scene diffusion experiment over an explicit legacy overlay.

The overlay is an experiment allowlist, not a migration plan. Source Markdown
and the real memory edge store are never edited. Reports omit source prose by
default and contain only ids, names, relation rationales, and path scores.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import frontmatter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_diffusion import DiffusionOptions, diffuse_memory
from memory_edges import RELATION_TYPES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buckets-dir", required=True)
    parser.add_argument("--overlay", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_overlay(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("mode") != "experiment_only":
        raise ValueError("overlay.mode must be experiment_only")
    if payload.get("writes_source") is not False:
        raise ValueError("overlay must explicitly declare writes_source=false")
    return payload


def load_buckets(root: Path) -> dict[str, dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*.md")):
        if ".tombstones" in path.relative_to(root).parts:
            continue
        post = frontmatter.load(path)
        metadata = dict(post.metadata)
        bucket_id = str(metadata.get("id") or path.stem).strip()
        if not bucket_id:
            continue
        buckets[bucket_id] = {
            "id": bucket_id,
            "content": str(post.content or ""),
            "metadata": metadata,
            "path": str(path),
        }
    return buckets


def build_scene_map(
    buckets: dict[str, dict[str, Any]],
    approved_ids: list[str],
) -> dict[str, dict[str, Any]]:
    scene_map: dict[str, dict[str, Any]] = {}
    for bucket_id in approved_ids:
        bucket = buckets.get(bucket_id)
        if bucket is None:
            raise KeyError(f"approved Scene bucket not found: {bucket_id}")
        metadata = dict(bucket.get("metadata") or {})
        metadata["memory_value_source"] = "authored_scene"
        metadata["scene_overlay_only"] = True
        # The copied record is an in-memory experiment node. Source frontmatter
        # remains byte-for-byte untouched.
        scene_map[bucket_id] = {**bucket, "metadata": metadata}
    return scene_map


def validate_edges(edges: list[dict[str, Any]], scene_ids: set[str]) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for index, edge in enumerate(edges):
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        relation_type = str(edge.get("relation_type") or "")
        if source not in scene_ids or target not in scene_ids:
            raise ValueError(f"edge[{index}] has a non-approved endpoint")
        if relation_type not in RELATION_TYPES:
            raise ValueError(f"edge[{index}] has unsupported relation_type={relation_type}")
        if relation_type == "emotional_echo":
            raise ValueError("experiment overlay forbids emotional_echo")
        validated.append(
            {
                "source": source,
                "target": target,
                "relation_type": relation_type,
                "confidence": float(edge.get("confidence", 0.5)),
                "reason": str(edge.get("reason") or ""),
            }
        )
    return validated


def node_name(scene_map: dict[str, dict[str, Any]], bucket_id: str) -> str:
    metadata = scene_map.get(bucket_id, {}).get("metadata", {})
    return str(metadata.get("name") or bucket_id)


def path_payload(scene_map: dict[str, dict[str, Any]], hit: Any) -> dict[str, Any]:
    path = hit.best_path
    return {
        "target_id": hit.bucket_id,
        "target_name": node_name(scene_map, hit.bucket_id),
        "activation": round(float(hit.activation), 4),
        "hop_count": len(path.steps),
        "nodes": [
            {"bucket_id": bucket_id, "name": node_name(scene_map, bucket_id)}
            for bucket_id in path.nodes
        ],
        "steps": [
            {
                "source": step.source,
                "target": step.target,
                "relation_type": step.relation_type,
                "confidence": round(float(step.confidence), 4),
                "reason": step.reason,
                "direction": step.direction,
            }
            for step in path.steps
        ],
    }


def main() -> int:
    args = parse_args()
    buckets_dir = Path(args.buckets_dir).resolve()
    overlay_path = Path(args.overlay).resolve()
    output_path = Path(args.output).resolve()
    overlay = load_overlay(overlay_path)
    buckets = load_buckets(buckets_dir)
    approved_ids = [str(item) for item in overlay.get("approved_scene_ids") or []]
    scene_map = build_scene_map(buckets, approved_ids)
    edges = validate_edges(list(overlay.get("edges") or []), set(scene_map))
    options_payload = dict(overlay.get("options") or {})
    options = DiffusionOptions(
        chain_walk_enabled=True,
        chain_max_hops=int(options_payload.get("chain_max_hops", 6)),
        chain_min_strength=float(options_payload.get("chain_min_strength", 0.12)),
        chain_min_confidence=float(options_payload.get("chain_min_confidence", 0.72)),
        chain_max_frontier=int(options_payload.get("chain_max_frontier", 24)),
        min_activation=float(options_payload.get("min_activation", 0.0)),
        top_k=int(options_payload.get("top_k", 20)),
        include_incoming=bool(options_payload.get("include_incoming", False)),
        max_paths_per_hit=int(options_payload.get("max_paths_per_hit", 4)),
    )

    case_results: list[dict[str, Any]] = []
    for case in overlay.get("cases") or []:
        seed_id = str(case.get("seed_scene_id") or "")
        if seed_id not in scene_map:
            raise ValueError(f"case seed is not approved: {seed_id}")
        hits = diffuse_memory(
            {seed_id: float(case.get("seed_score", 1.0))},
            edges,
            scene_map,
            options=options,
            exclude_ids={seed_id},
            query_text="",
        )
        paths = [path_payload(scene_map, hit) for hit in hits]
        deepest = sorted(
            paths,
            key=lambda row: (int(row["hop_count"]), float(row["activation"])),
            reverse=True,
        )[:5]
        case_results.append(
            {
                "case_id": str(case.get("case_id") or seed_id),
                "prompt": str(case.get("prompt") or ""),
                "seed_scene_id": seed_id,
                "seed_scene_name": node_name(scene_map, seed_id),
                "ranked_paths": paths,
                "deepest_paths": deepest,
            }
        )

    report = {
        "mode": "experiment_only",
        "source_buckets_dir": str(buckets_dir),
        "overlay": str(overlay_path),
        "writes_source": False,
        "writes_edge_store": False,
        "approved_scene_count": len(scene_map),
        "edge_count": len(edges),
        "options": {
            "chain_walk_enabled": True,
            "chain_max_hops": options.chain_max_hops,
            "include_incoming": options.include_incoming,
            "top_k": options.top_k,
        },
        "scenes": [
            {
                "bucket_id": bucket_id,
                "name": node_name(scene_map, bucket_id),
                "source_path": str(scene_map[bucket_id]["path"]),
            }
            for bucket_id in approved_ids
        ],
        "edges": edges,
        "cases": case_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
