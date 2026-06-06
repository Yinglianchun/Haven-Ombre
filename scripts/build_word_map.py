#!/usr/bin/env python3
"""Build derived Word Map and private identity alias indexes from buckets."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bucket_manager import BucketManager
from identity_semantics import IdentitySemanticStore
from utils import load_config
from word_map import WordMapStore


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build local Word Map Lite and private identity semantic alias indexes."
    )
    parser.add_argument("--include-archive", action="store_true", help="Include archived buckets.")
    parser.add_argument("--skip-word-map", action="store_true", help="Do not rebuild the generic word map.")
    parser.add_argument("--skip-identity", action="store_true", help="Do not rebuild private identity aliases.")
    parser.add_argument(
        "--identity-config",
        default="",
        help="Private canonical YAML/JSON path. Overrides OMBRE_IDENTITY_SEMANTICS_PATH for this run.",
    )
    parser.add_argument("--nodes", type=int, default=12, help="How many word nodes to print.")
    parser.add_argument("--edges", type=int, default=12, help="How many word edges to print.")
    parser.add_argument("--aliases", type=int, default=20, help="How many private aliases to print.")
    return parser.parse_args(argv)


def _with_word_map_enabled(config: dict[str, Any], private_terms: set[str]) -> dict[str, Any]:
    next_config = dict(config)
    word_map_cfg = dict(next_config.get("word_map") or {})
    word_map_cfg["enabled"] = True
    existing_private_terms = {
        str(item).strip()
        for item in (word_map_cfg.get("private_terms") or [])
        if str(item).strip()
    }
    word_map_cfg["private_terms"] = sorted(existing_private_terms | private_terms)
    next_config["word_map"] = word_map_cfg
    return next_config


def _identity_config(config: dict[str, Any], private_path: str) -> dict[str, Any]:
    next_config = dict(config)
    identity_cfg = dict(next_config.get("identity_semantics") or {})
    if private_path:
        identity_cfg["enabled"] = True
        identity_cfg["private_config_path"] = private_path
    next_config["identity_semantics"] = identity_cfg
    return next_config


def _identity_seed_terms(store: IdentitySemanticStore) -> set[str]:
    return {
        alias
        for node in store.load_private_nodes()
        for alias in node.seed_aliases
        if str(alias).strip()
    }


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config()
    if args.identity_config:
        config = _identity_config(config, os.path.abspath(args.identity_config))

    bucket_mgr = BucketManager(config)
    buckets = await bucket_mgr.list_all(include_archive=args.include_archive)

    identity_store = IdentitySemanticStore(config)
    private_terms = _identity_seed_terms(identity_store)

    result: dict[str, Any] = {
        "bucket_count": len(buckets),
        "include_archive": bool(args.include_archive),
    }

    if args.skip_word_map:
        result["word_map"] = {"skipped": True}
    else:
        word_map_store = WordMapStore(_with_word_map_enabled(config, private_terms))
        result["word_map"] = {
            "stats": word_map_store.rebuild(buckets),
            "top_nodes": word_map_store.list_nodes(args.nodes),
            "top_edges": word_map_store.list_edges(args.edges),
            "private_terms_excluded": sorted(private_terms),
        }

    if args.skip_identity:
        result["identity_semantics"] = {"skipped": True}
    else:
        result["identity_semantics"] = {
            "enabled": identity_store.enabled,
            "stats": identity_store.rebuild_alias_index(buckets),
            "aliases": identity_store.list_aliases()[: max(0, args.aliases)],
        }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
