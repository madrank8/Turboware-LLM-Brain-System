"""Command-line entry point for Brain admin tasks.

    python brain_cli.py init               # create schema
    python brain_cli.py stats              # row counts per table
    python brain_cli.py graphify <repo>    # AST ingest a local repo
    python brain_cli.py skill-ingest       # fetch curated SKILL.md set
    python brain_cli.py skill-list         # list stored skills
    python brain_cli.py persona-ingest     # fetch agency-agents personas
    python brain_cli.py persona-list       # list stored personas
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging

import brain_db as db
from brain_config import CONFIG


def _setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, CONFIG.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


async def cmd_init(_: argparse.Namespace) -> None:
    await db.init_pool()
    print("schema initialized")


async def cmd_stats(_: argparse.Namespace) -> None:
    await db.init_pool()
    print(json.dumps(await db.stats(), indent=2))


async def cmd_graphify(args: argparse.Namespace) -> None:
    await db.init_pool()
    from brain_graphify import graphify_repo
    summary = await graphify_repo(args.repo)
    print(json.dumps(summary, indent=2))


async def cmd_skill_ingest(_: argparse.Namespace) -> None:
    await db.init_pool()
    from brain_skill_ingester import ingest_skills
    print(json.dumps(await ingest_skills(), indent=2))


async def cmd_skill_list(_: argparse.Namespace) -> None:
    await db.init_pool()
    async with db.acquire() as conn:
        rows = await conn.fetch(
            "SELECT name, source_repo, updated_at FROM brain_skills ORDER BY name"
        )
    for r in rows:
        print(f"{r['name']:40s}  {r['source_repo']:40s}  {r['updated_at']}")


async def cmd_persona_ingest(_: argparse.Namespace) -> None:
    await db.init_pool()
    from brain_persona_ingester import ingest_personas
    print(json.dumps(await ingest_personas(), indent=2))


async def cmd_persona_list(_: argparse.Namespace) -> None:
    await db.init_pool()
    async with db.acquire() as conn:
        rows = await conn.fetch(
            "SELECT name, division, cardinality(domain_keywords) AS kw FROM brain_personas ORDER BY division, name"
        )
    for r in rows:
        print(f"{r['division']:20s}  {r['name']:40s}  kw={r['kw']}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="brain_cli")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init").set_defaults(func=cmd_init)
    sub.add_parser("stats").set_defaults(func=cmd_stats)
    g = sub.add_parser("graphify")
    g.add_argument("repo")
    g.set_defaults(func=cmd_graphify)
    sub.add_parser("skill-ingest").set_defaults(func=cmd_skill_ingest)
    sub.add_parser("skill-list").set_defaults(func=cmd_skill_list)
    sub.add_parser("persona-ingest").set_defaults(func=cmd_persona_ingest)
    sub.add_parser("persona-list").set_defaults(func=cmd_persona_list)
    return p


def main() -> None:
    _setup_logging()
    args = build_parser().parse_args()
    try:
        asyncio.run(args.func(args))
    finally:
        asyncio.run(db.close_pool())


if __name__ == "__main__":
    main()
