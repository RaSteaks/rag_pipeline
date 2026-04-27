#!/usr/bin/env python3
"""CLI entry point for RAG search, sync, and stats.

Usage:
  python cli.py search "query" [--top-k 5] [--debug]
  python cli.py sync [--source NAME] [--rebuild]
  python cli.py stats
"""
import sys
import json
import argparse
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

from config import get_config


def search(query: str, top_k: int = 5, debug: bool = False):
    """Search the knowledge base via API."""
    import requests
    config = get_config()
    url = f"http://{config.server.host}:{config.server.port}/search"
    try:
        resp = requests.post(url, json={"query": query, "top_k": top_k, "debug": debug}, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"error": f"RAG service not running on {url}. Start: python rag_service.py"}
    except Exception as e:
        return {"error": str(e)}


def sync(source_name: str = None, rebuild: bool = False):
    """Trigger index synchronization."""
    import requests
    config = get_config()
    url = f"http://{config.server.host}:{config.server.port}/sync"
    try:
        payload = {}
        if source_name:
            payload["source_name"] = source_name
        if rebuild:
            payload["rebuild"] = True
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"error": "RAG service not running"}
    except Exception as e:
        return {"error": str(e)}


def stats():
    """Get index statistics."""
    import requests
    config = get_config()
    url = f"http://{config.server.host}:{config.server.port}/stats"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Search command
    search_cmd = subparsers.add_parser("search", help="Search the knowledge base")
    search_cmd.add_argument("query", help="Search query")
    search_cmd.add_argument("--top-k", type=int, default=5, help="Number of results")
    search_cmd.add_argument("--debug", action="store_true", help="Show debug info")

    # Sync command
    sync_cmd = subparsers.add_parser("sync", help="Synchronize index")
    sync_cmd.add_argument("--source", help="Rebuild specific source only")
    sync_cmd.add_argument("--rebuild", action="store_true", help="Full rebuild from scratch")

    # Stats command
    subparsers.add_parser("stats", help="Show index statistics")

    args = parser.parse_args()

    if args.command == "search":
        result = search(args.query, args.top_k, args.debug)
    elif args.command == "sync":
        result = sync(args.source, args.rebuild)
    elif args.command == "stats":
        result = stats()
    else:
        parser.print_help()
        return

    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.stdout.flush()


if __name__ == "__main__":
    main()