"""
main.py
───────
CLI entry point for the GraphRAG Code Analyzer.

Commands
────────
  python main.py ingest  <repo_path>   — ingest a repository
  python main.py query   "<question>"  — ask a question
  python main.py shell                 — interactive REPL
"""

from __future__ import annotations

import asyncio
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def cmd_ingest(repo_path: str) -> None:
    from ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline()
    try:
        summary = asyncio.run(pipeline.ingest_repo(repo_path))
        print("\n✅  Ingestion complete")
        print(f"   Files     : {summary['total_files']}")
        print(f"   Succeeded : {summary['succeeded']}")
        print(f"   Failed    : {summary['failed']}")
        print(f"   Chunks    : {summary['vector_chunks']}")
        print(f"   Time      : {summary['elapsed_seconds']}s")
    finally:
        pipeline.close()


def cmd_query(question: str) -> None:
    from core.config import cfg
    from agent.graph_agent import CodeAnalyzerAgent

    cfg.validate()
    agent = CodeAnalyzerAgent()
    try:
        print(f"\n🔍  Query  : {question}")
        result = agent.run(question)
        print(f"📡  Route  : {result['route']}\n")
        print("─" * 60)
        print(result["answer"])
        print("─" * 60)
    finally:
        agent.close()


def cmd_shell() -> None:
    from core.config import cfg
    from agent.graph_agent import CodeAnalyzerAgent

    cfg.validate()
    agent = CodeAnalyzerAgent()
    print("\n🤖  GraphRAG Code Analyzer — interactive shell")
    print("   Type your question and press Enter. 'exit' to quit.\n")
    try:
        while True:
            try:
                question = input("❯ ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if not question:
                continue
            if question.lower() in ("exit", "quit", "q"):
                print("Bye!")
                break
            result = agent.run(question)
            print(f"\n[route: {result['route']}]\n")
            print(result["answer"])
            print()
    finally:
        agent.close()


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    command = args[0].lower()

    if command == "ingest":
        if len(args) < 2:
            print("Usage: python main.py ingest <repo_path>")
            sys.exit(1)
        cmd_ingest(args[1])

    elif command == "query":
        if len(args) < 2:
            print('Usage: python main.py query "<your question>"')
            sys.exit(1)
        cmd_query(" ".join(args[1:]))

    elif command == "shell":
        cmd_shell()

    else:
        print(f"Unknown command: {command}")
        print("Available: ingest | query | shell")
        sys.exit(1)


if __name__ == "__main__":
    main()
