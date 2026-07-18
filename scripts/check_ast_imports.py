"""Parse tracked Python and smoke-import the lightweight demo modules."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def tracked_python_files() -> list[Path]:
    output = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.py"],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    ).stdout
    return [Path(name) for name in output.splitlines()]


def main() -> int:
    files = tracked_python_files()
    for path in files:
        source = ROOT / path
        ast.parse(source.read_text(encoding="utf-8"), filename=str(path))

    __import__("components.settings")
    __import__("components.response_schema")
    __import__("components.demo_retrieval")
    __import__("components.s5_rag_llm")
    __import__("components.ui_helpers")
    __import__("components.evaluate")

    heavy = {"torch", "transformers", "pymilvus", "elasticsearch"} & set(sys.modules)
    if heavy:
        raise RuntimeError(f"demo smoke imports loaded full-mode dependencies: {sorted(heavy)}")
    print(f"Parsed {len(files)} tracked Python files; lightweight imports passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
