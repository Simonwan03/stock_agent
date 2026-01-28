from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict


DEFAULT_PORTFOLIO_PATH = Path("outputs/portfolio.json")
DEFAULT_TEMPLATE = {
    "holdings": [
        {"ticker": "AAPL", "shares": 10},
        {"ticker": "TSLA", "shares": 5},
    ],
    "benchmarks": ["SPY"],
}

# 生成组合 JSON 模板（供日报分析使用）
def build_portfolio_template(example: bool = True) -> Dict[str, Any]:
    if example:
        return DEFAULT_TEMPLATE
    return {"holdings": [], "benchmarks": []}


def write_template(path: Path, payload: Dict[str, Any], force: bool = False) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"File already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate portfolio.json template for daily analysis.")
    p.add_argument("--file", default=str(DEFAULT_PORTFOLIO_PATH), help="Portfolio file path (default: data/portfolio.json)")

    sub = p.add_subparsers(dest="cmd", required=True)

    p_tpl = sub.add_parser("template", help="Generate portfolio.json template for daily analysis.")
    p_tpl.add_argument("--empty", action="store_true", help="Generate empty holdings/benchmarks.")
    p_tpl.add_argument("--stdout", action="store_true", help="Print template JSON to stdout.")
    p_tpl.add_argument("--force", action="store_true", help="Overwrite target file if it exists.")

    return p.parse_args()


def main() -> int:
    args = parse_args()
    path = Path(args.file)

    if args.cmd == "template":
        payload = build_portfolio_template(example=not args.empty)
        if args.stdout:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        write_template(path, payload, force=args.force)
        print(f"Template written: {path}")
        return 0

    raise RuntimeError("Unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
