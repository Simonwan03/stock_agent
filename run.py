import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional


def load_dotenv(path: Optional[Path] = None) -> None:
    env_path = path or Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())

ROOT = Path(__file__).resolve().parent
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agent.config import get_settings  # noqa: E402
from agent.multiagent.orchestrator import MultiAgentOrchestrator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-agent stock analysis pipeline.")
    parser.add_argument("--ticker", default="", help="Ticker symbol, e.g. AAPL")
    parser.add_argument(
        "--outputs-dir",
        default="",
        help="Directory with module JSON outputs (default: src/tools/outputs).",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output JSON path (default: out/<TICKER>_multiagent.json).",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    settings = get_settings()
    args = parse_args()

    # Use CLI arguments if provided, otherwise fall back to settings/env defaults.
    ticker = args.ticker.strip().upper() if args.ticker else settings.default_ticker
    outputs_dir = Path(args.outputs_dir) if args.outputs_dir else settings.outputs_dir
    report_path = Path(args.out) if args.out else settings.report_dir / f"{ticker}_multiagent.json"

    # Build the pipeline and execute the module agents.
    orchestrator = MultiAgentOrchestrator(outputs_dir=outputs_dir)
    report = orchestrator.run(ticker=ticker)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved:", report_path)

if __name__ == "__main__":
    main()
