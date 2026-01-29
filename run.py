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
from agent.llm_client import llm_client_from_settings  # noqa: E402
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
    parser.add_argument(
        "--advice-md-out",
        default="",
        help="Write a markdown advice section to this path (default: out/<TICKER>_advice.md).",
    )
    parser.add_argument("--advice-md", action="store_true", help="Generate markdown advice via LLM.")
    llm_group = parser.add_mutually_exclusive_group()
    llm_group.add_argument("--llm", action="store_true", help="Force-enable LLM calls.")
    llm_group.add_argument("--no-llm", action="store_true", help="Disable LLM calls.")
    parser.add_argument("--verbose", action="store_true", help="Print pipeline progress.")
    parser.add_argument("--quiet", action="store_true", help="Disable pipeline progress output.")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    settings = get_settings()
    args = parse_args()

    # Use CLI arguments if provided, otherwise fall back to settings/env defaults.
    ticker = args.ticker.strip().upper() if args.ticker else settings.default_ticker
    outputs_dir = Path(args.outputs_dir) if args.outputs_dir else settings.outputs_dir
    report_path = Path(args.out) if args.out else settings.report_dir / f"{ticker}_multiagent.json"
    advice_md_path = Path(args.advice_md_out) if args.advice_md_out else settings.report_dir / f"{ticker}_advice.md"

    # Build the pipeline and execute the module agents.
    llm = llm_client_from_settings(settings.llm)
    if args.llm:
        if llm is None:
            raise SystemExit("LLM enabled but missing config (base_url/api_key/model).")
    if args.no_llm:
        llm = None
    def _progress(msg: str) -> None:
        print(msg, flush=True)

    orchestrator = MultiAgentOrchestrator(outputs_dir=outputs_dir, llm=llm)
    show_progress = True
    if args.quiet:
        show_progress = False
    if args.verbose:
        show_progress = True
    report = orchestrator.run(ticker=ticker, progress=_progress if show_progress else None)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved:", report_path)

    if args.advice_md:
        if llm is None:
            raise SystemExit("LLM is required for markdown advice. Enable with --llm or set LLM config.")
        if args.verbose:
            _progress("[advice-md] calling LLM for markdown summary")
        prompt = [
            {
                "role": "system",
                "content": "You are a financial advisor. Return ONLY markdown text.",
            },
            {
                "role": "user",
                "content": (
                    f"Ticker: {ticker}\n"
                    "Given the pipeline JSON, write a concise markdown advice section for a human reader.\n"
                    "Requirements:\n"
                    "- Output markdown only, no code fences.\n"
                    "- Use headings and short bullet lists.\n"
                    "- Include: Summary, Key Signals, Risks, Action Items.\n"
                    "- Avoid making up data; base on provided JSON.\n"
                    "- Keep it under 200 words.\n"
                    "Pipeline JSON:\n"
                    f"{json.dumps(report, ensure_ascii=False)}"
                ),
            },
        ]
        md_text = llm.chat(prompt, temperature=0.3)
        advice_md_path.parent.mkdir(parents=True, exist_ok=True)
        advice_md_path.write_text(md_text.strip(), encoding="utf-8")
        print("Saved:", advice_md_path)

if __name__ == "__main__":
    main()
