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

from agent.graph import build_graph  # noqa: E402

def main():
    load_dotenv()
    graph = build_graph()

    state = {
        "ticker": "AAPL",
        "start_date": "2022-01-01",
        "filing_form": "10-Q",
    }
    out = graph.invoke(state)

    os.makedirs("out", exist_ok=True)
    path = f"out/{state['ticker']}.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(out["report_md"])

    print("Saved:", path)

if __name__ == "__main__":
    main()
