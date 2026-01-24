import os
from dotenv import load_dotenv
from agent.graph import build_graph

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
