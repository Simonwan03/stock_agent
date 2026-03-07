# stock_agent

A modular stock analysis workflow for individual investors.

The project has two parts:
1. `src/tools/*`: fetch and transform market/financial/news/portfolio JSON data.
2. `run.py`: orchestrate module agents, summarize each module, and generate a final advice payload.

## Disclaimer
- This project is for learning and research only.
- Outputs are not investment advice.
- Always verify results with trusted data sources before making decisions.

## Project Structure

```text
stock_agent/
├─ run.py
├─ config/
│  └─ config.toml
├─ src/
│  ├─ agent/
│  │  ├─ config.py
│  │  ├─ llm_client.py
│  │  └─ multiagent/
│  │     ├─ agents.py
│  │     ├─ orchestrator.py
│  │     └─ schema.py
│  └─ tools/
│     ├─ market_data.py
│     ├─ financial_statement.py
│     ├─ compact_financials.py
│     ├─ indicators.py
│     ├─ news.py
│     └─ portfolio.py
└─ out/
```

## Requirements
- Python `>=3.11`
- Recommended: virtual environment

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip

# base dependencies (pipeline + indicators)
pip install -e .

# install OpenBB if you want to fetch market/news/financial raw data
pip install -e '.[openbb]'
```

## Configuration

You can configure with environment variables or `config/config.toml`.

### `config/config.toml` example

```toml
[llm]
base_url = "https://api.deepseek.com"
api_key = "sk-***"
model = "deepseek-chat"
timeout = 60

[outputs]
dir = "src/tools/outputs"
```

### Environment variables (optional)

```bash
export STOCK_AGENT_TICKER=AAPL
export STOCK_AGENT_OUTPUTS_DIR=src/tools/outputs
export STOCK_AGENT_REPORT_DIR=out

export STOCK_AGENT_LLM_BASE_URL=https://api.deepseek.com
export STOCK_AGENT_LLM_API_KEY=your_key
export STOCK_AGENT_LLM_MODEL=deepseek-chat
export STOCK_AGENT_LLM_ENABLED=1
export STOCK_AGENT_LLM_TIMEOUT=60
```

## Step 1: Generate Tool Outputs

All examples below write into `src/tools/outputs`.

### 1) Market data

```bash
python src/tools/market_data.py \
  --ticker AAPL \
  --n-days 30 \
  --out-dir src/tools/outputs
```

### 2) Financial statements (raw) + compact payload

```bash
python src/tools/financial_statement.py \
  --ticker AAPL \
  --period quarter \
  --limit 4 \
  --provider yfinance \
  --out src/tools/outputs/openbb_financials_AAPL_quarter_yfinance_income-balance-cash.json

python src/tools/compact_financials.py \
  --in src/tools/outputs/openbb_financials_AAPL_quarter_yfinance_income-balance-cash.json \
  --out src/tools/outputs/aapl_compact.json
```

### 3) Technical indicators

```bash
python src/tools/indicators.py \
  --ticker AAPL \
  --outputs-dir src/tools/outputs \
  --out src/tools/outputs/aapl_indicators.json
```

### 4) News

```bash
python src/tools/news.py \
  --tickers AAPL \
  --limit 30 \
  --out AAPL_news.json \
  --out_dir src/tools/outputs
```

### 5) Portfolio template

```bash
python src/tools/portfolio.py \
  --file src/tools/outputs/portfolio.json \
  template --force
```

## Step 2: Run the Multi-Agent Pipeline

```bash
python run.py --ticker AAPL
```

Default output:

```text
out/AAPL_multiagent.json
```

## Useful CLI Options

```bash
python run.py \
  --ticker AAPL \
  --outputs-dir src/tools/outputs \
  --out out/AAPL_multiagent.json \
  --advice-md \
  --advice-md-out out/AAPL_advice.md
```

Options:
- `--llm`: force-enable LLM calls.
- `--no-llm`: disable LLM calls and use rule-based summaries.
- `--verbose`: print module progress.
- `--quiet`: suppress progress logs.

## Output Schema (high level)

Pipeline output includes:
- `ticker`
- `asof_utc`
- `modules[]` with each module summary/data/source file
- `advice` final merged recommendation block

## Troubleshooting

- `run.py` fails before execution:
  Check `config/config.toml` format and remove duplicate keys.

- `No source file found` for a module:
  Ensure the expected JSON exists in `--outputs-dir` and file naming matches the ticker.

- LLM errors:
  Confirm `base_url`, `api_key`, and `model` are valid. Use `--no-llm` to run pipeline without LLM.

- OpenBB import/runtime errors:
  Install OpenBB extras with `pip install -e '.[openbb]'`.

## Current Workflow Notes

- Module file selection is ticker-aware and validates JSON shape before use.
- Financial module supports both compact JSON and raw `openbb_financials_*.json` input.
- Advisor LLM parsing supports direct advice fields in JSON output.
