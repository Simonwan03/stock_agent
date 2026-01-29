# stock_agent
**一个面向个人投资人的股票多模块分析 agent（Multi-Agent Pipeline）：**
本项目将行情、财务、技术指标、新闻和组合等模块拆分成独立 agent。每个 agent 负责读取
`src/tools/outputs/` 中的 JSON 输出，并按照统一 schema 结构化输出。最终由 advisor agent 汇总
生成投资建议，整个流程由 `run.py` 一键驱动。

---

## 快速开始

### 1) 生成工具输出（JSON）
请先运行 `src/tools` 下的脚本，确保它们将结果写入 `src/tools/outputs/`。例如：
```bash
# 行情数据（脚本默认保存到 outputs/，下面示例写入 src/tools/outputs）
python - <<'PY'
from src.tools.market_data import build_equity_json, save_equity_json
payload = build_equity_json("AAPL", n_days=30)
path = save_equity_json(payload, out_dir="src/tools/outputs")
print("saved:", path)
PY

# 财务报表与压缩（可选）
python src/tools/financial_statement.py --ticker AAPL --period quarter --limit 4 --provider yfinance \
  --out src/tools/outputs/openbb_financials_AAPL_quarter_yfinance_income-balance-cash.json
python src/tools/compact_financials.py --in src/tools/outputs/openbb_financials_AAPL_quarter_yfinance_income-balance-cash.json \
  --out src/tools/outputs/aapl_compact.json

# 技术指标（可选）
python src/tools/indicators.py --prices src/tools/outputs/AAPL_20260128T125306Z.json \
  --out src/tools/outputs/aapl_indicators.json

# 新闻（可选）
python src/tools/news.py --ticker AAPL --limit 30 --out src/tools/outputs/AAPL.json

# 组合模板（可选）
python src/tools/portfolio.py --file src/tools/outputs/portfolio.json template --force
```

### 2) 运行 Multi-Agent Pipeline
```bash
python run.py --ticker AAPL
```
默认会读取 `src/tools/outputs/` 下的最新 JSON，并输出到：
```
out/AAPL_multiagent.json
```

---

## 运行参数与环境变量

### CLI 参数
```bash
python run.py \
  --ticker AAPL \
  --outputs-dir src/tools/outputs \
  --out out/AAPL_multiagent.json
```

### 环境变量（可选）
```bash
export STOCK_AGENT_TICKER=AAPL
export STOCK_AGENT_OUTPUTS_DIR=src/tools/outputs
export STOCK_AGENT_REPORT_DIR=out
```

---

## Pipeline 逻辑（端到端）
1. `run.py` 加载 `.env` 和环境变量配置，并初始化 orchestrator。
2. 每个模块 agent 从 `outputs` 中挑选**最新**的 JSON 文件并做结构化摘要：
   - `market_data`: 行情价格快照
   - `financials`: 财务摘要（支持压缩后的 compact 输出）
   - `indicators`: 技术指标 + 性能指标
   - `news`: 新闻聚合
   - `portfolio`: 持仓与基准
3. advisor agent 基于各模块输出生成投资建议摘要。
4. 最终将所有模块 + 建议合并写入 `out/<TICKER>_multiagent.json`。

---

## 目录结构（核心）
```
stock_agent/
├─ run.py                       # ✅ Multi-Agent 入口
├─ src/
│  ├─ agent/
│  │  ├─ config.py              # 环境变量配置
│  │  └─ multiagent/
│  │     ├─ agents.py           # 各模块 agent + advisor
│  │     ├─ orchestrator.py     # 流程编排
│  │     └─ schema.py           # 统一输出 schema
│  └─ tools/
│     ├─ market_data.py
│     ├─ financial_statement.py
│     ├─ compact_financials.py
│     ├─ indicators.py
│     ├─ news.py
│     ├─ portfolio.py
│     └─ outputs/               # ✅ 工具输出 JSON 目录
└─ out/                          # ✅ pipeline 汇总输出
```

---

## 输出示例（schema 结构）
```json
{
  "schema_version": "1.0",
  "ticker": "AAPL",
  "asof_utc": "2026-01-28T12:53:05Z",
  "modules": [
    {
      "module": "market_data",
      "schema_version": "1.0",
      "asof_utc": "2026-01-28T12:53:05Z",
      "source_files": ["src/tools/outputs/AAPL_20260128T125306Z.json"],
      "summary": "Latest close 274.61, change -0.55%.",
      "data": { "latest_close": 274.61, "daily_change_pct": -0.0055 }
    }
  ],
  "advice": {
    "agent": "advisor",
    "summary": "基于现有模块数据生成初步投资建议。",
    "signals": [],
    "risk_notes": [],
    "action_items": []
  }
}
```

---

## 常见问题

### 1) outputs 里没有对应的 JSON 会怎样？
对应模块会输出 `status: missing`，其余模块仍正常运行。

### 2) 如何扩展新的模块？
在 `src/agent/multiagent/agents.py` 中继承 `ModuleAgent`，实现 `summarize()`，
并在 `orchestrator._agents()` 里注册对应文件匹配规则即可。
