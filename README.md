# stock_agent
**一个面向个人投资人的股票市场分析agent：**
我们做的是一个“投研助理型”个人投资 Agent：它每天自动汇总市场热点，把新闻从碎片化报道整理成结构化事件卡（发生了什么、为什么重要、影响链条、关键观察点），并结合用户的投资组合与风险约束，输出“与你相关”的影响解读与可执行的组合指导（该关注什么、需要降低哪些风险暴露、是否触发再平衡）。它提供可追溯的依据与反证条件，让个人投资者用更少时间获得更专业的投研支持，而不是黑盒信号。

## MVP功能
1. 每日晨报：Top 5 + For You（与你持仓最相关的 3–8 条）
    - 已经完成最简单的版本，详见src/tools/news.py
2. 事件卡结构化（一句话、影响链条、观察点）
3. 组合暴露画像（行业/地区/币种/单票集中度）
4. 组合指导（以再平衡/风险控制/关注清单为主）
5. 跟踪清单：把“观察点”变成提醒（比如财报、数据发布、价格/利率阈值）
    - 实现可以用openbb拉取**价格**、**财报**等数据

---

## 快速开始

### 1) 配置文件
复制并修改 `config/config.toml`：
- 填写 LLM 的 `base_url`、`api_key`、`model`
- 选择行情数据源（目前推荐 `stooq`，无需 API Key）
- 指定 portfolio 路径（默认 `data/portfolio.json`）

### 2) 准备组合文件
编辑 `data/portfolio.json` 填入你的持仓（示例已提供）。

### 3) 运行
```bash
python run.py
```
运行成功后会输出 `out/AAPL.md`（基础报表）以及 `outputs/<date>_daily.md`（完整日报）。

---

## Pipeline 逻辑（端到端）
1. `run.py` 加载 `.env` 和配置文件，并构建日报 graph。
2. `agent.graph` 根据 state 判断是否有 `portfolio_path`：
   - 有：进入完整日报 pipeline
   - 无：输出最小 Markdown 报告用于快速验证
3. `agent.orchestrator` 执行主流程：
   - 读取组合（`portfolio.store`）
   - 拉行情与新闻（`tools.market_data` / `tools.news`）
   - 计算风险与快照（`analysis.risk`）
   - 调用 LLM 输出结构化 JSON（`llm.client`）
   - 渲染并写入 Markdown（`render.render`）

---

## 主要模块说明

### `run.py`
入口脚本，加载配置并执行日报。适合本地跑通 pipeline。

### `src/agent/config.py`
读取 `config/config.toml`，并允许环境变量覆盖。

### `src/agent/graph.py`
日报 graph 接口：支持最小报表或完整日报。

### `src/agent/orchestrator.py`
日报主流程编排（组合 → 行情/新闻 → 风险 → LLM → 渲染）。

### `src/tools/market_data.py`
行情数据抓取。目前支持 `stooq` 免费日线。

### `src/tools/news.py`
新闻数据抓取（GDELT）。

### `src/analysis/risk.py`
计算波动率、最大回撤、平均收益等指标，并生成组合快照。

### `src/llm/client.py`
通过 OpenAI-compatible API 生成结构化 JSON 报告。

### `src/llm/prompts/daily_report.md`
日报 Prompt 模板，要求 LLM 输出严格 JSON。

### `src/render/render.py`
渲染 Markdown 模板并写入输出目录。

### `data/portfolio.json`
你的持仓模板，供 pipeline 读取。

### `config/config.toml`
总配置文件（行情源、LLM、输出路径等）。

---

## 使用教程

### 切换行情数据源
在 `config/config.toml` 中修改：
```toml
[market_data]
provider = "stooq"
```
目前推荐 `stooq`（无需 API key）。

### 切换 LLM
```toml
[llm]
base_url = "https://api.openai.com/v1"
api_key = "YOUR_LLM_API_KEY"
model = "gpt-4o-mini"
```

### 自定义组合
编辑 `data/portfolio.json`：
```json
{
  "holdings": [
    { "ticker": "AAPL", "shares": 10 },
    { "ticker": "TSLA", "shares": 5 }
  ],
  "benchmarks": ["SPY"]
}
```

### 输出位置
`outputs/` 会生成当天日报 Markdown 与来源索引。
