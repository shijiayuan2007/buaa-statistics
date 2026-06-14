# 伊以冲突对国际油价冲击效应研究

本项目内容为选题3“中东冲突对国际油价冲击——事件研究法与回归分析”

## 一、项目结构

```text
homework/
├── report.md                         # 完整研究报告 Markdown 版本
├── requirements.txt                  # Python 依赖包
├── README.md                         # 运行说明
├── scripts/
│   ├── config.py                     # 路径、事件、窗口参数配置
│   ├── 01_fetch_data.py              # 唯一需要联网运行的数据抓取脚本
│   └── 02_run_analysis.py            # 本地分析脚本：事件研究、ITS、断点检测、绘图
├── data/
│   ├── raw/                          # 01_fetch_data.py 保存的原始数据
│   └── processed/                    # 01_fetch_data.py 保存的合并后分析数据
├── outputs/
│   ├── figures/                      # 02_run_analysis.py 生成的图像
│   └── tables/                       # 02_run_analysis.py 生成的结果表
└── references/
    └── 2409.19307v3.pdf              # 参考文献：arXiv:2409.19307
```

## 二、安装依赖

建议使用 Python 3.10 或以上版本。

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## 三、运行方法

### 第一步：联网抓取数据

```bash
python scripts/01_fetch_data.py
```
该脚本会抓取并保存：
- `data/raw/oil_prices.csv`：WTI 与 Brent 原油价格；优先使用 FRED 公共 CSV，备用 Yahoo Finance 的 `CL=F` 与 `BZ=F`。
- `data/raw/market_benchmark.csv`：市场基准；优先使用 FRED `SP500`，备用 Yahoo Finance `SPY`。
- `data/raw/events_manual.csv`：本文选取的 7 个事件。
- `data/raw/gdelt_timeline.csv`：GDELT 相关新闻热度辅助数据，若网络接口可用。
- `data/raw/acled_events.csv`：ACLED 辅助数据；若设置了环境变量 `ACLED_EMAIL` 与 `ACLED_KEY`，则尝试下载，否则保存为空表。
- `data/processed/analysis_data.csv`：合并后的本地分析数据，包含价格与对数收益率。

### 第二步：本地运行分析
```bash
python scripts/02_run_analysis.py
```
该脚本不联网，只读取 `data/processed/analysis_data.csv` 与 `data/raw/events_manual.csv`，并输出：
- `outputs/tables/event_study_results.csv`
- `outputs/tables/individual_its_results.csv`
- `outputs/tables/multiple_its_results.csv`
- `outputs/tables/multiple_its_diagnostics.csv`
- `outputs/tables/breakpoint_results.csv`
- `outputs/figures/fig1_oil_price_events.png` 至 `fig9_residual_diagnostics.png`

## 四、核心方法
1. 事件研究法：估计窗口 120 个交易日，事件窗口 `[-5,+10]`，正常收益模型为市场模型。
2. CAR 显著性检验：基于估计窗口残差标准差的标准化 CAR t 检验。
3. 中断时间序列 ITS：每个事件前后各 60 个交易日，估计事件后的水平变化与趋势变化。
4. 残差诊断：Durbin-Watson、Breusch-Pagan、Breusch-Godfrey。
5. Newey-West 修正：对 ITS 回归采用 HAC 标准误。
6. 结构断点检测：使用 `ruptures` 实现 Bai-Perron 思路下的多重断点检测，包含固定断点数动态规划法与 PELT 惩罚法。
