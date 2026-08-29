# Image Trend

基于 A 股日频 OHLCV 数据生成二值蜡烛图，并使用传统特征与 2D CNN 预测未来股票收益方向。项目流程参考 Jiang, Kelly and Xiu (2023) 的价格图像建模思路，并扩展为矩阵收益研究。

当前版本：`v1.3.1`

## 项目内容

| 实验 | 图像窗口 | 预测收益窗口 | 图像形状 |
| --- | ---: | ---: | --- |
| I5R5 | 5 个交易日 | 5 个交易日 | `[N, 32, 15, 1]` |
| I20R5 | 20 个交易日 | 5 个交易日 | `[N, 64, 60, 1]` |
| I60R5 | 60 个交易日 | 5 个交易日 | `[N, 96, 180, 1]` |
| I20R20 | 20 个交易日 | 20 个交易日 | `[N, 64, 60, 1]` |
| I60R20 | 60 个交易日 | 20 个交易日 | `[N, 96, 180, 1]` |

V1.0 的收益与回测口径改为更接近实盘的执行假设：信号在日期 `t` 收盘后形成，次一交易日 `t+1` 按可交易开盘价买入，并在持有期结束日收盘卖出。

```text
future_ret_h = close_adj[t + h] / open_adj[t + 1] - 1
```

GitHub 仓库保存 Python 源码、`README.md`、`AGENTS.md`、`05_train_cnn2d.md` 和 `.gitignore`。原始数据、生成特征、图像矩阵、模型权重、预测结果和论文 PDF 不上传。

## 文件夹结构

```text
image_trend/
├── AGENTS.md
├── config.py
├── 01_build_panel.py
├── 02_make_labels_and_baselines.py
├── 03_make_images.py
├── 03_make_images_fast.py
├── 04_train_logistic.py
├── 05_train_cnn2d.py
├── 05_train_cnn2d_4090_fast.py
├── 05_train_cnn2d_v131.py
├── aggregate_cnn_v131.py
├── 05_train_cnn2d.md
├── 05_train_cnn2d_v131.md
├── 06_backtest_decile.py
├── 07_backtest_nonoverlap_portfolio.py
├── 08_qa_v131.py
├── slurm/
│   └── run_v131_background.sh  # 远端幂等、脱离 SSH 的后台提交器
├── README.md
├── data/                    # 本地生成，不上传
│   ├── processed/
│   ├── features/
│   └── images/
├── outputs/                 # 本地生成，不上传
│   ├── predictions/
│   ├── models/
│   └── tables/
├── presentation_work/        # 本地材料，不上传
└── Jiang 等 - 2023 - (Re‐)Imag(in)ing Price Trends.pdf  # 本地论文，不上传
```

项目外部原始数据默认位于：

```text
N:\quant\A_share\daily_OHLVC\
├── 不复权\
└── 后复权\
```

## 输入输出

| 文件 | 作用 | 输入 | 输出 | 数据格式 |
| --- | --- | --- | --- | --- |
| `config.py` | 统一配置路径、样本区间、训练切分、实验矩阵、手续费网格和 universe split 参数 | 无直接输入 | 自动创建 `data/`、`outputs/` 及其子目录 | Python 配置对象 |
| `01_build_panel.py` | 合并每只股票的不复权和后复权日频行情，生成标准股票-日期面板 dataset | `../daily_OHLVC/不复权/*.csv`；`../daily_OHLVC/后复权/*.csv` | `data/processed/panel_by_code/code=*/part.parquet`；`data/processed/panel_by_year/year=*/part-*.parquet` | 输入为逐股票 CSV；输出为按 `code` 和按 `year` 分区的 Parquet dataset，包含 raw/adjusted OHLCV、成交额、市值、行业、ST、涨停等字段 |
| `02_make_labels_and_baselines.py` | 生成未来收益标签、传统量价基线特征和可执行回测收益字段 | `data/processed/panel_by_code/code=*/part.parquet` | `data/features/features_by_code_bucket/bucket=*/part-*.parquet`；`data/features/features_by_year/year=*/part-*.parquet` | 双 Parquet dataset；按 `code bucket` 分区供 rolling/image 使用，按 `year` 分区供训练和横截面扫描使用；包含 `future_ret_{h}d`、`label_{h}d`、`ret_1d`、`open_to_close_ret_1d`、动量/反转/波动率/流动性/市值特征、`is_tradable` 和低量涨跌停交易约束字段 |
| `03_make_images.py` | 按唯一图像窗口生成 Jiang 风格二值价格图像 shard，并把多个 horizon 标签挂到同一张图像 metadata 上 | `data/features/features_by_code_bucket/bucket=*/part-*.parquet` | `data/images/window_{window}/shard_*/images.npy`；`data/images/window_{window}/shard_*/meta.parquet` | 每个 shard 的 `.npy` 为 `uint8` 图像矩阵 `[N,H,W,1]`，像素值 `0/255`；metadata 为 Parquet，每行对应同 shard 中一张图，并包含 `label_{h}d/future_ret_{h}d` |
| `04_train_logistic.py` | 对每个实验训练传统特征 Logistic 基线 | `data/features/features_by_year/year=*/part-*.parquet` | `outputs/predictions/pred_{experiment}_logistic.parquet` | Parquet；包含测试集 `date`、`code`、`future_ret`、`label`、`experiment_name`、`window`、`horizon`、`model_name`、`pred_prob` |
| `05_train_cnn2d.py` | 官方 CNN 训练入口，面向 4080/4090 优化，并支持论文式多 run 概率平均 | `data/images/window_{window}/shard_*/images.npy`；`data/images/window_{window}/shard_*/meta.parquet` | `outputs/models/jiang_cnn2d_{experiment}.pt` 或 `outputs/models/jiang_cnn2d_{experiment}_run*.pt`；`outputs/predictions/pred_{experiment}_jiang_cnn2d.parquet`；`outputs/tables/cnn_training_log_{experiment}*.csv`；`outputs/tables/cnn_ensemble_summary_{experiment}.csv` | 支持 lazy shard memmap、AMP/TF32、训练诊断、AdamW、RankIC checkpoint、可选 spatial dropout/reslite；默认 `--ensemble-runs 1`，使用 `--ensemble-runs 5` 时独立训练 5 次并对 `pred_prob` 做算术平均 |
| `05_train_cnn2d_4090_fast.py` | 兼容入口，转发执行 `05_train_cnn2d.py` | 同 `05_train_cnn2d.py` | 同 `05_train_cnn2d.py` | 保留给旧集群脚本使用，不再维护第二份训练逻辑 |
| `05_train_cnn2d_v131.py` | v1.3.1 Purged 五折 CNN 消融训练 | `data_v1_3_1/images/window_{window}`；`data_v1_3_1/features/features_by_year` | `outputs/v1_3_1/models`、`outputs/v1_3_1/predictions/members`、训练日志和完成清单 | BCE、Huber、Huber+IC；连续预2021五折、±20交易日 purge、42/43/44/45 四种 seed；日期级逻辑 batch、按显存自动选择微批、向量化 shard 读取、warm-up、ReduceLROnPlateau、25轮上限和 patience=4/min_delta=1e-3 |
| `aggregate_cnn_v131.py` | 校验20个成员并聚合 v1.3.1 信号 | `outputs/v1_3_1/predictions/members/{loss}/{experiment}` | `outputs/v1_3_1/predictions/pred_*_k5s4.parquet`、ensemble summary | 每个成员原始 logit 逐日 z-score 后算术平均；同时写入 `pred_score`、兼容列 `pred_prob`、平均概率诊断和成员计数 |
| `06_backtest_decile.py` | 评估预测效果、Decile 单调性、D1-D10 long-only 组合和 D10-D1 long-short 组合 | `outputs/predictions/pred_*.parquet`；`data/features/features_by_year/year=*/part-*.parquet` | `outputs/tables/*.csv` | CSV；包含 IC、RankIC、累计 IC、Decile 未来收益、单调性、组合收益、换手、逐日净值和回撤、手续费敏感度、绩效汇总、有效收益覆盖率和收益磨损归因 |
| `07_backtest_nonoverlap_portfolio.py` | 评估严格非重叠持仓组合，每隔 horizon 个交易日一次性全仓换股，并构建 D10-D1 自筹资多空组合 | `outputs/predictions/pred_*.parquet`；`data/features/features_by_year/year=*/part-*.parquet` | `outputs/tables/nonoverlap_*.csv` | CSV；包含非重叠组合收益、D10 long-only 专用收益、换手、逐日净值和回撤、手续费敏感度、绩效汇总、有效收益覆盖率、收益磨损归因，以及 Logistic/CNN 回测组合持仓重叠度；默认不覆盖 `06_backtest_decile.py` 输出 |
| `08_qa_v131.py` | v1.3.1 数据、图像、标签、键和聚合产物 QA | `data_v1_3_1`、可选旧版 `data` | `outputs/v1_3_1/tables/qa/data_qa_v131.json` | 检查 2009–2024 年份、date/code 唯一键、图像与 metadata 行数/形状/二值像素、标签尾部完整性；可抽样比较新旧共同样本 |
| `slurm/run_v131_background.sh` | 在远端脱离 SSH 安全提交 v1.3.1 依赖链 | 已提交的数据重建和 smoke 作业 ID | `outputs/v1_3_1/logs/background/` 下的 PID、日志、启动/提交清单 | 原子锁和 marker 防重复提交；父进程立即返回，子进程由 `nohup` 脱离终端运行 |
| `slurm/sub_v131_perf_smoke.sh` | 4080/4090 输入管线与显存压力测试 | 已完成的 v1.3.1 图像和特征 | `outputs/v1_3_1_perf_smoke/` 与正式日志目录 | 分别覆盖 I5 BCE 和 I60 Huber+IC，不写入正式成员清单；可用 `V131_PERF_MODE=i60_contiguous` 比较内存格式 |
| `slurm/sub_v131_i5_pack_probe.sh` | I5 同卡 2–5 进程基准 | 期末 64 个最大股票截面 | `outputs/v1_3_1_pack_probe/` | 独立比较整卡吞吐、墙钟时间和 OOM；不写入正式成员清单 |
| `slurm/sub_v131_train_i5_packed.sh` | I5 正式同卡并行训练 | I5R5 的 20 个 fold/seed 成员 | 正式模型、成员预测、manifest 与 packed 日志 | 4 个数组任务，每卡并发 5 个成员；若偶发失败，仅将缺失 manifest 对应成员顺序重试 |
| `slurm/migrate_v131_to_optimized.sh` | 从旧 4090-only 队列安全迁移到优化队列 | 旧作业 ID 与两个在途成员 | `outputs/v1_3_1/logs/migration/` | 等待成员的 manifest/model/prediction 全部原子落盘后才取消精确旧作业并提交训练-only依赖链 |

`06_backtest_decile.py` 输出表：

```text
outputs/tables/ic_by_period.csv
outputs/tables/ic_summary.csv
outputs/tables/cumulative_ic.csv
outputs/tables/decile_returns.csv
outputs/tables/decile_summary.csv
outputs/tables/decile_monotonicity.csv
outputs/tables/portfolio_returns.csv
outputs/tables/portfolio_turnover.csv
outputs/tables/portfolio_nav.csv
outputs/tables/performance_summary.csv
outputs/tables/cost_sensitivity.csv
outputs/tables/return_attribution.csv
```

其中组合回测相关表的当前口径如下：

- `portfolio_returns.csv`：逐日组合收益明细，包含 D1-D10 long-only 组合和 D10-D1 long-short 组合；保留 `gross_return/net_return`、换手、交易阻塞、强制持有、缺失收益覆盖率等诊断字段。
- `portfolio_turnover.csv`：逐日换手明细，买入和卖出 turnover 分开记录；低量涨跌停、停牌和缺失收益导致的执行问题会计入诊断字段。
- `portfolio_nav.csv`：逐日净值曲线，字段为 `date/experiment_name/model_name/universe_group/portfolio_name/cost_bps/gross_nav/net_nav/drawdown`；净值在剔除 warmup 期后从 1 开始复利计算，`drawdown` 使用 `net_nav` 相对历史高点计算。
- `performance_summary.csv`：按组合汇总年化收益、年化波动率、Sharpe、最大回撤、胜率、平均换手、gross/net 累计收益和最终 NAV；累计收益和 NAV 使用复利，年化收益仍使用日均净收益乘以 252。
- `return_attribution.csv`：按组合汇总收益磨损归因，拆分信号毛收益、买入阻塞损失、卖出阻塞强制持有损失、缺失收益数据影响和交易成本。

`06_backtest_decile.py` 当前实现为了降低大样本回测开销，组合持仓内部使用整数化 `code_id`、数组化权重向量、到期卖出队列和 forced-hold 队列管理 active cohorts。该优化只改变执行效率，不改变组合收益、涨跌停/停牌处理、手续费或 warmup 统计口径。

`07_backtest_nonoverlap_portfolio.py` 输出表：

```text
outputs/tables/nonoverlap_portfolio_returns.csv
outputs/tables/nonoverlap_portfolio_turnover.csv
outputs/tables/nonoverlap_portfolio_nav.csv
outputs/tables/nonoverlap_performance_summary.csv
outputs/tables/nonoverlap_cost_sensitivity.csv
outputs/tables/nonoverlap_return_attribution.csv
outputs/tables/nonoverlap_d10_long_only_returns.csv
outputs/tables/nonoverlap_d10_long_only_nav.csv
outputs/tables/nonoverlap_d10_long_only_performance.csv
outputs/tables/nonoverlap_portfolio_holdings.csv
outputs/tables/nonoverlap_holding_overlap.csv
outputs/tables/nonoverlap_holding_overlap_summary.csv
```

`07_backtest_nonoverlap_portfolio.py` 的组合口径为严格非重叠持仓：对 horizon 为 `H` 的预测，只在每个实验/模型/universe 组的第一天可用信号起，每隔 `H` 个交易日调仓一次。D10 多头腿等权到总权重 `+1`，D1 空头腿等权到总权重 `-1`，D10-D1 为自筹资多空组合，不使用 `1/H` 子组合权重。

`nonoverlap_d10_long_only_*.csv` 是从现有 D10 long-only 结果中过滤出的专用表，收益、净值、手续费和绩效字段与 `nonoverlap_portfolio_*` 原表一致。`nonoverlap_portfolio_holdings.csv` 只记录每次实际建仓后的回测组合腿持仓：`D10/long`、`D10_minus_D1/long` 和 `D10_minus_D1/short`。`nonoverlap_holding_overlap.csv` 按相同 experiment、universe、portfolio、leg 和 rebalance date 配对 Logistic 与 CNN，输出共同持仓数、并集持仓数和 Jaccard 重叠度；summary 表按组合腿汇总重叠度分布。

## v1.3.1 训练与发布口径

v1.3.1 使用独立的 `data_v1_3_1/` 和 `outputs/v1_3_1/`，不会覆盖 v1.2.x 产物。数据从 2009-01-01 重建到 2024-12-31；不引入 2008 年预热行情，I5/I20/I60 在各自完整回看窗口后自然产生首个图像样本。五折边界由 2009–2020 的统一交易日历按交易日数量切分，验证块前后各剔除 20 个交易日，所有成员统一推断 2021–2024，并仅保留具有完整未来收益的预测行。

每种损失包含 5 folds × 4 seeds（42、43、44、45），五组实验按 `BCE → Huber → Huber+IC` 串行执行。逻辑 batch 是一个完整交易日截面，`shuffle=False`、代码顺序确定；物理微批默认按 GPU 总显存和实验窗口自动选择，也可显式覆盖并用 `--max-micro-batch-size` 限制。I5/I20 使用完整单日截面；I60 在 RTX 4080/4090 上分别自动使用 896/1216，实测对应约 14.63 GiB/目标约 20 GiB。DataLoader 对一个日期涉及的每个 shard 只执行一次向量化 memmap 读取，避免逐样本打开和 Python tensor 分配。Huber 目标按日期做 1%/99% 去极值和总体标准差 z-score。训练使用 AdamW (`lr=1e-4`, `weight_decay=3e-5`)、Kaiming normal、AMP/TF32、pinned memory、persistent workers、双缓冲 CUDA 预取、2轮线性 warm-up、ReduceLROnPlateau（factor=0.5、patience=1、min_lr=1e-6），最多 25 轮；无最少轮数，验证目标连续 4 轮未超过 `1e-3` 改善即早停。Huber+IC 的目标为 `Huber + 1.0 × (1 - PearsonIC)`，IC 在完整日期截面计算，外部 Pearson 梯度显式保持零和并以 float32 重放；IC 退化检查留在 GPU，日期内日志标量合并同步，重放之间不执行全设备 barrier。每个逻辑模型在启动或断点续跑时重置对应 seed，CPU 兼容路径的归一化不修改 DataLoader 原始 batch。

远端集群可从项目根目录提交：

```bash
bash slurm/submit_v131_pipeline.sh
```

提交脚本按 `afterok` 串联数据重建、三种损失及各阶段聚合和两套回测；每个 loss 内，I5 使用 4 个 packed 数组任务、每卡并发 5 个成员，I20R5/I60R5/I20R20/I60R20 则拆成四个独立的 20 成员数组并行调度，全部完成后才聚合。GPU 任务使用通用 `gpu:1`，因此可调度 RTX 4080 或 RTX 4090，同时排除 V100 节点。普通数组默认最多并发 16 个任务，单任务申请 10 个 CPU；作业将文件描述符软限制提高到 65536，使用 8 个 persistent workers、`prefetch_factor=2`、4096 shard mmap cache，并设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。单独 smoke 测试使用 `sbatch slurm/sub_v131_smoke.sh`，输出到 `outputs/v1_3_1_smoke/`，不会污染全量完成清单。

如果数据重建任务已经单独提交，可通过 `V131_DATA_JOB_ID=<jobid> bash slurm/submit_v131_pipeline.sh` 复用该任务作为 `afterok` 起点，避免重复重建面板、特征和图像；首阶段也可同时设置 `V131_SMOKE_JOB_ID=<jobid>`，使训练等待数据和 smoke 均成功。不设置数据变量时，提交脚本会自动新建数据重建任务。

退出 SSH/Codex 前推荐使用自后台化提交器（示例中的 `84642` 和 `84643` 分别是数据与 smoke 作业 ID）：

```bash
bash slurm/run_v131_background.sh 84642 84643
```

脚本会立即返回并把子进程重定向到 `outputs/v1_3_1/logs/background/launcher_*.log`；成功提交后写入 `pipeline_submitted.txt`，重复执行会直接显示清单而不会再创建训练作业。若提交中途失败，会保留 `pipeline_started.txt` 以阻止不安全的重复提交，需先检查对应日志和 Slurm 状态。

## 推荐运行顺序

```powershell
uv run python 01_build_panel.py
uv run python 02_make_labels_and_baselines.py
uv run python 03_make_images.py
uv run python 04_train_logistic.py
uv run python 05_train_cnn2d.py
uv run python 06_backtest_decile.py
```

如需验证严格非重叠持仓口径，额外运行：

```powershell
uv run python 07_backtest_nonoverlap_portfolio.py
```

若要复现 Jiang 等论文中“同一模型配置独立训练 5 次、最终概率取算术平均”的 CNN 口径，运行：

```powershell
uv run python 05_train_cnn2d.py --ensemble-runs 5
```

如果从 v0.1.x 升级到 v1.0，需要从 `02_make_labels_and_baselines.py` 开始重新生成下游数据，因为未来收益标签和组合回测收益字段已经改变。

如果从 v1.0 升级到 v1.1，也需要从 `02_make_labels_and_baselines.py` 开始重新生成下游数据，因为新增了 `limit_pct`、`volume_mean_20d_prev`、`volume_ratio_to_20d_prev`、`is_low_volume_limit_up` 和 `is_low_volume_limit_down`。

如果从 v1.1 升级到 v1.2，需要重新运行 `01_build_panel.py`。当前面板格式已从旧的单文件 `data/processed/panel_daily.parquet` 迁移为 `panel_by_code` 和 `panel_by_year` 两个 Parquet dataset。后续脚本不再读取旧单文件。

如果从旧的单文件特征格式升级到 v1.2.1，需要重新运行 `02_make_labels_and_baselines.py`。当前特征格式已从 `data/features/baseline_features.parquet` 迁移为 `features_by_code_bucket` 和 `features_by_year` 两个 Parquet dataset。后续脚本不再读取旧单文件。

当前图像格式已按唯一 window 去重存储：`I20R5/I20R20` 共享 `data/images/window_20`，`I60R5/I60R20` 共享 `data/images/window_60`。如果本地仍有旧的 `data/images/i20r5`、`data/images/i20r20` 等目录，需要重新运行 `03_make_images.py` 生成新的 window 级 shard。

如果从 v1.2.1 升级到 v1.2.2，需要重新运行 `03_make_images.py` 或兼容入口 `03_make_images_fast.py`。当前图像生成已将“图像 window”和“标签 horizon”解耦，同一个 window 图像只生成一次，多个 horizon 标签写入同一份 metadata。

如果从 v1.2.3 升级到 v1.2.4，只需要重新运行 `06_backtest_decile.py`。当前回测新增 D10-D1 long-short、缺失收益覆盖率诊断、warmup 期过滤和 `return_attribution.csv` 收益归因输出，不需要重新生成面板、特征、图像或模型预测。

如果从 v1.2.4 升级到 v1.2.5，不需要重新生成图像；`05_train_cnn2d_4090_fast.py` 已改为兼容入口，正式训练入口统一为 `05_train_cnn2d.py`。默认仍训练 1 次；如需论文式 ensemble，需要重新运行 `05_train_cnn2d.py --ensemble-runs 5` 生成新的 CNN 预测。由于 `06_backtest_decile.py` 新增 `portfolio_nav.csv` 并优化了组合回测执行路径，如需生成逐日净值和回撤曲线，需要重新运行 `06_backtest_decile.py`。

如果从 v1.2.5 升级到 v1.2.6，不需要重新生成面板、特征、图像或模型预测。新增的非重叠持仓回测只需要运行 `07_backtest_nonoverlap_portfolio.py`，输出文件使用 `nonoverlap_` 前缀，不覆盖 `06_backtest_decile.py` 的重叠持仓结果。

如果从 v1.2.6 升级到 v1.2.7，不需要重新生成面板、特征、图像或模型预测。只需重新运行 `07_backtest_nonoverlap_portfolio.py`，即可补充 D10 long-only 专用输出和 Logistic/CNN 回测组合持仓重叠度输出。

若依赖缺失，请统一安装后再运行。当前项目约定不在脚本中自动安装依赖。

## 版本记录

后续每次推送前应更新本表，并创建同名 Git tag。

| 日期 | 版本 | 推送内容 | 新增功能 | 待更新功能 |
| --- | --- | --- | --- | --- |
| 2026-08-29 | `v1.3.1` | 4080/4090 GPU 输入与同步优化 | 日期级读取改为按 shard 向量化批量加载；I60 微批按卡型设为 896/1216，4090 目标约 20 GiB；消除日期级 device barrier 和微批级 `.item()` 同步；提高 worker、mmap cache、文件描述符上限和数组并发；Slurm 改为同时接受 RTX 4080/4090 并继续排除 V100；增加独立性能 smoke 与安全迁移脚本 | I5 输入优化实测吞吐约提升 1.65 倍、等待占比约从 94% 降至 8%；I60/Huber+IC 在 4080 上 batch=896 峰值 14.63 GiB、等待约 4.1%；继续观察全量训练的共享存储压力 |
| 2026-08-29 | `v1.3.1` | I5 同卡并发与实验级数组拆分 | 用最大期末截面测试同卡 2/3/4/5 个 I5 模型；正式选择每卡 5 个以降低饱和集群下的总 GPU 时间，20 个成员仅占 4 张卡；I20/I60 四组实验分别提交独立数组 | 2/3/4/5 路分别用 78/91/107/124 秒完成对应数量的完整 smoke，均无 OOM；5 路正式卡实测 util 100%，并保留缺失成员顺序重试保护 |
| 2026-08-28 | `v1.3.1` | 2009–2024 数据重建与 Purged 五折 CNN 消融 | 新增独立版本数据/输出目录；BCE、Huber、Huber+IC 三阶段；5 folds × 4 seeds；日期级 batch、±20 交易日 purge、warm-up、Kaiming、Pinned/CUDA 预取、成员校验聚合与两套回测；新增数据 QA 和 Slurm 编排 | 保留远端 checkpoint、成员预测和图像；后续可接入现成策略平台做外部执行回测 |
| 2026-06-15 | `v1.2.7` | 非重叠回测 D10 long-only 与持仓对比诊断 | `07_backtest_nonoverlap_portfolio.py` 新增 D10 long-only 专用收益、净值和绩效表；新增回测组合构建时的持仓集合输出，并按 `D10/long`、`D10_minus_D1/long`、`D10_minus_D1/short` 计算 Logistic/CNN 持仓 Jaccard 重叠度和汇总表 | 将持仓重叠度接入可视化报告；评估是否将大型明细输出改为 Parquet |
| 2026-06-13 | `v1.2.6` | 非重叠持仓组合回测 | 新增 `07_backtest_nonoverlap_portfolio.py`，按每个实验/模型/universe 的首个可用信号作为锚点，每隔 horizon 个交易日一次性全仓换股；D10 多头腿等权到 `+1`，D1 空头腿等权到 `-1`，直接构建 D10-D1 自筹资多空组合；输出 `nonoverlap_*.csv`，避免覆盖 `06_backtest_decile.py` 的重叠持仓结果 | 对非重叠结果补充分年度、行业暴露和 beta 暴露报表；评估是否将大型明细输出改为 Parquet |
| 2026-06-11 | `v1.2.5` | CNN 训练入口合并、论文式 ensemble 支持和回测净值输出 | `05_train_cnn2d.py` 合并 4090 fast 训练逻辑，成为唯一正式 CNN 训练入口；`05_train_cnn2d_4090_fast.py` 改为兼容 wrapper；新增 `--ensemble-runs`，支持每个 experiment 独立训练多次并对测试集概率做算术平均，`--ensemble-runs 5` 对应论文式 5-run 预测信号；`06_backtest_decile.py` 新增 `portfolio_nav.csv`，输出逐日 `gross_nav/net_nav/drawdown`；组合回测 hot path 改为整数化 code id、数组化权重向量、到期卖出队列和 forced-hold 队列，降低 Python dict/list 遍历开销且不改变既有回测逻辑 | 评估 5-run ensemble 的训练耗时和预测收益表现；如默认启用 5-run，需要同步调整 Slurm walltime 和模型存储策略；评估是否将大型明细输出改为 Parquet；补充分年度、行业暴露和 beta 暴露报表 |
| 2026-06-11 | `v1.2.4` | 回测执行诊断和收益归因升级 | `06_backtest_decile.py` 缺失收益不再导致整日组合收益变成 NaN，改为有效收益股票加权平均；买入受阻股票不进入 cohort，剩余股票在 cohort 内重新等权；卖出受阻股票继续持有，并继续 mark-to-market；新增 `D10_minus_D1` long-short 组合；增加收益归因输出，拆分 `signal_gross_alpha - buy_blocked_loss - sell_blocked_forced_hold_loss - missing_return_data_issue - turnover_cost = attributed_net_return`；多周期组合权重改为独立子组合法，并在绩效统计中剔除前 `horizon` warmup 期 | 进一步数组化 active cohort 和 daily lookup；评估是否将大型明细输出改为 Parquet；补充分年度、行业暴露和 beta 暴露报表 |
| 2026-06-02 | `v1.2.3` | 4090 CNN 训练诊断和策略升级 | 新增 `05_train_cnn2d_4090_fast.py`；支持 lazy shard memmap、writable uint8 copy、AMP/TF32、AdamW、可配置学习率/weight decay/scheduler/warmup、`fc_dropout`、默认关闭的 `spatial_dropout`、可选 `reslite` 架构、validation RankIC/decile 诊断、batch 级效率日志和 `cnn_training_log_{experiment}.csv`；移除 tqdm 依赖，改用 `--log-interval` 输出集群日志 | 根据训练日志判断是否实现 shard-aware sampler 或合并训练 shard；补充分年度、行业暴露和 beta 暴露报表 |
| 2026-06-01 | `v1.2.2` | 图像生成去重与 fast 入口正式化 | `03_make_images.py` 采用优化后的 feature part 进程池路径，并按唯一 `window` 输出 `data/images/window_{window}`；`I20R5/I20R20`、`I60R5/I60R20` 共享物理图像，metadata 同时保存 `label_{h}d/future_ret_{h}d`；`05_train_cnn2d.py` 按 experiment horizon 选择标签列；`03_make_images_fast.py` 改为兼容入口，避免维护两份图像生成逻辑 | 增加结构化 experiment log；补充分年度、行业暴露和 beta 暴露报表 |
| 2026-05-27 | `v1.2.1` | 图像与特征分区 IO 优化 | `03_make_images.py` 改为每个实验输出 `shard_*/images.npy` 和 `shard_*/meta.parquet`；新增 `IMAGE_SHARD_SIZE`；`05_train_cnn2d.py` 改为跨 shard 读取 memmap，并用 `shard_id/local_index` 定位样本；`02_make_labels_and_baselines.py` 改为逐股票计算、按 `features_by_code_bucket` 和 `features_by_year` 批量写 Parquet dataset；`03/04/06` 同步改为从新 feature dataset 投影读取，降低小文件 IO、内存峰值和全量读写开销 | 增加结构化 experiment log；补充分年度、行业暴露和 beta 暴露报表 |
| 2026-05-25 | `v1.2` | 面板构建性能和 Parquet 数据格式升级 | `01_build_panel.py` 改为输出 `panel_by_code` 和 `panel_by_year` 双 Parquet dataset；默认启用 12 进程按股票读取、清洗和合并；保留 `--workers 1` 单进程路径；新增 `--limit-codes` 小样本测试入口；CSV 默认 `gbk` 并保留 fallback；数值转换加速；面板 schema 下压为 `float32/float64/int8`；`02_make_labels_and_baselines.py` 改为从 `panel_by_code` 读取；文档同步新数据格式 | 将图像生成进一步 shard 化；增加结构化 experiment log；补充分年度、行业暴露和 beta 暴露报表；进一步处理北交所 30% 涨跌停和更细滑点模型 |
| 2026-05-25 | `v1.1` | 防过拟合与可交易性修正 | 增加按 horizon 的 purge/embargo 切分；增加随机种子和 CNN weight decay；新增按日期+代码板块的涨跌停阈值与低量涨跌停标记；回测中低量涨停不可买、低量跌停/停牌/缺收益延迟卖出；缺失收益不再填 0；新增 blocked buy/sell、forced hold、data missing 诊断字段；增加未来收益复利一致性检查 | 增加结构化 experiment log；补充分年度、行业暴露和 beta 暴露报表；进一步处理北交所 30% 涨跌停和更细滑点模型 |
| 2026-05-09 | `v1.0` | 重大版本：收益标签、回测评估和组合绩效体系升级 | 未来收益改为次日开盘买入、持有期末收盘卖出；新增 `open_to_close_ret_1d`；新增 IC/RankIC、ICIR、累计 IC、各期 IC；新增 D1-D10 decile 平均未来收益和单调性监测；新增 D1-D10 long-only 重叠持仓组合；新增 turnover、gross/net return、手续费敏感度曲线；新增 large-cap vs small/mid-cap universe split；移除多空组合收益输出；新增预测和日收益字段检查 | 完善训练日志与独立参数配置文件；补充正式单元测试和全链路 schema 检查；加入混合精度和多卡训练支持；加入更严格交易约束，如涨停无法买入、停牌处理和滑点模型 |
| 2026-05-08 | `v0.1.1` | 上传 CNN 训练脚本详细说明文档 | 将 `05_train_cnn2d.md` 纳入 Git 版本管理，便于同步查看 05 脚本的逐函数、全流程和变量解释 | ~~按持有期修正回测年化因子~~；~~增加 RankIC、ICIR 等测评~~；~~实现重叠持仓组合~~；增加训练日志与独立参数配置文件；补充正式单元测试和全链路 schema 检查；加入混合精度和多卡训练支持 |
| 2026-05-08 | `v0.1.0` | 首次源码入库 | 建立 A 股日频面板、标签与基线特征、I5/I20/I60 图像生成、矩阵收益实验、Logistic 基线、Jiang 风格 2D CNN、十分组回测 | ~~按持有期修正回测年化因子~~；~~实现重叠持仓组合~~；增加训练日志与独立参数配置文件；补充正式单元测试和全链路 schema 检查；加入混合精度和多卡训练支持 |
