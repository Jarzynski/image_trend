# Image Trend

基于 A 股日频 OHLCV 数据生成二值蜡烛图，并使用传统特征与 2D CNN 预测未来股票收益方向。项目流程参考 Jiang, Kelly and Xiu (2023) 的价格图像建模思路，并扩展为矩阵收益研究。

当前版本：`v1.2.4`

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

GitHub 仓库保存 Python 源码、`README.md`、`05_train_cnn2d.md` 和 `.gitignore`。原始数据、生成特征、图像矩阵、模型权重、预测结果和论文 PDF 不上传。

## 文件夹结构

```text
image_trend/
├── config.py
├── 01_build_panel.py
├── 02_make_labels_and_baselines.py
├── 03_make_images.py
├── 03_make_images_fast.py
├── 04_train_logistic.py
├── 05_train_cnn2d.py
├── 05_train_cnn2d_4090_fast.py
├── 05_train_cnn2d.md
├── 06_backtest_decile.py
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
| `05_train_cnn2d.py` | 对每个实验训练 Jiang/Kelly/Xiu 风格 2D CNN | `data/images/window_{window}/shard_*/images.npy`；`data/images/window_{window}/shard_*/meta.parquet` | `outputs/models/jiang_cnn2d_{experiment}.pt`；`outputs/predictions/pred_{experiment}_jiang_cnn2d.parquet` | `.pt` 为 PyTorch `state_dict`；同一 window 图像可供多个 horizon 实验复用，训练时按 experiment 选择对应标签列 |
| `05_train_cnn2d_4090_fast.py` | 面向 4080/4090 的 CNN 训练脚本，增加 lazy memmap、AMP/TF32、训练诊断、AdamW、RankIC checkpoint、可选 spatial dropout 和 reslite ablation | `data/images/window_{window}/shard_*/images.npy`；`data/images/window_{window}/shard_*/meta.parquet` | `outputs/models/jiang_cnn2d_{experiment}.pt`；`outputs/predictions/pred_{experiment}_jiang_cnn2d.parquet`；`outputs/tables/cnn_training_log_{experiment}.csv` | 默认保持 Jiang baseline 输出兼容；非默认 `--arch` 会在输出文件名和 `model_name` 中标记架构 |
| `06_backtest_decile.py` | 评估预测效果、Decile 单调性、D1-D10 long-only 组合和 D10-D1 long-short 组合 | `outputs/predictions/pred_*.parquet`；`data/features/features_by_year/year=*/part-*.parquet` | `outputs/tables/*.csv` | CSV；包含 IC、RankIC、累计 IC、Decile 未来收益、单调性、组合收益、换手、手续费敏感度、绩效汇总、有效收益覆盖率和收益磨损归因 |

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
outputs/tables/performance_summary.csv
outputs/tables/cost_sensitivity.csv
outputs/tables/return_attribution.csv
```

## 推荐运行顺序

```powershell
uv run python 01_build_panel.py
uv run python 02_make_labels_and_baselines.py
uv run python 03_make_images.py
uv run python 04_train_logistic.py
uv run python 05_train_cnn2d.py
uv run python 06_backtest_decile.py
```

如果从 v0.1.x 升级到 v1.0，需要从 `02_make_labels_and_baselines.py` 开始重新生成下游数据，因为未来收益标签和组合回测收益字段已经改变。

如果从 v1.0 升级到 v1.1，也需要从 `02_make_labels_and_baselines.py` 开始重新生成下游数据，因为新增了 `limit_pct`、`volume_mean_20d_prev`、`volume_ratio_to_20d_prev`、`is_low_volume_limit_up` 和 `is_low_volume_limit_down`。

如果从 v1.1 升级到 v1.2，需要重新运行 `01_build_panel.py`。当前面板格式已从旧的单文件 `data/processed/panel_daily.parquet` 迁移为 `panel_by_code` 和 `panel_by_year` 两个 Parquet dataset。后续脚本不再读取旧单文件。

如果从旧的单文件特征格式升级到 v1.2.1，需要重新运行 `02_make_labels_and_baselines.py`。当前特征格式已从 `data/features/baseline_features.parquet` 迁移为 `features_by_code_bucket` 和 `features_by_year` 两个 Parquet dataset。后续脚本不再读取旧单文件。

当前图像格式已按唯一 window 去重存储：`I20R5/I20R20` 共享 `data/images/window_20`，`I60R5/I60R20` 共享 `data/images/window_60`。如果本地仍有旧的 `data/images/i20r5`、`data/images/i20r20` 等目录，需要重新运行 `03_make_images.py` 生成新的 window 级 shard。

如果从 v1.2.1 升级到 v1.2.2，需要重新运行 `03_make_images.py` 或兼容入口 `03_make_images_fast.py`。当前图像生成已将“图像 window”和“标签 horizon”解耦，同一个 window 图像只生成一次，多个 horizon 标签写入同一份 metadata。

如果从 v1.2.3 升级到 v1.2.4，只需要重新运行 `06_backtest_decile.py`。当前回测新增 D10-D1 long-short、缺失收益覆盖率诊断、warmup 期过滤和 `return_attribution.csv` 收益归因输出，不需要重新生成面板、特征、图像或模型预测。

若依赖缺失，请统一安装后再运行。当前项目约定不在脚本中自动安装依赖。

## 版本记录

后续每次推送前应更新本表，并创建同名 Git tag。

| 日期 | 版本 | 推送内容 | 新增功能 | 待更新功能 |
| --- | --- | --- | --- | --- |
| 2026-06-11 | `v1.2.4` | 回测执行诊断和收益归因升级 | `06_backtest_decile.py` 缺失持仓收益改为按有效收益股票加权平均，并记录 `valid_weight/missing_weight`；`performance_summary.csv` 增加 `valid_day_ratio`、`avg_valid_weight`、`low_coverage_day_ratio`；新增 `D10_minus_D1` long-short 组合；新增 `return_attribution.csv`，拆分 `signal_gross_alpha - buy_blocked_loss - sell_blocked_forced_hold_loss - missing_return_data_issue - turnover_cost = attributed_net_return`；多周期组合权重改为独立子组合法，并在绩效统计中剔除前 `horizon` warmup 期 | 进一步数组化 active cohort 和 daily lookup；评估是否将大型明细输出改为 Parquet；补充分年度、行业暴露和 beta 暴露报表 |
| 2026-06-02 | `v1.2.3` | 4090 CNN 训练诊断和策略升级 | 新增 `05_train_cnn2d_4090_fast.py`；支持 lazy shard memmap、writable uint8 copy、AMP/TF32、AdamW、可配置学习率/weight decay/scheduler/warmup、`fc_dropout`、默认关闭的 `spatial_dropout`、可选 `reslite` 架构、validation RankIC/decile 诊断、batch 级效率日志和 `cnn_training_log_{experiment}.csv`；移除 tqdm 依赖，改用 `--log-interval` 输出集群日志 | 根据训练日志判断是否实现 shard-aware sampler 或合并训练 shard；补充分年度、行业暴露和 beta 暴露报表 |
| 2026-06-01 | `v1.2.2` | 图像生成去重与 fast 入口正式化 | `03_make_images.py` 采用优化后的 feature part 进程池路径，并按唯一 `window` 输出 `data/images/window_{window}`；`I20R5/I20R20`、`I60R5/I60R20` 共享物理图像，metadata 同时保存 `label_{h}d/future_ret_{h}d`；`05_train_cnn2d.py` 按 experiment horizon 选择标签列；`03_make_images_fast.py` 改为兼容入口，避免维护两份图像生成逻辑 | 增加结构化 experiment log；补充分年度、行业暴露和 beta 暴露报表 |
| 2026-05-27 | `v1.2.1` | 图像与特征分区 IO 优化 | `03_make_images.py` 改为每个实验输出 `shard_*/images.npy` 和 `shard_*/meta.parquet`；新增 `IMAGE_SHARD_SIZE`；`05_train_cnn2d.py` 改为跨 shard 读取 memmap，并用 `shard_id/local_index` 定位样本；`02_make_labels_and_baselines.py` 改为逐股票计算、按 `features_by_code_bucket` 和 `features_by_year` 批量写 Parquet dataset；`03/04/06` 同步改为从新 feature dataset 投影读取，降低小文件 IO、内存峰值和全量读写开销 | 增加结构化 experiment log；补充分年度、行业暴露和 beta 暴露报表 |
| 2026-05-25 | `v1.2` | 面板构建性能和 Parquet 数据格式升级 | `01_build_panel.py` 改为输出 `panel_by_code` 和 `panel_by_year` 双 Parquet dataset；默认启用 12 进程按股票读取、清洗和合并；保留 `--workers 1` 单进程路径；新增 `--limit-codes` 小样本测试入口；CSV 默认 `gbk` 并保留 fallback；数值转换加速；面板 schema 下压为 `float32/float64/int8`；`02_make_labels_and_baselines.py` 改为从 `panel_by_code` 读取；文档同步新数据格式 | 将图像生成进一步 shard 化；增加结构化 experiment log；补充分年度、行业暴露和 beta 暴露报表；进一步处理北交所 30% 涨跌停和更细滑点模型 |
| 2026-05-25 | `v1.1` | 防过拟合与可交易性修正 | 增加按 horizon 的 purge/embargo 切分；增加随机种子和 CNN weight decay；新增按日期+代码板块的涨跌停阈值与低量涨跌停标记；回测中低量涨停不可买、低量跌停/停牌/缺收益延迟卖出；缺失收益不再填 0；新增 blocked buy/sell、forced hold、data missing 诊断字段；增加未来收益复利一致性检查 | 增加结构化 experiment log；补充分年度、行业暴露和 beta 暴露报表；进一步处理北交所 30% 涨跌停和更细滑点模型 |
| 2026-05-09 | `v1.0` | 重大版本：收益标签、回测评估和组合绩效体系升级 | 未来收益改为次日开盘买入、持有期末收盘卖出；新增 `open_to_close_ret_1d`；新增 IC/RankIC、ICIR、累计 IC、各期 IC；新增 D1-D10 decile 平均未来收益和单调性监测；新增 D1-D10 long-only 重叠持仓组合；新增 turnover、gross/net return、手续费敏感度曲线；新增 large-cap vs small/mid-cap universe split；移除多空组合收益输出；新增预测和日收益字段检查 | 完善训练日志与独立参数配置文件；补充正式单元测试和全链路 schema 检查；加入混合精度和多卡训练支持；加入更严格交易约束，如涨停无法买入、停牌处理和滑点模型 |
| 2026-05-08 | `v0.1.1` | 上传 CNN 训练脚本详细说明文档 | 将 `05_train_cnn2d.md` 纳入 Git 版本管理，便于同步查看 05 脚本的逐函数、全流程和变量解释 | ~~按持有期修正回测年化因子~~；~~增加 RankIC、ICIR 等测评~~；~~实现重叠持仓组合~~；增加训练日志与独立参数配置文件；补充正式单元测试和全链路 schema 检查；加入混合精度和多卡训练支持 |
| 2026-05-08 | `v0.1.0` | 首次源码入库 | 建立 A 股日频面板、标签与基线特征、I5/I20/I60 图像生成、矩阵收益实验、Logistic 基线、Jiang 风格 2D CNN、十分组回测 | ~~按持有期修正回测年化因子~~；~~实现重叠持仓组合~~；增加训练日志与独立参数配置文件；补充正式单元测试和全链路 schema 检查；加入混合精度和多卡训练支持 |
