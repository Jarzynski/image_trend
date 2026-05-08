# Image Trend

基于 A 股日频 OHLCV 数据生成二值蜡烛图，并使用传统特征与 2D CNN 预测未来股票收益方向。项目流程参考 Jiang, Kelly and Xiu (2023) 的价格图像建模思路，并扩展为矩阵收益研究：

| 实验 | 图像窗口 | 预测收益窗口 | 图像形状 |
| --- | ---: | ---: | --- |
| I5R5 | 5 个交易日 | 5 个交易日 | `[N, 32, 15, 1]` |
| I20R5 | 20 个交易日 | 5 个交易日 | `[N, 64, 60, 1]` |
| I60R5 | 60 个交易日 | 5 个交易日 | `[N, 96, 180, 1]` |
| I20R20 | 20 个交易日 | 20 个交易日 | `[N, 64, 60, 1]` |
| I60R20 | 60 个交易日 | 20 个交易日 | `[N, 96, 180, 1]` |

GitHub 仓库只保存 Python 源码、`README.md`、`05_train_cnn2d.md` 和 `.gitignore`。原始数据、生成特征、图像矩阵、模型权重、预测结果和论文 PDF 不上传。

## 文件夹结构

```text
image_trend/
├── config.py
├── 01_build_panel.py
├── 02_make_labels_and_baselines.py
├── 03_make_images.py
├── 04_train_logistic.py
├── 05_train_cnn2d.py
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
├── 05_train_cnn2d.md         # CNN 训练脚本详细说明文档，上传
└── Jiang 等 - 2023 - (Re‐)Imag(in)ing Price Trends.pdf  # 本地论文，不上传
```

项目外部原始数据默认位于：

```text
N:\quant\A_share\daily_OHLVC\
├── 不复权\
└── 后复权\
```

## 脚本输入输出

| 文件 | 作用 | 输入 | 输出 | 数据格式 |
| --- | --- | --- | --- | --- |
| `config.py` | 统一配置路径、样本区间、训练切分、实验矩阵和回测参数 | 无直接输入 | 自动创建 `data/`、`outputs/` 及其子目录 | Python 配置对象 |
| `01_build_panel.py` | 合并每只股票的不复权和后复权日频行情，生成标准股票-日期面板 | `../daily_OHLVC/不复权/*.csv`；`../daily_OHLVC/后复权/*.csv` | `data/processed/panel_daily.parquet` | 输入为逐股票 CSV；输出为 Parquet，包含 raw/adjusted OHLCV、成交额、市值、行业、ST、涨停等字段 |
| `02_make_labels_and_baselines.py` | 生成未来收益标签和传统量价基线特征 | `data/processed/panel_daily.parquet` | `data/features/baseline_features.parquet` | Parquet；包含 `future_ret_{h}d`、`label_{h}d`、收益率、反转、动量、均线偏离、价格位置、波动率、流动性和 `is_tradable` |
| `03_make_images.py` | 按 `EXPERIMENTS` 生成 Jiang 风格二值价格图像 | `data/features/baseline_features.parquet` | `data/images/images_{experiment}.npy`；`data/images/meta_{experiment}.parquet` | `.npy` 为 `uint8` 图像矩阵 `[N,H,W,1]`，像素值 `0/255`；metadata 为 Parquet，每行对应一张图 |
| `04_train_logistic.py` | 对每个实验训练传统特征 Logistic 基线 | `data/features/baseline_features.parquet` | `outputs/predictions/pred_{experiment}_logistic.parquet` | Parquet；包含测试集 `date`、`code`、`future_ret`、`label`、`experiment_name`、`window`、`horizon`、`model_name`、`pred_prob` |
| `05_train_cnn2d.py` | 对每个实验训练 Jiang/Kelly/Xiu 风格 2D CNN | `data/images/images_{experiment}.npy`；`data/images/meta_{experiment}.parquet` | `outputs/models/jiang_cnn2d_{experiment}.pt`；`outputs/predictions/pred_{experiment}_jiang_cnn2d.parquet` | `.pt` 为 PyTorch `state_dict`；预测 Parquet 字段与 Logistic 输出对齐 |
| `06_backtest_decile.py` | 对所有预测文件做横截面十分组回测 | `outputs/predictions/pred_*.parquet` | `outputs/tables/decile_returns.csv`；`outputs/tables/long_short_returns.csv`；`outputs/tables/performance_summary.csv` | CSV；分别保存分组收益、多空收益和绩效指标 |

## 推荐运行顺序

```powershell
uv run python 01_build_panel.py
uv run python 02_make_labels_and_baselines.py
uv run python 03_make_images.py
uv run python 04_train_logistic.py
uv run python 05_train_cnn2d.py
uv run python 06_backtest_decile.py
```

若依赖缺失，请统一安装后再运行。当前项目约定不在脚本中自动安装依赖。

## 版本记录

当前版本：`v0.1.1`

后续每次推送前应更新本表，并创建同名 Git tag。

| 日期 | 版本 | 推送内容 | 新增功能 | 待更新功能 |
| --- | --- | --- | --- | --- |
| 2026-05-08 | `v0.1.1` | 上传 CNN 训练脚本详细说明文档 | 将 `05_train_cnn2d.md` 纳入 Git 版本管理，便于同步查看 05 脚本的逐函数、全流程和变量解释 | 优先：按持有期修正回测年化因子；增加RankIC，ICIR等测评；后续：实现重叠持仓组合；增加训练日志与参数配置文件；补充单元测试和数据 schema 检查|
| 2026-05-08 | `v0.1.0` | 首次源码入库 | 建立 A 股日频面板、标签与基线特征、I5/I20/I60 图像生成、矩阵收益实验、Logistic 基线、Jiang 风格 2D CNN、十分组回测 | 按持有期修正回测年化因子；实现重叠持仓组合；增加训练日志与参数配置文件；补充单元测试和数据 schema 检查；加入混合精度和多卡训练支持 |
