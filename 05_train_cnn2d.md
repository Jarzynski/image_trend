# `05_train_cnn2d.py` 说明文档

本文档对应当前版本的 `05_train_cnn2d.py`，说明脚本定位、输入输出、执行流程、模型结构、CLI 参数和主要函数职责。

脚本位置：

```text
N:\quant\A_share\image_trend\05_train_cnn2d.py
```

## 1. 脚本定位

`05_train_cnn2d.py` 是项目里的官方 CNN 训练入口。自 v1.2.5 起，原 `05_train_cnn2d_4090_fast.py` 的 4080/4090 优化训练逻辑已经合并到本文件，`05_train_cnn2d_4090_fast.py` 只保留为兼容 wrapper。

脚本读取 `03_make_images.py` 生成的二值价格图像 shard 和 metadata，按 `config.EXPERIMENTS` 中的实验矩阵训练 Jiang, Kelly, and Xiu (2023) 风格的 2D CNN，并输出测试集预测概率。默认每个实验训练 1 次；如需论文式“同一配置独立训练 5 次并平均概率”，使用 `--ensemble-runs 5`。

当前脚本的主要工程特性：

- 按 `window` 共享读取图像 metadata，`I20R5/I20R20` 和 `I60R5/I60R20` 不重复加载同一窗口数据。
- 图像 shard 使用 `np.load(..., mmap_mode="r")` lazy 打开。
- `Dataset` 返回 `uint8` CHW tensor，batch 到 GPU 后再转 `float32` 并除以 255。
- 默认启用 CUDA AMP mixed precision 和 TF32。
- 默认使用 AdamW、cosine scheduler、warmup、weight decay 和 FC dropout。
- 支持 `jiang` 和 `reslite` 两种模型架构。
- 支持 validation RankIC/IC/decile 诊断，并优先按 validation RankIC 保存 checkpoint。
- 支持多 run ensemble、单独训练某个 ensemble run、以及 aggregation-only 汇总模式。

当前脚本覆盖的实验来自 `config.EXPERIMENTS`：

| 实验名 | 输入窗口 | 预测 horizon | 图像尺寸 |
|---|---:|---:|---|
| `I5R5` | 5 日 | 5 日 | `[N, 32, 15, 1]` |
| `I20R5` | 20 日 | 5 日 | `[N, 64, 60, 1]` |
| `I60R5` | 60 日 | 5 日 | `[N, 96, 180, 1]` |
| `I20R20` | 20 日 | 20 日 | `[N, 64, 60, 1]` |
| `I60R20` | 60 日 | 20 日 | `[N, 96, 180, 1]` |

## 2. 上游和下游关系

完整流程中，`05_train_cnn2d.py` 位于图像生成之后、组合回测之前。

```text
01_build_panel.py
    -> data/processed/panel_by_code/code=*/part.parquet
    -> data/processed/panel_by_year/year=*/part-*.parquet

02_make_labels_and_baselines.py
    -> data/features/features_by_code_bucket/bucket=*/part-*.parquet
    -> data/features/features_by_year/year=*/part-*.parquet

03_make_images.py
    -> data/images/window_{window}/shard_*/images.npy
    -> data/images/window_{window}/shard_*/meta.parquet

05_train_cnn2d.py
    -> outputs/models/jiang_cnn2d_{experiment}.pt
    -> outputs/models/jiang_cnn2d_{experiment}_run*.pt
    -> outputs/predictions/ensemble_runs/{experiment}/run*.parquet
    -> outputs/predictions/pred_{experiment}_jiang_cnn2d.parquet
    -> outputs/tables/cnn_training_log_{experiment}*.csv
    -> outputs/tables/cnn_ensemble_summary_{experiment}.csv

06_backtest_decile.py
    -> outputs/tables/*.csv
```

## 3. 输入文件

图像路径由 `config.image_dir_for_window(window)` 决定。当前图像按唯一 `window` 存储，而不是按 experiment 重复存储：

```text
data/images/window_{window}/shard_*/images.npy
data/images/window_{window}/shard_*/meta.parquet
```

示例：

```text
data/images/window_5/shard_00000/images.npy
data/images/window_20/shard_00000/images.npy
data/images/window_60/shard_00000/images.npy
```

### 3.1 `images.npy`

每个 `images.npy` 是一个 `uint8` NumPy 数组，维度顺序为 `NHWC`：

```text
[N, H, W, C]
```

| 维度 | 含义 |
|---|---|
| `N` | 当前 shard 中的样本数，每个样本对应一个 `code-date` |
| `H` | 图像高度，5/20/60 日窗口分别为 32/64/96 |
| `W` | 图像宽度，等于 `window * 3` |
| `C` | 通道数，当前为 1 |

像素值为 `0/255`。脚本不会在 `Dataset.__getitem__` 中把每张图转成 float，而是在 `prepare_batch()` 中对整个 batch 执行：

```python
x = x.float().div_(255.0)
```

这样可以减少 CPU 端拷贝和 host-to-device 传输量。

### 3.2 `meta.parquet`

每个 metadata shard 与同目录 `images.npy` 一一对应，行数必须等于图像数组第 0 维。脚本会优先只读取训练需要的列，如果旧 shard 缺列导致读取失败，会 fallback 到读取完整 parquet。

关键列：

| 列名 | 含义 |
|---|---|
| `date` | 图像窗口结束日，也是预测发出日 |
| `code` | 股票代码 |
| `industry` | 行业 |
| `shard_id` | 样本所在 shard 编号；缺失时由脚本按目录顺序补齐 |
| `local_index` | 样本在 shard 内的行号；缺失时由脚本按行号补齐 |
| `label_{h}d` | horizon 为 `h` 的二分类标签 |
| `future_ret_{h}d` | horizon 为 `h` 的未来收益 |
| `amount` | 当日成交额 |
| `float_mktcap` | 流通市值 |
| `is_low_volume_limit_up` | 低量涨停不可买标记 |
| `is_low_volume_limit_down` | 低量跌停不可卖标记 |

对每个实验，`select_experiment_label_view()` 会按 `cfg["horizon"]` 选择 `label_{h}d` 和 `future_ret_{h}d`，生成临时列：

```text
label
future_ret
experiment_name
window
horizon
```

## 4. 输出文件

### 4.1 模型权重

默认单 run：

```text
outputs/models/jiang_cnn2d_{experiment}.pt
```

启用多 run ensemble 时：

```text
outputs/models/jiang_cnn2d_{experiment}_run01.pt
outputs/models/jiang_cnn2d_{experiment}_run02.pt
...
```

如果使用 `--arch reslite`，输出 stem 会带架构后缀：

```text
outputs/models/jiang_cnn2d_{experiment}_reslite.pt
outputs/models/jiang_cnn2d_{experiment}_reslite_run01.pt
```

模型文件内容是 PyTorch `state_dict`，不包含 optimizer 状态。

### 4.2 单 run 预测中间文件

每次独立 run 都会写出一份中间预测：

```text
outputs/predictions/ensemble_runs/{experiment}/run01.parquet
```

`reslite` 架构会写到：

```text
outputs/predictions/ensemble_runs/{experiment}_reslite/run01.parquet
```

单 run 文件中包含 `pred_prob_run_XX`，用于后续平均。

### 4.3 最终测试集预测文件

最终预测文件路径：

```text
outputs/predictions/pred_{experiment}_jiang_cnn2d.parquet
```

`reslite` 架构路径：

```text
outputs/predictions/pred_{experiment}_reslite_jiang_cnn2d.parquet
```

关键列：

| 列名 | 含义 |
|---|---|
| `date` | 预测日期 |
| `code` | 股票代码 |
| `industry` | 行业 |
| `future_ret` | 实际未来收益 |
| `label` | 实际二分类标签 |
| `amount` | 成交额 |
| `float_mktcap` | 流通市值 |
| `is_low_volume_limit_up` | 低量涨停不可买标记 |
| `is_low_volume_limit_down` | 低量跌停不可卖标记 |
| `experiment_name` | 实验名 |
| `window` | 图像窗口 |
| `horizon` | 预测周期 |
| `model_name` | `JiangCNN2D`、`JiangCNN2D_reslite` 或带 `_ensN` 的 ensemble 名称 |
| `ensemble_runs` | 汇总使用的独立 run 数量 |
| `pred_prob_run_XX` | 第 `XX` 个独立 run 的预测概率 |
| `pred_prob` | 最终预测概率；多 run 时为各 run 概率算术平均 |

`06_backtest_decile.py` 会读取这些预测文件，并按 `pred_prob` 做横截面排序和回测。

### 4.4 训练日志

每个实验会写出 epoch 级 CSV 日志：

```text
outputs/tables/cnn_training_log_{experiment}.csv
outputs/tables/cnn_training_log_{experiment}_run01.csv
```

主要字段包括：

- `epoch`
- `lr`
- `epoch_seconds`
- `train_seconds`
- `valid_seconds`
- `train_samples_per_sec`
- `valid_samples_per_sec`
- `avg_data_wait_ms`
- `avg_h2d_ms`
- `avg_gpu_step_ms`
- `train_loss`
- `valid_loss`
- `valid_auc`
- `valid_acc`
- `valid_brier`
- `valid_rankic_mean`
- `valid_rankic_positive_rate`
- `valid_ic_mean`
- `valid_decile_spearman`
- `valid_decile_violations`
- `checkpoint_score`
- `checkpoint_metric`

多 run ensemble 还会输出：

```text
outputs/tables/cnn_ensemble_summary_{experiment}.csv
```

## 5. 依赖库

脚本依赖：

| 包 | 用途 |
|---|---|
| `numpy` | 读取 `.npy`、数组索引和数值处理 |
| `pandas` | 读取 metadata parquet、处理日期、写预测和日志 |
| `torch` | CNN、Dataset、DataLoader、训练、AMP 和保存权重 |
| `scikit-learn` | 计算 AUC、accuracy、Brier score |
| `pyarrow` 或其他 parquet engine | 被 pandas 用于读写 parquet |

项目约定不在脚本中自动安装依赖。运行时使用当前 `uv` 环境。

## 6. 运行方式

在 `image_trend` 目录下运行：

```powershell
uv run python 05_train_cnn2d.py
```

只训练指定实验：

```powershell
uv run python 05_train_cnn2d.py --experiments I20R5
```

只训练指定窗口对应的实验：

```powershell
uv run python 05_train_cnn2d.py --windows 20
```

论文式 5 次独立训练并平均预测概率：

```powershell
uv run python 05_train_cnn2d.py --ensemble-runs 5
```

在 Slurm array 或手工拆分任务时，只训练其中一个 run：

```powershell
uv run python 05_train_cnn2d.py --experiments I20R5 --ensemble-runs 5 --ensemble-run-id 3
```

所有 run 完成后，只做概率平均、不重新训练：

```powershell
uv run python 05_train_cnn2d.py --experiments I20R5 --ensemble-runs 5 --ensemble-aggregate-only
```

快速 smoke test 示例：

```powershell
uv run python 05_train_cnn2d.py --experiments I5R5 --epochs 1 --max-valid-batches 2 --max-test-batches 2
```

## 7. CLI 参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--experiments` | `None` | 逗号分隔实验名，例如 `I5R5,I20R5` |
| `--windows` | `None` | 逗号分隔窗口，例如 `5,20` |
| `--epochs` | `50` | 最大训练 epoch |
| `--batch-size` | `256` | batch size |
| `--lr` | `1e-4` | 基础学习率 |
| `--workers` | `default_num_workers()` | DataLoader worker 数，默认 2 到 4 之间 |
| `--prefetch-factor` | `2` | 每个 worker 预取 batch 数 |
| `--patience` | `8` | early stopping 容忍轮数 |
| `--min-epochs` | `8` | early stopping 生效前最少训练轮数 |
| `--min-delta` | `1e-4` | checkpoint score 最小改善幅度 |
| `--optimizer` | `adamw` | `adamw` 或 `adam` |
| `--weight-decay` | `3e-5` | AdamW/Adam weight decay |
| `--scheduler` | `cosine` | `none`、`cosine` 或 `plateau` |
| `--warmup-epochs` | `1` | warmup epoch 数 |
| `--fc-dropout` | `0.20` | 分类头 dropout |
| `--spatial-dropout` | `0.0` | CNN feature map dropout，0 表示关闭 |
| `--arch` | `jiang` | `jiang` 或 `reslite` |
| `--ensemble-runs` | `1` | 每个实验独立训练次数 |
| `--ensemble-run-id` | `None` | 只训练第几个 ensemble member，1-based |
| `--ensemble-aggregate-only` | `False` | 只读取 run parquet 并写最终平均预测 |
| `--valid-metric-interval` | `1` | 每多少个 epoch 跑一次完整验证 |
| `--max-valid-batches` | `None` | 验证 batch 上限，主要用于 smoke test |
| `--max-test-batches` | `None` | 测试 batch 上限，主要用于 smoke test |
| `--no-amp` | `False` | 关闭 CUDA AMP |
| `--no-tf32` | `False` | 关闭 TF32 |
| `--compile` | `False` | 使用 `torch.compile` |
| `--channels-last` | `False` | CUDA 下使用 channels_last memory format |
| `--no-pin-memory` | `False` | 关闭 DataLoader pin_memory |
| `--no-persistent-workers` | `False` | 关闭 persistent workers |
| `--drop-last-train` | `False` | 丢弃训练集最后一个不完整 batch |
| `--log-interval` | `200` | 训练 batch 级日志间隔 |
| `--profile-batches` | `0` | 对每个 epoch 前 N 个 batch 做同步计时，0 表示关闭 |
| `--shard-cache-size` | `32` | 每个 DataLoader worker 最多缓存的 shard memmap 数 |

## 8. 全流程概览

入口是：

```python
if __name__ == "__main__":
    main()
```

`main()` 的流程：

1. 解析 CLI 参数。
2. 通过 `selected_experiments()` 和 `group_experiments_by_window()` 选出实验并按 `window` 分组。
3. 如果指定 `--ensemble-aggregate-only`，直接读取 run parquet 并调用 `aggregate_ensemble_predictions()` 写最终预测。
4. 调用 `configure_torch()` 设置随机种子、cuDNN benchmark、TF32。
5. 对每个 `window` 调用 `load_image_window()` 读取 shard 路径和合并 metadata。
6. 对该窗口下的每个实验调用 `fit_one_experiment()`。
7. `fit_one_experiment()` 根据 `--ensemble-runs` 决定训练一个或多个独立 run。
8. 每个 run 调用 `fit_one_experiment_run()` 完成训练、验证、测试、保存模型和 run 预测。
9. 如果不是单独 run 模式，调用 `aggregate_ensemble_predictions()` 生成最终预测文件。
10. 释放当前 window 的对象和 CUDA cache，再进入下一个 window。

简化伪代码：

```text
experiments = selected_experiments(args.experiments, args.windows)
grouped = group_experiments_by_window(experiments)

if ensemble_aggregate_only:
    for exp_name, cfg in experiments:
        aggregate_ensemble_predictions(exp_name, cfg, options)
    return

configure_torch(RANDOM_SEED)

for window in grouped:
    image_paths, meta_window, image_shape = load_image_window(window)

    for exp_name, cfg in grouped[window]:
        for run_id in selected run ids:
            pred_run = fit_one_experiment_run(...)
            save_run_prediction(...)

        aggregate_ensemble_predictions(...)
```

## 9. 数据读取和切分

### 9.1 `load_image_window`

`load_image_window(window, window_experiments)` 会：

- 定位 `data/images/window_{window}/shard_*`；
- 检查每个 shard 是否同时包含 `images.npy` 和 `meta.parquet`；
- 用 `np.load(..., mmap_mode="r")` 读取 shape 后立即关闭 memmap；
- 检查同一 window 下所有 shard 的 trailing shape 一致；
- 读取 metadata；
- 补齐缺失的 `shard_id/local_index`；
- 检查 metadata 行数与图像数一致；
- 压缩 metadata dtype；
- 返回 `image_paths`、`meta_window`、`image_shape`。

注意：主进程这里只保存 `images.npy` 路径，不长期持有所有 shard memmap。真正的图像 memmap 由 `ImageDataset` 在 DataLoader worker 内 lazy 打开。

### 9.2 `compress_meta_dtypes`

该函数降低大样本 metadata 的内存占用：

- `date` 转为 datetime；
- `code/industry/experiment_name` 转为 category；
- `shard_id/local_index/window/horizon` 转为 `int32`；
- `label_*` 和 `future_ret_*` 转为 `float32`；
- `amount/float_mktcap` 转为 `float32`；
- 低量涨跌停标记转为 `int8`。

### 9.3 `get_split_indices`

`get_split_indices(meta, horizon)` 按日期生成 train/valid/test 行号。

训练集：

```text
date <= TRAIN_END
date < VALID_START - embargo_gap
```

验证集：

```text
VALID_START <= date <= VALID_END
date < TEST_START - embargo_gap
```

测试集：

```text
date >= TEST_START
```

其中 `embargo_gap` 来自 `config.EMBARGO_DAYS_BY_HORIZON`。这样可以减少不同样本未来收益窗口重叠导致的泄漏风险。

## 10. Dataset 和 DataLoader

## 10.1 `ImageDataset`

`ImageDataset` 保存：

| 属性 | 含义 |
|---|---|
| `image_paths` | 每个 shard 的 `images.npy` 路径 |
| `labels` | 全量标签数组，`float32` |
| `shard_ids` | 每行样本所在 shard |
| `local_indices` | 每行样本在 shard 内的位置 |
| `indices` | 当前 split 使用的全量行号 |
| `shard_cache_size` | 每个 Dataset 实例最多打开的 shard memmap 数 |
| `_shard_cache` | LRU shard memmap cache |

`__getitem__()` 的关键流程：

1. 用 split 内部 `idx` 找到全量 `real_idx`。
2. 根据 `shard_id/local_index` 定位图片。
3. 如果 shard 不在 cache 中，用 `np.load(path, mmap_mode="r")` 打开。
4. 从 NHWC 单样本图像转为 CHW。
5. 显式复制为 writable `uint8` contiguous array。
6. 返回 `torch.uint8` 图像 tensor 和 `float32` label。

脚本显式复制 writable uint8，是为了避免 read-only memmap 传给 `torch.from_numpy` 后在某些 PyTorch/Numpy 组合中引发 worker 崩溃。

### 10.2 `make_loader`

`make_loader()` 统一创建 DataLoader：

- train loader 默认 `shuffle=True`；
- validation/test 默认 `shuffle=False`；
- `num_workers`、`prefetch_factor`、`pin_memory`、`persistent_workers` 来自 CLI；
- 只有 `num_workers > 0` 时才设置 `prefetch_factor` 和 `persistent_workers`。

### 10.3 `prepare_batch`

`prepare_batch()` 负责把 batch 移动到训练设备：

```python
x = x.to(device, non_blocking=True)
y = y.to(device, non_blocking=True)
x = x.float().div_(255.0)
```

如果启用 `--channels-last` 且设备是 CUDA，会将图像 tensor 转为 channels_last memory format。

## 11. 模型结构

### 11.1 `JiangCNNBlock`

Jiang block 结构：

```text
Conv2d -> BatchNorm2d -> LeakyReLU -> MaxPool2d
```

核心参数：

| 参数 | 值 |
|---|---|
| kernel | `(5, 3)` |
| activation | `LeakyReLU(negative_slope=0.01)` |
| pooling | `MaxPool2d(kernel_size=(2,1), stride=(2,1), ceil_mode=True)` |

### 11.2 `ResLiteCNNBlock`

`ResLiteCNNBlock` 是可选轻量残差变体，通过 `--arch reslite` 启用。它保留 Jiang block 的 pooling schedule，但在 pooling 前使用两层卷积和 residual connection。

如果输入通道或 stride 与输出不匹配，skip path 使用 `1x1 Conv2d + BatchNorm2d`；否则使用 `Identity`。

### 11.3 `JiangCNN2D`

`JiangCNN2D` 根据 `window` 自动选择 block 数、第一层 vertical stride 和 dilation：

| window | blocks | 第一层 vertical stride | 第一层 vertical dilation | channels |
|---:|---:|---:|---:|---|
| 5 | 2 | 1 | 1 | 64, 128 |
| 20 | 3 | 3 | 2 | 64, 128, 256 |
| 60 | 4 | 3 | 3 | 64, 128, 256, 512 |

初始化参数：

```python
JiangCNN2D(
    window,
    image_height,
    image_width,
    in_channels=1,
    fc_dropout=0.20,
    spatial_dropout=0.0,
    arch="jiang",
)
```

`fc_dropout` 和 `spatial_dropout` 由 CLI 控制。`feature_dim` 通过 dummy tensor 自动推导：

```python
dummy = torch.zeros(1, in_channels, image_height, image_width)
feature_dim = self.spatial_dropout(self.features(dummy)).flatten(1).shape[1]
```

分类头：

```text
Flatten -> Dropout(fc_dropout) -> Linear(feature_dim, 1)
```

模型输出单个 logit，训练使用 `BCEWithLogitsLoss`；评估和预测时用 `sigmoid(logit)` 得到 `pred_prob`。

权重初始化：

- `Conv2d` 和 `Linear` 使用 Xavier uniform；
- bias 初始化为 0。

## 12. 训练和 checkpoint

### 12.1 `build_optimizer`

默认优化器：

```text
AdamW(lr=1e-4, weight_decay=3e-5)
```

也可以通过 `--optimizer adam` 使用 Adam。

### 12.2 `build_scheduler`

支持三种 scheduler：

| 参数 | 行为 |
|---|---|
| `--scheduler none` | 不使用 scheduler |
| `--scheduler cosine` | `CosineAnnealingLR` |
| `--scheduler plateau` | `ReduceLROnPlateau` |

warmup 由 `set_warmup_lr()` 实现，默认 `--warmup-epochs 1`。

### 12.3 AMP 和 TF32

默认：

- CUDA 可用时，训练和评估在 `torch.amp.autocast(device_type="cuda")` 下执行；
- 使用 `torch.amp.GradScaler("cuda")`；
- 允许 matmul 和 cuDNN TF32；
- `torch.set_float32_matmul_precision("high")` 在可用时生效。

可用 `--no-amp` 和 `--no-tf32` 关闭。

### 12.4 validation 评估

`evaluate_model()` 返回：

- `loss`
- `auc`
- `acc`
- `brier`
- `probs`
- `labels`
- `seconds`
- `samples_per_sec`

`signal_metrics()` 基于验证集预测概率计算：

- 每日 IC 均值；
- 每日 RankIC 均值；
- RankIC 为正的日期占比；
- decile 平均收益单调性 Spearman；
- decile 单调性违反次数。

`probability_summary()` 记录预测概率分布，例如均值、标准差、分位数、最小值和最大值。

### 12.5 checkpoint 选择

`checkpoint_score()` 的优先级：

1. 如果 validation RankIC 可用，使用 `valid_rankic_mean`；
2. 否则如果 validation AUC 可用，使用 `valid_auc`；
3. 否则使用 `-valid_loss`。

当 `score > best_score + min_delta` 时保存当前模型参数到 CPU。只有在达到 `--min-epochs` 后，连续 `--patience` 个 epoch 没有改善才会 early stop。

## 13. Ensemble 逻辑

### 13.1 多 run 训练

`fit_one_experiment()` 根据 `--ensemble-runs` 决定每个实验训练几次。第 `run_id` 次训练使用：

```text
seed = RANDOM_SEED + run_id - 1
```

每次 run 会保存：

- 对应模型权重；
- `outputs/predictions/ensemble_runs/{stem}/runXX.parquet`；
- 对应训练日志。

### 13.2 单独训练某个 run

`--ensemble-run-id K` 只训练第 `K` 个 run。该模式不会写最终平均预测文件，适合 GPU array job。

所有 run 完成后，使用 `--ensemble-aggregate-only` 汇总。

### 13.3 概率平均

`aggregate_ensemble_predictions()` 会：

1. 读取或接收所有 run 的预测 DataFrame；
2. 检查每个 run 的预测行数和关键列顺序一致；
3. 将各 run 的 `pred_prob_run_XX` 合并到同一张表；
4. 计算：

```text
pred_prob = mean(pred_prob_run_01, pred_prob_run_02, ...)
```

5. 写出最终 `pred_{stem}_jiang_cnn2d.parquet`。

## 14. 与论文设定的对应关系

| 论文设定 | 当前脚本实现 |
|---|---|
| 使用价格图像预测未来收益方向 | `label` 为 0/1，模型输出 `pred_prob` |
| 5/20/60 日图像尺寸 | 上游生成 `[N,32,15,1]`、`[N,64,60,1]`、`[N,96,180,1]` |
| 不同窗口使用不同 CNN 深度 | `JiangCNN2D.WINDOW_CONFIG` |
| CNN building block | `JiangCNNBlock` |
| Conv + BN + LeakyReLU + Pool | `JiangCNNBlock.block` |
| LeakyReLU slope = 0.01 | `nn.LeakyReLU(negative_slope=0.01)` |
| Xavier 初始化 | `JiangCNN2D._init_weights()` |
| 同一模型配置独立训练 5 次并平均概率 | `--ensemble-runs 5` |
| softmax up/down 二分类 | 当前用单 logit + `BCEWithLogitsLoss`，`sigmoid` 后得到 up 概率 |

当前默认是工程优化配置，不是完全复刻旧 baseline 参数：

| 项目 | 当前默认 |
|---|---|
| optimizer | AdamW |
| lr | `1e-4` |
| weight decay | `3e-5` |
| FC dropout | `0.20` |
| scheduler | cosine |
| AMP | 开启 |
| TF32 | 开启 |
| checkpoint metric | validation RankIC 优先 |

如需更接近旧 baseline 风格，可显式指定：

```powershell
uv run python 05_train_cnn2d.py --optimizer adam --lr 1e-5 --fc-dropout 0.50 --scheduler none
```

## 15. 常见问题

### 15.1 找不到图像 shard

报错通常来自 `load_image_window()`：

```text
No image shards found for window=...
```

常见原因：

- 未运行 `03_make_images.py`；
- 图像仍是旧的按 experiment 目录存储；
- `config.image_dir_for_window(window)` 指向的目录不存在。

处理方式：

```powershell
uv run python 03_make_images.py
```

### 15.2 metadata 行数和图像数不一致

`load_image_window()` 会检查每个 shard 的 `len(meta)` 是否等于 `images.shape[0]`。如果不一致，说明 shard 生成不完整或文件混用，需要重新生成图像。

### 15.3 split 为空

如果 train/valid/test 任一 split 没有样本，`fit_one_experiment_run()` 会报错。常见原因：

- 上游数据日期范围不足；
- `TRAIN_END/VALID_START/VALID_END/TEST_START` 设置不合理；
- horizon 对应的 purge/embargo 后样本被过滤；
- 某个 window 的图像尚未生成完整。

### 15.4 AUC 显示为 `nan`

如果某个评估 split 的 label 只有一个类别，`roc_auc_score` 无法定义，脚本会返回 `nan`。这不是程序崩溃，而是指标本身不可计算。

### 15.5 GPU 没有被使用

`get_device()` 只检查：

```python
torch.cuda.is_available()
```

如果返回 false，脚本会自动使用 CPU。常见原因是当前 uv 环境安装了 CPU 版 PyTorch，或 CUDA driver / PyTorch CUDA 版本不匹配。

### 15.6 I60 训练慢或 DataLoader 慢

I60 图像更大、模型更深，训练和 IO 都更重。可优先调整：

- `--batch-size`
- `--workers`
- `--prefetch-factor`
- `--shard-cache-size`
- `--channels-last`
- `--profile-batches`

如果 `num_workers > 0` 带来不稳定，可尝试：

```powershell
uv run python 05_train_cnn2d.py --workers 0 --no-persistent-workers
```

### 15.7 只想汇总已训练的 ensemble

确认所有 run 文件存在，例如：

```text
outputs/predictions/ensemble_runs/i20r5/run01.parquet
outputs/predictions/ensemble_runs/i20r5/run02.parquet
...
```

然后运行：

```powershell
uv run python 05_train_cnn2d.py --experiments I20R5 --ensemble-runs 5 --ensemble-aggregate-only
```

## 16. 运行前检查清单

运行前建议确认：

- `03_make_images.py` 已按当前配置生成 `data/images/window_{window}/shard_*`；
- 每个 shard 同时有 `images.npy` 和 `meta.parquet`；
- `.npy` 样本数与 metadata 行数一致；
- metadata 覆盖 train/valid/test 日期区间；
- metadata 中存在对应 horizon 的 `label_{h}d` 和 `future_ret_{h}d`；
- 当前 uv 环境能 import `torch`、`sklearn` 和 parquet engine；
- CUDA 训练时 `torch.cuda.is_available()` 为 true。

检查当前 Python 解释器：

```powershell
uv run python -c "import sys; print(sys.executable)"
```

检查 PyTorch CUDA：

```powershell
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 17. 当前实现边界

当前脚本仍然不做以下事情：

1. 不做行业中性化。
2. 不做市值中性化。
3. 不在训练中显式使用样本权重。
4. 不保存 optimizer checkpoint。
5. 不做多 GPU DDP。
6. 不保存训练集或验证集逐样本预测。

这些是当前训练入口的设计边界，不代表预测文件和回测接口缺失。

## 18. 总结

`05_train_cnn2d.py` 的核心任务是：

1. 按 window 读取图像 shard 和 metadata；
2. 按 experiment horizon 选择标签和未来收益；
3. 用 purge/embargo 后的日期切分构造 train/valid/test；
4. 训练 Jiang-style 或 ResLite CNN；
5. 用 validation RankIC/AUC/loss 选择 checkpoint；
6. 在测试集输出 `pred_prob`；
7. 可选执行多 run ensemble 并平均预测概率；
8. 为 `06_backtest_decile.py` 提供标准预测 parquet。
