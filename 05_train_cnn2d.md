# `05_train_cnn2d.py` 说明文档

本文档解释 `05_train_cnn2d.py` 的用途、输入输出、整体执行流程、模型结构、每个类/函数的职责，以及主要变量的含义。

脚本位置：

```text
N:\quant\A_share\image_trend\05_train_cnn2d.py
```

## 1. 脚本定位

`05_train_cnn2d.py` 是整个图像趋势预测项目中的 CNN 训练脚本。

它接收 `03_make_images.py` 生成的二值价格图像和对应 metadata，按 `config.py` 中定义的实验矩阵逐一训练 Jiang, Kelly, and Xiu (2023) 风格的 2D CNN 模型，并输出：

- 每个实验的 PyTorch 模型权重；
- 每个实验在测试集上的逐股票预测概率；
- 训练、验证、测试过程中的 AUC、准确率、Brier score 和 loss 日志。

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
    -> data/images/{experiment}/shard_*/images.npy
    -> data/images/{experiment}/shard_*/meta.parquet

05_train_cnn2d.py
    -> outputs/models/jiang_cnn2d_{experiment}.pt
    -> outputs/predictions/pred_{experiment}_jiang_cnn2d.parquet

06_backtest_decile.py
    -> outputs/tables/*.csv
```

## 3. 输入文件

对每个 `experiment_name`，脚本读取两类输入文件。

### 3.1 图像数组

路径由 `config.image_dir_for_experiment(exp_name)` 生成：

```text
data/images/{experiment_name.lower()}/shard_*/images.npy
```

示例：

```text
data/images/i5r5/shard_00000/images.npy
data/images/i20r5/shard_00000/images.npy
data/images/i60r5/shard_00000/images.npy
data/images/i20r20/shard_00000/images.npy
data/images/i60r20/shard_00000/images.npy
```

格式：

- NumPy `.npy`
- dtype: `uint8`
- 背景像素为 `0`
- 可见 OHLC、MA、volume 像素为 `255`
- 维度顺序为 `NHWC`

含义：

```text
[N, H, W, C]
```

| 维度 | 含义 |
|---|---|
| `N` | 样本数量，每个样本对应一个 `code-date` |
| `H` | 图像高度，I5/I20/I60 分别为 32/64/96 |
| `W` | 图像宽度，等于 `window * 3` |
| `C` | 通道数，当前为 1，即黑白单通道 |

每个 shard 内部仍是标准 `.npy` 数组，训练脚本逐个 shard 以内存映射方式打开：

```python
images = np.load(shard_image_path, mmap_mode="r")
```

这里使用 `mmap_mode="r"`，表示以内存映射方式只读加载图像文件。好处是不会一次性把单个 shard 的完整 `.npy` 全部复制进内存；同时 shard 化避免把全部样本集中在一个超大 `.npy` 文件中，尤其对 I60 图像更重要。

### 3.2 图像 metadata

路径由 `config.image_dir_for_experiment(exp_name)` 生成：

```text
data/images/{experiment_name.lower()}/shard_*/meta.parquet
```

示例：

```text
data/images/i5r5/shard_00000/meta.parquet
data/images/i20r5/shard_00000/meta.parquet
data/images/i60r5/shard_00000/meta.parquet
data/images/i20r20/shard_00000/meta.parquet
data/images/i60r20/shard_00000/meta.parquet
```

格式：

- Parquet
- 每一行对应 `.npy` 中同位置的一张图像
- 行顺序必须与图像数组第 0 维严格一致

关键列：

| 列名 | 含义 |
|---|---|
| `date` | 当前图像窗口结束日，也是预测发出日 |
| `code` | 股票代码 |
| `industry` | 行业 |
| `experiment_name` | 实验名 |
| `window` | 图像窗口长度，例如 5、20、60 |
| `horizon` | 预测未来收益周期，例如 5、20 |
| `future_ret` | 未来收益率，来自 `future_ret_{horizon}d` |
| `label` | 二分类标签，未来收益大于 0 为 1，否则为 0 |
| `amount` | 当日成交额 |
| `float_mktcap` | 流通市值 |
| `is_limit_up` | 是否涨停 |
| `image_height` | 图像高度 |
| `image_width` | 图像宽度 |
| `price_height` | 价格区域高度 |
| `volume_height` | 成交量区域高度 |

## 4. 输出文件

### 4.1 模型权重

路径：

```text
outputs/models/jiang_cnn2d_{experiment_name.lower()}.pt
```

示例：

```text
outputs/models/jiang_cnn2d_i5r5.pt
outputs/models/jiang_cnn2d_i20r5.pt
outputs/models/jiang_cnn2d_i60r5.pt
outputs/models/jiang_cnn2d_i20r20.pt
outputs/models/jiang_cnn2d_i60r20.pt
```

内容：

- PyTorch `state_dict`
- 只保存模型参数，不保存 optimizer 状态

保存代码：

```python
torch.save(model.state_dict(), model_path)
```

### 4.2 测试集预测文件

路径：

```text
outputs/predictions/pred_{experiment_name.lower()}_jiang_cnn2d.parquet
```

示例：

```text
outputs/predictions/pred_i5r5_jiang_cnn2d.parquet
outputs/predictions/pred_i20r5_jiang_cnn2d.parquet
outputs/predictions/pred_i60r5_jiang_cnn2d.parquet
outputs/predictions/pred_i20r20_jiang_cnn2d.parquet
outputs/predictions/pred_i60r20_jiang_cnn2d.parquet
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
| `is_limit_up` | 是否涨停 |
| `experiment_name` | 实验名 |
| `window` | 图像窗口 |
| `horizon` | 预测周期 |
| `model_name` | 当前为 `JiangCNN2D` |
| `pred_prob` | 模型预测未来收益为正的概率 |

`06_backtest_decile.py` 会读取这些预测文件，并根据 `pred_prob` 做横截面十分组回测。

## 5. 依赖库

脚本依赖以下 Python 包：

| 包 | 用途 |
|---|---|
| `numpy` | 读取 `.npy` 图像、数组索引、数值处理 |
| `pandas` | 读取 parquet metadata、处理日期和预测表 |
| `torch` | CNN 模型、Dataset、DataLoader、训练和保存权重 |
| `scikit-learn` | 计算 AUC、accuracy、Brier score |
| `pyarrow` 或其他 parquet engine | 被 pandas 用于读写 parquet |

当前约定：不在脚本里安装依赖，依赖由环境统一管理。

## 6. 导入模块解释

```python
import numpy as np
import pandas as pd
```

`numpy` 用于：

- 读取 `.npy`；
- 生成 split indices；
- 转换 dtype；
- 拼接评估结果；
- 判断 label 是否只有单一类别。

`pandas` 用于：

- 读取 metadata parquet；
- 日期转换；
- 根据 test mask 构建预测输出表；
- 写出预测 parquet。

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
```

`torch` 用于：

- 张量计算；
- 模型训练；
- CUDA/CPU 设备选择；
- 保存模型权重。

`torch.nn` 用于定义 CNN 层、loss 和模型结构。

`Dataset` 和 `DataLoader` 用于把 `.npy` 图像和标签包装成可迭代 mini-batch。

```python
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
```

这三个函数分别计算：

- AUC；
- 0.5 阈值分类准确率；
- 概率预测校准误差 Brier score。

```python
from config import (...)
```

从 `config.py` 读取：

| 名称 | 含义 |
|---|---|
| `PRED_DIR` | 预测结果输出目录 |
| `MODEL_DIR` | 模型权重输出目录 |
| `TRAIN_END` | 训练集结束日期 |
| `VALID_START` | 验证集开始日期 |
| `VALID_END` | 验证集结束日期 |
| `TEST_START` | 测试集开始日期 |
| `EXPERIMENTS` | 实验矩阵配置 |
| `image_dir_for_experiment` | 根据实验名生成 shard 根目录 |

## 7. 全流程概览

脚本运行入口是：

```python
if __name__ == "__main__":
    main()
```

`main()` 执行以下流程：

1. 遍历 `config.EXPERIMENTS` 中的所有实验。
2. 对每个实验定位 shard 根目录。
3. 调用 `train_one_experiment(...)`。
4. 在 `train_one_experiment(...)` 内部：
   - 读取全部图像 shard；
   - 合并全部 metadata shard；
   - 按日期切分 train/valid/test；
   - 构建 Dataset 和 DataLoader；
   - 初始化 JiangCNN2D；
   - 用训练集训练；
   - 用验证集 early stopping；
   - 恢复验证集 loss 最低的模型；
   - 在测试集上预测；
   - 保存模型权重；
   - 保存测试集预测结果。

伪代码：

```text
for exp_name, cfg in EXPERIMENTS:
    image_shards, meta = load_image_shards(exp_name)

    train_mask, valid_mask, test_mask = get_split_masks(meta)

    train_loader = DataLoader(ImageDataset(... train_idx ...))
    valid_loader = DataLoader(ImageDataset(... valid_idx ...))
    test_loader = DataLoader(ImageDataset(... test_idx ...))

    model = JiangCNN2D(window=cfg["window"], image_height=H, image_width=W)
    optimizer = Adam(model.parameters(), lr=1e-5)
    criterion = BCEWithLogitsLoss()

    for epoch in 1..n_epochs:
        train one epoch
        evaluate validation loss
        save best state if validation loss improves
        stop if no improvement for patience epochs

    restore best state
    evaluate test set
    save model and predictions
```

## 8. 类和函数逐项说明

## 8.1 `ImageDataset`

定义：

```python
class ImageDataset(Dataset):
```

用途：

把图像数组、标签数组和样本索引包装成 PyTorch Dataset，供 DataLoader 按 batch 读取。

为什么需要自定义 Dataset：

- 图像文件可能很大，尤其是 I60；
- 脚本使用 `np.load(..., mmap_mode="r")`，不希望提前复制 train/valid/test 三份数组；
- Dataset 只保存 shard memmap 列表、样本定位索引和当前 split 的 indices；
- 每次 `__getitem__` 只读取一个样本。

### `ImageDataset.__init__`

定义：

```python
def __init__(self, image_shards, labels, shard_ids, local_indices, indices):
```

输入变量：

| 变量 | 类型 | 含义 |
|---|---|---|
| `image_shards` | list[memmap] | 图像 shard 列表，每个 shard 形状 `[N, H, W, C]` |
| `labels` | NumPy array | 全量标签数组，形状 `[N]` |
| `shard_ids` | NumPy array | 全量样本对应的 shard 编号 |
| `local_indices` | NumPy array | 全量样本在各自 shard 内的行号 |
| `indices` | array-like | 当前 split 使用的行号 |

内部变量：

```python
self.image_shards = image_shards
```

保存图像 shard 引用。这里不复制图像。

```python
self.labels = labels.astype(np.float32)
```

把标签转成 `float32`，因为 `BCEWithLogitsLoss` 需要浮点标签。

```python
self.shard_ids = np.asarray(shard_ids, dtype=np.int64)
self.local_indices = np.asarray(local_indices, dtype=np.int64)
```

保存每个全量样本的 shard 定位信息。

```python
self.indices = np.asarray(indices, dtype=np.int64)
```

把索引转成 `int64` NumPy 数组，保证后续可稳定索引。

### `ImageDataset.__len__`

定义：

```python
def __len__(self):
    return len(self.indices)
```

返回当前 split 的样本数。

例如：

- train dataset 返回训练样本数；
- valid dataset 返回验证样本数；
- test dataset 返回测试样本数。

### `ImageDataset.__getitem__`

定义：

```python
def __getitem__(self, idx):
```

输入：

| 变量 | 含义 |
|---|---|
| `idx` | DataLoader 传入的局部索引，范围是 `[0, len(indices)-1]` |

关键步骤：

```python
real_idx = self.indices[idx]
```

把 split 内部索引转换成全量 metadata 中的真实行号。

```python
shard_id = self.shard_ids[real_idx]
local_idx = self.local_indices[real_idx]
x = self.image_shards[shard_id][local_idx].astype(np.float32)
```

定位到具体 shard，读取单张图像，并转为 `float32`。

注意：当前图像像素是 `0/255`。这里没有除以 255，因此模型看到的是 `0` 或 `255`。这贴近论文中 “0 or 255 for black or white pixels” 的表述。

```python
x = np.ascontiguousarray(np.transpose(x, (2, 0, 1)))
```

把图像从 NumPy/图像常用的 `NHWC` 单样本形状：

```text
[H, W, C]
```

转成 PyTorch 卷积层要求的：

```text
[C, H, W]
```

`np.ascontiguousarray` 用于确保内存连续，避免 `torch.from_numpy` 在非连续数组上出现性能或兼容问题。

```python
y = self.labels[real_idx]
```

读取对应标签。

```python
return torch.from_numpy(x), torch.tensor(y)
```

返回：

- 图像张量，形状 `[1, H, W]`；
- 标签张量，标量。

## 8.2 `JiangCNNBlock`

定义：

```python
class JiangCNNBlock(nn.Module):
```

用途：

表示 Jiang et al. CNN 的一个 building block。

结构：

```text
Conv2d -> BatchNorm2d -> LeakyReLU -> MaxPool2d
```

### `JiangCNNBlock.__init__`

定义：

```python
def __init__(self, in_channels, out_channels, stride=(1, 1), dilation=(1, 1)):
```

输入变量：

| 变量 | 含义 |
|---|---|
| `in_channels` | 输入通道数 |
| `out_channels` | 输出通道数，也就是卷积 filter 数量 |
| `stride` | 卷积步长，格式 `(vertical_stride, horizontal_stride)` |
| `dilation` | 卷积 dilation，格式 `(vertical_dilation, horizontal_dilation)` |

局部变量：

```python
kernel_size = (5, 3)
```

卷积核大小为 `5 x 3`。这是根据论文 Appendix 的 baseline model 设置。

```python
padding = (
    ((kernel_size[0] - 1) * dilation[0]) // 2,
    ((kernel_size[1] - 1) * dilation[1]) // 2,
)
```

计算 padding，使卷积在 stride 为 1 时尽量保持空间尺寸。考虑 dilation 后，实际感受野会变大，因此 padding 也需要随 dilation 调整。

`self.block`：

```python
self.block = nn.Sequential(...)
```

把若干层按顺序组合。

#### `nn.Conv2d`

```python
nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=kernel_size,
    stride=stride,
    dilation=dilation,
    padding=padding,
)
```

作用：

- 在图像上滑动卷积核；
- 学习局部价格形态；
- 把输入通道映射到更多输出通道。

参数解释：

| 参数 | 含义 |
|---|---|
| `in_channels` | 输入图像或上一层 feature map 的通道数 |
| `out_channels` | 当前层卷积核数量 |
| `kernel_size=(5,3)` | 每个卷积核覆盖 5 个 vertical pixels 和 3 个 horizontal pixels |
| `stride` | 每次卷积窗口移动步长 |
| `dilation` | 扩张卷积间距 |
| `padding` | 边界补零 |

#### `nn.BatchNorm2d`

```python
nn.BatchNorm2d(out_channels)
```

作用：

- 对卷积输出做 batch normalization；
- 稳定训练；
- 减少 covariate shift；
- 对应论文中卷积和激活之间的 batch normalization。

#### `nn.LeakyReLU`

```python
nn.LeakyReLU(negative_slope=0.01)
```

作用：

- 引入非线性；
- 负半轴保留 `0.01 * x`，避免普通 ReLU 的 dead neuron 问题；
- 对应论文中 Leaky ReLU 的设定。

#### `nn.MaxPool2d`

```python
nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), ceil_mode=True)
```

作用：

- 在 vertical 方向下采样；
- horizontal 方向不下采样；
- 保留时间宽度上的细粒度结构。

参数解释：

| 参数 | 含义 |
|---|---|
| `kernel_size=(2,1)` | 每次 pooling 覆盖 2 行、1 列 |
| `stride=(2,1)` | vertical 方向步长为 2，horizontal 方向步长为 1 |
| `ceil_mode=True` | 当尺寸不能整除时向上取整，贴近论文图示中的输出尺寸 |

### `JiangCNNBlock.forward`

定义：

```python
def forward(self, x):
    return self.block(x)
```

输入：

- `x`: PyTorch tensor，形状 `[B, C, H, W]`

输出：

- 经过 block 后的 feature map。

## 8.3 `JiangCNN2D`

定义：

```python
class JiangCNN2D(nn.Module):
```

用途：

根据输入窗口 `window` 自动构造 5/20/60 日图像对应的 CNN。

这是脚本的核心模型类。

### `JiangCNN2D.WINDOW_CONFIG`

定义：

```python
WINDOW_CONFIG = {
    5: {"num_blocks": 2, "first_stride_v": 1, "first_dilation_v": 1},
    20: {"num_blocks": 3, "first_stride_v": 3, "first_dilation_v": 2},
    60: {"num_blocks": 4, "first_stride_v": 3, "first_dilation_v": 3},
}
```

含义：

| window | blocks | 第一层 vertical stride | 第一层 vertical dilation |
|---:|---:|---:|---:|
| 5 | 2 | 1 | 1 |
| 20 | 3 | 3 | 2 |
| 60 | 4 | 3 | 3 |

对应论文设定：

- 5 日图像模型较浅；
- 20 日图像模型中等；
- 60 日图像模型更深；
- 第一层针对较长窗口使用更粗的 vertical stride 和更大的 vertical dilation。

### `JiangCNN2D.__init__`

定义：

```python
def __init__(self, window, image_height, image_width, in_channels=1):
```

输入变量：

| 变量 | 含义 |
|---|---|
| `window` | 图像窗口长度，必须是 5、20 或 60 |
| `image_height` | 图像高度 |
| `image_width` | 图像宽度 |
| `in_channels` | 输入通道数，当前默认为 1 |

#### window 检查

```python
if window not in self.WINDOW_CONFIG:
    raise ValueError(f"Unsupported CNN image window: {window}")
```

作用：

- 防止配置中出现模型不支持的窗口；
- 当前仅支持 5、20、60。

#### 读取当前 window 配置

```python
cfg = self.WINDOW_CONFIG[window]
```

`cfg` 包含：

- `num_blocks`
- `first_stride_v`
- `first_dilation_v`

#### 构造通道数

```python
channels = [64 * (2 ** i) for i in range(cfg["num_blocks"])]
```

生成每个 block 的输出通道数。

例子：

| window | `num_blocks` | `channels` |
|---:|---:|---|
| 5 | 2 | `[64, 128]` |
| 20 | 3 | `[64, 128, 256]` |
| 60 | 4 | `[64, 128, 256, 512]` |

#### 逐层构建 block

```python
prev_channels = in_channels
for i, out_channels in enumerate(channels):
```

`prev_channels` 表示当前 block 的输入通道数。

第一层：

```python
if i == 0:
    stride = (cfg["first_stride_v"], 1)
    dilation = (cfg["first_dilation_v"], 1)
```

只有第一层使用论文指定的 vertical stride 和 dilation。

后续层：

```python
else:
    stride = (1, 1)
    dilation = (1, 1)
```

后续层使用普通卷积。

添加 block：

```python
blocks.append(
    JiangCNNBlock(
        in_channels=prev_channels,
        out_channels=out_channels,
        stride=stride,
        dilation=dilation,
    )
)
```

更新下一层输入通道：

```python
prev_channels = out_channels
```

#### 组合 feature extractor

```python
self.features = nn.Sequential(*blocks)
```

`self.features` 是所有 CNN block 的顺序组合。

#### 自动计算 FC 输入维度

```python
with torch.no_grad():
    dummy = torch.zeros(1, in_channels, image_height, image_width)
    feature_dim = self.features(dummy).flatten(1).shape[1]
```

作用：

- 用一张虚拟空图像跑一遍 CNN；
- 自动得到 flatten 后的维度；
- 避免手工计算不同窗口下的 FC 输入长度。

变量解释：

| 变量 | 含义 |
|---|---|
| `dummy` | 形状 `[1, C, H, W]` 的全零张量 |
| `feature_dim` | CNN 输出 flatten 后的特征长度 |

#### 分类器

```python
self.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Dropout(0.50),
    nn.Linear(feature_dim, 1),
)
```

结构：

1. `Flatten`: 把 CNN feature map 展平成向量；
2. `Dropout(0.50)`: 随机丢弃 50% 特征，降低过拟合；
3. `Linear(feature_dim, 1)`: 输出一个 logit。

为什么输出 1 个 logit：

- 当前任务是二分类；
- `BCEWithLogitsLoss` 接收一个 logit；
- `torch.sigmoid(logit)` 得到正类概率；
- 等价于二分类 softmax 的正类概率，但实现更简洁。

#### Xavier 初始化

```python
self.apply(self._init_weights)
```

对模型中的卷积层和线性层应用 Xavier 初始化。

### `JiangCNN2D._init_weights`

定义：

```python
@staticmethod
def _init_weights(module):
```

作用：

初始化权重。

逻辑：

```python
if isinstance(module, (nn.Conv2d, nn.Linear)):
    nn.init.xavier_uniform_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
```

含义：

- 对 `Conv2d` 和 `Linear` 的权重使用 Xavier uniform；
- bias 初始化为 0；
- 对 BatchNorm 等层不做额外处理，使用 PyTorch 默认初始化。

### `JiangCNN2D.forward`

定义：

```python
def forward(self, x):
    x = self.features(x)
    logit = self.classifier(x).squeeze(-1)
    return logit
```

输入：

- `x`: 形状 `[B, 1, H, W]`

输出：

- `logit`: 形状 `[B]`

执行流程：

1. `self.features(x)` 提取图像特征；
2. `self.classifier(x)` 输出 `[B, 1]`；
3. `.squeeze(-1)` 转为 `[B]`；
4. 返回 logits。

注意：

- `forward` 不做 sigmoid；
- 训练 loss 使用 raw logits；
- 评估时再用 `torch.sigmoid(logit)` 转概率。

## 8.4 `get_device`

定义：

```python
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

用途：

自动选择训练设备。

返回：

| 条件 | 返回 |
|---|---|
| CUDA 可用 | `cuda` |
| CUDA 不可用 | `cpu` |

影响：

- 模型会被 `.to(device)` 移动到该设备；
- mini-batch 中的 `x` 和 `y` 也会被 `.to(device)`；
- 如果是 CUDA，会启用 `pin_memory=True`。

## 8.5 `get_split_masks`

定义：

```python
def get_split_masks(meta):
```

用途：

根据 `config.py` 中的日期边界，把样本切成训练集、验证集、测试集。

输入：

| 变量 | 含义 |
|---|---|
| `meta` | 当前实验的 metadata DataFrame |

关键代码：

```python
date = pd.to_datetime(meta["date"])
```

确保 `date` 是 pandas datetime 类型。

```python
train_mask = date <= TRAIN_END
```

训练集：

```text
date <= TRAIN_END
```

```python
valid_mask = (date >= VALID_START) & (date <= VALID_END)
```

验证集：

```text
VALID_START <= date <= VALID_END
```

```python
test_mask = date >= TEST_START
```

测试集：

```text
date >= TEST_START
```

返回：

```python
return train_mask.values, valid_mask.values, test_mask.values
```

返回三个 NumPy boolean arrays。

当前配置来自 `config.py`：

| 名称 | 当前值 |
|---|---|
| `TRAIN_END` | `2019-12-31` |
| `VALID_START` | `2020-01-01` |
| `VALID_END` | `2020-12-31` |
| `TEST_START` | `2021-01-01` |

## 8.6 `evaluate_model`

定义：

```python
def evaluate_model(model, loader, device, criterion):
```

用途：

在验证集或测试集上评估模型。

输入变量：

| 变量 | 含义 |
|---|---|
| `model` | 已训练或正在训练的 PyTorch 模型 |
| `loader` | valid_loader 或 test_loader |
| `device` | `cuda` 或 `cpu` |
| `criterion` | loss 函数，当前为 `BCEWithLogitsLoss` |

内部变量：

```python
probs = []
labels = []
total_loss = 0.0
n_obs = 0
```

| 变量 | 含义 |
|---|---|
| `probs` | 存放每个 batch 的预测概率 |
| `labels` | 存放每个 batch 的真实标签 |
| `total_loss` | loss 总和，按样本数加权 |
| `n_obs` | 已评估样本数 |

关键步骤：

```python
model.eval()
```

切换到评估模式：

- Dropout 停止随机丢弃；
- BatchNorm 使用评估模式统计。

```python
with torch.no_grad():
```

关闭梯度计算，节省显存和计算。

batch 循环：

```python
for x, y in loader:
    x = x.to(device)
    y = y.to(device)
    logit = model(x)
    loss = criterion(logit, y)
    prob = torch.sigmoid(logit).cpu().numpy()
```

含义：

1. 把数据移动到设备；
2. 前向传播得到 logit；
3. 计算 loss；
4. 用 sigmoid 得到正类概率。

收集结果：

```python
probs.append(prob)
labels.append(y.cpu().numpy())
total_loss += loss.item() * len(y)
n_obs += len(y)
```

最后拼接：

```python
probs = np.concatenate(probs)
labels = np.concatenate(labels)
```

指标计算：

```python
if len(np.unique(labels)) < 2:
    auc = np.nan
else:
    auc = roc_auc_score(labels, probs)
```

如果标签只有一个类别，AUC 无法定义，返回 `nan`。

```python
acc = accuracy_score(labels, probs > 0.5)
```

以 0.5 为阈值计算分类准确率。

```python
brier = brier_score_loss(labels, probs)
```

计算 Brier score，衡量概率预测误差。

```python
avg_loss = total_loss / max(n_obs, 1)
```

计算平均 loss。

返回：

```python
return probs, labels, auc, acc, brier, avg_loss
```

| 返回值 | 含义 |
|---|---|
| `probs` | 正类预测概率 |
| `labels` | 真实标签 |
| `auc` | ROC AUC |
| `acc` | accuracy |
| `brier` | Brier score |
| `avg_loss` | 平均 BCE loss |

## 8.7 `train_one_experiment`

定义：

```python
def train_one_experiment(exp_name, cfg, n_epochs=50, batch_size=128, lr=1e-5):
```

用途：

训练单个实验的 CNN 模型。

这是脚本中最重要的流程函数。

输入变量：

| 变量 | 默认值 | 含义 |
|---|---:|---|
| `exp_name` | 无 | 实验名，例如 `I20R5` |
| `cfg` | 无 | 当前实验在 `EXPERIMENTS` 中的配置 |
| `n_epochs` | `50` | 最大训练 epoch 数 |
| `batch_size` | `128` | mini-batch 样本数 |
| `lr` | `1e-5` | Adam 初始学习率 |

### 8.7.1 加载图像 shard 和 metadata

```python
print(f"Loading image shards: {image_dir_for_experiment(exp_name)}")
image_shards, meta = load_image_shards(exp_name)
image_height, image_width = image_shards[0].shape[1], image_shards[0].shape[2]
```

变量解释：

| 变量 | 含义 |
|---|---|
| `image_shards` | 多个 shard 的图像 memmap 列表 |
| `meta` | 合并后的 metadata 表 |
| `image_height` | 图像高度 |
| `image_width` | 图像宽度 |

这里不复制图像数据，适合大样本。每个样本通过 `shard_id` 和 `local_index` 定位到具体 shard 内的位置。

### 8.7.2 提取标签与 shard 索引

```python
labels = meta["label"].values.astype(np.float32)
shard_ids = meta["shard_id"].values.astype(np.int64)
local_indices = meta["local_index"].values.astype(np.int64)
```

`labels` 是长度为 `N` 的 NumPy 数组。

值域：

- `1.0`: 未来收益为正；
- `0.0`: 未来收益不为正。

### 8.7.4 生成时间切分 mask

```python
train_mask, valid_mask, test_mask = get_split_masks(meta)
```

得到三个 boolean arrays。

### 8.7.5 mask 转 indices

```python
train_idx = np.flatnonzero(train_mask)
valid_idx = np.flatnonzero(valid_mask)
test_idx = np.flatnonzero(test_mask)
```

变量解释：

| 变量 | 含义 |
|---|---|
| `train_idx` | 训练集样本在全量数组中的行号 |
| `valid_idx` | 验证集样本在全量数组中的行号 |
| `test_idx` | 测试集样本在全量数组中的行号 |

为什么用 indices：

- 避免 `images[train_mask]` 这种复制大数组的操作；
- Dataset 每次只根据 index 读取需要的样本。

### 8.7.6 检查 split 是否为空

```python
if len(train_idx) == 0 or len(valid_idx) == 0 or len(test_idx) == 0:
    raise RuntimeError(...)
```

如果任意 split 为空，训练没有意义，直接报错。

常见原因：

- 数据日期范围没有覆盖 train/valid/test；
- 过滤条件过严；
- 某个 experiment 样本不足。

### 8.7.7 选择设备

```python
device = get_device()
pin_memory = device.type == "cuda"
```

变量解释：

| 变量 | 含义 |
|---|---|
| `device` | 训练设备 |
| `pin_memory` | CUDA 下启用 pinned memory，加快 CPU 到 GPU 的数据传输 |

### 8.7.8 构造 DataLoader

训练集：

```python
train_loader = DataLoader(
    ImageDataset(image_shards, labels, shard_ids, local_indices, train_idx),
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=pin_memory,
)
```

验证集：

```python
valid_loader = DataLoader(
    ImageDataset(image_shards, labels, shard_ids, local_indices, valid_idx),
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=pin_memory,
)
```

测试集：

```python
test_loader = DataLoader(
    ImageDataset(image_shards, labels, shard_ids, local_indices, test_idx),
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=pin_memory,
)
```

参数解释：

| 参数 | 含义 |
|---|---|
| `batch_size` | 每个 batch 的样本数 |
| `shuffle=True` | 训练集打乱顺序 |
| `shuffle=False` | 验证/测试保持顺序 |
| `num_workers=0` | 不开多进程读取，Windows 下更稳定 |
| `pin_memory` | CUDA 时加速数据传输 |

### 8.7.9 初始化模型

```python
model = JiangCNN2D(
    window=cfg["window"],
    image_height=image_height,
    image_width=image_width,
).to(device)
```

变量解释：

| 变量 | 含义 |
|---|---|
| `cfg["window"]` | 决定模型使用 2/3/4 个 CNN blocks |
| `image_height` | 决定 FC 输入维度 |
| `image_width` | 决定 FC 输入维度 |

`.to(device)` 把模型移动到 GPU 或 CPU。

### 8.7.10 optimizer 和 loss

```python
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()
```

`optimizer`：

- 使用 Adam；
- 默认学习率 `1e-5`；
- 更新模型参数。

`criterion`：

- 二分类交叉熵；
- 输入为 raw logit；
- 内部包含 sigmoid 的数值稳定形式。

### 8.7.11 early stopping 变量

```python
best_valid_loss = np.inf
best_state = None
patience = 2
bad_epochs = 0
```

变量解释：

| 变量 | 含义 |
|---|---|
| `best_valid_loss` | 当前最好的验证集 loss |
| `best_state` | 验证 loss 最低时的模型权重 |
| `patience` | 允许验证 loss 连续不改善的 epoch 数 |
| `bad_epochs` | 当前连续不改善次数 |

当前规则：

- 如果验证 loss 下降，保存模型；
- 如果连续 2 个 epoch 没有下降，提前停止。

### 8.7.12 训练循环

```python
for epoch in range(1, n_epochs + 1):
```

最多训练 `n_epochs` 轮。

每轮开始：

```python
model.train()
total_loss = 0.0
n_obs = 0
```

`model.train()` 启用训练模式：

- Dropout 生效；
- BatchNorm 使用当前 batch 统计。

batch 训练：

```python
for x, y in train_loader:
    x = x.to(device)
    y = y.to(device)

    optimizer.zero_grad()
    logit = model(x)
    loss = criterion(logit, y)
    loss.backward()
    optimizer.step()

    total_loss += loss.item() * len(y)
    n_obs += len(y)
```

逐步解释：

| 步骤 | 含义 |
|---|---|
| `x.to(device)` | 图像移动到 GPU/CPU |
| `y.to(device)` | 标签移动到 GPU/CPU |
| `optimizer.zero_grad()` | 清空上一 batch 梯度 |
| `model(x)` | 前向传播 |
| `criterion(logit, y)` | 计算 BCE loss |
| `loss.backward()` | 反向传播 |
| `optimizer.step()` | 参数更新 |
| `total_loss += ...` | 累积训练 loss |
| `n_obs += ...` | 累积样本数 |

训练集平均 loss：

```python
train_loss = total_loss / max(n_obs, 1)
```

### 8.7.13 验证集评估

```python
valid_prob, valid_y, valid_auc, valid_acc, valid_brier, valid_loss = evaluate_model(
    model, valid_loader, device, criterion
)
```

得到：

| 变量 | 含义 |
|---|---|
| `valid_prob` | 验证集预测概率 |
| `valid_y` | 验证集真实标签 |
| `valid_auc` | 验证集 AUC |
| `valid_acc` | 验证集 accuracy |
| `valid_brier` | 验证集 Brier score |
| `valid_loss` | 验证集平均 loss |

日志输出：

```python
print(
    f"Epoch {epoch:03d} | "
    f"loss={train_loss:.5f} | "
    f"valid loss={valid_loss:.5f} | "
    f"valid AUC={valid_auc:.4f} | "
    f"valid ACC={valid_acc:.4f} | "
    f"valid Brier={valid_brier:.4f}"
)
```

### 8.7.14 保存最佳模型

```python
if valid_loss < best_valid_loss:
    best_valid_loss = valid_loss
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    bad_epochs = 0
else:
    bad_epochs += 1
```

如果验证 loss 改善：

- 更新 `best_valid_loss`；
- 保存当前模型参数到 CPU；
- 重置 `bad_epochs`。

如果没有改善：

- `bad_epochs += 1`。

为什么保存到 CPU：

- 避免占用额外 GPU 显存；
- 后续恢复时再 `.to(device)`。

### 8.7.15 early stopping

```python
if bad_epochs >= patience:
    print("Early stopping triggered.")
    break
```

如果验证 loss 连续 `patience` 个 epoch 没有下降，提前停止训练。

### 8.7.16 恢复最佳模型

```python
if best_state is None:
    raise RuntimeError(f"{exp_name} did not produce a valid checkpoint.")

model.load_state_dict(best_state)
model = model.to(device)
```

如果训练过程中从未保存过模型，报错。

否则，恢复验证集 loss 最低的权重。

### 8.7.17 测试集评估

```python
test_prob, test_y, test_auc, test_acc, test_brier, test_loss = evaluate_model(
    model, test_loader, device, criterion
)
```

得到测试集指标。

日志：

```python
print(
    f"{exp_name} TEST | "
    f"loss={test_loss:.5f} | "
    f"AUC={test_auc:.4f} | ACC={test_acc:.4f} | Brier={test_brier:.4f}"
)
```

### 8.7.18 保存模型权重

```python
model_path = MODEL_DIR / f"jiang_cnn2d_{exp_name.lower()}.pt"
torch.save(model.state_dict(), model_path)
```

输出文件：

```text
outputs/models/jiang_cnn2d_{experiment}.pt
```

### 8.7.19 保存测试集预测结果

选取测试集 metadata：

```python
pred = meta.loc[test_mask, [
    "date", "code", "industry",
    "future_ret", "label",
    "amount", "float_mktcap", "is_limit_up",
]].copy()
```

新增模型信息：

```python
pred["experiment_name"] = exp_name
```

如果 metadata 有 `window`：

```python
pred["window"] = meta.loc[test_mask, "window"].values
```

如果 metadata 有 `horizon`：

```python
pred["horizon"] = meta.loc[test_mask, "horizon"].values
```

模型名：

```python
pred["model_name"] = "JiangCNN2D"
```

预测概率：

```python
pred["pred_prob"] = test_prob
```

保存路径：

```python
out_path = PRED_DIR / f"pred_{exp_name.lower()}_jiang_cnn2d.parquet"
pred.to_parquet(out_path, index=False)
```

## 8.8 `main`

定义：

```python
def main():
    for exp_name, cfg in EXPERIMENTS.items():
        train_one_experiment(exp_name, cfg)
```

用途：

遍历所有实验并逐个训练。

变量解释：

| 变量 | 含义 |
|---|---|
| `exp_name` | 实验名，例如 `I60R20` |
| `cfg` | 当前实验配置 |
| `image_dir_for_experiment(exp_name)` | 当前实验 shard 根目录 |

当前执行顺序由 `config.EXPERIMENTS` 的字典顺序决定。

## 9. 关键变量总表

## 9.1 数据相关变量

| 变量 | 位置 | 含义 |
|---|---|---|
| `image_shards` | `train_one_experiment` | 多个 shard 图像数组 memmap 列表 |
| `shard_ids` | `train_one_experiment` | 每行样本所在 shard 的编号 |
| `local_indices` | `train_one_experiment` | 每行样本在 shard 内的行号 |
| `image_height` | `train_one_experiment` | 图像高度 |
| `image_width` | `train_one_experiment` | 图像宽度 |
| `meta` | `train_one_experiment` | metadata DataFrame |
| `labels` | `train_one_experiment` | 二分类标签数组 |
| `train_mask` | `train_one_experiment` | 训练集 boolean mask |
| `valid_mask` | `train_one_experiment` | 验证集 boolean mask |
| `test_mask` | `train_one_experiment` | 测试集 boolean mask |
| `train_idx` | `train_one_experiment` | 训练集整数行号 |
| `valid_idx` | `train_one_experiment` | 验证集整数行号 |
| `test_idx` | `train_one_experiment` | 测试集整数行号 |

## 9.2 DataLoader 相关变量

| 变量 | 含义 |
|---|---|
| `train_loader` | 训练集 mini-batch 迭代器 |
| `valid_loader` | 验证集 mini-batch 迭代器 |
| `test_loader` | 测试集 mini-batch 迭代器 |
| `batch_size` | 每个 mini-batch 的样本数 |
| `pin_memory` | CUDA 下是否启用 pinned memory |
| `num_workers` | DataLoader 子进程数量，当前为 0 |

## 9.3 模型相关变量

| 变量 | 含义 |
|---|---|
| `model` | `JiangCNN2D` 实例 |
| `window` | 图像窗口长度，决定 CNN 深度 |
| `in_channels` | 输入通道数，当前为 1 |
| `channels` | 每个 CNN block 的输出通道数 |
| `prev_channels` | 当前 block 的输入通道数 |
| `out_channels` | 当前 block 的输出通道数 |
| `stride` | 当前卷积层 stride |
| `dilation` | 当前卷积层 dilation |
| `kernel_size` | 卷积核大小，当前为 `(5,3)` |
| `padding` | 卷积 padding |
| `feature_dim` | CNN 输出 flatten 后的维度 |
| `self.features` | CNN blocks |
| `self.classifier` | Flatten + Dropout + Linear |

## 9.4 训练相关变量

| 变量 | 含义 |
|---|---|
| `n_epochs` | 最大训练轮数 |
| `lr` | 学习率 |
| `optimizer` | Adam optimizer |
| `criterion` | BCEWithLogitsLoss |
| `epoch` | 当前 epoch |
| `x` | 当前 batch 图像 |
| `y` | 当前 batch 标签 |
| `logit` | 模型 raw output |
| `loss` | 当前 batch loss |
| `total_loss` | 当前 epoch 累积 loss |
| `n_obs` | 当前 epoch 累积样本数 |
| `train_loss` | 当前 epoch 平均训练 loss |

## 9.5 early stopping 变量

| 变量 | 含义 |
|---|---|
| `best_valid_loss` | 历史最低验证 loss |
| `best_state` | 历史最佳模型参数 |
| `patience` | 连续不改善容忍轮数 |
| `bad_epochs` | 当前连续不改善轮数 |

## 9.6 评估相关变量

| 变量 | 含义 |
|---|---|
| `probs` | 预测概率 |
| `labels` | 真实标签 |
| `auc` | ROC AUC |
| `acc` | Accuracy |
| `brier` | Brier score |
| `avg_loss` | 平均 loss |
| `valid_prob` | 验证集预测概率 |
| `valid_y` | 验证集真实标签 |
| `valid_auc` | 验证集 AUC |
| `valid_acc` | 验证集 accuracy |
| `valid_brier` | 验证集 Brier score |
| `valid_loss` | 验证集 loss |
| `test_prob` | 测试集预测概率 |
| `test_y` | 测试集真实标签 |
| `test_auc` | 测试集 AUC |
| `test_acc` | 测试集 accuracy |
| `test_brier` | 测试集 Brier score |
| `test_loss` | 测试集 loss |

## 9.7 输出相关变量

| 变量 | 含义 |
|---|---|
| `model_path` | 模型权重保存路径 |
| `pred` | 测试集预测结果 DataFrame |
| `out_path` | 预测 parquet 保存路径 |
| `pred_prob` | 未来收益为正的预测概率 |

## 10. 模型结构尺寸解释

当前图像结构来自 `config.py`：

| window | image height | image width | channels |
|---:|---:|---:|---:|
| 5 | 32 | 15 | 1 |
| 20 | 64 | 60 | 1 |
| 60 | 96 | 180 | 1 |

CNN block 数：

| window | blocks | channels |
|---:|---:|---|
| 5 | 2 | 64, 128 |
| 20 | 3 | 64, 128, 256 |
| 60 | 4 | 64, 128, 256, 512 |

第一层特殊设置：

| window | first vertical stride | first vertical dilation |
|---:|---:|---:|
| 5 | 1 | 1 |
| 20 | 3 | 2 |
| 60 | 3 | 3 |

后续层：

| 参数 | 值 |
|---|---|
| stride | `(1,1)` |
| dilation | `(1,1)` |
| kernel | `(5,3)` |
| pooling | `(2,1)` |

## 11. 与论文设定的对应关系

当前脚本对应 Jiang, Kelly, and Xiu (2023) 的主要设计：

| 论文设定 | 当前脚本实现 |
|---|---|
| 使用价格图像预测未来收益方向 | `label` 为 0/1，模型输出 `pred_prob` |
| 5/20/60 日图像 | `EXPERIMENTS` 中的 `window` |
| 5/20/60 日模型使用不同深度 | `JiangCNN2D.WINDOW_CONFIG` |
| CNN building block | `JiangCNNBlock` |
| Conv + BN + LeakyReLU + Pool | `JiangCNNBlock.block` |
| LeakyReLU slope = 0.01 | `nn.LeakyReLU(negative_slope=0.01)` |
| FC 前 dropout 50% | `nn.Dropout(0.50)` |
| Xavier 初始化 | `_init_weights` |
| Adam, lr=1e-5 | `torch.optim.Adam(..., lr=lr)` 默认 `lr=1e-5` |
| batch size 128 | `batch_size=128` |
| validation early stopping | `best_valid_loss`, `patience=2` |

实现上的一个工程选择：

- 论文描述最终 softmax 输出 up/down 两类概率；
- 当前脚本使用单 logit + `BCEWithLogitsLoss`；
- 评估时用 `sigmoid(logit)` 得到 up 概率；
- 对二分类任务，这与二分类 softmax 在预测概率意义上等价；
- 好处是输出文件天然只有一列 `pred_prob`，和回测脚本兼容。

## 12. 运行方式

在依赖和上游数据都已准备好后，可在项目目录运行：

```powershell
python 05_train_cnn2d.py
```

如果你使用 uv 管理环境，运行命令由你的环境设置决定，例如：

```powershell
uv run python 05_train_cnn2d.py
```

前提：

1. 已运行 `03_make_images.py`；
2. `data/images/{experiment}/shard_*/` 下已有所有实验的 `images.npy` 和 `meta.parquet`；
3. Python 环境中已有 `numpy/pandas/pyarrow/torch/scikit-learn`；
4. GPU 环境可用时，PyTorch 能正确识别 CUDA。

## 13. 常见问题

## 13.1 报错：找不到图像文件

可能原因：

- 未运行 `03_make_images.py`；
- 实验配置改过，但旧图像没有重新生成；
- 文件名与 `image_dir_for_experiment(exp_name)` 不一致。

处理：

- 先重新运行 `03_make_images.py`。

## 13.2 报错：某个 split 为空

报错来自：

```python
if len(train_idx) == 0 or len(valid_idx) == 0 or len(test_idx) == 0:
```

可能原因：

- 日期切分区间不合理；
- 数据覆盖时间太短；
- I60 图像过滤后样本过少；
- 训练/验证/测试边界与数据日期不匹配。

处理：

- 检查 `config.py` 中的 `TRAIN_END/VALID_START/VALID_END/TEST_START`；
- 检查对应 `shard_*/meta.parquet` 的日期范围。

## 13.3 AUC 显示为 `nan`

原因：

- 当前验证集或测试集 label 只有一个类别；
- `roc_auc_score` 无法在单类别标签上定义。

脚本处理：

```python
if len(np.unique(labels)) < 2:
    auc = np.nan
```

这不是程序崩溃，而是指标本身无法计算。

## 13.4 GPU 没有被使用

`get_device()` 只检查：

```python
torch.cuda.is_available()
```

如果返回 false，会自动使用 CPU。

可能原因：

- 安装的是 CPU 版 PyTorch；
- CUDA driver 或 PyTorch CUDA 版本不匹配；
- 当前环境不可见 GPU。

## 13.5 I60 训练慢

原因：

- I60 图像更大：`96 x 180`；
- 模型更深：4 个 CNN blocks；
- 通道数最高到 512；
- 样本读取和卷积计算都更重。

可调参数：

- `batch_size`
- `n_epochs`
- `lr`
- `num_workers`

注意：Windows 下 `num_workers > 0` 可能引入多进程兼容问题，当前脚本保守设置为 0。

## 14. 修改建议

### 14.1 想只训练某一个实验

可以临时修改 `main()`：

```python
def main():
    exp_name = "I20R5"
    cfg = EXPERIMENTS[exp_name]
    train_one_experiment(exp_name, cfg)
```

### 14.2 想调整 batch size

改 `train_one_experiment` 默认参数：

```python
def train_one_experiment(..., batch_size=128, ...):
```

如果 4090 显存充足，可以尝试：

```python
batch_size=256
batch_size=512
```

但 batch size 变大可能影响优化表现，不只是速度问题。

### 14.3 想调整训练轮数

改：

```python
n_epochs=50
```

实际训练通常会被 early stopping 截断。

### 14.4 想改 early stopping

改：

```python
patience = 2
```

如果验证 loss 波动较大，可以尝试 3 或 5。

### 14.5 想保存 optimizer 状态

当前只保存：

```python
torch.save(model.state_dict(), model_path)
```

如果需要断点恢复，可改成保存 checkpoint：

```python
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_valid_loss": best_valid_loss,
    },
    model_path,
)
```

但这会改变加载逻辑，需要同步修改后续推理代码。

## 15. 运行前检查清单

运行 `05_train_cnn2d.py` 前，建议确认：

- `config.py` 中 `EXPERIMENTS` 正确；
- `03_make_images.py` 已按当前配置重新生成图像；
- `data/images/i5r5/shard_00000/images.npy` 存在；
- `data/images/i5r5/shard_00000/meta.parquet` 存在；
- 其他实验对应文件也存在；
- `.npy` 的样本数与 metadata 行数一致；
- `meta["date"]` 覆盖 train/valid/test 三个区间；
- 当前 Python 环境能 import `torch`；
- 当前 Python 环境能 import `sklearn`；
- 当前 Python 环境能读写 parquet。

## 16. 文件输出和后续回测

`05_train_cnn2d.py` 只输出测试集预测。训练集和验证集预测不会保存。

输出 parquet 会被 `06_backtest_decile.py` 读取。回测脚本按每个交易日的 `pred_prob` 做横截面排序：

```text
pred_prob 越高 -> 模型越看好未来收益为正
```

因此 `pred_prob` 是从 CNN 到组合回测之间最核心的接口字段。

## 17. 当前实现的边界

1. 未做行业中性化。
2. 未做市值中性化。
3. 未在训练中显式使用样本权重。
4. 未保存 validation/test 的逐 epoch 详细日志文件。
5. 未保存 optimizer checkpoint。
6. 未做多 GPU DDP。
7. 未做混合精度训练。
8. 未做 seed 固定。

这些不是 bug，而是当前版本为了保持流程清晰和复现论文主结构所做的简化。

## 18. 总结

`05_train_cnn2d.py` 的核心任务是：

1. 对每个 `I/R` 矩阵实验读取图像和标签；
2. 根据图像窗口自动选择 Jiang-style CNN 深度；
3. 用训练集拟合模型；
4. 用验证集 loss 做 early stopping；
5. 用测试集输出预测概率；
6. 保存模型权重和测试集预测结果；
7. 为后续十分组回测提供 `pred_prob`。

该脚本是图像识别信号进入投资组合评价之前的关键模型训练环节。
