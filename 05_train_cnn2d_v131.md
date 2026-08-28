# Image Trend v1.3.1 CNN 训练说明

## 目录和复现边界

所有 v1.3.1 作业通过环境变量把 `config.py` 的数据和输出根目录指向 `data_v1_3_1/` 与 `outputs/v1_3_1/`。默认未设置环境变量时，旧入口仍使用 `data/` 和 `outputs/`，因此不会覆盖 v1.2.x 产物。

训练入口也支持显式覆盖：`--data-dir /path/to/data_v1_3_1 --output-dir /path/to/outputs/v1_3_1`；Slurm 作业使用环境变量以便在作业链中保持路径一致。

数据阶段按以下顺序执行：

```bash
python -u 01_build_panel.py --workers 24
python -u 02_make_labels_and_baselines.py --workers 24
python -u 03_make_images.py --workers 24
python -u 08_qa_v131.py --data-root data_v1_3_1 --output-root outputs/v1_3_1
```

`START_DATE=2009-01-01`、`END_DATE=2024-12-31`。不向旧 shard 追加；每个阶段只在版本化目录内清理并完整重建。图像的首个有效日期由窗口回看自然决定，未来收益标签只在完整 horizon 可评估时保留。

## Purged 五折

`load_master_calendar()` 从 `features_by_year/year=2009..2020` 读取统一交易日历，`np.array_split` 产生五个连续验证块。对于第 `k` 折：

- 验证集是第 `k` 个日期块；
- 训练集是其余所有预 2021 样本；
- 验证块前 20 个和后 20 个交易日从训练集中剔除；
- 测试集固定为 2021-01-01 至 2024 年最后一个具有完整未来收益标签的日期。

`DateBatchSampler` 先按 `date` 再按零填充后的 `code` 稳定排序，每次 yield 一个完整日期截面。DataLoader 的 worker 只负责从 uint8 memmap shard 读取；GPU 端以固定 256 的物理微批处理，不改变日期级损失权重。

## 三种损失和目标

`--loss bce` 使用方向标签和 `binary_cross_entropy_with_logits`。`--loss huber` 对每个交易日的未来收益先做 1%/99% winsorize，再用总体标准差（`ddof=0`）截面 z-score，使用 `SmoothL1/Huber(beta=1.0)`。`--loss huber_ic` 使用：

```text
Huber + ic_weight * (1 - PearsonIC)
```

默认 `ic_weight=1.0`。IC 目标在完整日期截面计算；实现先无梯度收集整日 logits，再恢复 RNG/BatchNorm 状态重放一次，通过解析的 Huber 和 Pearson IC 外部梯度反传，避免把微批内相关性误当成整日 IC。

## 固定训练参数

| 参数 | 值 |
|---|---:|
| optimizer | AdamW |
| learning rate / weight decay | `1e-4` / `3e-5` |
| epochs | 最多 `25`，无最少轮数 |
| warm-up | 线性 `2` 轮 |
| early stopping | patience `4`，min_delta `1e-3`，从第 1 轮开始 |
| scheduler | ReduceLROnPlateau(mode=min, factor=0.5, patience=1, min_lr=1e-6) |
| seeds | `42, 43, 44, 45` |
| workers / prefetch | `2` / `2`，persistent workers |
| micro batch | `256` |
| memory | `pin_memory=True`，双缓冲 CUDA stream，`copy_(..., non_blocking=True)` |
| numerical | AMP、TF32、FC dropout `0.20` |

卷积层和线性层使用 LeakyReLU `a=0.01` 的 Kaiming normal（fan-out），bias 为零；BatchNorm weight 为一、bias 为零。训练日志记录每轮目标、组件、日期/行吞吐、等待时间和峰值显存（作业日志中由 CUDA/Slurm 环境保留）。

## 输出与断点

一个 GPU 数组任务对应一个 `fold × seed`，并顺序训练五组实验。每个损失 20 个任务，完成清单为：

```text
outputs/v1_3_1/manifests/{loss}/{experiment}/fold##_seed##.json
outputs/v1_3_1/models/{loss}/{experiment}/fold##_seed##.pt
outputs/v1_3_1/predictions/members/{loss}/{experiment}/fold##_seed##.parquet
```

完成清单、checkpoint 和成员预测均使用临时文件原子重命名；存在完整三者时可安全断点续跑。`aggregate_cnn_v131.py` 要求恰好 5×4 个成员、预测键和顺序完全一致，逐成员逐日 z-score 原始 logit 后做算术平均，输出 `pred_score`、兼容的 `pred_prob`、BCE 平均概率诊断和成员计数。

## 集群编排

```bash
bash slurm/submit_v131_pipeline.sh
```

该脚本提交数据阶段后，严格用 `afterok` 串联 `bce → huber → huber_ic`；每阶段完成聚合、`06_backtest_decile.py` 和 `07_backtest_nonoverlap_portfolio.py`。GPU 数组限制为最多并发 8 个 RTX 4090 任务并排除 V100。smoke 测试使用：

若数据阶段已单独提交并成功完成，可设置 `V131_DATA_JOB_ID=<jobid>`；若 smoke 也已提交，可再设置 `V131_SMOKE_JOB_ID=<jobid>`，首个训练阶段会等待两个作业均以 `afterok` 成功结束，避免重复构建或跳过 smoke。

退出 SSH/Codex 前可使用自后台化、幂等的提交器：

```bash
bash slurm/run_v131_background.sh <data-job-id> <smoke-job-id>
```

父进程立即返回，子进程通过 `nohup` 脱离终端；PID、日志和 `pipeline_started.txt`/`pipeline_submitted.txt` 位于 `outputs/v1_3_1/logs/background/`。原子锁和提交 marker 会阻止并发或重复创建 300 个模型的任务。若启动后只看到 `pipeline_started.txt` 而没有 `pipeline_submitted.txt`，先检查 launcher 日志和 Slurm 状态，不要直接重跑。

```bash
sbatch slurm/sub_v131_smoke.sh
```

smoke 输出在 `outputs/v1_3_1_smoke/`，不会写入全量 v1.3.1 完成清单。
