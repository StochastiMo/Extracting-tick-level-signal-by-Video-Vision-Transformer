# ViT Universe、A2 与多损失调参流

本目录把 17×8×8 的日度 Level-2 股票图片训练为年度滚动 ViT 因子。A2 仍为 2018–2025、每年 seeds 42/43/44，4 张 RTX 5090 上每卡并发 2 个独立单 GPU 任务。

## 源码阶段

| 顺序 | Python 模块 | 函数归属 |
|---:|---|---|
| 01 | [`vit_stage01_foundation.py`](vit_stage01_foundation.py) | 数据路径、签名、日期规范与原子写入 |
| 02 | [`vit_stage02_universe.py`](vit_stage02_universe.py) | 图像候选、交易日历、行情状态与 Universe |
| 03 | [`vit_stage03_labels.py`](vit_stage03_labels.py) | T+1→T+11 收益、Winsorize 与 Z-score |
| 04 | [`vit_stage04_feature_store.py`](vit_stage04_feature_store.py) | 图像归一化、feature store 与 rolling split |
| 05 | [`vit_stage05_model.py`](vit_stage05_model.py) | Patch embedding 与 ViT 主干 |
| 06 | [`vit_stage06_ranking.py`](vit_stage06_ranking.py) | Pearson、SoftRank、换手约束与统一损失 |
| 07 | [`vit_stage07_training.py`](vit_stage07_training.py) | DataLoader、训练、预检、checkpoint 与三种子集成 |
| 08 | [`vit_stage08_factor_evaluation.py`](vit_stage08_factor_evaluation.py) | 样本外 IC、分组收益、相对绩效与实验报告 |

各模块按从数据底座到评价的依赖顺序排列。`validate_*`、覆盖率汇总、预检和产物验收位于对应模块主流程之后。

## 数据口径

- Universe 候选来自每日 NPZ 的全部股票行；`in_universe_t` 只剔除 T 日停牌、涨停、跌停和状态缺失。
- `can_enter_t1` 是独立执行掩码，不参与 T 日预测、五分位分组或 RankIC；仅回测口径 `mode1_t_with_t1_filter` 会在五分位固定后将它用于组合收益和换手。
- 标签为 `adjclose(T+11) / adjclose(T+1) - 1`，每日 1%/99% Winsorize 后以 `ddof=0` 标准化。
- 特征样本为 `feature_valid & in_universe_t`；训练和验证只使用 `label_valid=True`。
- 回测默认 `mode2_t_no_filter`：使用 T 日 `s_dq_adjclose`，1/5/10D 对应 T→T+1/T+5/T+10，且不使用未来状态过滤。`mode1_t_with_t1_filter` 保留相同价格起点并应用 `can_enter_t1`；`mode3_t1_no_filter` 改为 T+1→T+2/T+6/T+11。
- 每种口径都自动计算完整因子覆盖池的全市场等权基准，以及 Q5、Q5−Q1 的超额年化收益、跟踪误差和信息比；Q5−Q1 保持 200% gross exposure。

## 损失、batch 与验证开关

Notebook 03 的 `LOSS_NAME`、`TRAIN_BATCH_MODE`、`VALIDATION_METRIC` 分别显式配置；除时序换手损失的完整截面约束外不会自动联动：

- 损失支持 `pearson_ic`、`sigmoid_rank_ic`、`neural_sort_rank_ic`、`mse`、`huber`、`smooth_l1`、`pearson_huber` 和 `sigmoid_rank_ic_turnover`。
- `fixed_samples` 保留固定 `batch_size=5000`、样本级 shuffle 和 `drop_last=True`；普通 `trade_date` 损失使用完整交易日，并在每个 epoch 按 `seed + epoch` 重排日期。
- 可微 RankIC 对原始预测和已完成 Winsorize/Z-score 的标签分别软排序，损失内部不额外标准化；软排序损失均为 O(n²)。
- `sigmoid_rank_ic_turnover` 使用 `-SoftRankIC(pred_t, label_t) + lambda * (1 - SoftRankIC(pred_t, stopgrad(pred_t-1)))`。它必须搭配 `trade_date`，按自然日期升序且不 reshuffle；前一日预测缓存为固定锚点，每个训练日只前向一次，每个 epoch 累积全部相邻日期对的平均梯度后只更新一次，默认 `lambda=0.1`。
- 验证记录日均硬 Spearman RankIC、日均 Pearson IC 和 pooled Pearson IC；默认按日均 RankIC 早停。

日截面模式使用 `torch.compile(dynamic=True, mode="default")`；时序换手损失禁止 `reduce-overhead`，避免动态截面为每个进程保留 CUDAGraph 私有显存池。Notebook 03 的预检会先在同一张 GPU 并发启动 `TASKS_PER_GPU` 个进程，各自覆盖含最大截面在内的 12 组动态 shape。`INFERENCE_BATCH_SIZE` 只控制验证读取与 prediction，不改变训练 batch 口径。

## Notebook 执行顺序

1. `00_build_vit_universe.ipynb`
2. `01_build_t10_labels.ipynb`
3. `02_build_vit_dataset.ipynb`
4. `03_train_rolling_vit.ipynb`：参数 → 预检 → 训练 → prediction → 验收
5. `04_evaluate_oos_backtest.ipynb`：使用 factor-eval 内核，选择同一 `FOLDER_NAME` 和 `BACKTEST_MODE`；默认口径 2，口径 1/3 可选。比较两个模型 run 前需用相同口径分别重新生成新版评价

历史实验 `baseline` 和 `date_batch` 均为训练只读。Notebook 03 默认新调参文件夹为：

```text
sigmoid_rank_penalty_turnover_stopgrad
```

## 等价 CLI

训练接口支持独立的损失、batch 和验证指标参数。完整参数以 Notebook 03 为准；评估和报告必须显式选择 run：

```bash
python vit_stage02_universe.py
python vit_stage03_labels.py
python vit_stage04_feature_store.py
python vit_stage07_training.py preflight --run-name <run> --loss-name sigmoid_rank_ic_turnover --turnover-lambda 0.1 --train-batch-mode trade_date --validation-metric daily_mean_spearman_ic --compile-mode default
python vit_stage07_training.py a2 --run-name <run> --loss-name sigmoid_rank_ic_turnover --turnover-lambda 0.1 --train-batch-mode trade_date --validation-metric daily_mean_spearman_ic --compile-mode default
python vit_stage07_training.py validate --run-name <run>
python vit_stage08_factor_evaluation.py backtest --run-name <run> --data-root ./data --backtest-mode mode2_t_no_filter
python vit_stage08_factor_evaluation.py report --run-name <run> --data-root ./data
```

## 主要产物

- `models/runs/<run>/`：年度×种子 checkpoint、history 和完成标记
- `data/predictions/runs/<run>/vit_predictions.YYYY.parquet`
- `data/quality/runs/<run>/a2_training_manifest.json`
- `data/quality/runs/<run>/factor_evaluation/`：RankIC、分层收益、含全市场等权的 `portfolio_returns.parquet`/Pyfolio 指标、`relative_performance.parquet`、累计收益图，以及 `pic/` 下的最大回撤图和 Alphalens tear sheet
- `data/quality/vit_hyperparameter_factor_summary.xlsx`：不同配置的指标与图片汇总；运行 `python build_factor_summary_xlsx.py` 可原子重建，No.11 完成因子评估后会自动加入
- `data/quality/runs/<run>/EXPERIMENT_REPORT.md`

图片数据截止 2024-12-31，因此 2025 会训练三种子模型但没有 2025 年 prediction。流程不会生成 delivery，也不会修改 `Wind/` 或 `level2_pic/`；只有显式设置 `FOLDER_NAME` 和 `OVERWRITE=1` 才会覆盖同名实验。
