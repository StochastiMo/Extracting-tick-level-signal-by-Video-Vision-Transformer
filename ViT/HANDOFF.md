# ViT 调参训练交接

## 当前口径

- Universe、T+1 执行掩码、T+1→T+11 标签和 feature store 均保持现有定义。
- `baseline` 与 `date_batch` 是训练只读历史实验；新调参 run 必须使用新文件夹名。
- 损失、batch、验证指标分别显式配置；原有七种损失允许任意搭配 batch。新增 `sigmoid_rank_ic_turnover` 因需要相邻完整截面，明确只允许 `trade_date`。
- 普通 `trade_date` 每个 epoch 以 `seed + epoch` 重排日期；时序换手损失固定自然日期升序、不 reshuffle。前一交易日预测缓存为 stop-gradient 固定锚点，每个训练日只前向一次；全部相邻日期对累计平均梯度后每个 epoch 只执行一次 Adam step。
- 时序复合损失为 `-SoftRankIC(pred_t, label_t) + lambda * (1 - SoftRankIC(pred_t, stopgrad(pred_t-1)))`，共同股票按 `instrument_id` 对齐；梯度只更新当期预测分支，默认 `lambda=0.1`、`tau=0.1`。
- 当前 seed 原本只能保证样本级 DataLoader 重排；日截面模式由 `TradeDateBatchSampler` 明确保证日期级重排、覆盖守恒和断点续训一致。
- 日截面模式使用 `torch.compile(dynamic=True, mode="default")`；时序换手损失禁止 `reduce-overhead`，避免动态 shape 的 CUDAGraph 私有显存池。预检会在第一张 GPU 同时启动 `TASKS_PER_GPU` 个进程，每个进程测试含最大截面在内的 12 组 shape。
- 验证记录每日硬 Spearman RankIC、每日 Pearson IC 和 pooled Pearson IC；新调参默认按日均 RankIC 选择最佳 checkpoint。

## 用户执行入口

Notebook 03 默认 `sigmoid_rank_penalty_turnover_stopgrad`、学习率 0.01、最多 500 epochs、patience 10、日均硬 Spearman RankIC 早停。参数区仍显式提供 run、损失参数、batch、prediction batch、验证指标、年度、seeds、GPU、学习率和早停配置。Run All 顺序为：

```text
preflight → 8年×3 seeds训练 → 年度prediction集成 → validate
```

同一命令重启可断点续训。run signature 包含损失名称及参数、batch 模式、验证指标和执行配置；任一配置不一致的 checkpoint 都不会被复用。

训练完成后切换到 factor-eval 内核运行 Notebook 04，并设置相同 `FOLDER_NAME`。Notebook 04 会：

1. 以完整 T 日预测截面固定五分位并计算 RankIC，再对分层收益、组合回测和换手应用 `can_enter_t1`；
2. 保存 Alphalens/Pyfolio 产物及 run 专属实验报告；
3. 校验所选 run 与 baseline 均为新分组口径，再展示 RankIC、Q5−Q1 和回测指标对比；旧 baseline factor_evaluation 需单独刷新一次。

factor-eval 环境没有 PyTorch 是预期状态；评价入口 `vit_stage08_factor_evaluation.py` 不依赖 PyTorch；训练命令独立位于 `vit_stage07_training.py`。

## 产物边界

- prediction：`data/predictions/runs/<run>/vit_predictions.YYYY.parquet`
- 训练审计：`data/quality/runs/<run>/a2_training_manifest.json`
- 因子评价：`data/quality/runs/<run>/factor_evaluation/`（包含中文累计收益图 `portfolio_cumulative_returns.png`，以及 `pic/` 下的 `portfolio_drawdown.png` 和 Alphalens returns、information、turnover tear sheet）
- 实验报告：`data/quality/runs/<run>/EXPERIMENT_REPORT.md`

图片截止 2024-12-31，因此 2025 只有模型、没有 prediction。覆盖已有实验必须显式设置 `OVERWRITE=1`；不生成 delivery。
