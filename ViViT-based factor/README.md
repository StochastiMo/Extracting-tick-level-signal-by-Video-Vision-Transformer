# ViViT 基线因子

本项目提供可独立交付的数据清洗、ViViT 基线模型训练与年度样本外因子生成流程。

## 输入目录

项目文件夹需与 `level2_pic/`、`Wind/` 位于同一级目录：

```text
level2_pic/stock_pixels_cn.YYYYMMDD.npz
Wind/ASHAREEODPRICES.parquet
Wind/ASHAREEODDERIVATIVEINDICATOR.parquet
Wind/ASHARECALENDAR.parquet
ViViT-based factor/
```

所需 Wind 字段为 `s_dq_tradestatus`、`up_down_limit_status`、`s_dq_avgprice` 和 `s_dq_adjfactor`。

## 数据口径

- Universe：T 日图片有效，T 日与 T+1 日均不停牌，且 T+1 日不涨停、不跌停；状态缺失时不入选。
- 标签：按股票计算 `T+11 复权 VWAP / T+1 复权 VWAP - 1`。
- 标签标准化：每日横截面先按“中位数 ± 3×1.4826×MAD”去极值，再进行 z-score 标准化。
- 特征：每只股票、每个通道的 8×8 图片独立做归一化；20 日历史缺帧保留为空，不向前填充。
- 训练：使用轻量 ViViT 参数；滚动训练按标签结束日隔离训练、验证和预测区间。
- 因子：2018—2025 年各使用 3 个随机种子训练；某年度三个 seed 全部完成后立即输出该年度因子，每个种子的预测分数先逐日标准化，再等权平均。

## 运行方式

在本项目目录执行：

```bash
python -m vivit_factor check-inputs
python -m vivit_factor prepare-data
python -m vivit_factor smoke-test
python -m vivit_factor train-parallel --device-ids 0,1,2,3
python -m vivit_factor validate
```

也可以用 `python -m vivit_factor run-all` 按顺序执行全部步骤。训练中断后会自动从签名一致的 `last.pt` 恢复；如需忽略已有产物，请使用相应的 `--overwrite` 参数。

## 产物

数据、模型、日志和因子统一写入 `artifacts/`。最终年度因子文件为：

```text
artifacts/factors/vivit_oos_factor.YYYY.parquet
```

Notebook `notebooks/end_to_end_pipeline.ipynb` 逐步调用同一组 Python 函数，适合检查中间结果；命令行入口适合正式批量运行。
