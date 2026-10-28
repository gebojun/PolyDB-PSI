# AutoDL 部署与运行说明

## 上传内容

上传整个文件夹，不要只上传 `scripts`：

```text
v11_repeated5fold_cv_reliability_validation/
```

该目录已包含数据、8 个分类表、MACCS 缓存和 TabPFN checkpoint，不依赖本机其他
V10/V11 路径。

## 环境要求

开发和目标环境使用：

- Python 3.12
- TabPFN 8.0.8
- Torch 2.8 + CUDA 12.8
- NumPy 2.x
- pandas 2.x/3.x
- RDKit
- pyarrow

检查：

```bash
python scripts/00_check_environment.py
```

V11 的 packed-bit Tanimoto 计算要求 NumPy 2.x。若检查失败：

```bash
pip install -r requirements.txt
```

## 正式命令

推荐在 `screen` 内运行：

```bash
screen -S v11
cd ~/autodl-tmp/v11_repeated5fold_cv_reliability_validation
bash run_autodl.sh 2>&1 | tee run_all.log
```

按 `Ctrl+A`、再按 `D` 退出 screen；恢复：

```bash
screen -r v11
```

也可使用 nohup：

```bash
cd ~/autodl-tmp/v11_repeated5fold_cv_reliability_validation
nohup bash run_autodl.sh > run_all.log 2>&1 &
echo $!
```

## 线程变量

`run_autodl.sh` 固定设置：

```bash
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

这些设置限制 CPU 数值库的内部线程，避免多套线程池争抢资源，也避免此前
`libgomp: Invalid value for environment variable OMP_NUM_THREADS`。TabPFN 主推理仍在 GPU。

## 运行阶段

`run_autodl.sh` 调用 `scripts/run_all.py`，顺序为：

1. 环境、数据、缓存、checkpoint 检查。
2. 67 个性质的五次重复五折严格交叉拟合。
3. OOF 与真实缺失格的 SI 对齐可靠性计算。
4. P0-A 至 P0-D 可靠性验证图。
5. P1-A 至 P1-D 数据库知识空缺图。

真实缺失预测已在训练阶段按 repeat 对应 fold 生成，因此无需再运行额外的缺失预测脚本。

## 监控

```bash
tail -f run_all.log
nvidia-smi
watch -n 30 "find outputs/01_oof_predictions -name '*_metadata.json' | wc -l"
```

训练日志：

```text
outputs/logs/01_train_repeated5fold_cv_*.log
```

每个性质完成后会原子写入 CSV 和 JSON，因此日志中出现
`Completed <property>` 后该性质才算完成。

## 断点续跑

默认一键入口使用 `--skip-existing`。运行中断后再次执行：

```bash
nohup bash run_autodl.sh > run_all_resume.log 2>&1 &
```

以下三个训练文件同时存在时才跳过该性质：

```text
outputs/01_oof_predictions/{property}_all_folds.csv
outputs/01_oof_predictions/{property}_metadata.json
outputs/03_missing_predictions/{property}_missing_predictions.csv
```

可靠性阶段也要求 OOF 和 missing 两个可靠性文件同时存在才跳过。

## 局部运行与调试

三个代表性质：

```bash
python scripts/run_all.py --device cuda --test-mode --skip-plots
```

指定性质：

```bash
python scripts/run_all.py --device cuda --properties Tg_K dielectric_constant
```

强制重算已有结果：

```bash
python scripts/run_all.py --device cuda --properties Tg_K --overwrite-existing
```

## 完成检查

```bash
find outputs/01_oof_predictions -name '*_metadata.json' | wc -l
find outputs/02_oof_reliability_scores -name '*_oof_reliability.csv' | wc -l
find outputs/03_missing_predictions -name '*_missing_reliability.csv' | wc -l
find outputs/04_p0_figures -type f | wc -l
find outputs/06_p1_figures -type f | wc -l
```

前三项正式完整运行应各为 67。
