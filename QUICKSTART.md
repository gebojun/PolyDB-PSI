# V11 快速启动

## 1. 上传

将整个 `v11_repeated5fold_cv_reliability_validation` 文件夹上传到 AutoDL，进入目录：

```bash
cd ~/autodl-tmp/v11_repeated5fold_cv_reliability_validation
```

## 2. 环境检查

```bash
python scripts/00_check_environment.py
```

预期关键结果：

```text
workflow: v11_repeated5fold_cv_reliability_validation
cuda.available: true
data.rows: 33142
fingerprint_cache.valid: 33140
data.selected_targets_after_valid_fp_filter: 67
status: ok
```

如果缺包：

```bash
pip install -r requirements.txt
```

## 3. CPU 冒烟测试

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python scripts/00_smoke_test_cpu_150rows.py
```

最后应看到：

```text
"smoke_test": "ok"
```

冒烟测试仅使用前 150 行和临时输出，不修改正式 `outputs`。

## 4. 正式后台运行

```bash
nohup bash run_autodl.sh > run_all.log 2>&1 &
echo $!
tail -f run_all.log
```

这条命令依次完成环境检查、67 个性质的 5x5 严格交叉拟合、可靠性计算和全部
P0/P1 可视化。

## 5. 查看进度

```bash
tail -n 100 run_all.log
find outputs/01_oof_predictions -name '*_metadata.json' | wc -l
find outputs/02_oof_reliability_scores -name '*_oof_reliability.csv' | wc -l
find outputs/03_missing_predictions -name '*_missing_reliability.csv' | wc -l
nvidia-smi
```

## 6. 中断后续跑

重新执行同一命令即可：

```bash
nohup bash run_autodl.sh > run_all_resume.log 2>&1 &
tail -f run_all_resume.log
```

完整的性质输出会被跳过；不完整性质会重新运行。

## 单独运行某个性质

```bash
python scripts/run_all.py \
  --device cuda \
  --properties Tg_K
```

## 仅训练，不画图

```bash
python scripts/run_all.py \
  --device cuda \
  --skip-plots
```

## 只重新计算可靠性和画图

```bash
python scripts/run_all.py \
  --skip-training \
  --device cuda
```
