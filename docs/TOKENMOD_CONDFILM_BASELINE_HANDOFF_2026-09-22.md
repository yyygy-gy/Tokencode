# TokenMod 条件 FiLM Baseline 交接文档（2026-09-22）

## 0. 新窗口先读这三段

1. **当前 baseline 已切换为条件 FiLM**：
   - `MODULATION=film`
   - `USE_CONDITIONAL_CODES=True`
   - `USE_RESIDUAL=True`
   - `USE_DISTILL=False`
   - `RESIDUAL_GATE=False`
   - 建议 `MAX_EPOCH=40`
2. **StanfordCars seed1 当前主结果**：
   - 条件 FiLM：Base `80.73%` / Novel `68.80%` / HM `~74.29%`
   - 无条件 FiLM：Base `77.99%` / Novel `69.10%` / HM `~73.28%`
3. **eval-only 时必须忽略 checkpoint 中的 `fixed_embeddings`**。否则 base 权重拿到 novel 上会测崩。旧的无条件 FiLM novel `28.22%` 已判定为加载 bug；修复后是 `69.10%`。

---

## 1. 当前目标

在 **不插入额外 token、不做 t2i/i2t 交叉注入** 的前提下，用 short-code 结构化调制替换插入式 prompt。

这一轮已经验证：

- FiLM 比增强版 attn-bias 更有前途
- **条件化 short code** 比全局 short code 更能抬 base
- 但条件化目前 **没有同步抬 novel**
- 下一阶段 baseline 定为：**条件 FiLM + residual + no distill**

---

## 2. Baseline 定义

### 2.1 正式 baseline

名称建议：

- `CondFiLM-R-NoDist`

固定设置：

| 项 | 值 |
|---|---|
| Trainer | `TokenModHiCroPL` |
| Backbone | `ViT-B/16` |
| Dataset probe | `StanfordCars` |
| Shots | `16` |
| Seed | `1` |
| Split | CoOp/Zhou `base` / `new` |
| `MODULATION` | `film` |
| `USE_CONDITIONAL_CODES` | `True` |
| `USE_RESIDUAL` | `True` |
| `USE_DISTILL` | `False` |
| `RESIDUAL_GATE` | `False` |
| `CODE_LEN` | `16` |
| `RANK` | `8` |
| `PROMPT_DEPTH` | `12` |
| `LR` | `0.0025` |
| `MAX_EPOCH` | `40` |
| Train batch | `32` |
| Test batch | `64` |
| Workers | `2` |

### 2.2 条件化具体含义

- **Text codes**：由当前 split 的 frozen class embeddings 生成
  - 输入：`prompt_learner.fixed_embeddings`
  - 生成器：`text_code_generator`
- **Visual codes**：由当前样本的 frozen CLIP image features 生成
  - 输入：`ZS_image_encoder(image)`
  - 生成器：`visual_code_generator`
- 生成器是 per-layer 线性：
  - `code = Linear(normalize(feature))`
  - 每层一组 `(W, b)`
- 无条件 FiLM 则仍使用全局可学习 `text_codes / visual_codes`

### 2.3 为什么把它定为 baseline

相对无条件 FiLM：

- Base：`77.99 -> 80.73`（+2.74）
- Novel：`69.10 -> 68.80`（-0.30）
- HM：`73.28 -> 74.29`（+1.01）

结论：

- 条件化 **已经证明有效**，至少对 base / HM 有效
- 但它当前更像 “base specialization 更强”，还不是 “base-new 双升”
- 所以后续所有新模块，都应相对 **CondFiLM-R-NoDist** 比，而不是再回到 attn-bias 或无条件 FiLM

---

## 3. 关键代码状态

主文件：

- [`trainers/token_mod_hicropl.py`](E:/Project/Hicropl_3/HiCroPL/trainers/token_mod_hicropl.py)
- [`clip/model.py`](E:/Project/Hicropl_3/HiCroPL/clip/model.py)
- [`train.py`](E:/Project/Hicropl_3/HiCroPL/train.py)
- [`configs/trainers/TokenModHiCroPL/vit_b16_ep50_k16.yaml`](E:/Project/Hicropl_3/HiCroPL/configs/trainers/TokenModHiCroPL/vit_b16_ep50_k16.yaml)

### 3.1 重要开关

`train.py` 里已新增：

```python
cfg.TRAINER.TOKENMOD.USE_CONDITIONAL_CODES = False
```

注意：

- **代码默认仍是 False**
- 跑条件 FiLM 时必须显式传：
  - `TRAINER.TOKENMOD.USE_CONDITIONAL_CODES True`
- 这样无条件 FiLM checkpoint 仍然能正常加载

### 3.2 关键关键加载修复

`TokenModHiCroPL.load_model()` 现在会删除：

- class/token 相关 buffer
- 所有 `fixed_embeddings`
- 所有 `ZS_image_encoder`

原因：

- `fixed_embeddings` 是 class-specific 的 frozen classifier
- 若把 base checkpoint 的 embeddings 带进 novel 评测，会用 base 类名去给 novel 图打分
- 这会把 novel 假性打崩

已确认：

- 无条件 FiLM 旧 novel `28.22%` = 加载 bug
- 修复后重测 = `69.10%`

### 3.3 residual / distill

当前 residual：

```text
feat = normalize(feat_adapt + feat_fixed)
```

distill：

```text
loss = CE + λ * [(1 - cos(text, frozen_text)) + (1 - cos(image, frozen_image))]
```

默认 `λ=12`。

当前 baseline **不开 distill**。

---

## 4. 实验设计（到目前为止）

### 4.1 协议

StanfordCars 按 CoOp/Zhou 协议：

1. 读 `split_zhou_StanfordCars.json`
2. 16-shot 只在构建时抽一次，缓存为 `shot_16-seed_1.pkl`
3. 再按类 id 排序后对半切：
   - base = 前 98 类
   - new = 后 98 类
4. 选中后 label remap 到 `0..97`

本地规模：

| Split | Classes | Images |
|---|---:|---:|
| few-shot train | 196 | 3136 |
| few-shot val | 196 | 784 |
| base test | 98 | 4002 |
| novel test | 98 | 4039 |

说明：

- **不是每个 epoch 重新抽 16 张**
- epoch 间变化只有 shuffle + augmentation

### 4.2 这一轮比较轴

只在 StanfordCars / seed1 上做主探针，比较：

1. 无条件 FiLM + residual + no distill
2. 条件 FiLM + residual + no distill

都要求：

- 先在 `base` 上训
- 再 `eval-only` 到 `new`
- 用 `model-best.pth.tar`
- 忽略 checkpoint 内 `fixed_embeddings`

### 4.3 训练轮数结论

无条件 FiLM 50 轮 log 显示：

- best val = epoch 37，`81.12%`
- 41–50 基本平台，无明显新高
- train loss 还在降，但 val 不涨

因此：

- **下次默认训 40 轮**
- 不再默认 50
- 20 轮明显不够
- 30 轮接近，但可能吃不到峰值

条件 FiLM 已按这个结论直接训了 40 轮。

---

## 5. 关键主结果

### 5.1 无条件 FiLM

目录：

- train: `output/token_mod/train_base/stanford_cars/shots_16/TokenModHiCroPL/vit_b16_ep50_k16/seed1_film_r_nodist`
- novel retest: `output/token_mod/logs/cars_seed1_film_r_nodist_full_novel_refixed.out.txt`

设置：

- `film`
- `conditional=False`
- `residual=True`
- `distill=False`
- `MAX_EPOCH=50`
- best epoch = `37`

结果：

| Split | Acc |
|---|---:|
| Base test | 77.99% |
| Novel test | 69.10% |
| HM | ~73.28% |

### 5.2 条件 FiLM（当前 baseline）

目录：

- train: `output/token_mod/train_base/stanford_cars/shots_16/TokenModHiCroPL/vit_b16_ep50_k16/seed1_condfilm_r_nodist_e40`
- novel: `output/token_mod/logs/cars_seed1_condfilm_r_nodist_e40_novel.out.txt`

设置：

- `film`
- `conditional=True`
- `residual=True`
- `distill=False`
- `MAX_EPOCH=40`
- best epoch = `32`

结果：

| Split | Acc |
|---|---:|
| Base test | **80.73%** |
| Novel test | **68.80%** |
| HM | **~74.29%** |

训练侧补充：

- best val 出现在 epoch 32，val `82.14%`
- 总耗时约 `35m25s`
- trainable params：`688512`
  - 高于无条件 FiLM 的 `491904`
  - 多出来的是两个 conditional code generator

### 5.3 并排结论

| 模型 | Base | Novel | HM | 解读 |
|---|---:|---:|---:|---|
| Uncond FiLM-R-NoDist | 77.99 | 69.10 | 73.28 | 更稳的双端起点 |
| **Cond FiLM-R-NoDist** | **80.73** | 68.80 | **74.29** | **当前 baseline** |

核心判断：

1. 条件化把 base 明显抬起来了
2. novel 基本持平，甚至略降
3. HM 仍优于无条件
4. 所以 baseline 用条件 FiLM；下一问是 **怎么把 novel 一起抬上去**

---

## 6. 对话里已经形成的核心判断

### 6.1 该坚持的

1. **继续走调制，不回插入式 token**
2. **FiLM 优于增强 attn-bias**
3. **residual 先保留**，它是保底锚
4. **默认先关 distill**
5. **条件 FiLM 已成为新 baseline**
6. **StanfordCars 默认 40 epoch**
7. **novel 评测必须重建 class embeddings**

### 6.2 该警惕的

1. 条件化目前更偏 base specialization
2. residual 已经在锚 frozen CLIP；如果再加满配 distill（`λ=12`），可能过约束
3. 只看 base 会高估方法；必须同时报 novel / HM
4. 代码默认值还没切到条件 FiLM，命令行必须显式打开

### 6.3 关于“如果加 distill”

当前还没跑条件 FiLM + distill。基于机制和已有经验，预期是：

- novel：有机会涨到约 `70–72`
- base：大概率从 `80.73` 回落到约 `78–80`
- HM：有机会更好，但不保证

建议若试：

1. 先跑 `USE_DISTILL=True, LAMBD=12, MAX_EPOCH=40`
2. 若 base 掉太多而 novel 涨得少，再试 `LAMBD=4/6`

不要默认假设 distill 一定更好。

---

## 7. 可直接复现的命令

### 7.1 训练条件 FiLM baseline

```bat
set PYTHONPATH=E:\Project\Hicropl_3\HiCroPL
python train.py ^
  --root E:\Project\dataset ^
  --seed 1 ^
  --trainer TokenModHiCroPL ^
  --dataset-config-file configs\datasets\stanford_cars.yaml ^
  --config-file configs\trainers\TokenModHiCroPL\vit_b16_ep50_k16.yaml ^
  --output-dir output\token_mod\train_base\stanford_cars\shots_16\TokenModHiCroPL\vit_b16_ep50_k16\seed1_condfilm_r_nodist_e40 ^
  DATASET.NUM_SHOTS 16 ^
  DATASET.SUBSAMPLE_CLASSES base ^
  OPTIM.MAX_EPOCH 40 ^
  TRAIN.PRINT_FREQ 20 ^
  DATALOADER.TEST.BATCH_SIZE 64 ^
  DATALOADER.NUM_WORKERS 2 ^
  TRAINER.TOKENMOD.MODULATION film ^
  TRAINER.TOKENMOD.USE_RESIDUAL True ^
  TRAINER.TOKENMOD.USE_DISTILL False ^
  TRAINER.TOKENMOD.RESIDUAL_GATE False ^
  TRAINER.TOKENMOD.USE_CONDITIONAL_CODES True
```

### 7.2 Novel 评测

```bat
set PYTHONPATH=E:\Project\Hicropl_3\HiCroPL
python train.py ^
  --root E:\Project\dataset ^
  --seed 1 ^
  --trainer TokenModHiCroPL ^
  --dataset-config-file configs\datasets\stanford_cars.yaml ^
  --config-file configs\trainers\TokenModHiCroPL\vit_b16_ep50_k16.yaml ^
  --output-dir output\token_mod\test_new\stanford_cars\shots_16\TokenModHiCroPL\vit_b16_ep50_k16\seed1_condfilm_r_nodist_e40 ^
  --model-dir output\token_mod\train_base\stanford_cars\shots_16\TokenModHiCroPL\vit_b16_ep50_k16\seed1_condfilm_r_nodist_e40 ^
  --eval-only ^
  DATASET.NUM_SHOTS 16 ^
  DATASET.SUBSAMPLE_CLASSES new ^
  DATALOADER.TEST.BATCH_SIZE 64 ^
  DATALOADER.NUM_WORKERS 2 ^
  TRAINER.TOKENMOD.MODULATION film ^
  TRAINER.TOKENMOD.USE_RESIDUAL True ^
  TRAINER.TOKENMOD.USE_DISTILL False ^
  TRAINER.TOKENMOD.RESIDUAL_GATE False ^
  TRAINER.TOKENMOD.USE_CONDITIONAL_CODES True
```

### 7.3 看训练进度

```powershell
cd E:\Project\Hicropl_3\HiCroPL
Get-Content .\output\token_mod\logs\cars_seed1_condfilm_r_nodist_e40.out.txt -Wait
```

或只看关键行：

```powershell
Select-String -Path .\output\token_mod\logs\cars_seed1_condfilm_r_nodist_e40.out.txt -Pattern 'epoch \[|accuracy:|Checkpoint saved|Evaluate on the \*test\*'
```

---

## 8. 关键产物路径

### 8.1 条件 FiLM baseline

- 权重目录：
  [`E:/Project/Hicropl_3/HiCroPL/output/token_mod/train_base/stanford_cars/shots_16/TokenModHiCroPL/vit_b16_ep50_k16/seed1_condfilm_r_nodist_e40`](E:/Project/Hicropl_3/HiCroPL/output/token_mod/train_base/stanford_cars/shots_16/TokenModHiCroPL/vit_b16_ep50_k16/seed1_condfilm_r_nodist_e40)
- 训练日志：
  [`E:/Project/Hicropl_3/HiCroPL/output/token_mod/logs/cars_seed1_condfilm_r_nodist_e40.out.txt`](E:/Project/Hicropl_3/HiCroPL/output/token_mod/logs/cars_seed1_condfilm_r_nodist_e40.out.txt)
- Novel 日志：
  [`E:/Project/Hicropl_3/HiCroPL/output/token_mod/logs/cars_seed1_condfilm_r_nodist_e40_novel.out.txt`](E:/Project/Hicropl_3/HiCroPL/output/token_mod/logs/cars_seed1_condfilm_r_nodist_e40_novel.out.txt)

### 8.2 无条件 FiLM 对照

- 权重目录：
  [`E:/Project/Hicropl_3/HiCroPL/output/token_mod/train_base/stanford_cars/shots_16/TokenModHiCroPL/vit_b16_ep50_k16/seed1_film_r_nodist`](E:/Project/Hicropl_3/HiCroPL/output/token_mod/train_base/stanford_cars/shots_16/TokenModHiCroPL/vit_b16_ep50_k16/seed1_film_r_nodist)
- 修复后 novel 日志：
  [`E:/Project/Hicropl_3/HiCroPL/output/token_mod/logs/cars_seed1_film_r_nodist_full_novel_refixed.out.txt`](E:/Project/Hicropl_3/HiCroPL/output/token_mod/logs/cars_seed1_film_r_nodist_full_novel_refixed.out.txt)

### 8.3 旧交接文档

- 早期 attn-bias 交接：
  [`E:/Project/Hicropl_3/HiCroPL/docs/TOKENMOD_EXPERIMENT_HANDOFF_2026-09-22.md`](E:/Project/Hicropl_3/HiCroPL/docs/TOKENMOD_EXPERIMENT_HANDOFF_2026-09-22.md)
- 设计总文档：
  [`E:/Project/Hicropl_3/HiCroPL/TOKEN_MODULATION_PROMPT_DESIGN_AND_VALIDATION_ZH.md`](E:/Project/Hicropl_3/HiCroPL/TOKEN_MODULATION_PROMPT_DESIGN_AND_VALIDATION_ZH.md)

---

## 9. 新窗口建议下一步

按优先级：

1. **先把 CondFiLM-R-NoDist 当固定对照**
2. 若追 HM / novel：
   - 试 `CondFiLM + distill`，先 `λ=12`，不行再降到 `4/6`
3. 若追机制而不是调参：
   - 分析条件 codes 是否过度依赖 base class embeddings
   - 考虑 novel 更友好的 conditioning（例如只条件化 visual，或 text 用更弱/共享 generator）
4. 不要回头主攻增强 attn-bias，除非只做负对照
5. 任何新结果都至少报：
   - Base
   - Novel
   - HM
   - best epoch

### 一句话给新窗口

> 现在的主线不是“再证明调制有没有用”，而是：**在条件 FiLM baseline 上，如何提升 novel，同时尽量保住已经拿到的 base 增益。**
