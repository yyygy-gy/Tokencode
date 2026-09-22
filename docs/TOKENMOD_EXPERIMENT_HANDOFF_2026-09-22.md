# TokenModHiCroPL 实验交接文档（2026-09-22）

## 1. 目标与当前结论

目标是在 **不插入额外 token、不做 t2i/i2t 交叉注入** 的前提下，用短码（short code）对 CLIP Transformer 做结构化调制，验证是否能超过 residual 基线。

当前阶段性结论：

1. **调制不是完全学不动**，但默认设定下增益很弱，容易被 residual / distill 盖住。
2. **强 distill 有害**，会把模型往 frozen CLIP 拉回去，伤害泛化。
3. **硬 residual 能保底**，但也会压住 adapted 分支，让调制很难真正超过基线。
4. **attn-bias 表达力偏弱**；直接增强后出现 val 微涨、test 变差，说明“继续堆 attn-bias 表达力”这条路不稳定。
5. 现阶段最稳的结果是：
   - `residual=True`
   - `distill=False`
   - `modulation=attn_bias`
   - `bias_scale=5.0`
   - best val `66.33%`, best test `66.29%`

---

## 2. 当前代码状态

主文件：

- `trainers/token_mod_hicropl.py`
- `clip/model.py`
- `train.py`
- `configs/trainers/TokenModHiCroPL/vit_b16_ep50_k16.yaml`

当前实现要点：

### 2.1 调制方式

- 默认 `MODULATION: attn_bias`
- 备选已实现：`film`

### 2.2 attn-bias 已改成增强版

相对最初版本，当前代码已经改成：

1. **Visual bias**
   - 不再只改 `CLS -> patch`
   - 已改成 **双向 CLS-centered bias**：
     - `bias[:, 0, 1:] = patch_bias`
     - `bias[:, 1:, 0] = patch_bias`
2. **Text / Visual modulator**
   - 从单层线性改成了 **两层 MLP**
3. **Text bias**
   - 浅层：`SOS <-> template`
   - 深层：`EOT <-> template`
   - 仍然硬屏蔽 class-name / padding keys

### 2.3 residual / distill 开关

已支持：

- `USE_RESIDUAL`
- `USE_DISTILL`
- `BIAS_SCALE`
- `RESIDUAL_GATE`
- `RESIDUAL_GATE_INIT`

当前 residual 逻辑：

- 若 `RESIDUAL_GATE=False`：
  - `feat = normalize(feat_adapt + feat_fixed)`
- 若 `RESIDUAL_GATE=True`：
  - `alpha = sigmoid(residual_gate_logit)`
  - `feat = normalize(feat_fixed + alpha * feat_adapt)`
  - 初始 `alpha ≈ RESIDUAL_GATE_INIT`

注意：

- `residual_gate_logit` 挂在 `CustomCLIP` 上，不在 `prompt_learner` 里
- `build_model` 已显式允许它参与训练

### 2.4 默认配置

`configs/trainers/TokenModHiCroPL/vit_b16_ep50_k16.yaml` 当前默认值：

- `LR: 0.0025`
- `WARMUP_EPOCH: 1`
- `WARMUP_CONS_LR: 1e-5`
- `MODULATION: attn_bias`
- `BIAS_SCALE: 5.0`
- `USE_RESIDUAL: True`
- `USE_DISTILL: True`
- `RESIDUAL_GATE: True`
- `RESIDUAL_GATE_INIT: 0.1`

也就是说：**代码默认已经变成增强版 attn-bias + residual gate**。
后续新实验如果不显式覆盖，会带着这些改动跑。

---

## 3. 关键实验设置

统一设定：

- Dataset: `StanfordCars`
- Shots: `16`
- Seed: `1`
- Subsample: `base`
- Backbone: `ViT-B/16`
- Trainer: `TokenModHiCroPL`
- Batch size train: `32`
- Test batch size 短训时常用：`64`
- Workers 短训时常用：`2`

常用短训命令模板：

```bat
set PYTHONPATH=E:\Project\Hicropl_3\HiCroPL
python train.py ^
  --root E:\Project\dataset ^
  --seed 1 ^
  --trainer TokenModHiCroPL ^
  --dataset-config-file configs\datasets\stanford_cars.yaml ^
  --config-file configs\trainers\TokenModHiCroPL\vit_b16_ep50_k16.yaml ^
  --output-dir <OUTPUT_DIR> ^
  DATASET.NUM_SHOTS 16 ^
  DATASET.SUBSAMPLE_CLASSES base ^
  OPTIM.MAX_EPOCH 8 ^
  TRAIN.PRINT_FREQ 20 ^
  DATALOADER.TEST.BATCH_SIZE 64 ^
  DATALOADER.NUM_WORKERS 2 ^
  TRAINER.TOKENMOD.USE_RESIDUAL True ^
  TRAINER.TOKENMOD.USE_DISTILL False ^
  TRAINER.TOKENMOD.BIAS_SCALE 5.0
```

---

## 4. 关键实验结果

### 4.1 无 residual / 无 distill

目录：

- `output/token_mod/train_base/stanford_cars/shots_16/TokenModHiCroPL/vit_b16_ep50_k16/seed1_nrd_b5_e3`

设置：

- `residual=False`
- `distill=False`
- `bias_scale=5.0`
- `3 epoch`

结果：

- best val: `66.07%`
- best test: `65.34%`

说明：

- 关掉 residual 后，模型仍能训练
- 但稳定性与上限一般

### 4.2 residual + no distill（当前最稳）

目录：

- `output/token_mod/train_base/stanford_cars/shots_16/TokenModHiCroPL/vit_b16_ep50_k16/seed1_r_nodist_b5_e8`

设置：

- `residual=True`
- `distill=False`
- `bias_scale=5.0`
- `8 epoch`
- 当时还是较早期的 attn-bias（未上增强版 / gate）

结果：

| Epoch | Val Acc |
|------|---------|
| 1 | 66.33% |
| 2 | 65.82% |
| 3 | 65.56% |
| 4 | 65.56% |
| 5 | 65.31% |
| 6 | 66.07% |
| 7 | 66.33% |
| 8 | 65.82% |

- best test: `66.29%`
- 耗时约 `7m57s`

结论：

- 关掉 distill 明显有帮助
- residual 保住了 baseline
- 但调制本身没有真正把性能明显推上去

### 4.3 增强版 attn-bias + residual gate

目录：

- `output/token_mod/train_base/stanford_cars/shots_16/TokenModHiCroPL/vit_b16_ep50_k16/seed1_enh_bias_gate_e8`

设置：

- 双向 CLS bias
- MLP modulator
- `residual=True`
- `distill=False`
- `residual_gate=True`
- `gate_init=0.1`
- `bias_scale=5.0`
- `8 epoch`
- LR 仍为 `2.5e-3`

结果：

| Epoch | Val Acc |
|------|---------|
| 1 | 66.58% |
| 2 | 66.84% |
| 3 | **67.09%** |
| 4 | 66.33% |
| 5 | 66.07% |
| 6 | 66.33% |
| 7 | 66.07% |
| 8 | 66.07% |

- best test: `65.02%`
- 总耗时约 `7m48s`

gate 实际值：

- best ckpt（epoch 3）：`alpha ≈ 0.118`
- epoch 8：`alpha ≈ 0.156`

问题：

1. **val 微涨，test 反而掉**
2. **过早 peaking**
3. **loss 抖动更大**
4. **gate 几乎没打开**，但泛化已经变差

因此这版不应解读为“增强成功”，更像是：

> 增强后的 attn-bias 更容易在小 val 上抖出高峰，但不稳定，泛化变差。

---

## 5. 关于学习率的判断

当前优化器：

- Adam
- `LR=0.0025`
- warmup 1 epoch constant `1e-5`
- 然后直接跳到 `2.5e-3`，再 cosine

判断：

1. **不完全是 LR 设错**
   - 因为同一 LR 下，`residual + no distill` 仍能跑到 test `66.29%`
2. **但对增强版 attn-bias，这个 LR 很可能偏大**
   - loss 更抖
   - 更早 peaking
   - test 掉点
3. 更准确说法：
   - LR 是放大器
   - 根因更像是增强后的 attn-bias 方向不稳 / 收益差

如果以后还想验证 LR，最干净的做法是：

- 固定增强结构
- 只把 `LR` 降到 `5e-4` 或 `1e-3`
- 可顺便把 warmup 拉长到 2~3 epoch

但本次对话结束时，决定 **先不继续试**。

---

## 6. 已经形成的技术判断

### 6.1 有效观察

1. **distill 太强是负向因素**
2. **residual 是保底器，不是增益器**
3. **attn-bias 当前主要瓶颈不是“完全没梯度”，而是“改写 logits 的能力弱 / 不稳”**
4. 单纯加大 `bias_scale` 帮助有限
5. 同时改太多东西（双向 bias + MLP + gate）会让归因变糊

### 6.2 不太建议继续的方向

1. 继续盲目加大 `bias_scale`
2. 在增强版 attn-bias 上继续叠更多复杂改动却不拆开对照
3. 继续保留强 distill

### 6.3 更值得考虑的下一步（尚未做）

按优先级：

1. **切到 FiLM**
   - `modulation=film`
   - `residual=True`
   - `distill=False`
   - 先不叠加太多花活
2. **只保留 residual gate，不改 bias 结构**
   - 验证 gate 本身有没有用
3. **只做 visual 双向 bias，不加 MLP**
   - 拆开归因
4. **降 LR 复测增强版 attn-bias**
   - 验证是不是 LR 放大了不稳定性
5. 更远一点：
   - value-side 低秩增量
   - 条件化 code（样本相关）
   - 更深 / 更稳的 modulator

当前更倾向的下一跳是：

> **FiLM + residual + no distill**
> 因为“继续在 attn-bias 内部堆表达力”已经显得收益可疑。

---

## 7. 日志与产物位置

短训日志：

- `output/token_mod/logs/cars_seed1_nrd_b5_e3.out.txt`
- `output/token_mod/logs/cars_seed1_r_nodist_b5_e8.out.txt`
- `output/token_mod/logs/cars_seed1_enh_bias_gate_e8.out.txt`

对应训练目录：

- `.../seed1_nrd_b5_e3`
- `.../seed1_r_nodist_b5_e8`
- `.../seed1_enh_bias_gate_e8`

临时检查脚本：

- `tmp_check_gate.py`
  - 用于读取 residual gate 的 logit / alpha

---

## 8. 新窗口接手时建议先做什么

1. 先读本文件，不要直接继续改代码
2. 明确当前代码已经是：
   - 增强版 attn-bias
   - residual gate 已接入
3. 如果要开新实验，先决定走哪条线：
   - A. 回退到简单 attn-bias，只做 gate / LR 对照
   - B. 直接切 FiLM
4. 任何新实验都尽量 **只改一个变量**
5. 短训优先看：
   - val 是否稳定上升
   - best test 是否超过 `66.29%`
   - residual gate 是否真的打开
   - loss 是否明显抖

---

## 9. 一句话交接

目前最稳基线是 `residual + no distill + attn_bias` 的 `66.29%` test；增强版 attn-bias 没有证明自己，反而暴露了不稳定。下一阶段更合理的是换调制形式（优先 FiLM）或把改动拆开做单变量对照，而不是继续同时堆复杂结构。

