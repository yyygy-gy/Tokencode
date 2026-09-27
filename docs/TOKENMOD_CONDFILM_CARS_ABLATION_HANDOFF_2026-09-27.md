# TokenMod CondFiLM Cars 消融交接（2026-09-27）

## 0. 新窗口先读这四段

1. **当前最强 Cars seed1 组合**是：
   - `CondFiLM + task_text + preserve_norm + distill`
   - Base `79.86%` / Novel `76.06%` / HM `~77.91%`
2. **主线判断已经从“条件化抬 Base”推进到“保范数 + 任务级视觉码抬 Novel”**。
3. **`best@epoch40` 不是中途塌掉，而是后段平台**：val 在 38/40 并列最高 `80.10`，29-40 基本在 `79.5~80.1`。
4. **代码开关都已落地**，可直接复现：
   - `TRAINER.TOKENMOD.VISUAL_CODE_SOURCE=task_text|image`
   - `TRAINER.TOKENMOD.FILM_PRESERVE_NORM=True|False`
   - `TRAINER.TOKENMOD.USE_DISTILL=True|False`

---

## 1. 本轮要回答的问题

在不插入额外 token、不做 t2i/i2t 交叉注入的前提下：

1. 条件 FiLM 的视觉码是否必须样本级 `image`，还是任务级 `task_text` 也够？
2. FiLM 的有害通道是不是主要在范数，而不是方向？
3. `task_text + preserve_norm + distill` 叠在一起，Novel 能不能明显上来？

---

## 2. 固定实验设置

| 项 | 值 |
|---|---|
| Trainer | `TokenModHiCroPL` |
| Backbone | `ViT-B/16` |
| Dataset | `StanfordCars` |
| Shots | `16` |
| Seed | `1` |
| Split | CoOp/Zhou `base` / `new` |
| `MODULATION` | `film` |
| `USE_CONDITIONAL_CODES` | `True` |
| `USE_RESIDUAL` | `True` |
| `RESIDUAL_GATE` | `False` |
| `CODE_LEN` | `16` |
| `PROMPT_DEPTH` | `12` |
| `MAX_EPOCH` | `40` |

公共约束：

- 训练只看 base classes
- novel 用 `--eval-only --model-dir <train_dir>`
- eval-only 时继续忽略 checkpoint 中的 `fixed_embeddings`

---

## 3. 新增/确认的代码能力

### 3.1 `VISUAL_CODE_SOURCE`

位置：`trainers/token_mod_hicropl.py` / `train.py`

- `image`：样本级，frozen CLIP image features -> visual codes
- `task_text`：任务级，候选类表 `Y` 的 mean frozen text embeddings -> visual codes
  - 只用候选集合，不用 `y*`
  - 训练时 `Y=base`，novel eval 时 `Y=new`

### 3.2 `FILM_PRESERVE_NORM`

位置：`VisualFiLMGenerator` / `TextFiLMGenerator`

公式：

```text
y = (1 + γ) ⊙ x + β
y <- y * (||x|| / ||y||)
```

含义：

- 仍允许改方向/内容
- 但把有害 scale 通道掐掉
- 零初始化仍是恒等

### 3.3 Distill

沿用已有 residual distill：

- `USE_DISTILL=True`
- 默认 `LAMBD=12`

---

## 4. Cars seed1 结果总表

| Variant | Base | Novel | HM | Best Epoch | 备注 |
|---|---:|---:|---:|---:|---|
| CondFiLM (`image`, no distill) | 80.73 | 68.80 | ~74.29 | 既有 baseline | 抬 Base，Novel 一般 |
| CondFiLM + `task_text` | 80.88 | 70.09 | ~75.10 | 31 | 任务级视觉码不塌，Novel +1.3 |
| CondFiLM + `preserve_norm` | 81.11 | 70.83 | ~75.62 | 35 | Novel +2.0，Base 不掉 |
| **CondFiLM + `task_text` + `preserve_norm` + distill** | **79.86** | **76.06** | **~77.91** | **40** | 当前最强 HM / Novel |

相对普通 CondFiLM：

- `task_text`：Base 持平，Novel +1.29
- `preserve_norm`：Base +0.38，Novel +2.03
- 三件套：Base -0.87，Novel +7.26，HM +3.62

---

## 5. 关键结论

### 5.1 `task_text` 是通的

任务级视觉码没有塌：

- 说明视觉调制不一定依赖样本图像条件
- 候选类集合 `Y` 的文本均值已经能提供有效任务先验
- 这对“插入/普通 FiLM 都在改内容，但条件来源可换成任务级”这条叙事很关键

### 5.2 `preserve_norm` 打中了有害通道

保范数后：

- Novel 先动
- Base 没掉，还略升

这支持：

- AlignedNorm 的判断：有害通道往往是范数，不是方向
- FiLM 比插 token 更容易干净做这件事

### 5.3 三件套换来了 Novel 大幅提升

`task_text + preserve_norm + distill`：

- Novel 从 `68.80 -> 76.06`
- HM 从 `~74.29 -> ~77.91`
- Base 从 `80.73 -> 79.86`，有小幅代价

当前解释：

- `preserve_norm` 负责掐掉有害 scale
- `task_text` 负责给更稳的任务先验
- `distill` 进一步把表示拉回 CLIP 先验，放大 Novel

### 5.4 `best@40` 怎么理解

该 run 的 val：

```text
29: 79.59
30: 79.34
31-32: 79.59
33: 79.08
34: 78.83
35: 79.59
36-37: 79.85
38: 80.10
39: 79.59
40: 80.10
```

所以：

- 不是明显欠拟合后的“碰巧最后一拍”
- 更像后段平台，38/40 并列最好
- 若担心上限，下一步可把 `MAX_EPOCH` 拉到 `50/60` 复核；但现有证据不支持“明显没训够”

---

## 6. 日志与复现入口

### 6.1 关键日志

- `output/token_mod/logs/cars_seed1_condfilm_tasktext_vis_e40.out.txt`
- `output/token_mod/logs/cars_seed1_condfilm_preservenorm_e40.out.txt`
- `output/token_mod/logs/cars_seed1_condfilm_tasktext_preservenorm_distill_e40.out.txt`
- 对应 `*_novel.out.txt`

### 6.2 复现脚本

- `scripts/token_mod/run_condfilm_tasktext_vis_cars.py`
- `scripts/token_mod/run_condfilm_preservenorm_cars.py`
- `scripts/token_mod/run_condfilm_tasktext_preservenorm_distill_cars.py`

### 6.3 当前最强组合命令要点

```text
TRAINER.TOKENMOD.MODULATION film
TRAINER.TOKENMOD.USE_CONDITIONAL_CODES True
TRAINER.TOKENMOD.VISUAL_CODE_SOURCE task_text
TRAINER.TOKENMOD.FILM_PRESERVE_NORM True
TRAINER.TOKENMOD.USE_DISTILL True
TRAINER.TOKENMOD.USE_RESIDUAL True
TRAINER.TOKENMOD.RESIDUAL_GATE False
OPTIM.MAX_EPOCH 40
```

---

## 7. 新窗口建议下一步

优先级建议：

1. **先复核三件套是否稳**
   - 同设置再跑 seed2 / seed3
   - 或把 epoch 拉到 50/60，确认不是 40-budget 假象
2. **拆开交互**
   - `task_text + preserve_norm`（无 distill）
   - `image + preserve_norm + distill`
   - 看 Novel 大涨主要来自哪两项交互
3. **补一张范数图**
   - 插入 / 普通 FiLM / 保范数 FiLM 的 token L2
   - 预期：前两者范数飙，保范数版贴 CLIP
4. **若三件套站稳，再扩到第二数据集**
   - 优先 `FGVCAircraft` 或 `OxfordPets`，验证 Novel 提升是否迁移

---

## 8. 一句话现状

Cars 上，CondFiLM 已经从“条件化抬 Base”走到“任务级视觉码 + 保范数 + distill 抬 Novel”；当前最强点是 `79.86 / 76.06 / ~77.91`，下一步先确认它稳不稳，再决定是否升格为新 baseline。
