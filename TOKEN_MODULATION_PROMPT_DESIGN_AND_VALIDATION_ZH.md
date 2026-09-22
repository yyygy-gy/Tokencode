# Token Modulation Prompt：设计与落地规格

日期：2026-03-25
状态：待实现、待验证。本文档只覆盖 Token Modulation 这一条。
仓库：E:/Project/HiCropl_3/HiCroPL
骨架：本仓库已经消融掉双向流动的 HiCroPL。T2I / I2T mapper 和 LKP 已从 trainers/hicropl.py 删除，但深层 prompt 仍按 token 插入，蒸馏和 frozen residual 仍保留。

新窗口接到本文件后，只实现和验证这一条。不要并入 EMA、第二条 prompt、谱损失、反 prompt、几何正则、暗类蒸馏，也不要把 mapper / LKP 加回来。不要改坏现有 HiCroPL 分支。

---

## 0. 本仓库的真实骨架

这里没有 AblatedHiCroPL trainer，也没有 train_ablate_hicropl.py。消融是直接写进现有 HiCroPL 的：

关闭的模块：

```text
Text-to-Image Prompt mapper / Hierarchical Knowledge Mapper = 删除
Image-to-Text Prompt mapper                                 = 删除
Layer-specific Knowledge Proxy (LKP / AttentionPooling)     = 类还在文件里，但不再实例化、不再前向
```

证据：trainers/hicropl.py 的 CrossModalPromptLearner.__init__ 在初始化独立 text/visual prompt 之后直接打印

```text
Ablation enabled: bidirectional cross-modal knowledge flow is disabled
```

forward() 里也不再做 T->I / I->T mapping，只返回：

```text
text_input, current_visual_prompts[0], cross_prompts_text_deeper, cross_prompts_visual_deeper
```

仍保留的部分：

```text
12 层独立文本 Prompt token      (cross_prompts_text, 含共享 ctx)
12 层独立视觉 Prompt token      (cross_prompts_visual)
VisionTransformer_HiCroPL 插入  [CLS + patches] cat visual_ctx
ResidualAttentionBlock_HiCroPL  每层抠掉上一层 prompt 再 cat 当前层
Frozen image residual           image_features += ZS_image_encoder(image)
Frozen text residual            text_features  += fixed_embeddings
原始蒸馏                        CE + LAMBD * (1 - cos)
```

默认配置 configs/trainers/HiCroPL/vit_b16_c2_ep50_batch32_16ctx.yaml：

```text
N_CTX: 16
CROSS_LAYER: 6
PROMPT_DEPTH: 12
PREC: fp16
TEACHER_NAME: ViT-B/16
LAMBD: 12.
OPTIM.LR: 0.0025
MAX_EPOCH: 50
BATCH_SIZE: 32
```

因此 A0 就是本仓库当前的 HiCroPL：独立深层视觉 prompt + 独立深层文本 prompt + 插入接口 + frozen residual + 蒸馏。验证必须挂在这个骨架上。不要把方法加回 T2I/I2T 流动，也不要把 A0 改成纯 CE；本仓库基线本来就带着 LAMBD=12 和 residual。

CROSS_LAYER 在当前 hicropl.py 里只被读入 self.cross_layer，mapper 删除后已经没有前向用途。本方法重新启用它，但只用来划分浅层/深层调制权限，不表示跨模态流动。

---

## 1. 要证明什么

现有 prompt tuning 默认把可学习向量插入冻结 CLIP 的 token 序列。本仓库的消融版 HiCroPL 也是这样，只是已经关掉跨模态 prompt 交换：

- 文本：[SOS] + n_ctx + class/EOT，见 trainers/hicropl.py 的 construct_prompts()
- 视觉：[CLS + patches] cat visual_ctx，见 clip/model.py 的 VisionTransformer_HiCroPL.forward
- 深层：每层先抠掉上一层 prompt，再 cat 当前层 prompt，见 ResidualAttentionBlock_HiCroPL.forward
- 已关闭：Text-to-Image mapper、Image-to-Text mapper、LKP
- 仍开启：frozen image/text residual、原始蒸馏

插入之后，这些新 token 和 CLS / patch / class name 进入同一个 softmax。AlignedNorm 的纠缠塌缩、DeAR 的头污染、SkipT 的“多出来的 context 并不关键”，根因都是这个接口。

本方法的主张不是“换一种更好的 prompt 排法”，而是：

> base-new 冲突有一部分是插入接口带来的。新向量一进序列，就和预训练 token 抢同一组 softmax 槽位；base 类会把这些槽位占满，novel 再想用预训练几何就晚了。把 prompt 从“插入的 token”改成“对已有 token 的调制”后，序列长度不变，塌缩应明显减弱，novel 随训练下降的斜率应变缓。

成立条件必须同时满足，缺一不可：

1. 视觉序列长度始终是 1 + 14*14 = 197，不再出现 197 + n_ctx。
2. 文本不再插入 n_ctx 个可学习 token；class name token 始终不被调制。
3. 相对本仓库当前 HiCroPL（仍插入 token、但无双向流动），prompt/CLS 范数比更接近 1；“全局 token 对 prompt 的注意力”这项应消失或变得无意义，因为已经没有 prompt token。
4. 在 Stanford Cars / FGVC / DTD 上，novel 掉得比插入版慢；base 允许略降，但不能崩。
5. 把调制容量从 k=4 加到 k=64，novel 不应像插入版把 n_ctx 从 2 加到 16 那样明显变差。

如果只涨了 HM、诊断没变，视为失败，不是“另一种 prompt 碰巧变强”。

---

## 2. 明确不要做什么

- 不要在序列末尾或 SOS 后面 torch.cat 任何可学习 token。
- 不要把调制做成满的 N x N 注意力偏置矩阵；那会退化成另一种 prompt。
- 不要把已经删除的 mapper 输出重新变成 [n_ctx, dim] 再插回去，也不要为了本方法把 CrossPromptAttention / AttentionPooling 实例化回来。
- 第一轮损失对齐本仓库当前 HiCroPL：CE + LAMBD * distill，并保留 frozen residual。不要额外并入谱损失、几何正则、EMA。
- 不要上 Six-View、水平约束、法向约束、DPC 平行 prompt。
- 不要把 class name token 设成可学习，也不要对它做 FiLM。
- 不要改坏 trainers/hicropl.py、clip/model.py 的 HiCroPL 分支，以及 configs/trainers/HiCroPL/vit_b16_c2_ep50_batch32_16ctx.yaml。A0 必须还能用现有 train.py --trainer HiCroPL 跑。
- 不要实现 Slow/Fast Prompt。那是另一条线。

---

## 3. 核心设计

### 3.1 可学习对象

不再学习长度为 n_ctx 的 prompt token，改为每层一个短 code，再用很小的线性层把 code 变成结构化注意力偏置。

| 对象 | 形状 | 作用 |
|---|---|---|
| text_codes[l] | [k]，默认 k=16，dtype=fp16 | 第 l 层文本调制码 |
| visual_codes[l] | [k] | 第 l 层视觉调制码 |
| 偏置生成器 | 从 code 生成结构化注意力偏置 | 真正作用到已有 token 的 softmax |
| 跨模态通路 | 关闭 | 本仓库已经关掉 T2I/I2T。text_codes 和 visual_codes 独立更新，互不改写 |

层数仍为 PROMPT_DEPTH=12。CROSS_LAYER=6 只划分调制权限：

- 层 0 .. 5：浅层。文本只调 template，视觉只调 patch。
- 层 6 .. 11：深层。文本只调 EOT，视觉只调 CLS。

不要恢复 text2visual_net / visual2text_net / proxy_token。

### 3.2 文本序列怎么构造

冻结 CLIP 模板，不再插入 X X ... X。

```text
prompt = "a photo of a {class}."
```

用 CLIP tokenizer 得到长度为 77 的 token id，token_embedding 全部冻结。序列里只有预训练过的词：SOS、模板词、class name、EOT、padding。

位置约定（实现时用 tokenized_prompts 动态计算，不要写死下标）：

- sos_idx = 0
- eot_idx = tokenized_prompts.argmax(dim=-1)
- template_span：SOS 之后、class name 之前，对应 "a photo of a"
- class_span：模板之后、EOT 之前
- padding：EOT 之后

调制权限：

- 浅层（l < CROSS_LAYER）：只允许调制模板词，允许很小的 SOS 偏置
- 深层（l >= CROSS_LAYER）：只允许调制 EOT
- 任何层都禁止调制 class name 和 padding

### 3.3 视觉序列怎么构造

VisionTransformer 只保留：

```text
[CLS] + 196 patches + positional embedding
```

不要复制 VisionTransformer_HiCroPL.forward 里这段：

```python
visual_ctx = img_prompts.expand(...).half()
x = torch.cat([x, visual_ctx], dim=1)
```

也不要复制 ResidualAttentionBlock_HiCroPL 里抠掉旧 prompt 再 cat 新 prompt 的逻辑。

调制权限：

- 浅层：只调制 patch token（下标 1:）
- 深层：只调制 CLS（下标 0）
- 不调制并不存在的 prompt token

### 3.4 默认调制：结构化注意力偏置

默认实现走注意力偏置，不走满矩阵，也不先上 FiLM。原因：主张针对的是 softmax 竞争，偏置直接改的就是这件事。

CLIP 现有注意力是：

```python
self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
```

nn.MultiheadAttention 的 float attn_mask 会加到 logit 上。文本分支已有因果 mask（clip/model.py 的 CLIP.build_attention_mask，上三角为 -inf），新偏置必须加在因果 mask 之上，不能覆盖它。视觉分支原来 attn_mask=None，本方法只在允许的位置填有限偏置，其余保持 0。

视觉层 l

序列 [CLS, p1, ..., p196]。只允许改 CLS 作为 query、patch 作为 key 的那一行：

```text
c = visual_codes[l]                         # [k]
a = tanh(W_a c)                             # [n_head]
b = tanh(W_b c)                             # [n_head, r], r=8
patch_pos = positional_embedding[1:]        # [196, 768], 冻结
s = (patch_pos @ W_pos)                     # [196, r]
B[h, 0, 1:] = a[h] * (s @ b[h])             # 只填 CLS 行的 patch 列
其余位置 = 0
```

W_pos 每层独立、可学习，但 positional_embedding 本身冻结。不要学习 [197, 197] 的偏置表。浅层可以按同样公式作用到 patch query 对 patch key，但仍禁止引入新 token。第一轮为了实现简单：浅层和深层都只改 CLS 行；若诊断显示浅层完全没起作用，再把浅层改成 patch 行，不要一开始就做满矩阵。

文本层 l

在已有因果 mask 上，只给允许的 query 行、允许的 key 列加偏置：

```text
深层: B[h, eot, class_span] += tanh(W_eot c)[h]
浅层: B[h, sos, template_span] += tanh(W_sos c)[h]
```

class_span / template_span 必须从 tokenizer 算。不同类的 class name 长度不同，要按类做 mask。padding 列必须保持 0。

偏置幅度用 tanh 限制。初始化这些线性层为零，训练开始时调制为 0，模型等于冻结 CLIP 手工模板再加本仓库原有的 frozen residual。不要用随机偏置一上来打乱注意力。

### 3.5 对照调制：FiLM（只作为消融，不是主方法）

若注意力偏置的 base 完全涨不动，再开 FiLM 消融，不要和偏置叠满。

```text
x_i <- (1 + gamma_l(c)) * x_i + beta_l(c)
```

同样只作用在 3.2 / 3.3 允许的位置。gamma, beta 从 code 线性生成，零初始化。

主表只报注意力偏置。FiLM 放在消融表。

### 3.6 损失与推理

对齐本仓库当前 HiCroPL，不要改成纯 CE：

```text
image_features = L2(prompted_image) + L2(ZS_image_encoder(image))
image_features = L2(image_features)
text_features  = L2(prompted_text)  + L2(fixed_embeddings)
text_features  = L2(text_features)
logits = logit_scale * image_features @ text_features.T
loss = CE(logits, y) + LAMBD * [(1 - cos(text, fixed_text)) + (1 - cos(image, fixed_image))]
LAMBD = 12
```

推理仍是单视图、无 TTA、base 和 new 用同一套 code。本方法不在推理期拆通路。

如果后面要对比“关掉 residual / 蒸馏的调制版”，那是额外消融，不是 A0/A1 主表。主表必须和当前仓库基线共用同一套损失，否则分不清是接口有效还是把 teacher 拿掉了。

### 3.7 参数量预算

本仓库当前 HiCroPL（A0）大约：

```text
12 * 16 * 512  (独立 text tokens)
12 * 16 * 768  (独立 visual tokens)
无 mapper / LKP
另有冻结的 ZS_image_encoder 和 fixed_embeddings，不进优化器
```

本方法应明显更小。粗算：

```text
codes: 12 * 2 * k
每层 W_a / W_b / W_pos（视觉）和 W_sos / W_eot（文本）
无跨模态 MLP
```

k=16, r=8 时，可训练参数应低于 A0 的 1/5。在 log 里打印 trainable 数量，并写入结果。参数量更大却没有诊断改善，视为接口没改干净，尤其要查是否又把 prompt token 或 mapper 加回来。

---

## 4. 具体实现

### 4.1 新增 / 修改文件

| 文件 | 做什么 |
|---|---|
| clip/model.py | 新增 ResidualAttentionBlock_TokenMod、VisionTransformer_TokenMod；Transformer.__init__ 和 CLIP.__init__ 增加 TokenModHiCroPL 分支。HiCroPL 分支保持不动 |
| trainers/token_mod_hicropl.py | 从 trainers/hicropl.py 复制后改。删除 prompt token / construct_prompts 插入逻辑，换成 code + mask。trainer 名 TokenModHiCroPL |
| configs/trainers/TokenModHiCroPL/vit_b16_ep50_k16.yaml | 主配置。优化器和 LAMBD 对齐当前 HiCroPL |
| scripts/token_mod/base2new_train.sh | 训练 |
| scripts/token_mod/base2new_test.sh | 测 new / base |
| tools/token_mod_diagnostic.py | 序列长度、注意力、范数、容量扫描 |
| train.py | import trainers.token_mod_hicropl，extend_cfg() 增加 cfg.TRAINER.TOKENMOD |

不要改 configs/trainers/HiCroPL/*.yaml。A0 继续用现有脚本。

### 4.2 train.py

在现有 import trainers.hicropl 后增加：

```python
import trainers.token_mod_hicropl
```

在 extend_cfg() 末尾增加：

```python
cfg.TRAINER.TOKENMOD = CN()
cfg.TRAINER.TOKENMOD.N_CTX = 16          # 仅作兼容字段，实现中不得用来插入 token
cfg.TRAINER.TOKENMOD.CROSS_LAYER = 6
cfg.TRAINER.TOKENMOD.CTX_INIT = "a photo of a"
cfg.TRAINER.TOKENMOD.PREC = "fp16"
cfg.TRAINER.TOKENMOD.PROMPT_DEPTH = 12
cfg.TRAINER.TOKENMOD.TEACHER_NAME = "ViT-B/16"
cfg.TRAINER.TOKENMOD.LAMBD = 12.
cfg.TRAINER.TOKENMOD.CODE_LEN = 16       # k
cfg.TRAINER.TOKENMOD.RANK = 8            # r
cfg.TRAINER.TOKENMOD.MODULATION = "attn_bias"  # attn_bias | film
```

load_clip_to_cpu() 里 design_details['trainer'] 必须写成 'TokenModHiCroPL'。不要复用 'HiCroPL'，否则会走到 VisionTransformer_HiCroPL 的插入分支。

### 4.3 clip/model.py

保持 ResidualAttentionBlock_HiCroPL 和 VisionTransformer_HiCroPL 不动。新增两条路径。

1. Transformer.__init__ 在 HiCroPL 分支之后、else 之前插入：

```python
elif current_trainer == 'TokenModHiCroPL':
    self.resblocks = nn.Sequential(*[
        ResidualAttentionBlock_TokenMod(
            width, heads, attn_mask, text_layer, i, design_details
        )
        for i in range(layers)
    ])
```

2. CLIP.__init__ 在 trainer == "HiCroPL" 之后增加：

```python
elif trainer == "TokenModHiCroPL":
    self.visual = VisionTransformer_TokenMod(
        input_resolution=image_resolution,
        patch_size=vision_patch_size,
        width=vision_width,
        layers=vision_layers,
        heads=vision_heads,
        output_dim=embed_dim,
        design_details=design_details,
    )
```

3. VisionTransformer_TokenMod

从 VisionTransformer_HiCroPL 复制，但 forward 签名和插入逻辑改成：

```python
def forward(self, x, visual_codes, visual_modulators):
    # conv1 / reshape / cat CLS / add positional embedding
    # 不要 cat visual_ctx
    x = self.ln_pre(x)
    x = x.permute(1, 0, 2)  # [197, B, 768]
    outputs = self.transformer([x, visual_codes, visual_modulators, self.positional_embedding])
    x = outputs[0].permute(1, 0, 2)
    x = self.ln_post(x[:, 0, :])
    return x @ self.proj
```

断言：进入 transformer 之前 x.shape[1] == 197。

4. ResidualAttentionBlock_TokenMod

不要 add_prompt，不要 cat。inputs 约定：

```text
视觉: [x, visual_codes, visual_modulators, pos_embed]
文本: [x, text_codes, text_modulators, token_masks]
```

x 始终是 [L, B, C]。注意力改成：

```python
def attention(self, x, extra_mask=None):
    causal = None
    if self.attn_mask is not None:
        causal = self.attn_mask.to(dtype=x.dtype, device=x.device)
    if extra_mask is None:
        attn_mask = causal
    elif causal is None:
        attn_mask = extra_mask
    else:
        # extra_mask: [B*n_head, L, L] or [n_head, L, L]
        attn_mask = causal + extra_mask
    return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]
```

视觉偏置由 visual_modulators[self.i](visual_codes[self.i], pos_embed, n_head) 生成，形状 [n_head, 197, 197] 或 [B*n_head, 197, 197]，仅 [:, 0, 1:] 非零。

文本偏置由 text_modulators[self.i](text_codes[self.i], token_masks, n_head) 生成，仅允许位置非零。token_masks 至少包含：

```text
eot_idx:      [n_cls]
template_idx: [n_cls, 77] bool
class_idx:    [n_cls, 77] bool
padding_idx:  [n_cls, 77] bool
```

文本 transformer 的 batch 维是 n_cls，不是图像 batch。mask 按类广播即可。

PyTorch 的 MultiheadAttention 对 3D attn_mask 的期望是 [B*num_heads, L, S]。实现时把 [n_head, L, L] unsqueeze 到 batch 再 reshape，不要传 [n_head, L, L] 直接碰 2D causal mask 的广播坑。

### 4.4 trainers/token_mod_hicropl.py

从 trainers/hicropl.py 整文件复制，然后按下面改。保留：

- load_clip_to_cpu / load_clip_to_cpu_teacher
- gpt_clip_classifier 和 fixed_embeddings
- ZS_image_encoder
- CustomCLIP 里的 residual 加法和 LAMBD 蒸馏
- HiCroPL.build_model 的冻结逻辑、optimizer、load_model 忽略 token_prefix/suffix

必须改掉的部分：

1. 类名

```text
CrossModalPromptLearner -> TokenModPromptLearner
HiCroPL                 -> TokenModHiCroPL
```

cfg 命名空间从 cfg.TRAINER.HICROPL 改成 cfg.TRAINER.TOKENMOD。

2. load_clip_to_cpu 的 design_details

```python
design_details = {
    "trainer": "TokenModHiCroPL",
    "vision_depth": cfg.TRAINER.TOKENMOD.PROMPT_DEPTH,
    "language_depth": cfg.TRAINER.TOKENMOD.PROMPT_DEPTH,
    "vision_ctx": 0,
    "language_ctx": 0,
    "code_len": cfg.TRAINER.TOKENMOD.CODE_LEN,
}
```

vision_ctx / language_ctx 必须为 0，防止任何残留插入逻辑误用 n_ctx。教师模型仍用 trainer='IVLP' 且 depth=0。

3. TokenModPromptLearner.__init__

删除：

```text
self.ctx
self.cross_prompts_text
self.cross_prompts_visual
construct_prompts()
```

新增：

```python
k = cfg.TRAINER.TOKENMOD.CODE_LEN
d = cfg.TRAINER.TOKENMOD.PROMPT_DEPTH
self.text_codes = nn.ParameterList([
    nn.Parameter(torch.zeros(k, dtype=dtype)) for _ in range(d)
])
self.visual_codes = nn.ParameterList([
    nn.Parameter(torch.zeros(k, dtype=dtype)) for _ in range(d)
])
```

code 零初始化。不要 std=0.02 随机初始化，否则 step 0 就不是冻结 CLIP 模板。

文本构造改成手工模板，不再给 class name 前面留 n_ctx 个 X：

```python
prompt_prefix = cfg.TRAINER.TOKENMOD.CTX_INIT.replace("_", " ")  # "a photo of a"
prompts = [prompt_prefix + " " + name + "." for name in classnames]
tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # [n_cls, 77]
with torch.no_grad():
    embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
self.register_buffer("text_embedding", embedding)          # 冻结词向量，整段保存
self.register_buffer("tokenized_prompts", tokenized_prompts)
self._build_token_masks(tokenized_prompts, prompt_prefix, classnames)
```

_build_token_masks 必须动态算，不要写死。推荐算法：

```text
sos = 0
eot = tokenized_prompts.argmax(dim=-1)
tokenize("a photo of a") 得到模板 token 长度 L_tmp（不含 SOS/EOT）
template_span = [1, 1+L_tmp)
class_span    = [1+L_tmp, eot)
padding_span  = [eot+1, 77)
```

用 buffer 存：

```text
sos_idx, eot_idx, template_mask, class_mask, padding_mask
```

4. 调制器模块就放在同一文件

```python
class VisualBiasGenerator(nn.Module):
    def __init__(self, k, n_head, rank, vis_dim, layer_id, cross_layer):
        # W_a: [n_head, k]
        # W_b: [n_head * rank, k]
        # W_pos: [vis_dim, rank]
        # 全部 nn.init.zeros_

    def forward(self, code, pos_embed, seq_len=197):
        # 只填 B[:, 0, 1:]
        # 返回 [n_head, 197, 197]

class TextBiasGenerator(nn.Module):
    def __init__(self, k, n_head, layer_id, cross_layer):
        # 浅层 W_sos: [n_head, k]
        # 深层 W_eot: [n_head, k]
        # 全部 nn.init.zeros_

    def forward(self, code, token_masks):
        # 返回 [n_cls * n_head, 77, 77]
```

TokenModPromptLearner 里：

```python
self.visual_modulators = nn.ModuleList([
    VisualBiasGenerator(...) for i in range(d)
])
self.text_modulators = nn.ModuleList([
    TextBiasGenerator(...) for i in range(d)
])
```

5. TokenModPromptLearner.forward

```python
def forward(self):
    return (
        self.text_embedding,          # [n_cls, 77, 512], 无插入
        list(self.text_codes),
        list(self.visual_codes),
        self.text_modulators,
        self.visual_modulators,
        {
            "eot_idx": self.eot_idx,
            "template_mask": self.template_mask,
            "class_mask": self.class_mask,
            "padding_mask": self.padding_mask,
        },
    )
```

6. CustomCLIP.forward

图像和文本编码改成调制版，损失保持原样：

```python
text_emb, text_codes, visual_codes, text_mods, visual_mods, masks = self.prompt_learner()
text_features = self.text_encoder(text_emb, tokenized_prompts, text_codes, text_mods, masks)
image_features = self.image_encoder(image.type(self.dtype), visual_codes, visual_mods)

image_features = image_features / image_features.norm(dim=-1, keepdim=True)
image_features = image_features + image_features_fixed
image_features = image_features / image_features.norm(dim=-1, keepdim=True)
text_features = text_features / text_features.norm(dim=-1, keepdim=True)
text_features = text_features + self.prompt_learner.fixed_embeddings.half()
text_features = text_features / text_features.norm(dim=-1, keepdim=True)
```

TextEncoder.forward 对应改成把 [text_emb + pos, text_codes, text_mods, masks] 送进 transformer，仍从 EOT 取特征。

7. build_model 冻结规则

保持“只训练 prompt_learner”：

```python
name_to_update = "prompt_learner"
# ZS_image_encoder 仍冻结
# token_embedding / positional_embedding / transformer 权重仍冻结
```

打印 Parameters to be updated 时，集合里只应出现：

```text
prompt_learner.text_codes.*
prompt_learner.visual_codes.*
prompt_learner.text_modulators.*
prompt_learner.visual_modulators.*
```

若出现 cross_prompts_text、cross_prompts_visual、text2visual_net、visual2text_net、VPT，实现错了。启动时打印：

```text
TokenMod switches: insert_tokens=False, t2i=False, i2t=False, residual=True, distill=True, modulation=attn_bias, k=16, r=8
visual_seq_len=197, text_seq_len=77
```

8. load_model

继续忽略不存在的 token_prefix / token_suffix。如果 state_dict 里误存了 text_embedding，加载时也删掉，始终用当前 classnames 现场 tokenize。

### 4.5 主配置

新建 configs/trainers/TokenModHiCroPL/vit_b16_ep50_k16.yaml，除 TOKENMOD 字段外与 vit_b16_c2_ep50_batch32_16ctx.yaml 对齐：

```yaml
DATALOADER:
  TRAIN_X:
    BATCH_SIZE: 32
  TEST:
    BATCH_SIZE: 500
  NUM_WORKERS: 8

INPUT:
  SIZE: (224, 224)
  INTERPOLATION: "bicubic"
  PIXEL_MEAN: [0.48145466, 0.4578275, 0.40821073]
  PIXEL_STD: [0.26862954, 0.26130258, 0.27577711]
  TRANSFORMS: ["random_resized_crop", "random_flip", "normalize"]

OPTIM:
  NAME: "adam"
  LR: 0.0025
  MAX_EPOCH: 50
  LR_SCHEDULER: "cosine"
  WARMUP_EPOCH: 1
  WARMUP_TYPE: "constant"
  WARMUP_CONS_LR: 1e-5
  EPS: 1e-3

TRAIN:
  PRINT_FREQ: 20

MODEL:
  BACKBONE:
    NAME: "ViT-B/16"

TRAINER:
  TOKENMOD:
    N_CTX: 16
    CROSS_LAYER: 6
    CTX_INIT: "a photo of a"
    PREC: "fp16"
    PROMPT_DEPTH: 12
    TEACHER_NAME: "ViT-B/16"
    LAMBD: 12.
    CODE_LEN: 16
    RANK: 8
    MODULATION: "attn_bias"
```

### 4.6 训练 / 测试命令

数据根目录沿用 E:/Project/dataset。先 seed=1。不要一上来 3 seed。

A0 训练（本仓库当前消融 HiCroPL，插入 token，无 mapper）：

```bash
python train.py \
  --root E:/Project/dataset \
  --seed 1 \
  --trainer HiCroPL \
  --dataset-config-file configs/datasets/stanford_cars.yaml \
  --config-file configs/trainers/HiCroPL/vit_b16_c2_ep50_batch32_16ctx.yaml \
  --output-dir output/base2new/train_base/stanford_cars/shots_16/HiCroPL/vit_b16_c2_ep50_batch32_16ctx/seed1 \
  DATASET.NUM_SHOTS 16 \
  DATASET.SUBSAMPLE_CLASSES base
```

A1 训练：

```bash
python train.py \
  --root E:/Project/dataset \
  --seed 1 \
  --trainer TokenModHiCroPL \
  --dataset-config-file configs/datasets/stanford_cars.yaml \
  --config-file configs/trainers/TokenModHiCroPL/vit_b16_ep50_k16.yaml \
  --output-dir output/token_mod/train_base/stanford_cars/shots_16/TokenModHiCroPL/vit_b16_ep50_k16/seed1 \
  DATASET.NUM_SHOTS 16 \
  DATASET.SUBSAMPLE_CLASSES base
```

测 new：

```bash
python train.py \
  --root E:/Project/dataset \
  --seed 1 \
  --trainer TokenModHiCroPL \
  --dataset-config-file configs/datasets/stanford_cars.yaml \
  --config-file configs/trainers/TokenModHiCroPL/vit_b16_ep50_k16.yaml \
  --output-dir output/token_mod/test_new/stanford_cars/shots_16/TokenModHiCroPL/vit_b16_ep50_k16/seed1 \
  --model-dir output/token_mod/train_base/stanford_cars/shots_16/TokenModHiCroPL/vit_b16_ep50_k16/seed1 \
  --eval-only \
  DATASET.NUM_SHOTS 16 \
  DATASET.SUBSAMPLE_CLASSES new
```

测 base：同一命令，SUBSAMPLE_CLASSES base，output-dir 改成 test_base。A0 的测试把 trainer 和 config 换成现有 HiCroPL。

数据集顺序：stanford_cars -> fgvc_aircraft -> dtd。Cars 上诊断和斜率都没有改善就停，不要靠加容量或加几何损失硬抬 HM。

现有 output/base2new 里只有 DTD 的 HiCroPL 结果，没有 Stanford Cars。本规格不要求为了写文档去重跑 Cars；真正验证时 A0/A1 都从 seed=1 训练。DTD 的旧结果可以事后对照，但不能替代 Cars 主判定。

---

## 5. 必须先做的单元检查

训练前用 1 个 batch 卡住。写进 tools/token_mod_diagnostic.py，失败就不要开 50 epoch。

1. 视觉序列长度恒为 197。在 VisionTransformer_TokenMod.forward 里 assert x.shape[1] == 197。
2. 文本序列里不存在可学习 prompt token。text_embedding.requires_grad == False，且没有任何 Parameter 形状为 [16, 512] 或 [16, 768]。
3. class name token 的调制 mask 全 0。对 token_masks['class_mask'] 为 True 的列，偏置必须是 0。
4. 零初始化：step 0、不加载权重时，attn_bias 全 0。此时除去 frozen residual 外，prompted 特征应接近冻结 CLIP 手工模板。允许 residual 把特征拉开，但不允许 bias 非零。
5. 文本因果 mask 仍在。把 extra_mask 设为 0 时，eot 不得看到 padding 之后的未来位置；实现上 causal + extra 后，上三角仍是 -inf。
6. named_parameters() 里不应再出现 cross_prompts_text / cross_prompts_visual / text2visual / visual2text / VPT。
7. A0 日志仍应打印 Ablation enabled: bidirectional cross-modal knowledge flow is disabled。
8. 可训练参数量明显小于 A0。把两边的 numel 打进 log。

这些检查失败就不要开 50 epoch。

---

## 6. 对照实验

同一套数据、同一 seed、同一优化器、同一 LAMBD / residual。只改插入接口。

| ID | 名字 | 实现 | 目的 |
|---|---|---|---|
| A0 | 本仓库 HiCroPL | 插入独立深层 prompt，无 mapper，保留 residual+蒸馏 | 基线 |
| A1 | TokenMod attn-bias k=16 | 3.4 节默认 | 主方法 |
| A2 | 插入版 n_ctx=2 | 仍走 HiCroPL 插入，只改 N_CTX | 排除“少参数就好” |
| A3 | TokenMod k=4 | 更短 code | 容量是否关键 |
| A4 | TokenMod k=64 | 更长 code | 容量增加不应伤害 novel |
| A5 | 只调视觉，文本用冻结模板 | 视觉偏置 + 固定文本 | 定位模态 |
| A6 | 只调文本，视觉无偏置 | 文本偏置 + 冻结视觉 | 定位模态 |
| A7 | FiLM 替代 attn-bias | 3.5 节 | 接口形式 |

第一轮只跑 A0 和 A1，数据集 Stanford Cars。A1 的诊断优于 A0 且 novel 斜率更好，再跑 A2-A7 和另外两个数据集。

不要拿带 T2I/I2T 的完整 HiCroPL 当 A0。本仓库已经没有那条路径。也不要为了对齐别的仓库把 residual/蒸馏关掉。

A2 若接近 A0，说明病根仍是插入本身，不是 n_ctx 太大。A5/A6 用来判断主要污染在视觉还是文本。A7 只有在 A1 的 base 崩掉时才作为补救，不进主表。

---

## 7. 诊断，不是只看 HM

写 tools/token_mod_diagnostic.py。每个 epoch 把下列数写到 output_dir/diagnostics/token_mod.csv。

### 7.1 接口诊断（主张是否成立）

A0 仍插入 token，需要算：

- 视觉 prompt token 与 CLS 的平均 L2 范数比。A0 若明显大于 1，说明插入 token 在范数上压过 CLS
- CLS 对 prompt token 的注意力占比（12 头平均）
- 文本 EOT 对插入 context 的注意力占比

A1 没有 prompt token，对应项记 NA，改记：

- CLS 对 patch 的注意力熵，和冻结 CLIP 比
- EOT 对 class name 的注意力占比，和冻结 CLIP 比
- 视觉序列长度、文本是否插入

主张成立时：A1 的 CLS-patch 熵不应随训练单边塌到只看少数 patch；EOT 对 class name 的占比应高于 A0 中 EOT 对插入 context 的占比。

### 7.2 几何诊断（novel 有没有被拖走）

每 5 个 epoch：

- 当前文本特征与 step-0 冻结 CLIP 文本特征的平均余弦
- base 类文本中心与 new 类文本中心的余弦（把当前模型在 new 类名上跑一遍，不训练）
- 视觉 CLS 与冻结 CLIP CLS 的平均余弦

novel 被拖走的典型模式：base 文本越来越远、new 文本跟着 base 走、二者余弦升高。A1 应减缓这个过程。

### 7.3 训练轨迹

- 每个 epoch 的 train CE、distill、base acc
- 每 5 个 epoch 在 new 上做一次 eval-only（或至少用当前 classnames 扩到 new 类名算一个代理 acc）
- 画出 new acc 对 epoch 的斜率，不要只报最后 HM

关键图：base 还在涨时，A1 的 new 下降应浅于 A0。这是接口主张的直接证据。

---

## 8. 怎样算通过 / 失败

在 Stanford Cars、seed=1、50 epoch 上判定。对照是本仓库当前 HiCroPL，不是完整双向流动版，也不是别的仓库的 AblatedHiCroPL clean。

通过（可以继续 FGVC / DTD 和 A2-A7）

- A0 / A1 都没有 mapper / LKP
- A1 视觉序列长度恒为 197，无可学习 prompt token
- A1 的 novel 从峰值到最终的落差比 A0 小至少 1 个百分点，或最终 new acc 比 A0 高至少 0.5 个百分点
- A1 的 base 相对 A0 跌幅不超过 2 个百分点
- 7.1 的接口诊断支持“不再抢 softmax 槽位”

弱通过

- 诊断改善，但 new acc 只持平。可以跑 A3/A4：若加大 k 不会明显伤害 new，而 A0 加大 n_ctx 会伤害 new，主张仍可成立，继续 FGVC / DTD。

失败（停，不要加几何损失去救）

- 序列长度仍是 197+16，说明还在插入
- class name 被调制
- 又把 mapper 加回来了
- 只有 HM 涨，new 斜率没有变浅，base 却涨了
- base 崩掉超过 5 个点
- k=64 明显差于 k=16，且诊断重新恶化，说明容量一加又开始抢槽位，接口没有改干净

失败后优先查：VisionTransformer 是否仍 cat visual_ctx、文本是否仍 construct_prompts、attn_mask 是否覆盖了因果 mask、零初始化是否被覆盖、optimizer 是否误更新了 token_embedding。不要立刻把 k 调到 64，也不要并入 Slow/Fast。

---

## 9. 结果怎么记

每个 run 在 output_dir 留下：

- log.txt（必须含 TokenMod switches 那一行）
- diagnostics/unit_checks.json
- diagnostics/token_mod.csv
- 最终 base / new / HM

在仓库根目录追加 TOKEN_MODULATION_PROMPT_RESULTS_ZH.md（没有结果就不要先建空文件）。表格至少含：

```text
exp_id, dataset, seed, backbone, k, seq_len, t2i, i2t, residual, distill, base, new, hm, new_drop_from_peak, notes
```

backbone 一列写 HiCroPL-ablate-mapper。不要把完整双向流动 HiCroPL、Six-View 或 Slow/Fast 的数字写进对照表。对照只有 A0 当前插入版和本方法。

---

## 10. 新窗口工作顺序

1. 只读本文件、trainers/hicropl.py、clip/model.py 的 ResidualAttentionBlock_HiCroPL / VisionTransformer_HiCroPL。不要去翻几何实验文档，也不要把 mapper 加回来。
2. 改 clip/model.py 增加 TokenMod 分支；从当前 hicropl.py 复制 trainer，删掉插入和 CrossPromptAttention 实例化，加 code 和 bias generator。
3. 改 train.py 注册 TOKENMOD。
4. 跑 1-batch 单元检查，尤其是 seq_len=197、无 prompt token、无 mapper。
5. Stanford Cars seed=1：A0 用 vit_b16_c2_ep50_batch32_16ctx；A1 用 vit_b16_ep50_k16。DTD 旧结果不能代替 Cars。
6. 写诊断，按第 8 节判定。
7. 通过后再跑 A2、A5、A6 和 FGVC / DTD。
8. 不要开始实现 Slow/Fast Prompt。
