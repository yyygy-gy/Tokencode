import json
import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

CoPrompt_dataset_name_mapping = {
    "Caltech101": "caltech",
    "DescribableTextures": "dtd",
    "EuroSAT": "eurosat",
    "FGVCAircraft": "fgvc",
    "Food101": "food101",
    "ImageNet": "imagenet",
    "ImageNetA": "imagenet_a",
    "ImageNetR": "imagenet_r",
    "ImageNetSketch": "imagenet_sketch",
    "ImageNetV2": "imagenetv2",
    "OxfordFlowers": "oxford_flowers",
    "OxfordPets": "oxford_pets",
    "StanfordCars": "stanford_cars",
    "SUN397": "sun397",
    "UCF101": "ucf101",
}


def load_clip_to_cpu_teacher(cfg, zero_shot_model=False):
    backbone_name = cfg.TRAINER.TOKENMOD.TEACHER_NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    print(f"CLIP Teacher name is {backbone_name}")
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {
        "trainer": "IVLP",
        "vision_depth": 0,
        "language_depth": 0,
        "vision_ctx": 0,
        "language_ctx": 0,
    }
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model


def load_clip_to_cpu(cfg, zero_shot_model=False):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    if not zero_shot_model:
        design_details = {
            "trainer": "TokenModHiCroPL",
            "vision_depth": cfg.TRAINER.TOKENMOD.PROMPT_DEPTH,
            "language_depth": cfg.TRAINER.TOKENMOD.PROMPT_DEPTH,
            "vision_ctx": 0,
            "language_ctx": 0,
            "code_len": cfg.TRAINER.TOKENMOD.CODE_LEN,
            "modulation": cfg.TRAINER.TOKENMOD.MODULATION,
        }
        model = clip.build_model(state_dict or model.state_dict(), design_details)
    else:
        design_details = {
            "trainer": "IVLP",
            "vision_depth": 0,
            "language_depth": 0,
            "vision_ctx": 0,
            "language_ctx": 0,
        }
        model = clip.build_model(state_dict or model.state_dict(), design_details)
        return model
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, text_codes, text_modulators, token_masks):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        outputs = self.transformer([x, text_codes, text_modulators, token_masks])
        x = outputs[0]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class VisualBiasGenerator(nn.Module):
    def __init__(
        self, code_len, n_head, rank, vis_dim, layer_id, cross_layer, dtype=torch.float16, bias_scale=1.0
    ):
        super().__init__()
        self.n_head = n_head
        self.rank = rank
        self.layer_id = layer_id
        self.cross_layer = cross_layer
        self.mode = "attn_bias"
        self.bias_scale = float(bias_scale)
        hidden = max(code_len, n_head * rank)
        # Two-layer MLP gives the short code enough capacity to form spatial attention patterns.
        self.mlp = nn.Sequential(
            nn.Linear(code_len, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, n_head + n_head * rank, bias=True),
        )
        self.W_pos = nn.Parameter(torch.zeros(vis_dim, rank, dtype=dtype))
        self._init_dtype = dtype

    def forward(self, code, pos_embed, n_head=None, seq_len=197):
        dtype = pos_embed.dtype
        code = code.to(dtype=dtype, device=pos_embed.device)
        # Keep modulator math in fp32 for stable MLP, then cast back.
        h = self.mlp(code.float()).to(dtype=dtype)
        a = torch.tanh(h[: self.n_head])
        b = torch.tanh(h[self.n_head :].view(self.n_head, self.rank))
        patch_pos = pos_embed[1:seq_len].to(dtype=dtype)
        s = patch_pos @ self.W_pos.to(dtype=dtype)
        patch_bias = torch.einsum("nr,hr->hn", s, b)
        # Additive form avoids a dead zone when both a and b start at 0.
        patch_bias = patch_bias + a.unsqueeze(-1)
        patch_bias = self.bias_scale * patch_bias
        bias = torch.zeros(self.n_head, seq_len, seq_len, dtype=dtype, device=pos_embed.device)
        # Bidirectional CLS-centered modulation: CLS reads patches and patches read CLS.
        bias[:, 0, 1:] = patch_bias
        bias[:, 1:, 0] = patch_bias
        return bias


class TextBiasGenerator(nn.Module):
    def __init__(self, code_len, n_head, layer_id, cross_layer, dtype=torch.float16, bias_scale=1.0):
        super().__init__()
        self.n_head = n_head
        self.layer_id = layer_id
        self.cross_layer = cross_layer
        self.is_shallow = layer_id < cross_layer
        self.mode = "attn_bias"
        self.bias_scale = float(bias_scale)
        hidden = max(code_len, n_head * 2)
        self.mlp = nn.Sequential(
            nn.Linear(code_len, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, n_head, bias=True),
        )
        self._init_dtype = dtype

    def forward(self, code, token_masks, n_head=None):
        template_mask = token_masks["template_mask"]
        class_mask = token_masks["class_mask"]
        padding_mask = token_masks["padding_mask"]
        eot_idx = token_masks["eot_idx"]
        n_cls, seq_len = template_mask.shape
        dtype = self._init_dtype
        if torch.is_floating_point(code):
            dtype = code.dtype
        device = template_mask.device
        code = code.to(dtype=dtype, device=device)
        head_bias = self.bias_scale * torch.tanh(self.mlp(code.float()).to(dtype=dtype))

        bias = torch.zeros(n_cls, self.n_head, seq_len, seq_len, dtype=dtype, device=device)
        if self.is_shallow:
            key_mask = template_mask.to(dtype=dtype)
            bias[:, :, 0, :] = head_bias.view(1, self.n_head, 1) * key_mask.unsqueeze(1)
            # Allow template tokens to feed back into SOS.
            bias[:, :, :, 0] = bias[:, :, :, 0] + head_bias.view(1, self.n_head, 1) * key_mask.unsqueeze(1)
        else:
            # Deep layers modulate EOT as query. Class-name columns must stay zero
            # (unit check / no class-name modulation). Use template keys instead.
            key_mask = template_mask.to(dtype=dtype)
            rows = torch.zeros(n_cls, seq_len, dtype=dtype, device=device)
            rows.scatter_(1, eot_idx.view(n_cls, 1), 1.0)
            eot_from_template = (
                head_bias.view(1, self.n_head, 1, 1)
                * rows.view(n_cls, 1, seq_len, 1)
                * key_mask.view(n_cls, 1, 1, seq_len)
            )
            # Also allow template tokens to attend back to EOT.
            template_to_eot = (
                head_bias.view(1, self.n_head, 1, 1)
                * key_mask.view(n_cls, 1, seq_len, 1)
                * rows.view(n_cls, 1, 1, seq_len)
            )
            bias = eot_from_template + template_to_eot

        # Hard guarantee: never put bias on class-name or padding keys.
        bias = bias.masked_fill(class_mask.view(n_cls, 1, 1, seq_len), 0.0)
        bias = bias.masked_fill(padding_mask.view(n_cls, 1, 1, seq_len), 0.0)
        return bias.reshape(n_cls * self.n_head, seq_len, seq_len)


class VisualFiLMGenerator(nn.Module):
    def __init__(self, code_len, width, layer_id, cross_layer, dtype=torch.float16):
        super().__init__()
        self.layer_id = layer_id
        self.cross_layer = cross_layer
        self.is_shallow = layer_id < cross_layer
        self.mode = "film"
        self.W_gamma = nn.Parameter(torch.zeros(width, code_len, dtype=dtype))
        self.W_beta = nn.Parameter(torch.zeros(width, code_len, dtype=dtype))

    def forward(self, code, x, token_masks=None, n_head=None, seq_len=None):
        dtype = x.dtype
        code = code.to(dtype=dtype, device=x.device)
        if code.dim() == 1:
            gamma = self.W_gamma @ code
            beta = self.W_beta @ code
        elif code.dim() == 2:
            # code: [B, code_len] -> gamma/beta: [B, width]
            gamma = code @ self.W_gamma.t()
            beta = code @ self.W_beta.t()
        else:
            raise ValueError(f"Unexpected visual FiLM code shape: {tuple(code.shape)}")
        if self.is_shallow:
            patches = (1.0 + gamma) * x[1:] + beta
            y = torch.cat([x[:1], patches], dim=0)
        else:
            cls = (1.0 + gamma) * x[0] + beta
            cls = cls.unsqueeze(0)
            y = torch.cat([cls, x[1:]], dim=0)
        return y


class TextFiLMGenerator(nn.Module):
    def __init__(self, code_len, width, layer_id, cross_layer, dtype=torch.float16):
        super().__init__()
        self.layer_id = layer_id
        self.cross_layer = cross_layer
        self.is_shallow = layer_id < cross_layer
        self.mode = "film"
        self.W_gamma = nn.Parameter(torch.zeros(width, code_len, dtype=dtype))
        self.W_beta = nn.Parameter(torch.zeros(width, code_len, dtype=dtype))

    def forward(self, code, x, token_masks=None, n_head=None, seq_len=None):
        dtype = x.dtype
        code = code.to(dtype=dtype, device=x.device)
        if code.dim() == 1:
            gamma = self.W_gamma @ code
            beta = self.W_beta @ code
        elif code.dim() == 2:
            # code: [N, code_len] -> gamma/beta: [N, width]
            gamma = code @ self.W_gamma.t()
            beta = code @ self.W_beta.t()
        else:
            raise ValueError(f"Unexpected text FiLM code shape: {tuple(code.shape)}")
        if self.is_shallow:
            mask = token_masks["template_mask"].t().unsqueeze(-1).to(dtype=dtype)
            if gamma.dim() == 1:
                y = x * (1.0 + mask * gamma) + mask * beta
            else:
                gamma_b = gamma.unsqueeze(0)
                beta_b = beta.unsqueeze(0)
                y = x * (1.0 + mask * gamma_b) + mask * beta_b
        else:
            eot_idx = token_masks["eot_idx"]
            # Scatter onto a fresh tensor so we never inplace-write a view used by autograd.
            y = x.clone()
            for i in range(x.shape[1]):
                eot = int(eot_idx[i].item())
                if gamma.dim() == 1:
                    y[eot, i] = (1.0 + gamma) * x[eot, i] + beta
                else:
                    y[eot, i] = (1.0 + gamma[i]) * x[eot, i] + beta[i]
        return y


class ConditionalCodeGenerator(nn.Module):
    """Map frozen CLIP features to per-layer short codes."""

    def __init__(self, in_dim, code_len, depth, dtype=torch.float16):
        super().__init__()
        self.depth = depth
        self.code_len = code_len
        self.projs = nn.ParameterList(
            [nn.Parameter(torch.empty(code_len, in_dim, dtype=dtype)) for _ in range(depth)]
        )
        self.biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(code_len, dtype=dtype)) for _ in range(depth)]
        )
        for w in self.projs:
            nn.init.normal_(w, std=0.02)

    def forward(self, features):
        # features: [N, in_dim]
        feat = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        codes = []
        for w, b in zip(self.projs, self.biases):
            code = F.linear(feat, w, b)
            codes.append(code)
        return codes


class TokenModPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        assert cfg.TRAINER.TOKENMOD.PROMPT_DEPTH >= 1
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        v_dim = clip_model.visual.positional_embedding.shape[-1]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize

        self.prompt_depth = cfg.TRAINER.TOKENMOD.PROMPT_DEPTH
        self.cross_layer = cfg.TRAINER.TOKENMOD.CROSS_LAYER
        self.code_len = cfg.TRAINER.TOKENMOD.CODE_LEN
        self.rank = cfg.TRAINER.TOKENMOD.RANK
        self.modulation = cfg.TRAINER.TOKENMOD.MODULATION
        self.n_ctx = cfg.TRAINER.TOKENMOD.N_CTX
        self.bias_scale = float(cfg.TRAINER.TOKENMOD.BIAS_SCALE)
        self.use_residual = bool(cfg.TRAINER.TOKENMOD.USE_RESIDUAL)
        self.use_distill = bool(cfg.TRAINER.TOKENMOD.USE_DISTILL)

        prompt_prefix = cfg.TRAINER.TOKENMOD.CTX_INIT.replace("_", " ")
        print("TokenMod design: short-code structured modulation without token insertion")
        print(
            f"TokenMod switches: insert_tokens=False, t2i=False, i2t=False, residual={self.use_residual}, "
            f"distill={self.use_distill}, modulation={self.modulation}, k={self.code_len}, r={self.rank}, "
            f"bias_scale={self.bias_scale}"
        )
        print("visual_seq_len=197, text_seq_len=77")
        print(f'Frozen text template: "{prompt_prefix} <class>."')

        # Conditional codes are opt-in. Default film path keeps global codes
        # so unconditional FiLM checkpoints remain loadable.
        self.use_conditional_codes = bool(cfg.TRAINER.TOKENMOD.USE_CONDITIONAL_CODES)
        if self.use_conditional_codes and self.modulation != "film":
            raise ValueError("USE_CONDITIONAL_CODES=True currently requires MODULATION=film")
        print(f"TokenMod code mode: conditional={self.use_conditional_codes}")
        if self.use_conditional_codes:
            self.text_codes = None
            self.visual_codes = None
        else:
            self.text_codes = nn.ParameterList(
                [nn.Parameter(torch.empty(self.code_len, dtype=dtype)) for _ in range(self.prompt_depth)]
            )
            self.visual_codes = nn.ParameterList(
                [nn.Parameter(torch.empty(self.code_len, dtype=dtype)) for _ in range(self.prompt_depth)]
            )
            for p in self.text_codes:
                nn.init.normal_(p, std=0.02)
            for p in self.visual_codes:
                nn.init.normal_(p, std=0.02)

        n_head_text = clip_model.transformer.resblocks[0].attn.num_heads
        n_head_vis = clip_model.visual.transformer.resblocks[0].attn.num_heads

        if self.modulation == "attn_bias":
            self.text_modulators = nn.ModuleList(
                [
                    TextBiasGenerator(
                        self.code_len,
                        n_head_text,
                        i,
                        self.cross_layer,
                        dtype=dtype,
                        bias_scale=self.bias_scale,
                    )
                    for i in range(self.prompt_depth)
                ]
            )
            self.visual_modulators = nn.ModuleList(
                [
                    VisualBiasGenerator(
                        self.code_len,
                        n_head_vis,
                        self.rank,
                        v_dim,
                        i,
                        self.cross_layer,
                        dtype=dtype,
                        bias_scale=self.bias_scale,
                    )
                    for i in range(self.prompt_depth)
                ]
            )
            # Keep bias near zero at step 0, but give positional projector a
            # non-zero start so visual low-rank path is not identically dead.
            for mod in self.visual_modulators:
                for layer in mod.mlp:
                    if isinstance(layer, nn.Linear):
                        nn.init.normal_(layer.weight, std=0.02)
                        nn.init.zeros_(layer.bias)
                nn.init.normal_(mod.W_pos, std=0.02)
            for mod in self.text_modulators:
                for layer in mod.mlp:
                    if isinstance(layer, nn.Linear):
                        nn.init.normal_(layer.weight, std=0.02)
                        nn.init.zeros_(layer.bias)
        elif self.modulation == "film":
            self.text_modulators = nn.ModuleList(
                [
                    TextFiLMGenerator(self.code_len, ctx_dim, i, self.cross_layer, dtype=dtype)
                    for i in range(self.prompt_depth)
                ]
            )
            self.visual_modulators = nn.ModuleList(
                [
                    VisualFiLMGenerator(self.code_len, v_dim, i, self.cross_layer, dtype=dtype)
                    for i in range(self.prompt_depth)
                ]
            )
            for mod in list(self.text_modulators) + list(self.visual_modulators):
                nn.init.normal_(mod.W_gamma, std=0.02)
                nn.init.normal_(mod.W_beta, std=0.02)
        else:
            raise ValueError(f"Unsupported TOKENMOD.MODULATION={self.modulation}")

        clip_model_temp = load_clip_to_cpu(cfg, True).float().cuda()
        clip_model_temp_image = load_clip_to_cpu_teacher(cfg, True)
        with torch.no_grad():
            self.ZS_image_encoder = clip_model_temp_image.visual
        for p in self.ZS_image_encoder.parameters():
            p.requires_grad_(False)
        with open(f"gpt_file/{CoPrompt_dataset_name_mapping[cfg.DATASET.NAME]}_prompt.json") as f:
            gpt3_prompt = json.load(f)
        print("Getting textual features as CLIP's classifier.")
        clip_weights = gpt_clip_classifier(classnames, gpt3_prompt, clip_model_temp, cfg.DATASET.NAME)
        self.register_buffer("fixed_embeddings", clip_weights)

        if self.use_conditional_codes:
            feat_dim = int(self.fixed_embeddings.shape[-1])
            self.text_code_generator = ConditionalCodeGenerator(
                feat_dim, self.code_len, self.prompt_depth, dtype=dtype
            )
            self.visual_code_generator = ConditionalCodeGenerator(
                feat_dim, self.code_len, self.prompt_depth, dtype=dtype
            )
            print(
                "TokenMod conditional codes enabled: text<-frozen class embeddings, "
                "visual<-frozen image features"
            )

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("text_embedding", embedding)
        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.n_cls = n_cls
        self.name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        self._build_token_masks(tokenized_prompts, prompt_prefix)

        n_train = sum(
            p.numel()
            for n, p in self.named_parameters()
            if p.requires_grad and "ZS_image_encoder" not in n
        )
        print(f"TokenMod trainable parameters in prompt_learner: {n_train}")

    def _build_token_masks(self, tokenized_prompts, prompt_prefix):
        n_cls, seq_len = tokenized_prompts.shape
        eot_idx = tokenized_prompts.argmax(dim=-1)
        tmpl_tokens = clip.tokenize(prompt_prefix)[0]
        tmpl_eot = int(tmpl_tokens.argmax().item())
        l_tmp = tmpl_eot - 1

        template_mask = torch.zeros(n_cls, seq_len, dtype=torch.bool)
        class_mask = torch.zeros(n_cls, seq_len, dtype=torch.bool)
        padding_mask = torch.zeros(n_cls, seq_len, dtype=torch.bool)
        for i in range(n_cls):
            eot = int(eot_idx[i].item())
            template_mask[i, 1:1 + l_tmp] = True
            class_mask[i, 1 + l_tmp:eot] = True
            if eot + 1 < seq_len:
                padding_mask[i, eot + 1:] = True

        self.register_buffer("sos_idx", torch.zeros(n_cls, dtype=torch.long))
        self.register_buffer("eot_idx", eot_idx)
        self.register_buffer("template_mask", template_mask)
        self.register_buffer("class_mask", class_mask)
        self.register_buffer("padding_mask", padding_mask)

    def get_token_masks(self):
        return {
            "sos_idx": self.sos_idx,
            "eot_idx": self.eot_idx,
            "template_mask": self.template_mask,
            "class_mask": self.class_mask,
            "padding_mask": self.padding_mask,
        }

    def forward(self):
        return (
            self.text_embedding,
            None if self.use_conditional_codes else list(self.text_codes),
            None if self.use_conditional_codes else list(self.visual_codes),
            self.text_modulators,
            self.visual_modulators,
            self.get_token_masks(),
        )


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = TokenModPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.lambd = cfg.TRAINER.TOKENMOD.LAMBD
        self.use_residual = bool(cfg.TRAINER.TOKENMOD.USE_RESIDUAL)
        self.use_distill = bool(cfg.TRAINER.TOKENMOD.USE_DISTILL)
        self.use_residual_gate = bool(cfg.TRAINER.TOKENMOD.RESIDUAL_GATE)
        gate_init = float(cfg.TRAINER.TOKENMOD.RESIDUAL_GATE_INIT)
        # Store logit(alpha) so alpha starts near gate_init and stays in (0, 1).
        gate_init = min(max(gate_init, 1e-4), 1.0 - 1e-4)
        logit = torch.log(torch.tensor(gate_init / (1.0 - gate_init)))
        self.residual_gate_logit = nn.Parameter(logit.to(dtype=self.dtype))
        if not self.use_residual_gate:
            self.residual_gate_logit.requires_grad_(False)

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features_fixed = None
        need_fixed_image = (
            self.use_residual
            or (self.training and self.use_distill)
            or self.prompt_learner.use_conditional_codes
        )
        if need_fixed_image:
            with torch.no_grad():
                image_features_fixed = self.prompt_learner.ZS_image_encoder(image.type(self.dtype))
                image_features_fixed = image_features_fixed / image_features_fixed.norm(dim=-1, keepdim=True)

        text_emb, text_codes, visual_codes, text_mods, visual_mods, masks = self.prompt_learner()
        if self.prompt_learner.use_conditional_codes:
            # Class-conditional text codes from frozen CLIP class embeddings.
            text_codes = self.prompt_learner.text_code_generator(
                self.prompt_learner.fixed_embeddings.type(self.dtype)
            )
            # Sample-conditional visual codes from frozen CLIP image features.
            visual_codes = self.prompt_learner.visual_code_generator(
                image_features_fixed.type(self.dtype)
            )
        text_features = self.text_encoder(text_emb, tokenized_prompts, text_codes, text_mods, masks)
        image_features = self.image_encoder(image.type(self.dtype), visual_codes, visual_mods)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        if self.use_residual:
            text_features_fixed = self.prompt_learner.fixed_embeddings.half()
            if self.use_residual_gate:
                alpha = torch.sigmoid(self.residual_gate_logit).to(dtype=image_features.dtype)
                image_features = image_features_fixed + alpha * image_features
                text_features = text_features_fixed + alpha * text_features
            else:
                image_features = image_features + image_features_fixed
                text_features = text_features + text_features_fixed
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = logit_scale * image_features @ text_features.t()
        if self.prompt_learner.training:
            loss_cls = F.cross_entropy(logits, label)
            if self.use_distill:
                text_features_fixed = self.prompt_learner.fixed_embeddings
                cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
                loss_distill_text = 1.0 - torch.mean(cos(text_features, text_features_fixed))
                loss_distill_image = 1.0 - torch.mean(cos(image_features, image_features_fixed))
                return loss_cls + self.lambd * (loss_distill_text + loss_distill_image)
            return loss_cls
        return logits


def gpt_clip_classifier(classnames, gpt_prompts, clip_model, dataset_name):
    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            classname = classname.replace("_", " ")
            texts = [t for t in gpt_prompts[classname]]
            texts = clip.tokenize(texts)
            if torch.cuda.is_available():
                clip_model = clip_model.cuda()
                texts = texts.cuda()
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings /= class_embeddings.norm()
            clip_weights.append(class_embeddings)
        clip_weights = torch.stack(clip_weights, dim=0)
        if torch.cuda.is_available():
            clip_weights = clip_weights.cuda()
    return clip_weights


@TRAINER_REGISTRY.register()
class TokenModHiCroPL(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.TOKENMOD.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.TOKENMOD.PREC == "fp32" or cfg.TRAINER.TOKENMOD.PREC == "amp":
            clip_model.float()

        print("Building custom CLIP (TokenModHiCroPL)")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"
        for name, param in self.model.named_parameters():
            if name_to_update not in name and "residual_gate_logit" not in name:
                param.requires_grad_(False)
            else:
                if "ZS_image_encoder" in name:
                    param.requires_grad_(False)

        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        forbidden = ["cross_prompts_text", "cross_prompts_visual", "text2visual", "visual2text", "VPT"]
        for name in enabled:
            for bad in forbidden:
                assert bad not in name, f"Unexpected trainable parameter leaked into TokenMod: {name}"

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.TOKENMOD.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        model = self.model
        optim = self.optim
        scaler = self.scaler
        prec = self.cfg.TRAINER.TOKENMOD.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()
        loss_summary = {"loss": loss.item()}
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"].to(self.device)
        label = batch["label"].to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return
        names = self.get_model_names()
        model_file = "model-best.pth.tar"
        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)
        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))
            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            for key in [
                "prompt_learner.token_prefix",
                "prompt_learner.token_suffix",
                "prompt_learner.text_embedding",
                "prompt_learner.tokenized_prompts",
                "prompt_learner.sos_idx",
                "prompt_learner.eot_idx",
                "prompt_learner.template_mask",
                "prompt_learner.class_mask",
                "prompt_learner.padding_mask",
            ]:
                if key in state_dict:
                    del state_dict[key]
            # Class-specific frozen CLIP classifiers must stay with the current
            # evaluation split. Loading base embeddings into a novel model would
            # silently score novel images against base class names.
            for key in list(state_dict.keys()):
                if key.endswith("fixed_embeddings") or key.endswith("ZS_image_encoder"):
                    del state_dict[key]
            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)
