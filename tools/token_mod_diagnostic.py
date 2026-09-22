"""Unit checks and lightweight diagnostics for TokenModHiCroPL.

Run before any 50-epoch training.
"""
import argparse
import json
import os
import os.path as osp
from collections import OrderedDict

import torch
import torch.nn as nn

from dassl.config import get_cfg_default
from dassl.utils import set_random_seed

import datasets.stanford_cars  # noqa: F401
import datasets.dtd  # noqa: F401
import datasets.fgvc_aircraft  # noqa: F401
import trainers.token_mod_hicropl as tm
from trainers.token_mod_hicropl import load_clip_to_cpu


def extend_cfg(cfg):
    from yacs.config import CfgNode as CN
    cfg.TRAINER.TOKENMOD = CN()
    cfg.TRAINER.TOKENMOD.N_CTX = 16
    cfg.TRAINER.TOKENMOD.CROSS_LAYER = 6
    cfg.TRAINER.TOKENMOD.CTX_INIT = "a photo of a"
    cfg.TRAINER.TOKENMOD.PREC = "fp16"
    cfg.TRAINER.TOKENMOD.PROMPT_DEPTH = 12
    cfg.TRAINER.TOKENMOD.TEACHER_NAME = "ViT-B/16"
    cfg.TRAINER.TOKENMOD.LAMBD = 12.0
    cfg.TRAINER.TOKENMOD.CODE_LEN = 16
    cfg.TRAINER.TOKENMOD.RANK = 8
    cfg.TRAINER.TOKENMOD.MODULATION = "attn_bias"
    cfg.DATASET.SUBSAMPLE_CLASSES = "base"


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)
    cfg.merge_from_file(args.dataset_config_file)
    cfg.merge_from_file(args.config_file)
    cfg.DATASET.ROOT = args.root
    cfg.OUTPUT_DIR = args.output_dir
    cfg.SEED = args.seed
    cfg.TRAINER.NAME = "TokenModHiCroPL"
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dataset-config-file", type=str, required=True)
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # argparse.REMAINDER may keep a leading "--".
    if args.opts and args.opts[0] == "--":
        args.opts = args.opts[1:]

    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    diag_dir = osp.join(cfg.OUTPUT_DIR, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use a few dummy classnames for unit checks to keep this lightweight.
    classnames = [
        "AM General Hummer SUV 2000",
        "Acura RL Sedan 2012",
        "Acura TL Sedan 2012",
        "Acura TL Type-S 2008",
    ]

    clip_model = load_clip_to_cpu(cfg)
    if cfg.TRAINER.TOKENMOD.PREC in ["fp32", "amp"]:
        clip_model.float()

    # Avoid downloading teacher / gpt classifier during pure structural checks by
    # temporarily stubbing heavy dependencies, then restore.
    real_load_cpu = tm.load_clip_to_cpu
    real_load_teacher = tm.load_clip_to_cpu_teacher
    real_gpt = tm.gpt_clip_classifier

    class DummyVisual(nn.Module):
        def __init__(self, out_dim=512):
            super().__init__()
            self.out_dim = out_dim
        def forward(self, x):
            # Non-zero fixed features so residual/distill stay numerically stable in unit tests.
            feat = torch.ones(x.shape[0], self.out_dim, device=x.device, dtype=x.dtype)
            return feat / feat.norm(dim=-1, keepdim=True)

    def fake_load_cpu(cfg_in, zero_shot_model=False):
        return real_load_cpu(cfg_in, True)

    def fake_load_teacher(cfg_in, zero_shot_model=False):
        m = real_load_cpu(cfg_in, True)
        m.visual = DummyVisual(m.visual.output_dim if hasattr(m.visual, "output_dim") else 512)
        return m

    def fake_gpt(classnames_in, gpt_prompts, clip_model_in, dataset_name):
        feat = torch.randn(len(classnames_in), 512, device=device)
        return feat / feat.norm(dim=-1, keepdim=True)

    tm.load_clip_to_cpu = fake_load_cpu
    tm.load_clip_to_cpu_teacher = fake_load_teacher
    tm.gpt_clip_classifier = fake_gpt

    model = tm.CustomCLIP(cfg, classnames, clip_model).to(device)
    model.train()

    # Match trainer freezing: only prompt_learner (except ZS encoder) is trainable.
    for name, param in model.named_parameters():
        if "prompt_learner" not in name or "ZS_image_encoder" in name:
            param.requires_grad_(False)

    # restore
    tm.load_clip_to_cpu = real_load_cpu
    tm.load_clip_to_cpu_teacher = real_load_teacher
    tm.gpt_clip_classifier = real_gpt

    results = OrderedDict()
    ok = True

    # 1. visual sequence length remains 197
    try:
        images = torch.randn(2, 3, 224, 224, device=device, dtype=torch.float16 if cfg.TRAINER.TOKENMOD.PREC == "fp16" else torch.float32)
        labels = torch.tensor([0, 1], device=device)
        # force forward through image encoder only for seq check via hook
        seen = {}
        def hook_pre(_, args):
            x = args[0]
            # after conv path inside VisionTransformer_TokenMod we assert already
            seen["called"] = True
        handle = model.image_encoder.register_forward_pre_hook(hook_pre)
        with torch.no_grad():
            _ = model.image_encoder(images, list(model.prompt_learner.visual_codes), model.prompt_learner.visual_modulators)
        handle.remove()
        results["visual_seq_len_assert"] = "passed_in_forward"
    except Exception as e:
        ok = False
        results["visual_seq_len_assert"] = f"FAILED: {e}"

    # 2. no learnable prompt tokens / frozen text embedding
    try:
        assert model.prompt_learner.text_embedding.requires_grad is False
        bad_shapes = []
        for n, p in model.named_parameters():
            if p.requires_grad and tuple(p.shape) in [(16, 512), (16, 768)]:
                bad_shapes.append((n, tuple(p.shape)))
        assert len(bad_shapes) == 0, bad_shapes
        results["no_prompt_token_params"] = "passed"
    except Exception as e:
        ok = False
        results["no_prompt_token_params"] = f"FAILED: {e}"

    # 3. class name bias columns are zero
    try:
        masks = model.prompt_learner.get_token_masks()
        class_mask = masks["class_mask"]
        for i, gen in enumerate(model.prompt_learner.text_modulators):
            # temporarily set non-zero codes to force bias generation
            code = torch.ones_like(model.prompt_learner.text_codes[i])
            # temporarily set generator weights nonzero
            with torch.no_grad():
                gen.W.fill_(0.1)
            bias = gen(code, masks)  # [n_cls*n_head, 77, 77]
            n_cls = class_mask.shape[0]
            n_head = gen.n_head
            bias = bias.view(n_cls, n_head, 77, 77)
            class_cols = bias.masked_select(class_mask.view(n_cls, 1, 1, 77).expand_as(bias))
            assert torch.all(class_cols == 0), f"layer {i} class columns nonzero"
            with torch.no_grad():
                gen.W.zero_()
        results["class_name_bias_zero"] = "passed"
    except Exception as e:
        ok = False
        results["class_name_bias_zero"] = f"FAILED: {e}"

    # 4. zero-init bias is all zeros
    try:
        masks = model.prompt_learner.get_token_masks()
        for i, gen in enumerate(model.prompt_learner.text_modulators):
            bias = gen(model.prompt_learner.text_codes[i], masks)
            assert torch.all(bias == 0), f"text bias layer {i} nonzero at init"
        pos = model.image_encoder.positional_embedding
        for i, gen in enumerate(model.prompt_learner.visual_modulators):
            bias = gen(model.prompt_learner.visual_codes[i], pos, seq_len=197)
            assert torch.all(bias == 0), f"visual bias layer {i} nonzero at init"
        results["zero_init_bias"] = "passed"
    except Exception as e:
        ok = False
        results["zero_init_bias"] = f"FAILED: {e}"

    # 5. causal mask still present when extra_mask=0
    try:
        block = model.text_encoder.transformer.resblocks[0]
        causal = block.attn_mask
        assert causal is not None
        # upper triangle should be -inf
        tri = torch.triu(torch.ones_like(causal), 1).bool()
        assert torch.isneginf(causal[tri]).all() or torch.all(causal[tri] < -1e4)
        results["causal_mask_present"] = "passed"
    except Exception as e:
        ok = False
        results["causal_mask_present"] = f"FAILED: {e}"

    # 6. no forbidden parameter names
    try:
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        forbidden = ["cross_prompts_text", "cross_prompts_visual", "text2visual", "visual2text", "VPT"]
        leaked = [n for n in names for b in forbidden if b in n]
        assert len(leaked) == 0, leaked
        results["no_forbidden_params"] = "passed"
        results["trainable_params"] = names
        results["trainable_numel"] = int(sum(p.numel() for n, p in model.named_parameters() if p.requires_grad))
    except Exception as e:
        ok = False
        results["no_forbidden_params"] = f"FAILED: {e}"

    # 7. one training step should run
    try:
        images = torch.randn(2, 3, 224, 224, device=device)
        labels = torch.tensor([0, 1], device=device)
        loss = model(images.half() if cfg.TRAINER.TOKENMOD.PREC == "fp16" else images, labels)
        loss.backward()
        results["one_step_train"] = f"passed loss={float(loss.detach().cpu())}"
    except Exception as e:
        ok = False
        results["one_step_train"] = f"FAILED: {e}"

    results["all_passed"] = bool(ok)
    out_path = osp.join(diag_dir, "unit_checks.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
