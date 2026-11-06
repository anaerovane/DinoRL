#!/usr/bin/env python
"""LoRA SFT of GameScaling-0.8B-S3 on expert demonstrations of the Chrome
T-Rex Runner (collected by collect_expert.py from the DinoRL DQN expert).

Training sample = system prompt + k history user/assistant turns (k<=5,
expert actions in the official action format) + current user turn + assistant:
prefill with the empty-think block, completion = <|action_start|>...<|action_end|>.
Loss is computed on the completion only (everything up to and including the
last ``":-say after the prefill is masked).

Runs on Apple-Silicon MPS with bf16; frozen backbone + LoRA keeps memory small.

  python sft_lora.py --smoke              # 3 steps, timing check
  python sft_lora.py                      # full run, saves LoRA adapter
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import random
import sys
import time
from pathlib import Path

# Portable layout, see gamescaling_dino.py docstring.
BASE = Path(os.environ.get("GSD_BASE",
                           str(Path(__file__).resolve().parent)))
DATA = Path(os.environ.get("GSD_RESULTS", str(BASE / "results"))) / "expert_data.jsonl"
ADAPTER_DIR = Path(os.environ.get("GSD_RESULTS", str(BASE / "results"))) / "sft_lora"

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from transformers import AutoModelForMultimodalLM, AutoProcessor  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "GameScaling-client"))  # for `gamescaling`
from gamescaling.action import ACTION_END, ACTION_START, THINK_OPEN  # noqa: E402
from gamescaling.policy import NO_THINK_PREFILL, assistant_turn, user_turn  # noqa: E402

from gamescaling_dino import build_dino_prompt  # noqa: E402


def action_text_from_groups(groups: list[str]) -> str:
    body = "0 0 0 ; " + " ; ".join(groups)
    return f"{ACTION_START}{body}{ACTION_END}"


def load_samples(data_path=DATA):
    rows = []
    with open(data_path) as f:
        for line in f:
            r = json.loads(line)
            r["groups_list"] = r["groups"].split(";")
            r["action_text"] = action_text_from_groups(r["groups_list"])
            r["has_jump"] = "Up" in r["groups"]
            rows.append(r)
    by_seed: dict[int, list] = {}
    for r in rows:
        by_seed.setdefault(r["seed"], []).append(r)
    for seed in by_seed:
        pass  # rows are stored in play order already
    return by_seed


def build_windows(by_seed: dict, per_seed: int, w_probs: dict, seed: int):
    rng = random.Random(seed)
    items = []
    for ep_seed, rows in by_seed.items():
        pool = []
        i = 0
        while i < len(rows):
            w = rng.choices(list(w_probs), list(w_probs.values()))[0]
            w = min(w, len(rows) - i)
            window = rows[i:i + w]
            pool.append(window)
            i += w
        rng.shuffle(pool)
        chosen = pool[:per_seed]
        for window in chosen:
            neg = [None] * 1
            items.append(window)
            if window[-1]["has_jump"]:
                items.append(window)          # extra copies of rare positives
                items.append(window)
                items.append(window)
    rng.shuffle(items)
    return items


def render_sample(processor, system_prompt, window):
    history = window[:-1]
    target = window[-1]
    messages = [{"role": "system", "content": system_prompt}]
    url_prefix = "data:image/jpeg;base64,"
    for row in history:
        messages.append(user_turn(url_prefix + row["img"]))
        messages.append(assistant_turn(row["action_text"]))
    messages.append(user_turn(url_prefix + target["img"]))
    messages.append({"role": "assistant",
                     "content": NO_THINK_PREFILL + target["action_text"]})
    inputs = processor.apply_chat_template(messages, tokenize=True, return_dict=True,
                                           return_tensors="pt")
    ids = inputs["input_ids"][0]
    tokz = processor.tokenizer
    tid = tokz.convert_tokens_to_ids(THINK_OPEN)
    seq_ids = [tid] if (tid is not None and isinstance(tid, int) and tid >= 0) \
        else tokz(THINK_OPEN, add_special_tokens=False).input_ids
    assert seq_ids and any(ids[i:i+len(seq_ids)] == torch.tensor(seq_ids)
                           for i in range(len(ids) - len(seq_ids) + 1))
    # find the LAST occurrence of the think-open token sequence
    for i in range(len(ids) - len(seq_ids), -1, -1):
        if ids[i:i+len(seq_ids)] == torch.tensor(seq_ids):
            last_think = i + len(seq_ids) - 1   # index of its final token
            break
    return inputs, last_think


def make_lora(model):
    from peft import LoraConfig, get_peft_model
    names = {n.split(".")[-1] for n, _ in model.named_modules()}
    for cand in [["q_proj", "k_proj", "v_proj", "o_proj"],
                 ["q_proj", "v_proj"],
                 ["query_key_value"]]:
        if all(c in names for c in cand):
            targets = cand
            break
    else:
        raise SystemExit(f"unknown target modules; sample names={sorted(list(names))[:40]}")
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM", target_modules=targets)
    peft_model = get_peft_model(model, lcfg)
    peft_model.print_trainable_parameters()
    return peft_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--per-seed", type=int, default=250)
    ap.add_argument("--max-len", type=int, default=2400)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N micro-batches (smoke test)")
    ap.add_argument("--out", default=str(ADAPTER_DIR))
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16

    print("loading processor / model ...", flush=True)
    model_dir = os.environ.get("GSD_MODEL_DIR", str(BASE / "model"))
    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForMultimodalLM.from_pretrained(model_dir, dtype=dtype).to(device)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.eval()   # LoRA params still get grads via adapters
    lora = make_lora(model)
    lora.to(device)

    by_seed = load_samples(args.data)
    n_rows = sum(len(v) for v in by_seed.values())
    n_pos = sum(r["has_jump"] for v in by_seed.values() for r in v)
    print(f"data: {n_rows} rows across {len(by_seed)} seeds, {n_pos} jump rows", flush=True)
    items = build_windows(by_seed, args.per_seed,
                          {1: 0.4, 2: 0.25, 3: 0.15, 4: 0.1, 6: 0.1}, seed=args.seed)
    print(f"windows: {len(items)}", flush=True)
    items = items

    opt = torch.optim.AdamW([p for p in lora.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(items) / args.accum)
    total_steps = int(steps_per_epoch * args.epochs)
    warmup = 20
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warmup, 1.0) * 0.5 * (1 + math.cos(
            math.pi * min(max(s - 0, 0), total_steps) / max(total_steps - 1, 1))))

    print(f"steps/epoch={steps_per_epoch} total_steps={total_steps}", flush=True)

    t_last = time.monotonic()
    losses = []
    step = 0
    micro = 0
    t_start = time.monotonic()
    done = False
    epoch = 0
    while not done:
        epoch += 1
        for w in items:
            inputs, last_think = render_sample(processor, build_dino_prompt(), w)
            L = inputs["input_ids"].shape[-1]
            if L > args.max_len:
                continue
            inputs = inputs.to(device)
            take = L - (last_think + 1)          # number of completion target tokens
            if take <= 0:
                continue
            # Only the completion (after the last think-open token) needs logits.
            out = lora(**inputs, logits_to_keep=take + 1)
            logits = out.logits[:, :-1, :].float()     # positions last_think .. L-2
            r2 = time.monotonic()
            targets = inputs["input_ids"][:, -take:]   # ids[last_think+1 .. L-1]
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   targets.reshape(-1))
            (loss / args.accum).backward()
            torch.nn.utils.clip_grad_norm_([p for p in lora.parameters() if p.requires_grad], 1.0)
            losses.append(float(loss.detach()) * args.accum)
            micro += 1
            if args.limit:
                print(f"micro {micro}/{args.limit} len={L} take={take} "
                      f"fwd+bwd={(time.monotonic()-r2)*1000:.0f}ms "
                      f"loss={losses[-1]:.4f}", flush=True)
                if micro >= args.limit:
                    done = True
                    break
            if micro % args.accum == 0:
                opt.step(); opt.zero_grad(set_to_none=True); sched.step()
                step += 1
                if step % 20 == 0:
                    rate = micro / (time.monotonic() - t_start + 1e-9)
                    eta_min = (len(items) * args.epochs - micro) / max(rate, 1e-9) / 60
                    print(f"step {step}/{total_steps} epoch {epoch} "
                          f"loss={sum(losses[-80:])/min(80,len(losses)):.4f} "
                          f"window/s={rate:.2f} eta={eta_min:.0f}min", flush=True)
                if step >= total_steps:
                    done = True
                    break
        if done:
            break

    loss_final = sum(losses) / max(1, len(losses))
    print(f"mean micro loss over run: {loss_final:.4f}", flush=True)
    if not args.limit:
        outdir = Path(args.out)
        outdir.mkdir(parents=True, exist_ok=True)
        lora.save_pretrained(outdir)
        print(f"adapter saved -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
