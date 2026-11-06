#!/usr/bin/env python
"""Run miHoYo's GameScaling-0.8B-S3 on the Chrome T-Rex Runner used by DinoRL.

- The game environment is the one from DinoRL/dinorl.ipynb (Selenium + headless
  Firefox, local asset server), loaded from the notebook's own cells.
- The model is served locally with transformers (vLLM is not available on macOS).
  The prompt, multi-turn history and action parser are the official
  THUSI-Lab/GameScaling client's.
- One model call = one 200 ms chunk = 6 groups x 33 ms = 12 game frames
  (60 Hz). Each group is executed as 2 game frames; `Up`/`Space` in a group
  presses jump at the start of that group.

Usage:
  python gamescaling_dino.py --smoke              # timing check, 3 decisions
  python gamescaling_dino.py --episodes 3         # full evaluation
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

# Project layout (all paths relative to this file's directory):
#   GameScalingDino/
#     gamescaling_dino.py  collect_expert.py  sft_lora.py
#     model/                 <- GameScaling checkpoint files (download, see README)
#     DinoRL/                <- github.com/anaerovane/DinoRL  (game + expert checkpoint)
#     GameScaling-client/    <- huggingface.co/THUSI-Lab/GameScaling (client code)
#     results/               <- outputs (GIFs, summaries, adapters, data)
def _base() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


BASE = Path(os.environ.get("GSD_BASE", str(_base())))
DINORL = Path(os.environ.get("GSD_DINORL", str(BASE / "DinoRL")))
MODEL_DIR = os.environ.get("GSD_MODEL_DIR", str(BASE / "model"))
CLIENT_DIR = os.environ.get("GSD_CLIENT_DIR", str(BASE / "GameScaling-client"))
RESULTS = Path(os.environ.get("GSD_RESULTS", str(BASE / "results")))

os.chdir(DINORL)
sys.path.insert(0, CLIENT_DIR)

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor
try:
    from transformers import StoppingCriteria, StoppingCriteriaList
except ImportError:
    from transformers.generation.stopping_criteria import (
        StoppingCriteria,
        StoppingCriteriaList,
    )

from gamescaling import PolicyConfig, USER_TURN_TEXT, build_system_prompt, encode_image
from gamescaling.action import ACTION_END, parse_action, split_reasoning
from gamescaling.policy import assistant_turn, finalize, user_turn

# ---------------------------------------------------------------- environment
def load_game_env():
    """Exec the setup + LocalGitHubGame cells out of the DinoRL notebook."""
    nb = json.loads((DINORL / "dinorl.ipynb").read_text())

    def find(pred):
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            src = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
            if pred(src):
                return src
        raise SystemExit("notebook cell not found")

    setup = find(lambda s: "class LocalGameHandler" in s)
    env = find(lambda s: "class LocalGitHubGame" in s)
    ns: dict = {}
    exec(compile(setup, "dinorl_setup_cell", "exec"), ns)
    exec(compile(env, "dinorl_env_cell", "exec"), ns)
    # The notebook pins torch to one thread for its tiny CNN; undo that for the VLM.
    torch.set_num_threads(os.cpu_count() or 4)
    return ns


# ------------------------------------------------------------------ the model
class StopOnToken(StoppingCriteria):
    def __init__(self, token_id: int):
        self.token_id = token_id

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return bool(input_ids[0][-1] == self.token_id)


class LocalGameScalingPolicy:
    """transformers stand-in for GameScalingPolicy (same messages, same parser)."""

    def __init__(self, system_prompt: str, cfg: PolicyConfig, device: str = "cpu"):
        self.system_prompt = system_prompt
        self.cfg = cfg
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(MODEL_DIR)
        self.model = AutoModelForMultimodalLM.from_pretrained(MODEL_DIR, dtype="auto")
        if device != "cpu":
            self.model = self.model.to(device)
        self.model.eval()
        self._history: deque = deque()
        end_id = self.processor.tokenizer.convert_tokens_to_ids(ACTION_END)
        self._stop = StoppingCriteriaList([StopOnToken(end_id)])
        self.prompt_tokens = 0
        self.gen_tokens = 0

    def reset(self) -> None:
        self._history.clear()

    def messages(self, image_url: str) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": self.system_prompt}]
        n = len(self._history)
        for i, (url, reasoning, action_text) in enumerate(self._history):
            out.append(user_turn(url))
            age = n - i
            keep = self.cfg.history_think < 0 or age <= self.cfg.history_think
            out.append(assistant_turn(action_text, reasoning if keep else ""))
        out.append(user_turn(image_url))
        out.append({"role": "assistant", "content": self.cfg.prefill})
        return out

    @torch.inference_mode()
    def act(self, image: Image.Image):
        url = encode_image(image, self.cfg)
        inputs = self.processor.apply_chat_template(
            self.messages(url), tokenize=True, return_dict=True, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[-1]
        t0 = time.perf_counter()
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_tokens,
            do_sample=self.cfg.temperature > 0,
            temperature=max(self.cfg.temperature, 1e-4),
            stopping_criteria=self._stop,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        new_ids = out[0][prompt_len:]
        self.prompt_tokens += prompt_len
        self.gen_tokens += int(new_ids.shape[0])
        finished = bool(new_ids.shape[0]) and bool(new_ids[-1] == self._stop[0].token_id)
        text = self.processor.tokenizer.decode(new_ids, skip_special_tokens=False)
        text = finalize(text, "stop" if finished else "length", self.cfg)
        action = parse_action(text, lenient=self.cfg.lenient_parse)
        reasoning, _, closed = split_reasoning(text)
        self._history.append((url, reasoning if closed else "", action.history_text))
        while len(self._history) > self.cfg.history_len:
            self._history.popleft()
        return text, action, latency_ms


# --------------------------------------------------------------- system prompt
def build_dino_prompt() -> str:
    rules = (
        "You are playing the Chrome T-Rex Runner, the offline dinosaur game. The dinosaur "
        "runs automatically from left to right at a constant speed; it cannot stop or "
        "turn.\n"
        "## What you see on screen\n"
        "- The dinosaur stands on the left of the screen and is on the ground unless it "
        "is jumping.\n"
        "- Obstacles enter from the right: single or clustered cacti of different heights "
        "on the ground, and pterodactyls flying at low or high altitude.\n"
        "- The distance score is shown in the top-right corner.\n"
        "## Rules\n"
        "- The run ends the moment the dinosaur touches an obstacle; otherwise the score "
        "grows with distance.\n"
        "- Jump over ground cacti: press the jump key shortly before the dinosaur reaches "
        "the obstacle, and land before the next obstacle of a cluster.\n"
        "- Pterodactyls fly at two heights: jump over the low ones, and keep running "
        "under the high ones without jumping.\n"
        "- Jumping too early makes you land on the obstacle; jump only when it is close.\n"
        "- The speed is constant; ignore the score text and react only to the obstacles."
    )
    return build_system_prompt(
        "Chrome T-Rex Runner",
        rules=rules,
        key_bindings=[("Up", "Jump"), ("Space", "Jump (keyboard alternative)")],
        mouse_aim=False,
    )


JUMP_KEYS = {"Up", "Space"}


# ------------------------------------------------------------------- the loop
def run_episode(policy, game, seed, max_decisions, max_seconds,
                record_every=6, record_limit=400):
    policy.reset()
    game.reset(seed)
    capture = [game.capture()]
    capture_frames = [0]
    started = time.monotonic()
    decisions = 0
    game_frames = 0
    total_reward = 0.0
    score = 0.0
    done = False
    latencies = []
    unparsed = 0
    last_text = ""

    while (not done and decisions < max_decisions
           and time.monotonic() - started < max_seconds):
        frame = game.capture()
        text, action, ms = policy.act(frame)
        latencies.append(ms)
        last_text = text
        if not action.parsed:
            unparsed += 1
        for group in action.steps:
            jump = 1 if JUMP_KEYS.intersection(group) else 0
            _, reward, d, score, _, adv = game.step(jump)
            total_reward += reward
            game_frames += adv
            done = d
            if done:
                break
        decisions += 1
        if not done and decisions % record_every == 0 and len(capture) < record_limit:
            capture.append(game.capture())
            capture_frames.append(game_frames)
        if decisions % 50 == 0:
            print(f"    decision {decisions}: score={score:.0f} frames={game_frames} "
                  f"last_act={ms:.0f}ms", flush=True)

    terminal = ("collision" if done else
                ("frame_limit" if game_frames >= 20000 else
                 ("decision_limit" if decisions >= max_decisions else "time_limit")))
    if not done:
        capture.append(game.capture())
        capture_frames.append(game_frames)
    result = dict(
        seed=seed, score=float(score), game_frames=game_frames, decisions=decisions,
        total_reward=round(total_reward, 2), terminal=terminal,
        wall_s=round(time.monotonic() - started, 1),
        mean_act_ms=round(float(np.mean(latencies)), 1) if latencies else None,
        max_act_ms=round(float(np.max(latencies)), 1) if latencies else None,
        unparsed=unparsed, last_action_text=last_text[-300:],
    )
    return result, capture, capture_frames


def save_gif(images, capture_frames, path: Path):
    if len(images) < 2:
        return
    # Each recorded frame covers (frames delta)/60 seconds of game time.
    step_frames = [b - a for a, b in zip(capture_frames, capture_frames[1:])]
    durations = [max(20, round(f / 60 * 1000)) for f in step_frames] + [500]
    images[0].save(path, format="GIF", save_all=True, append_images=images[1:],
                   duration=durations, loop=0, optimize=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--max-decisions", type=int, default=200)
    ap.add_argument("--max-seconds", type=float, default=1800.0)
    ap.add_argument("--device",
                    default="mps" if torch.backends.mps.is_available() else "cpu",
                    choices=["cpu", "mps"])
    ap.add_argument("--history", type=int, default=5)
    ap.add_argument("--reasoning", action="store_true",
                    help="let the model think (default: empty think, act directly)")
    ap.add_argument("--adapter", default=None,
                    help="optional LoRA adapter dir (PEFT) to mount on the model")
    ap.add_argument("--tag", default="gamescaling", help="prefix for output files")
    ap.add_argument("--smoke", action="store_true", help="3 decisions, no GIF")
    ap.add_argument("--seed-offset", type=int, default=10000,
                    help="episode i uses DinoRL's eval seed SEED+offset+i")
    args = ap.parse_args()

    env = load_game_env()
    LocalGitHubGame = env["LocalGitHubGame"]
    SEED = env["SEED"]

    cfg = PolicyConfig(history_len=args.history, reasoning=args.reasoning,
                       max_tokens=512)
    print(f"loading GameScaling-0.8B-S3 (device={args.device}, "
          f"history={args.history}, reasoning={args.reasoning}"
          + (f", adapter={args.adapter}" if args.adapter else "") + ") ...", flush=True)
    policy = LocalGameScalingPolicy(build_dino_prompt(), cfg, device=args.device)
    if args.adapter:
        from peft import PeftModel
        policy.model = PeftModel.from_pretrained(policy.model, args.adapter)
        policy.model.eval()

    if args.smoke:
        game = LocalGitHubGame(SEED, action_repeat=2)
        game.reset(SEED)
        t0 = time.monotonic()
        for i in range(3):
            text, action, ms = policy.act(game.capture())
            print(f"smoke {i}: {ms:.0f}ms parsed={action.parsed} "
                  f"steps={action.steps} text={text!r}", flush=True)
            for group in action.steps:
                game.step(1 if JUMP_KEYS.intersection(group) else 0)
        print(f"smoke total {time.monotonic()-t0:.1f}s", flush=True)
        game.close()
        if env.get("asset_server") is not None:
            env["asset_server"].shutdown(); env["asset_server"].server_close()
        return

    RESULTS.mkdir(exist_ok=True)
    summary = []
    for ep in range(args.episodes):
        seed = SEED + args.seed_offset + ep
        print(f"=== episode {ep+1}/{args.episodes} seed={seed} ===", flush=True)
        game = LocalGitHubGame(SEED, action_repeat=2)
        try:
            result, images, img_frames = run_episode(
                policy, game, seed, args.max_decisions, args.max_seconds)
        finally:
            game.close()
        gif = RESULTS / f"{args.tag}_ep{ep+1}_score{int(result['score'])}.gif"
        save_gif(images, img_frames, gif)
        result["gif"] = str(gif)
        summary.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "last_action_text"},
                         ensure_ascii=False), flush=True)
        (RESULTS / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"    -> {gif}", flush=True)

    scores = [r["score"] for r in summary]
    agg = dict(episodes=len(scores), mean=float(np.mean(scores)), min=min(scores),
               max=max(scores), scores=scores,
               mean_act_ms=float(np.mean([r["mean_act_ms"] for r in summary
                                          if r["mean_act_ms"]])),
               unparsed=sum(r["unparsed"] for r in summary))
    print("AGGREGATE:", json.dumps(agg), flush=True)
    (RESULTS / "summary.json").write_text(
        json.dumps({"aggregate": agg, "episodes": summary}, indent=2,
                   ensure_ascii=False))
    if env.get("asset_server") is not None:
        env["asset_server"].shutdown(); env["asset_server"].server_close()


if __name__ == "__main__":
    main()
