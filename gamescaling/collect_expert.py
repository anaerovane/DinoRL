#!/usr/bin/env python
"""Collect expert demonstrations: the DinoRL DQN (2995-score checkpoint) plays
the Chrome T-Rex Runner; we record (frame, 200ms key-chunk) pairs aligned to
GameScaling's action format at 12-frame (200 ms) boundaries, to later SFT the
GameScaling model via LoRA."""

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path


# Portable layout, see gamescaling_dino.py docstring.
def _base() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


BASE = Path(os.environ.get("GSD_BASE", str(_base())))
DINORL = Path(os.environ.get("GSD_DINORL", str(BASE / "DinoRL")))
OUT = Path(os.environ.get("GSD_RESULTS", str(BASE / "results"))) / "expert_data.jsonl"

os.chdir(DINORL)

import torch  # noqa: E402


def load_dinorl_env():
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
    classes = find(lambda s: "class GitHubDinoNet" in s)
    env = find(lambda s: "class LocalGitHubGame" in s)
    ns: dict = {}
    exec(compile(setup, "setup_cell", "exec"), ns)
    exec(compile(classes, "classes_cell", "exec"), ns)
    exec(compile(env, "env_cell", "exec"), ns)
    return ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed-offset", type=int, default=30000)
    ap.add_argument("--max-frames", type=int, default=4000)
    args = ap.parse_args()

    env = load_dinorl_env()
    LocalGitHubGame = env["LocalGitHubGame"]
    GitHubDinoNet = env["GitHubDinoNet"]
    stack_frames = env["stack_frames"]
    SEED = env["SEED"]
    GAME_DIR_SEED = SEED

    net = GitHubDinoNet()
    saved = torch.load(io.BytesIO((DINORL / "dino_model.pt").read_bytes()),
                       map_location="cpu", weights_only=True)
    net.load_state_dict(saved["model"])
    net.eval().requires_grad_(False)
    print("expert checkpoint loaded:", saved.get("episodes", "?"), "episodes", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    t0 = time.monotonic()
    with OUT.open("w") as f:
        for ep in range(args.episodes):
            seed = GAME_DIR_SEED + args.seed_offset + ep
            game = LocalGitHubGame(GAME_DIR_SEED, action_repeat=4)
            observation = game.reset(seed)
            frames = [observation.copy() for _ in range(4)]
            done = False
            game_frames = 0
            pending = None  # frame_captured_at_boundary awaiting its 3 decisions
            n_ep = 0
            while not done and game_frames < args.max_frames:
                boundary = game_frames % 12 == 0
                if boundary:
                    if pending is not None:
                        pending["groups"] = ";".join(pending["groups"])
                        f.write(json.dumps(pending) + "\n")
                        n_ep += 1
                        total += 1
                    # frame the expert currently sees (canvas snapshot)
                    buf = io.BytesIO()
                    game.capture().save(buf, format="JPEG", quality=90)
                    pending = {"seed": seed, "img": base64.b64encode(buf.getvalue()).decode(),
                               "groups": ["", "", "", "", "", ""]}
                from collections import deque
                stack = stack_frames(deque(frames, maxlen=4))
                with torch.inference_mode():
                    a = int(net(torch.from_numpy(stack).float().unsqueeze(0)).argmax(1).item())
                obs, _, done, score, _, adv = game.step(a)
                frames.append(obs)
                game_frames += adv
                if pending is not None:
                    g = game_frames // 2  # finished groups after this decision
                    for gi in [g - 2, g - 1]:
                        if 0 <= gi < 6:
                            pending["groups"][gi] = "Up" if a == 1 else ""
            game.close()
            print(f"episode {ep} seed={seed}: frames={game_frames} score={score:.0f} "
                  f"samples={n_ep} done={done}", flush=True)
    print(f"total samples: {total} -> {OUT} ({time.monotonic()-t0:.0f}s)", flush=True)

    if env.get("asset_server") is not None:
        env["asset_server"].shutdown()
        env["asset_server"].server_close()


if __name__ == "__main__":
    main()
