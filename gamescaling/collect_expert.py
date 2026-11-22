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


JS_FEATS = r"""
  window.__dinoFeats=()=>{
    const rr=Runner.instance_,d=rr.tRex,gy=d.groundYPos||103;
    const arr=rr.horizon.obstacles.map(o=>({
      d:o.xPos-(d.xPos+d.config.WIDTH), w:o.width, top:o.yPos,
      h:o.typeConfig.height, ty:o.typeConfig.type
    })).filter(o=>o.d>-60).sort((a,b)=>a.d-b.d).slice(0,3);
    const slot=(o)=>{
      if(!o)return [1,0,0.5,0,0,0,0];
      return [Math.max(-1,Math.min(1,o.d/300)), o.w/80, o.top/150,
        o.ty==='CACTUS_SMALL'?1:0, o.ty==='CACTUS_LARGE'?1:0, o.ty==='PTERODACTYL'?1:0, o.h/150];
    };
    const base=[(gy-d.yPos)/150, d.jumpVelocity/20, d.jumping?1:0, d.ducking?1:0, (rr.currentSpeed-6)/7];
    return base.concat(slot(arr[0]),slot(arr[1]),slot(arr[2]));
  };
"""


class RulePolicy:
    """Cluster-aware jump heuristic over 26-dim engine features (score ~2995)."""

    def __init__(self, k_single=12, k_cluster=7, cluster_gap=60, horizon=32):
        self.k_single = k_single
        self.k_cluster = k_cluster
        self.cluster_gap = cluster_gap
        self.horizon = horizon

    def act(self, feat):
        if feat[2] > 0.5:
            return 0
        speed = float(feat[4]) * 7.0 + 6.0
        upcoming = []
        for i in range(3):
            s = feat[5 + 7 * i: 5 + 7 * (i + 1)]
            dist = float(s[0]) * 300.0
            if dist < -60:
                continue
            upcoming.append((dist, float(s[1]) * 80.0))
        if not upcoming:
            return 0
        d0, w0 = upcoming[0]
        if d0 > speed * self.horizon:
            return 0
        span_end = d0 + w0
        for d, w in upcoming[1:]:
            if d - span_end < self.cluster_gap:
                span_end = d + w
            else:
                break
        span = span_end - d0
        k = self.k_single if span <= 45 else self.k_cluster
        return 1 if d0 < speed * k else 0


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
    ns["FIREFOX_BINARY"] = os.environ.get("FIREFOX_BIN", "/root/tools/firefox/firefox")
    return ns


def load_expert(kind: str):
    env = load_dinorl_env()
    if kind == "rule":
        print("expert: rule heuristic (engine features)", flush=True)
        return env, RulePolicy(), None
    env["driver"] = None  # placeholder; feats injected after game reset
    net = env["GitHubDinoNet"]()
    saved = torch.load(io.BytesIO((DINORL / "dino_model.pt").read_bytes()),
                       map_location="cpu", weights_only=True)
    net.load_state_dict(saved["model"])
    net.eval().requires_grad_(False)
    print("expert checkpoint loaded:", saved.get("episodes", "?"), "episodes", flush=True)
    return env, net, env["stack_frames"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed-offset", type=int, default=30000)
    ap.add_argument("--max-frames", type=int, default=4000)
    ap.add_argument("--expert", choices=["rule", "dqn"], default="rule",
                    help="rule = near-optimal 2995 heuristic; dqn = vision checkpoint")
    args = ap.parse_args()

    env, expert, stack_frames = load_expert(args.expert)
    LocalGitHubGame = env["LocalGitHubGame"]
    SEED = env["SEED"]
    GAME_DIR_SEED = SEED
    rule = expert if args.expert == "rule" else None
    net = expert if args.expert == "dqn" else None

    OUT.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    t0 = time.monotonic()
    with OUT.open("w") as f:
        for ep in range(args.episodes):
            seed = GAME_DIR_SEED + args.seed_offset + ep
            game = LocalGitHubGame(GAME_DIR_SEED, action_repeat=4)
            observation = game.reset(seed)
            # Inject engine features for the rule expert.
            game.driver.execute_script(JS_FEATS)
            frames = [observation.copy() for _ in range(4)]
            done = False
            game_frames = 0
            pending = None
            n_ep = 0
            score = 0.0
            while not done and game_frames < args.max_frames:
                boundary = game_frames % 12 == 0
                if boundary:
                    if pending is not None:
                        pending["groups"] = ";".join(pending["groups"])
                        f.write(json.dumps(pending) + "\n")
                        n_ep += 1
                        total += 1
                    buf = io.BytesIO()
                    game.capture().save(buf, format="JPEG", quality=90)
                    pending = {"seed": seed, "img": base64.b64encode(buf.getvalue()).decode(),
                               "groups": ["", "", "", "", "", ""]}
                if rule is not None:
                    feat = game.driver.execute_script("return window.__dinoFeats()")
                    a = rule.act(feat)
                    obs = None
                else:
                    from collections import deque
                    stack = stack_frames(deque(frames, maxlen=4))
                    with torch.inference_mode():
                        a = int(net(torch.from_numpy(stack).float().unsqueeze(0)).argmax(1).item())
                    obs = None
                obs, _, done, score, _, adv = game.step(a)
                if obs is not None:
                    frames.append(obs)
                game_frames += adv
                if pending is not None:
                    g = ((game_frames % 12) or 12) // 2
                    for gi in [g - 2, g - 1]:
                        if 0 <= gi < 6:
                            pending["groups"][gi] = "Up" if a == 1 else ""
            game.close()
            print(f"episode {ep} seed={seed}: frames={game_frames} score={score:.0f} "
                  f"samples={n_ep} done={done} expert={args.expert}", flush=True)
    print(f"total samples: {total} -> {OUT} ({time.monotonic()-t0:.0f}s)", flush=True)

    if env.get("asset_server") is not None:
        env["asset_server"].shutdown()
        env["asset_server"].server_close()


if __name__ == "__main__":
    main()
