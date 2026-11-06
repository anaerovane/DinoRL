# GameScaling ⨉ DinoRL

Zero-shot transfer and fine-tuning experiments for **miHoYo's open-source GameScaling
game VLM** (THUSI-Lab/GameScaling, arXiv 2610.28698) played inside the **Chrome T-Rex
Runner environment from [anaerovane/DinoRL](https://github.com/anaerovane/DinoRL)**,
on Apple-Silicon Macs (MPS) — no vLLM / CUDA server required.

| File | What it does |
|---|---|
| `gamescaling_dino.py` | Runs a GameScaling checkpoint as the **policy** in the DinoRL dino game: renders the canvas, sends the official prompt + 5-frame history via the official client's prompt/action parser served by local `transformers` inference, executes each 200 ms chunk (6×33 ms) as 12 game frames, evaluates episodes and records GIFs. `--adapter` mounts an SFT/LoRA adapter. |
| `collect_expert.py` | **Expert demonstrations for SFT**: the DinoRL DQN checkpoint (≈2995 score) plays the game; frames are captured and labelled at 12-frame (200 ms) boundaries with keys in the official action format. |
| `sft_lora.py` | **LoRA SFT** (behaviour cloning) of GameScaling-0.8B-S3 on the expert data; loss on the action tokens only, gradient checkpointing + MPS bf16 so it trains in a few GB of RAM. |

## Results so far (M1, 8 GB, Nov 2026, eval seeds shared with the DQN baseline)

| Policy | Mean score | Notes |
|---|---|---|
| DinoRL DQN (specialised, same seeds) | **2995** (all 5 episodes hit the frame cap) | ms-level decisions |
| GameScaling-0.8B-S3 zero-shot | **42.5** | crashes at the first obstacle, never jumps; 100 % parse rate, ≈6.4 s/decision (MPS bf16) |
| GameScaling-0.8B-S3 + LoRA SFT (behaviour cloning from the DQN) | see `results/` after running | WIP |

The near-perfect zero-shot/untrained gap is expected: the game is far out of
distribution for the checkpoint, and 0.8B is the smallest scale.

## Setup

```bash
conda create -y -n dinorl python=3.12 pip
conda activate dinorl
pip install -r requirements.txt

# game + environment wrapper (notebook) - NOT redistributed here
git clone https://github.com/anaerovane/DinoRL.git DinoRL

# inference client (prompts, action parser, keymaps) - NOT redistributed here
git clone --depth 1 https://huggingface.co/THUSI-Lab/GameScaling GameScaling-client

# model weights (~2.2 GB bf16) - download into ./model
huggingface-cli download THUSI-Lab/GameScaling-0.8B-S3 --local-dir model

# browser stack (macOS; Selenium opens the game in Firefox)
brew install geckodriver
brew install --cask firefox        # or point FIREFOX_BINARY at another build
# geckodriver must be on PATH when running the scripts
```

## Run

```bash
export PATH="$(dirname "$(which geckodriver)"):$PATH"   # if needed

# 1) zero-shot policy evaluation (writes results/*.gif + results/summary.json)
python gamescaling_dino.py --episodes 2

# 2) collect expert demonstrations from the DQN checkpoint (dino_model.pt ships with DinoRL)
#    note: DinoRL's authors hold all rights until 2026-12-01; use for personal experiments.
python collect_expert.py --episodes 8 --seed-offset 40000 --max-frames 20000

# 3) LoRA SFT on the expert data (~1-2 h on M1 MPS for one epoch)
python sft_lora.py --epochs 1

# 4) evaluate the fine-tuned policy
python gamescaling_dino.py --episodes 2 --adapter results/sft_lora --tag sft
```

### Layout / configuration

```
GameScalingDino/
  model/              GameScaling checkpoint files (see download command above)
  DinoRL/             cloned DinoRL repo (game assets, notebook, expert checkpoint)
  GameScaling-client/ cloned THUSI-Lab/GameScaling client repo
  results/            GIFs, summaries, expert_data.jsonl, LoRA adapters
```

`GSD_BASE`, `GSD_DINORL`, `GSD_MODEL_DIR`, `GSD_CLIENT_DIR`, `GSD_RESULTS` override
each path; useful flags: `gamescaling_dino.py --device/--history/--max-decisions/
--adapter/--tag`, `collect_expert.py --episodes/--seed-offset/--max-frames`,
`sft_lora.py --epochs/--per-seed/--limit`.

## Licences

- This repo's code, the GameScaling model and its client: **Apache-2.0**.
- DinoRL's code and the bundled T-Rex Runner assets are **not** included here: DinoRL is
  copyright its author until 2026-12-01 (MIT planned after), game assets keep their
  BSD-3-Clause. Clone it yourself per the instructions above.
- Chrome T-Rex Runner assets belong to Google.
