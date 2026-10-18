# DinoRL In-Memory Notebook

A compact, reproducible Chrome Dino reinforcement-learning demo. The notebook runs the bundled offline game locally, loads the included PyTorch checkpoint, evaluates the policy, and renders a GIF in the notebook output.

## Quick start

1. Install Python 3.10+ and Firefox.
2. Install the Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Open `dinorl_in_memory.ipynb` from this repository root in Jupyter and run the cells from top to bottom.

Selenium Manager locates or downloads the matching geckodriver when one is not already on PATH. The first browser launch therefore needs internet access if no driver is cached.

## Files

- `dinorl_in_memory.ipynb`: game wrapper, DQN implementation, optional in-memory training, checkpoint evaluation, and GIF demonstration.
- `dino_model.pt`: pretrained PyTorch checkpoint used by default.
- `t-rex-runner-gh-pages/`: only the local game page, controller script, and two offline sprite sheets required by the notebook.
- `requirements.txt`: Python dependencies.

Training is disabled by default. Set `RUN_TRAINING = True` in the first code cell to run the 120-second experiment. Training updates remain in memory and do not overwrite or save to `dino_model.pt`; evaluation and the demo load the supplied checkpoint.

The included T-Rex Runner files retain their original BSD 3-Clause license in `t-rex-runner-gh-pages/LICENSE`.
