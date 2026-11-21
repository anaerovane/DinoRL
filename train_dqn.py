#!/usr/bin/env python3
"""Train GitHubDinoNet on the local T-Rex runner and log every episode."""
import base64, copy, io, json, os, random, shutil, sys, threading, time, traceback
from collections import deque
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ACTION_REPEAT = 4
GAMMA = 0.99
SEED = 20261015

TARGET_SCORE = 2000
MAX_EPISODES = 400
MAX_GAME_FRAMES = 20000
TIME_BUDGET_HOURS = float(os.environ.get('TRAIN_HOURS', '4'))

REPLAY_CAPACITY = 5000
BATCH_SIZE = 64
EARLY_CRASH_PENALTY = 5.0
STABILITY_SCORE = 1000
LR = 3e-5
EPS_START, EPS_END, EPS_DECAY_FRAMES = 0.20, 0.02, 1_000_000
TARGET_TAU = 0.001
GRAD_CLIP = 10.0

LOG_PATH = ROOT / 'results' / 'training_log.csv'
CKPT_DIR = ROOT / 'results' / 'checkpoints'
STATUS_PATH = ROOT / 'results' / 'training_status.txt'
BEST_PATH = ROOT / 'results' / 'dino_trained_best.pt'
FINAL_PATH = ROOT / 'results' / 'dino_trained_final.pt'

FIREFOX_BIN = os.environ.get('FIREFOX_BIN', '/root/tools/firefox/firefox')
GECKODRIVER = os.environ.get('GECKODRIVER', '/root/tools/geckodriver')
HEADLESS = os.environ.get('HEADLESS', '1') != '0'


def log_status(msg):
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
    print(line, flush=True)
    with open(STATUS_PATH, 'a') as f:
        f.write(line + '\n')


class GitHubDinoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4, padding=0)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=0)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=0)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64, 512)
        self.fc2 = nn.Linear(512, 2)

    @staticmethod
    def _keras_same(conv, x):
        h, w = x.shape[-2:]
        kh, kw = conv.kernel_size
        sh, sw = conv.stride
        oh, ow = (h + sh - 1) // sh, (w + sw - 1) // sw
        ph, pw = max((oh - 1) * sh + kh - h, 0), max((ow - 1) * sw + kw - w, 0)
        x = F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))
        return conv(x)

    def forward(self, x):
        x = F.relu(self.pool1(self._keras_same(self.conv1, x)))
        x = F.relu(self.pool2(self._keras_same(self.conv2, x)))
        x = F.relu(self.pool3(self._keras_same(self.conv3, x)))
        x = x.permute(0, 2, 3, 1).contiguous().flatten(1)
        return self.fc2(F.relu(self.fc1(x)))


class PrioritizedReplay:
    def __init__(self, cap=5000, alpha=0.6):
        self.items = []
        self.priorities = []
        self.cap = cap
        self.pos = 0
        self.alpha = alpha

    def add(self, item):
        p = max(self.priorities, default=1.0)
        if len(self.items) < self.cap:
            self.items.append(item)
            self.priorities.append(p)
        else:
            self.items[self.pos] = item
            self.priorities[self.pos] = p
            self.pos = (self.pos + 1) % self.cap

    def sample(self, n, beta):
        p = np.asarray(self.priorities, dtype=np.float64) ** self.alpha
        p /= p.sum()
        ids = np.random.choice(len(self.items), n, p=p)
        w = (len(self.items) * p[ids]) ** (-beta)
        w /= w.max()
        batch = [self.items[i] for i in ids]
        return batch, ids, w.astype(np.float32)

    def update(self, ids, errors):
        for i, e in zip(ids, errors):
            self.priorities[int(i)] = float(abs(e) + 1e-5)


def stack_frames(frames):
    return np.stack(list(frames)[::-1], axis=0)


GAME_DIR = (ROOT / 't-rex-runner-gh-pages').resolve()


class LocalGameHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(GAME_DIR), **kwargs)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def log_message(self, *args):
        pass


asset_server = None
asset_thread = None


def start_asset_server():
    global asset_server, asset_thread
    if asset_server is None:
        asset_server = ThreadingHTTPServer(('127.0.0.1', 0), LocalGameHandler)
        asset_thread = threading.Thread(target=asset_server.serve_forever, daemon=True)
        asset_thread.start()
    return f'http://127.0.0.1:{asset_server.server_port}/'


JS_SETUP = r"""
  const el=document.createElement('script');
  el.textContent='(()=>{let s='+String(arguments[0]>>>0)+'; Math.random=()=>{s=(1664525*s+1013904223)>>>0;return s/4294967296;};})();';
  document.head.appendChild(el);el.remove();
"""

JS_FREEZE = r"""
  const r=Runner.instance_;r.stop();Runner.config.ACCELERATION=0;r.config.ACCELERATION=0;
  let t=performance.now();
  Object.defineProperty(performance,'now',{configurable:true,value:()=>t});
  window.requestAnimationFrame=()=>1;window.cancelAnimationFrame=()=>{};
  const panel=document.querySelector('body > section');if(panel)panel.style.display='none';
  const message=document.getElementById('messageBox');if(message)message.style.display='none';
  let introStart=null;
  const off=document.createElement('canvas');off.width=80;off.height=80;
  const octx=off.getContext('2d',{willReadFrequently:true});
  octx.imageSmoothingEnabled=true;octx.imageSmoothingQuality='high';
  window.__dinoObs=()=>{
    const c=document.querySelector('canvas.runner-canvas');
    const w=Math.min(500,c.width),h=Math.min(300,c.height);
    octx.clearRect(0,0,80,80);
    octx.drawImage(c,0,0,w,h,0,0,80,80);
    const d=octx.getImageData(0,0,80,80).data;
    const out=new Uint8ClampedArray(6400);
    for(let i=0;i<6400;i++){const j=i<<2;out[i]=(d[j]*77+d[j+1]*151+d[j+2]*28)>>8;}
    let s='';for(let i=0;i<6400;i++)s+=String.fromCharCode(out[i]);
    return btoa(s);
  };
  window.__dinoAdvance=(n)=>{
    let advanced=0;
    for(let i=0;i<n && r.playing && !r.crashed;i++){
      if(r.playingIntro && introStart===null)introStart=t;
      if(r.playingIntro && t-introStart>=400){r.startGame();introStart=null;}
      t+=1000/60;r.update();advanced++;
      if(!r.playingIntro)introStart=null;
    }
    return {crashed:!!r.crashed,score:r.distanceMeter.getActualDistance(Math.ceil(r.distanceRan)),
      advanced,frame:window.__dinoObs()};
  };
  r.time=t;
  const e={keyCode:38,target:document.body,preventDefault(){}};
  r.onKeyDown(e);r.onKeyUp(e);
"""

JS_HISCORE = r"""const r=Runner.instance_;
  r.distanceMeter.setHighScore(363);
  r.distanceMeter.update(0,Math.ceil(r.distanceRan));"""

JS_JUMP = r"""const r=Runner.instance_;if(!r.tRex.jumping){
  const e={keyCode:38,target:document.body,preventDefault(){}};r.onKeyDown(e);r.onKeyUp(e);}"""


class LocalGitHubGame:
    def __init__(self):
        if not Path(GECKODRIVER).is_file():
            raise FileNotFoundError(f'geckodriver not found: {GECKODRIVER}')
        self.url = start_asset_server()
        options = Options()
        if HEADLESS:
            options.add_argument('-headless')
        if FIREFOX_BIN and Path(FIREFOX_BIN).is_file():
            options.binary_location = FIREFOX_BIN
        options.set_preference('media.volume_scale', '0.0')
        options.set_preference('dom.ipc.processCount', '1')
        service = Service(executable_path=GECKODRIVER, log_output=os.devnull)
        self.driver = webdriver.Firefox(service=service, options=options)
        self.driver.set_window_size(1000, 700)
        self._obs_cache = np.zeros((80, 80), dtype=np.uint8)

    def reset(self, seed):
        self.driver.get(self.url)
        deadline = time.monotonic() + 20
        while not self.driver.execute_script(
                'return !!(window.dinoControl && window.dinoControl.getState().ready)'):
            if time.monotonic() > deadline:
                raise RuntimeError('Game initialization timed out')
            time.sleep(0.05)
        self.driver.execute_script(JS_SETUP, int(seed))
        self.driver.execute_script(JS_FREEZE)
        self.driver.execute_script(JS_HISCORE)
        return self.observe()

    def observe(self):
        encoded = self.driver.execute_script('return window.__dinoObs()')
        return np.frombuffer(base64.b64decode(encoded), dtype=np.uint8).reshape(80, 80).copy()

    def step(self, action):
        if action == 1:
            self.driver.execute_script(JS_JUMP)
        result = self.driver.execute_script(
            'return window.__dinoAdvance(arguments[0])', ACTION_REPEAT)
        advanced = int(result['advanced'])
        if advanced == 0 and not result['crashed']:
            raise RuntimeError('Game stopped advancing')
        done = bool(result['crashed'])
        score = float(result['score'])
        reward = sum(GAMMA ** i * (-1.0 if done and i == advanced - 1 else 0.1)
                     for i in range(advanced))
        frame = np.frombuffer(base64.b64decode(result['frame']),
                              dtype=np.uint8).reshape(80, 80).copy()
        return frame, reward, done, score, GAMMA ** advanced, advanced

    def close(self):
        try:
            self.driver.quit()
        except Exception:
            pass


def append_log(row):
    new_file = not LOG_PATH.exists()
    with open(LOG_PATH, 'a') as f:
        if new_file:
            f.write(','.join(row.keys()) + '\n')
        f.write(','.join(str(row[k]) for k in row.keys()) + '\n')
        f.flush()
        os.fsync(f.fileno())


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / 'results').mkdir(exist_ok=True)
    log_status(f'start device={DEVICE} target_score={TARGET_SCORE} '
               f'max_episodes={MAX_EPISODES} budget_h={TIME_BUDGET_HOURS}')

    learner = GitHubDinoNet().to(DEVICE)
    target_net = copy.deepcopy(learner).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(learner.parameters(), lr=LR)
    replay = PrioritizedReplay(cap=REPLAY_CAPACITY)

    updates = 0
    total_frames = 0
    best_score = -1.0
    started = time.monotonic()
    deadline = started + TIME_BUDGET_HOURS * 3600
    game = None

    try:
        game = LocalGitHubGame()
        log_status('game ready')

        for episode in range(1, MAX_EPISODES + 1):
            if time.monotonic() >= deadline:
                log_status('time budget reached')
                break

            actor = copy.deepcopy(learner).eval().requires_grad_(False)
            epsilon = max(EPS_END, EPS_START - (EPS_START - EPS_END) * total_frames / EPS_DECAY_FRAMES)
            seed = SEED + episode
            observation = game.reset(seed)
            frames = deque([observation.copy() for _ in range(4)], maxlen=4)
            state = stack_frames(frames)
            losses = []
            game_frames = 0
            score = 0.0
            done = False
            ep_started = time.monotonic()

            while (not done and game_frames < MAX_GAME_FRAMES
                   and time.monotonic() < deadline):
                with torch.inference_mode():
                    greedy = int(actor(torch.from_numpy(state).float().unsqueeze(0).to(DEVICE))
                                 .argmax(1).item())
                action = random.randrange(2) if random.random() < epsilon else greedy
                image, reward, done, score, discount, advanced = game.step(action)
                frames.append(image)
                nxt = stack_frames(frames)
                if done and score < STABILITY_SCORE:
                    reward -= EARLY_CRASH_PENALTY * GAMMA ** max(0, advanced - 1)
                replay.add((state.copy(), action, reward, nxt.copy(), done, discount))
                state = nxt
                game_frames += advanced
                total_frames += advanced

                if len(replay.items) >= max(100, BATCH_SIZE):
                    beta = min(1.0, 0.4 + 0.6 * updates / 100_000)
                    batch, ids, weights = replay.sample(BATCH_SIZE, beta)
                    st = torch.from_numpy(np.stack([z[0] for z in batch])).float().to(DEVICE)
                    ns = torch.from_numpy(np.stack([z[3] for z in batch])).float().to(DEVICE)
                    actions = torch.tensor([z[1] for z in batch], dtype=torch.long, device=DEVICE)
                    rewards = torch.tensor([z[2] for z in batch], dtype=torch.float32, device=DEVICE)
                    terminals = torch.tensor([z[4] for z in batch], dtype=torch.float32, device=DEVICE)
                    discounts = torch.tensor([z[5] for z in batch], dtype=torch.float32, device=DEVICE)
                    q = learner(st).gather(1, actions[:, None]).squeeze(1)
                    with torch.no_grad():
                        next_actions = learner(ns).argmax(1)
                        expected = (rewards + discounts * (1 - terminals)
                                    * target_net(ns).gather(1, next_actions[:, None]).squeeze(1))
                    errors = (q - expected).detach().abs().cpu().numpy()
                    loss = (torch.from_numpy(weights).to(DEVICE)
                            * F.smooth_l1_loss(q, expected, reduction='none')).mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError('Non-finite training loss')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(learner.parameters(), GRAD_CLIP)
                    optimizer.step()
                    with torch.no_grad():
                        for tp, lp in zip(target_net.parameters(), learner.parameters()):
                            tp.lerp_(lp, TARGET_TAU)
                    replay.update(ids, errors)
                    updates += 1
                    losses.append(float(loss.detach()))

            terminal = ('collision' if done
                        else 'frame_limit' if game_frames >= MAX_GAME_FRAMES
                        else 'time_budget')
            mean_loss = float(np.mean(losses)) if losses else None
            row = dict(
                episode=episode, seed=seed, score=round(score, 2),
                frames=game_frames, total_frames=total_frames,
                epsilon=round(epsilon, 4),
                mean_loss=round(mean_loss, 6) if mean_loss is not None else '',
                updates=updates, terminal=terminal,
                episode_seconds=round(time.monotonic() - ep_started, 2),
                elapsed_seconds=round(time.monotonic() - started, 1),
            )
            append_log(row)

            is_best = terminal != 'time_budget' and score > best_score
            if is_best:
                best_score = score
                torch.save({'model': actor.state_dict(),
                            'score': score, 'episode': episode,
                            'action_repeat': ACTION_REPEAT,
                            'gamma': GAMMA}, BEST_PATH)

            if episode == 1 or episode % 5 == 0 or is_best or score >= TARGET_SCORE:
                log_status(
                    f'ep={episode} score={score:.1f} best={best_score:.1f} '
                    f'frames={game_frames} total={total_frames} eps={epsilon:.3f} '
                    f'loss={mean_loss if mean_loss is not None else float("nan"):.5f} '
                    f'updates={updates} {terminal}')

            if episode % 25 == 0:
                torch.save({'model': learner.state_dict(),
                            'episode': episode, 'total_frames': total_frames,
                            'updates': updates}, CKPT_DIR / f'learner_ep{episode}.pt')

            if best_score >= TARGET_SCORE:
                log_status(f'TARGET REACHED best_score={best_score:.1f} at episode {episode}')
                torch.save({'model': learner.state_dict(),
                            'score': best_score, 'episode': episode,
                            'total_frames': total_frames, 'updates': updates}, FINAL_PATH)
                break
        else:
            log_status('episode limit reached')

        if not FINAL_PATH.exists():
            torch.save({'model': learner.state_dict(),
                        'score': best_score, 'episode': episode,
                        'total_frames': total_frames, 'updates': updates}, FINAL_PATH)
        log_status(f'done best_score={best_score:.1f} total_frames={total_frames} '
                   f'updates={updates} elapsed={time.monotonic() - started:.0f}s')
    except Exception:
        log_status('ERROR:\n' + traceback.format_exc())
        raise
    finally:
        if game is not None:
            game.close()
        if asset_server is not None:
            asset_server.shutdown()
            asset_server.server_close()
            asset_thread.join(timeout=5)


if __name__ == '__main__':
    main()
