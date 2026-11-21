#!/usr/bin/env python3
"""Parallel-env DQN trainer for the T-Rex runner: maximise transitions/sec."""
import base64, copy, os, random, threading, time, traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
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

DEVICE = torch.device('cuda')
ACTION_REPEAT = 4
GAMMA = 0.99
SEED = 20261015

TARGET_SCORE = 2000
MAX_ROUNDS = 200_000
MAX_GAME_FRAMES = 20000
TIME_BUDGET_HOURS = float(os.environ.get('TRAIN_HOURS', '4'))
N_ENVS = int(os.environ.get('N_ENVS', '8'))
UPDATES_PER_ROUND = int(os.environ.get('UPDATES_PER_ROUND', '8'))
BATCH_SIZE = 256

REPLAY_CAPACITY = 50_000
EARLY_CRASH_PENALTY = 5.0
STABILITY_SCORE = 1000
LR = 3e-4
EPS_START, EPS_END, EPS_DECAY_FRAMES = 0.25, 0.02, 1_000_000
TARGET_TAU = 0.002
GRAD_CLIP = 10.0
WARMUP = 1000

LOG_PATH = ROOT / 'results' / 'training_log.csv'
STATUS_PATH = ROOT / 'results' / 'training_status.txt'
BEST_PATH = ROOT / 'results' / 'dino_trained_best.pt'
FINAL_PATH = ROOT / 'results' / 'dino_trained_final.pt'
CKPT_DIR = ROOT / 'results' / 'checkpoints'

FIREFOX_BIN = os.environ.get('FIREFOX_BIN', '/root/tools/firefox/firefox')
GECKODRIVER = os.environ.get('GECKODRIVER', '/root/tools/geckodriver')
HEADLESS = os.environ.get('HEADLESS', '1') != '0'


def log_status(msg):
    line = f'{time.strftime("%H:%M:%S")} {msg}'
    print(line, flush=True)
    with open(STATUS_PATH, 'a') as f:
        f.write(line + '\n')


class GitHubDinoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
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


def stack_frames(frames):
    return np.stack(list(frames)[::-1], axis=0)


class ReplayBuffer:
    """Preallocated ring buffer storing uint8 frames on CPU."""

    def __init__(self, capacity, obs_shape=(4, 80, 80)):
        self.cap = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.discounts = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0
        self._lock = threading.Lock()

    def add(self, obs, action, reward, next_obs, done, discount):
        with self._lock:
            i = self.pos
            self.obs[i] = obs
            self.next_obs[i] = next_obs
            self.actions[i] = action
            self.rewards[i] = reward
            self.dones[i] = float(done)
            self.discounts[i] = discount
            self.pos = (i + 1) % self.cap
            self.size = min(self.size + 1, self.cap)

    def sample(self, n):
        idx = np.random.randint(0, self.size, size=n)
        return (self.obs[idx], self.actions[idx], self.rewards[idx],
                self.next_obs[idx], self.dones[idx], self.discounts[idx])


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
_asset_lock = threading.Lock()


def start_asset_server():
    global asset_server, asset_thread
    with _asset_lock:
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


class GameEnv:
    """One browser instance. Not thread-safe; each worker thread owns one."""

    def __init__(self, env_id):
        self.env_id = env_id
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
        self.driver.set_window_size(640, 480)
        self.frames = deque(maxlen=4)
        self.game_frames = 0
        self.score = 0.0
        self.episode = 0
        self.seed = SEED + env_id * 10_000

    def reset(self):
        self.episode += 1
        self.seed = SEED + self.env_id * 10_000 + self.episode
        self.driver.get(self.url)
        deadline = time.monotonic() + 25
        while not self.driver.execute_script(
                'return !!(window.dinoControl && window.dinoControl.getState().ready)'):
            if time.monotonic() > deadline:
                raise RuntimeError(f'env {self.env_id} game init timed out')
            time.sleep(0.05)
        self.driver.execute_script(JS_SETUP, int(self.seed))
        self.driver.execute_script(JS_FREEZE)
        self.driver.execute_script(JS_HISCORE)
        self.game_frames = 0
        self.score = 0.0
        obs = self._decode(self.driver.execute_script('return window.__dinoObs()'))
        self.frames.clear()
        for _ in range(4):
            self.frames.append(obs.copy())
        return stack_frames(self.frames)

    @staticmethod
    def _decode(encoded):
        return np.frombuffer(base64.b64decode(encoded), dtype=np.uint8).reshape(80, 80).copy()

    def step(self, action):
        if action == 1:
            self.driver.execute_script(JS_JUMP)
        result = self.driver.execute_script(
            'return window.__dinoAdvance(arguments[0])', ACTION_REPEAT)
        advanced = int(result['advanced'])
        done = bool(result['crashed'])
        score = float(result['score'])
        if advanced == 0 and not done:
            raise RuntimeError(f'env {self.env_id} stopped advancing')
        reward = sum(GAMMA ** i * (-1.0 if done and i == advanced - 1 else 0.1)
                     for i in range(advanced))
        if done and score < STABILITY_SCORE:
            reward -= EARLY_CRASH_PENALTY * GAMMA ** max(0, advanced - 1)
        frame = self._decode(result['frame'])
        self.frames.append(frame)
        nxt = stack_frames(self.frames)
        self.game_frames += advanced
        self.score = score
        return nxt, reward, done, score, GAMMA ** advanced, advanced

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


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    log_status(f'start device={DEVICE} n_envs={N_ENVS} updates/round={UPDATES_PER_ROUND} '
               f'batch={BATCH_SIZE} target={TARGET_SCORE}')

    learner = GitHubDinoNet().to(DEVICE)
    target_net = copy.deepcopy(learner).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(learner.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    replay = ReplayBuffer(REPLAY_CAPACITY)
    cpu_policy = GitHubDinoNet().eval().requires_grad_(False)

    started = time.monotonic()
    deadline = started + TIME_BUDGET_HOURS * 3600
    total_frames = 0
    updates = 0
    best_score = -1.0
    episode_counter = 0

    log_status('spawning environments...')
    envs = [GameEnv(i) for i in range(N_ENVS)]
    log_status(f'{N_ENVS} envs ready')

    states = {}
    for env in envs:
        states[env.env_id] = env.reset()
    log_status('all envs reset')

    pool = ThreadPoolExecutor(max_workers=N_ENVS)

    def step_env(env):
        sid = states[env.env_id]
        x = torch.from_numpy(sid).float().unsqueeze(0)
        with torch.inference_mode():
            greedy = int(cpu_policy(x).argmax(1).item())
        epsilon = max(EPS_END,
                      EPS_START - (EPS_START - EPS_END) * total_frames / EPS_DECAY_FRAMES)
        action = random.randrange(2) if random.random() < epsilon else greedy
        nxt, reward, done, score, discount, advanced = env.step(action)
        return env, sid, action, reward, nxt, done, score, discount, advanced

    try:
        round_idx = 0
        while time.monotonic() < deadline and updates < 2_000_000:
            round_idx += 1
            cpu_policy.load_state_dict(learner.state_dict())

            futures = [pool.submit(step_env, env) for env in envs]
            results = [f.result() for f in futures]

            round_frames = 0
            for env, sid, action, reward, nxt, done, score, discount, advanced in results:
                replay.add(sid, action, reward, nxt, done, discount)
                total_frames += advanced
                round_frames += advanced
                if done:
                    episode_counter += 1
                    is_best = score > best_score
                    if is_best:
                        best_score = score
                        torch.save({'model': learner.state_dict(), 'score': score,
                                    'episode': episode_counter}, BEST_PATH)
                    mean_loss = ''
                    row = dict(episode=episode_counter, seed=env.seed, score=round(score, 2),
                               frames=env.game_frames, total_frames=total_frames,
                               epsilon=round(max(EPS_END, EPS_START -
                                                 (EPS_START - EPS_END) * total_frames
                                                 / EPS_DECAY_FRAMES), 4),
                               mean_loss='', updates=updates, terminal='collision',
                               episode_seconds=round(time.monotonic() - started, 2),
                               elapsed_seconds=round(time.monotonic() - started, 1),
                               env=env.env_id)
                    append_log(row)
                    if episode_counter % 10 == 0 or is_best or score >= TARGET_SCORE:
                        log_status(f'ep={episode_counter} score={score:.0f} best={best_score:.0f} '
                                   f'frames={total_frames} upd={updates} '
                                   f'sp={total_frames / max(1e-9, time.monotonic() - started):.0f}f/s')
                    if best_score >= TARGET_SCORE:
                        log_status(f'TARGET REACHED best={best_score:.0f} ep={episode_counter}')
                        torch.save({'model': learner.state_dict(), 'score': best_score,
                                    'episode': episode_counter, 'updates': updates}, FINAL_PATH)
                        pool.shutdown(wait=False)
                        for e in envs:
                            e.close()
                        return
                    states[env.env_id] = env.reset()
                else:
                    states[env.env_id] = nxt

            if replay.size >= max(WARMUP, BATCH_SIZE):
                for _ in range(UPDATES_PER_ROUND):
                    obs_b, act_b, rew_b, next_b, done_b, disc_b = replay.sample(BATCH_SIZE)
                    st = torch.from_numpy(obs_b).float().to(DEVICE, non_blocking=True) / 255.0
                    ns = torch.from_numpy(next_b).float().to(DEVICE, non_blocking=True) / 255.0
                    actions = torch.from_numpy(act_b).to(DEVICE, non_blocking=True)
                    rewards = torch.from_numpy(rew_b).to(DEVICE, non_blocking=True)
                    terminals = torch.from_numpy(done_b).to(DEVICE, non_blocking=True)
                    discounts = torch.from_numpy(disc_b).to(DEVICE, non_blocking=True)
                    with torch.cuda.amp.autocast():
                        q = learner(st).gather(1, actions[:, None]).squeeze(1)
                        with torch.no_grad():
                            next_actions = learner(ns).argmax(1)
                            expected = (rewards + discounts * (1 - terminals)
                                        * target_net(ns).gather(1, next_actions[:, None]).squeeze(1))
                        loss = F.smooth_l1_loss(q, expected)
                    if not torch.isfinite(loss):
                        raise RuntimeError('Non-finite loss')
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(learner.parameters(), GRAD_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                    with torch.no_grad():
                        for tp, lp in zip(target_net.parameters(), learner.parameters()):
                            tp.lerp_(lp, TARGET_TAU)
                    updates += 1

            if round_idx % 200 == 0:
                torch.save({'model': learner.state_dict(), 'updates': updates,
                            'total_frames': total_frames, 'best': best_score},
                           CKPT_DIR / 'learner_latest.pt')
                log_status(f'round={round_idx} frames={total_frames} upd={updates} '
                           f'best={best_score:.0f} replay={replay.size} '
                           f'speed={total_frames / max(1e-9, time.monotonic() - started):.0f} f/s')

        if best_score < TARGET_SCORE:
            torch.save({'model': learner.state_dict(), 'score': best_score,
                        'updates': updates, 'total_frames': total_frames}, FINAL_PATH)
            log_status(f'finished best={best_score:.0f} frames={total_frames} upd={updates}')
    except Exception:
        log_status('ERROR:\n' + traceback.format_exc())
        raise
    finally:
        pool.shutdown(wait=False)
        for e in envs:
            e.close()
        if asset_server is not None:
            asset_server.shutdown()
            asset_server.server_close()
        log_status('cleaned up')


if __name__ == '__main__':
    main()
