#!/usr/bin/env python3
"""Shared game environment + nets for the DinoRL comparison experiments.

Two observation modes:
  - vision: 4x80x80 uint8 frame stack (identical to train_fast.py)
  - state:  26-dim float feature vector read directly from the game engine
Reward, action space ({noop, jump}) and stepping match train_fast.py exactly.
"""
import base64, os, random, threading, time
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

DEVICE = torch.device('cuda')
ACTION_REPEAT = 4
GAMMA = 0.99
N_ACTIONS = 2
TARGET_SCORE = 2000
MAX_GAME_FRAMES = 20000
EARLY_CRASH_PENALTY = 5.0
STABILITY_SCORE = 1000
FEAT_DIM = 26

# 26-dim feature layout (see JS_FEATS_FN):
#   [0:5]   tRex y, jumpVel, jumping, ducking, speed
#   [5:12]  obstacle slot 0: dist, width, top, ty_small, ty_large, ty_ptero, height
#   [12:19] obstacle slot 1  (same layout)
#   [19:26] obstacle slot 2  (same layout)
FEAT_MASKS = {
    'notype': [8, 9, 10, 15, 16, 17, 22, 23, 24],
    'notrex': [0, 1, 2, 3, 4],
    'noheight': [6, 7, 11, 13, 14, 18, 20, 21, 25],
    # keeps only per-slot dist + the jumping flag
    'distonly': [0, 1, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18,
                 20, 21, 22, 23, 24, 25],
}


def feat_mask_array(name):
    """Boolean (FEAT_DIM,) mask: True = zero this dim out."""
    if not name:
        return None
    m = np.zeros(FEAT_DIM, dtype=bool)
    m[FEAT_MASKS[name]] = True
    return m

FIREFOX_BIN = os.environ.get('FIREFOX_BIN', '/root/tools/firefox/firefox')
GECKODRIVER = os.environ.get('GECKODRIVER', '/root/tools/geckodriver')
HEADLESS = os.environ.get('HEADLESS', '1') != '0'

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


def shutdown_asset_server():
    global asset_server
    if asset_server is not None:
        asset_server.shutdown()
        asset_server.server_close()
        asset_server = None


JS_SETUP = r"""
  const el=document.createElement('script');
  el.textContent='(()=>{let s='+String(arguments[0]>>>0)+'; Math.random=()=>{s=(1664525*s+1013904223)>>>0;return s/4294967296;};})();';
  document.head.appendChild(el);el.remove();
"""

JS_COMMON = r"""
  const r=Runner.instance_;r.stop();Runner.config.ACCELERATION=0;r.config.ACCELERATION=0;
  let t=performance.now();
  Object.defineProperty(performance,'now',{configurable:true,value:()=>t});
  window.requestAnimationFrame=()=>1;window.cancelAnimationFrame=()=>{};
  const panel=document.querySelector('body > section');if(panel)panel.style.display='none';
  const message=document.getElementById('messageBox');if(message)message.style.display='none';
  let introStart=null;
"""

JS_OBS_FN = r"""
  const off=document.createElement('canvas');off.width=__RES__;off.height=__RES__;
  const octx=off.getContext('2d',{willReadFrequently:true});
  octx.imageSmoothingEnabled=true;octx.imageSmoothingQuality='high';
  window.__dinoObs=()=>{
    const c=document.querySelector('canvas.runner-canvas');
    const w=Math.min(500,c.width),h=Math.min(300,c.height);
    octx.clearRect(0,0,__RES__,__RES__);
    octx.drawImage(c,0,0,w,h,0,0,__RES__,__RES__);
    const d=octx.getImageData(0,0,__RES__,__RES__).data;
    const out=new Uint8ClampedArray(__R2__);
    for(let i=0;i<__R2__;i++){const j=i<<2;out[i]=(d[j]*77+d[j+1]*151+d[j+2]*28)>>8;}
    let s='';for(let i=0;i<__R2__;i++)s+=String.fromCharCode(out[i]);
    return btoa(s);
  };
"""


def js_obs_fn(res):
    return JS_OBS_FN.replace('__R2__', str(res * res)).replace('__RES__', str(res))

JS_FEATS_FN = r"""
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

JS_ADVANCE_VISION = r"""
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
"""

JS_ADVANCE_STATE = r"""
  window.__dinoAdvance=(n)=>{
    let advanced=0;
    for(let i=0;i<n && r.playing && !r.crashed;i++){
      if(r.playingIntro && introStart===null)introStart=t;
      if(r.playingIntro && t-introStart>=400){r.startGame();introStart=null;}
      t+=1000/60;r.update();advanced++;
      if(!r.playingIntro)introStart=null;
    }
    return {crashed:!!r.crashed,score:r.distanceMeter.getActualDistance(Math.ceil(r.distanceRan)),
      advanced,feats:window.__dinoFeats()};
  };
"""

JS_TAIL = r"""
  r.time=t;
  const e={keyCode:38,target:document.body,preventDefault(){}};
  r.onKeyDown(e);r.onKeyUp(e);
"""

JS_HISCORE = r"""const r=Runner.instance_;
  r.distanceMeter.setHighScore(363);
  r.distanceMeter.update(0,Math.ceil(r.distanceRan));"""

JS_JUMP = r"""const r=Runner.instance_;if(!r.tRex.jumping){
  const e={keyCode:38,target:document.body,preventDefault(){}};r.onKeyDown(e);r.onKeyUp(e);}"""


def js_freeze(mode, obs_res=80):
    if mode == 'vision':
        return JS_COMMON + js_obs_fn(obs_res) + JS_FEATS_FN + JS_ADVANCE_VISION + JS_TAIL
    return JS_COMMON + JS_FEATS_FN + JS_ADVANCE_STATE + JS_TAIL


class GameEnv:
    """One headless Firefox running the game. Owned by one worker thread."""

    def __init__(self, env_id, mode='vision', seed_base=20261015,
                 frame_stack=4, obs_res=80, step_reward=0.1,
                 crash_penalty=1.0, early_penalty=5.0, feat_mask=None):
        assert mode in ('vision', 'state')
        self.env_id = env_id
        self.mode = mode
        self.seed_base = seed_base
        self.frame_stack = frame_stack
        self.obs_res = obs_res
        self.step_reward = step_reward
        self.crash_penalty = crash_penalty
        self.early_penalty = early_penalty
        self.feat_mask = feat_mask_array(feat_mask) if isinstance(feat_mask, str) else feat_mask
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
        self.frames = deque(maxlen=frame_stack)
        self.game_frames = 0
        self.score = 0.0
        self.episode = 0
        self.seed = seed_base + env_id * 10_000

    def reset(self, seed=None):
        self.episode += 1
        if seed is not None:
            self.seed = int(seed) & 0xFFFFFFFF
        else:
            self.seed = self.seed_base + self.env_id * 10_000 + self.episode
        self.driver.get(self.url)
        deadline = time.monotonic() + 25
        while not self.driver.execute_script(
                'return !!(window.dinoControl && window.dinoControl.getState().ready)'):
            if time.monotonic() > deadline:
                raise RuntimeError(f'env {self.env_id} game init timed out')
            time.sleep(0.05)
        self.driver.execute_script(JS_SETUP, int(self.seed))
        self.driver.execute_script(js_freeze(self.mode, self.obs_res))
        self.driver.execute_script(JS_HISCORE)
        self.game_frames = 0
        self.score = 0.0
        if self.mode == 'vision':
            obs = self._decode(self.driver.execute_script('return window.__dinoObs()'),
                               self.obs_res)
            self.frames.clear()
            for _ in range(self.frame_stack):
                self.frames.append(obs.copy())
            return stack_frames(self.frames)
        return self._masked(np.asarray(
            self.driver.execute_script('return window.__dinoFeats()'), dtype=np.float32))

    @staticmethod
    def _decode(encoded, res=80):
        return np.frombuffer(base64.b64decode(encoded), dtype=np.uint8).reshape(res, res).copy()

    def _masked(self, arr):
        if self.feat_mask is not None:
            arr = arr.copy()
            arr[self.feat_mask] = 0.0
        return arr

    def feats(self):
        """Engine feature vector (26,) — available in any mode once js_freeze ran."""
        return self._masked(np.asarray(
            self.driver.execute_script('return window.__dinoFeats()'), dtype=np.float32))

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
        reward = sum(GAMMA ** i * (-self.crash_penalty if done and i == advanced - 1
                                   else self.step_reward)
                     for i in range(advanced))
        if done and score < STABILITY_SCORE:
            reward -= self.early_penalty * GAMMA ** max(0, advanced - 1)
        if self.mode == 'vision':
            frame = self._decode(result['frame'], self.obs_res)
            self.frames.append(frame)
            nxt = stack_frames(self.frames)
        else:
            nxt = self._masked(np.asarray(result['feats'], dtype=np.float32))
        self.game_frames += advanced
        self.score = score
        return nxt, reward, done, score, GAMMA ** advanced, advanced

    def close(self):
        try:
            self.driver.quit()
        except Exception:
            pass


def stack_frames(frames):
    return np.stack(list(frames)[::-1], axis=0)


class ReplayBuffer:
    """Preallocated ring buffer (uint8 frames or float32 features)."""

    def __init__(self, capacity, obs_shape, obs_dtype):
        self.cap = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
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


class GitHubDinoNet(nn.Module):
    """Identical to train_fast.py so saved checkpoints stay loadable. out_dim=2 for Q."""

    def __init__(self, out_dim=2, in_channels=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 8, stride=4)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64, 512)
        self.fc2 = nn.Linear(512, out_dim)

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


class StateNet(nn.Module):
    """MLP over the 26-dim engine feature vector."""

    def __init__(self, in_dim=FEAT_DIM, out_dim=2, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PPOVisionNet(nn.Module):
    """Dino conv trunk + policy/value heads."""

    def __init__(self, in_channels=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 8, stride=4)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64, 512)
        self.policy = nn.Linear(512, N_ACTIONS)
        self.value = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.pool1(GitHubDinoNet._keras_same(self.conv1, x)))
        x = F.relu(self.pool2(GitHubDinoNet._keras_same(self.conv2, x)))
        x = F.relu(self.pool3(GitHubDinoNet._keras_same(self.conv3, x)))
        x = x.permute(0, 2, 3, 1).contiguous().flatten(1)
        h = F.relu(self.fc1(x))
        return self.policy(h), self.value(h).squeeze(-1)


def make_net(algo, obs_mode, out_dim=None, in_channels=4):
    if out_dim is None:
        out_dim = 2
    if obs_mode == 'vision':
        return GitHubDinoNet(out_dim, in_channels=in_channels)
    return StateNet(out_dim=out_dim)


def batched_to_tensor(obs_list, obs_mode):
    arr = np.stack(obs_list)
    if obs_mode == 'vision':
        return torch.from_numpy(arr).float().to(DEVICE, non_blocking=True) / 255.0
    return torch.from_numpy(arr).float().to(DEVICE, non_blocking=True)


def append_log(path, row):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new_file = not p.exists()
    with open(path, 'a') as f:
        if new_file:
            f.write(','.join(row.keys()) + '\n')
        f.write(','.join(str(row[k]) for k in row.keys()) + '\n')
        f.flush()


def log_status(path, msg):
    line = f'{time.strftime("%H:%M:%S")} {msg}'
    print(line, flush=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a') as f:
        f.write(line + '\n')
