# DinoRL 对比实验方案

## 1. 研究问题

1. 纯视觉（像素）输入下，不同深度强化学习算法在 T-Rex Runner 上的表现差异（样本效率、最终分数）。
2. 特权信息的价值：直接读取游戏引擎状态（非视觉）比纯视觉快多少、好多少。
3. 学习方法相对非学习基线（随机、规则启发式）的提升幅度。

## 2. 统一实验协议（所有方法共用）

| 项目 | 设定 |
|---|---|
| 游戏 | t-rex-runner-gh-pages（本地静态服务器 + headless Firefox/Selenium） |
| 动作空间 | `{noop, jump}`（与既有 `train_fast.py` 基线一致，保证可比） |
| 决策步 | 每个 agent step 推进 `ACTION_REPEAT=4` 个游戏帧 |
| 奖励 | 每游戏帧 +0.1 折扣累加；死亡帧 −1；`score<1000` 早死额外 −5（同 `train_fast.py`） |
| 折扣 | γ=0.99 / 游戏帧（transition 的 discount = γ^advanced） |
| episode 终止 | 碰撞（游戏原生 GameOver） |
| 随机性 | JS 侧 LCG 替换 `Math.random`，seed 驱动，完全可复现 |
| 训练预算 | **2,230,000 游戏帧**（对齐既有基线 2,229,322 帧），或 10h 硬超时 |
| 评估 | 50 个固定 seed（900000–900049），贪心策略（ε=0），报 mean/median/std/min/max/p25/p75 |
| 训练 seed | `20261015`（env seed = base + env_id×10000 + episode） |
| 硬件 | 1×RTX 4090D，15 CPU，8 并行 Firefox 环境 |

**观测模式**：
- `vision`：4×80×80 灰度帧堆叠（画布降采样），与既有基线逐位一致。
- `state`：26 维特征向量，JS 内直接读 `Runner.instance_`（恐龙高度/竖直速度/跳跃/下蹲/速度 + 前 3 个障碍物的距离/宽度/顶点高度/类型 one-hot/高度，均归一化）。

## 3. 方法清单

| 方法 | 观测 | 类型 | 说明 |
|---|---|---|---|
| `random` | state（仅驱动） | 无学习 | 每步均匀随机动作，分数下界 |
| `rule` | state | 无学习 | 规则启发式：仙人掌（含密集簇）近距起跳，起跳窗口 ~200px（恒速 6、跳跃滞空 ~34 帧） |
| `dqn` | vision | 值函数 | vanilla DQN（target-net max 目标），本方案新训 |
| `ddqn` | vision | 值函数 | Double DQN（online 选动作、target 评估）。**复用既有 `train_fast.py` 结果**（其目标本就是 Double 形式），2.23M 帧 |
| `c51` | vision | 值函数 | 分布式 DQN（51 原子，support [−20,20]，Double 选动作），代表 Rainbow 类改进 |
| `ppo` | vision | 策略梯度 | PPO-Clip，64×8 rollout，GAE λ=0.95，4 epochs，同款 CNN trunk + policy/value 头 |
| `state_dqn` | **state** | 值函数 | Double DQN + 26 维引擎特征 MLP（256-256）——非视觉直接接口读取版 |

除 `ddqn` 外均以同一代码基座（`dino_env.py`）新训，超参数对齐 `train_fast.py`：
Adam 3e-4、batch 256、replay 50k、每轮 8 次更新、soft target τ=0.002、grad clip 10、
ε: 0.25→0.02 线性退火（1M 帧）、warmup 1000。
PPO 专用：lr 2.5e-4、clip 0.2、entropy 0.01、value coef 0.5、grad-norm 0.5。

## 4. 评估指标

1. **训练曲线**：episode score vs 累计帧数（50 局滑动平均 + best-so-far），统一按帧数对齐。
2. **最终评估**：50 个固定 seed 贪心得分的 mean / median / std / min / max / p25 / p75。
3. **样本效率**：达到给定分数所需帧数（若达到）。
4. **视觉 vs 特权差距**：`state_dqn` 与 `ddqn`/`dqn` 的同预算对比。

## 5. 产物目录

```
results/exp/<name>/
  training_log.csv    # 逐 episode 训练日志
  status.txt          # 训练状态
  best.pt / final.pt  # 最优 / 最终权重
  eval_episodes.csv   # 50 seed 逐局评估
  eval_summary.csv    # 评估汇总
results/exp/compare_table.csv   # 总表
results/exp/compare_curves.png  # 训练曲线对比图
```

## 6. 执行顺序（`run_experiments.sh`）

1. `eval random`、`eval rule`（分钟级）
2. `train_offpolicy --algo dqn --obs vision`（≈4h）
3. `eval ddqn`（复用既有 best.pt）
4. `train_offpolicy --algo c51 --obs vision`（≈4h）
5. `train_ppo`（≈4h）
6. `train_offpolicy --algo ddqn --obs state --name state_dqn`（≈2-3h，无图像传输更快）
7. 评估全部 ckpt → 汇总表 + 对比图

预计总墙钟 ≈ 15h（顺序执行，避免 Firefox/CPU 争抢）。

## 7. 已知局限与约定

- **环境为恒速配置**：`ACCELERATION=0` 冻结速度为 6 px/帧，翼龙（minSpeed 8.5）永不出现，
  实际障碍物只有仙人掌（单/双/三连）。因此动作空间 `{noop, jump}` 足够，无需下蹲；
  2000 分对应连续躲过约 2 万游戏帧的仙人掌。JS 侧冻结时钟与随机数，完全确定性。
- **单训练 seed**：本轮每方法 1 个训练 seed（评估 50 seed）；论文级结论需补 2-3 个训练 seed。
- **规则基线**与学习方法同动作空间；密集仙人掌簇受 4 帧决策粒度限制，上限有限。
- `train_fast.py` 的目标函数实为 Double DQN，本表如实标注为 `ddqn`；`dqn` 为 vanilla 目标补跑，
  以保证算法标签准确。
