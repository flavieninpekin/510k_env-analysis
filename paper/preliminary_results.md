# 510K Benchmark — 初步结果汇总（合作者讨论版）

> 版本：2026-09-15 · 用途：ICLR 2027 投稿前的内部讨论稿
> 数据来源：`runs/*.eval.json`（本仓库实测），理论见 `notes/theory.md`，
> 定位见 `notes/benchmark_positioning.md`，正文草案见 `paper/draft.md`。

---

## 1. 项目目标（一句话）

510K 不是"又造一个难的游戏"，而是一个**可控消融的棋牌类 MARL testbed**：
把 **即时记分（随牌 5/10/K）**、**终局竞争（先出完）**、**部分可观测**、
**动态隐藏合作（队伍由红 A 分布决定、需推断）** 耦合在同一个决策过程里，
并让每条轴都能独立开关/调节。核心主张是：

> 这些挑战不是简单叠加，而是**互相改变彼此的价值**；现有 benchmark 通常
> 只隔离研究其中一条。510K 提供"组合 + 可控消融"，用来研究**不同算法的
> failure mode 出现在哪种 challenge combination 下**。

与已发现的先导现象 IIGC（AAAI 2027，*deceptive stability*）的关系：
**IIGC 研究现象，本文表征环境能力剖面**，二者互补而非重复。

---

## 2. 已完成的核心资产

| 资产 | 状态（含证据） |
|---|---|
| Gymnasium 单智能体接口 `510K-v0` | ✅ 4 种 mode 均通过 `env_checker` |
| PettingZoo AEC 多智能体接口 | ✅ 通过官方 `api_test` |
| 4 种可控 mode | ✅ SINGLE / STATIC / DYNAMIC / OBVIOUS |
| 信息揭示旋钮 `RevealEnv` | ✅ p=0≡DYNAMIC, p=1≡OBVIOUS，可连续调 |
| 内建 action masking | ✅ 所有策略 illegal-action rate = **0.0** |
| 规则 bot + 随机 bot | ✅ 作为固定评估对手 |
| 基线套件 | ✅ Masked PPO(MLP) / Recurrent PPO(LSTM) / 共享策略 IPPO |
| 环境测试 | ✅ 82/82 通过 |
| 理论结果 | ✅ 4 条命题（见 §3.4） |

---

## 3. 初步结果

范围：MLP = 300k steps，LSTM / IPPO = 200k steps，**3 seeds**，
评估协议统一为"训练策略作为 player 0 vs 规则 bot，100 局固定种子"。

### 3.1 Capability profile（胜率，vs rule bot）

| mode | MLP PPO | LSTM PPO | IPPO (shared) |
|---|---|---|---|
| SINGLE | **0.510 ± 0.022** | 0.437 ± 0.038 | 0.363 ± 0.017 |
| STATIC | 0.750 ± 0.016 | — | — |
| DYNAMIC | 0.847 ± 0.026 | 0.850 ± 0.000 | 0.820 ± 0.033 |
| OBVIOUS | 0.793 ± 0.019 | — | — |

**读法（论文 story 的关键）**
- **SINGLE 对每个方法都最难**：无队友分担，MLP 的胜率 ≈ 0.5（掷硬币）。
- **团队模式胜率被"抬高"**：规则 bot 队友的贡献被计入团队胜负，
  0.75–0.85 不代表任务简单。
- **SINGLE 上方法排序 MLP > LSTM > IPPO** 干净，符合 200–300k 预算下的样本效率预期；
  团队模式下排序被队友噪声淹没（见 §3.4 Prop D）。

### 3.2 信息揭示消融（MLP，DYNAMIC，`RevealEnv` 概率 p）

| reveal p | win rate | reward |
|---|---|---|
| 0.00 | 0.847 ± 0.026 | 77.2 |
| 0.25 | 0.850 ± 0.008 | 72.7 |
| 0.50 | 0.845 ± 0.009 | 73.5 |
| 0.75 | 0.827 ± 0.019 | 74.8 |
| 1.00 | 0.793 ± 0.019 | 67.4 |

**核心负结果**：把队友身份直接给短训练 PPO，**性能不升反略降**（曲线平坦/微降）。
这与 IIGC 的 *deceptive stability* 一致。注意 §3.4 Prop C 证明**旋钮确实被拧动了**
（信息量近线性增长），所以"平坦"是真的"没用起来信息"，而非"没给信息"。

### 3.3 隐藏关系鲁棒性（1v3 vs 2v2，MLP，DYNAMIC）

| seed | 2v2 win rate | 1v3 win rate |
|---|---|---|
| 0 | 0.859 | 0.909 |
| 1 | 0.842 | 0.864 |
| 2 | 0.850 | 0.909 |

结论对队伍构成不敏感 → 结论不依赖 76%/24% 的过滤选择；
latent variable 同时包含**队友是谁**和**是否有队友**。

### 3.4 理论结果（`notes/theory.md`，均数值验证）

- **Prop A 队伍规模先验（闭式）**：`P(|red|=2) = 1 − 4·C(50,11)/C(52,13) ≈ 0.765`，
  20k deals 精确吻合 → "拥抱 1v3"是**推导出的设计**，不是经验残差。
- **Prop B 成员概率**：`P(agent∈red) ≈ 0.441`，`P(agent 是唯一红方) ≈ 0.059`。
- **Prop C 揭示旋钮 ≈ 线性信息通道**：`MI(p) = p·H(T) − δ(p)`，
  实测 `MI(0.25)=0.62, MI(0.5)=1.26, MI(0.75)=1.91`，达理想值 ≥91%；
  δ(p)>0 恰是 1v3 隐藏态的签名。**这堵住了 reviewer "你没真扫信息轴"的质疑。**
- **Prop D 奖励稀释**：团队模式下队友贡献约一半分数方差（own/std share ≈ 0.51）
  → 解释了为什么 SINGLE 是最干净的探针、团队模式胜率是粗指标。

---

## 4. 目前能讲的最强故事

> 现有 MARL benchmark 各自隔离一种难点；510K 把它们耦合且可消融。
> 我们用统一协议扫出 capability profile，并给出**一个反直觉的负结果**：
> 在信息通道被证明近线性可用的前提下，短训练 PPO 依然无法利用队友身份信息
> （与 IIGC 的 deceptive stability 呼应）。环境同时提供闭式先验与奖励稀释理论，
> 使现象可复现、机理可研究。

对标表（`notes/benchmark_positioning.md` §6）显示：只有 510K 同时满足
即时记分 ● + 终局竞争 ● + 部分可观测 ● + **隐藏动态队伍 ●** + 可控消融 ●。

---

## 5. 距离 ICLR 2027 的差距（务必先看）

**时间**：按此前记录，ICLR 2027 abstract 截止 **2026-09-18**、paper 截止
**2026-09-25 AOE**。以今天 9/15 计，**abstract 只剩约 3 天，正文约 10 天**。
这是当前最大的风险，而非实验本身。

**已有 vs 投稿所需（缺口按风险排序）**

| 项 | 现状 | 投稿要求 | 风险 |
|---|---|---|---|
| 训练步数 | 200–300k | 1M（`PLAN.md` 目标） | 高：曲线可能变，尤其 reveal 平坦结论 |
| seeds | 3 | ≥5 + 95% CI + 显著性检验 | 中 |
| 评估对手 | 仅 rule bot | 加 random bot（便宜） | 低 |
| 指标 | 团队胜率（粗） | 加 finish position、自身分数贡献 | 中：直接撑起"信息有用但被稀释"的论证 |
| 矩阵覆盖 | 只跑了单/静/动/明（MLP）+ 单/动（LSTM/IPPO） | 4 modes × 3 policies | 中 |
| CTDE 基线 | 无 | MAPPO / QMIX | 中（可列 future work 降级） |
| 1M 运行环境 | 本机随机崩溃，靠 chunk 重启 | 干净机器 | 高（进度不可控） |
| 图表 | 已有 3 张 png | 最终数据重绘 | 低 |

---

## 6. 需要与合作者拍板的问题

1. **投稿定位**：Benchmark/Environment paper 冲击 ICLR 主会，还是先投
   workshop 保底？（`chatgpt0902.md` 判断：主会有机会但"环境介绍+几个 baseline"不够。）
2. **主 claim 取舍**：走"隐藏动态合作推断"（新颖，但需要信念推断证据）
   还是"多目标记分×胜负的 shaping 稳定性"（有 Ng1999/Skalse2022 理论锚点）？
   还是两者都留、以 capability profile 串起来？——**建议主线选隐藏合作，shaping 作 §7 future work**。
3. **1v3 处理**：拥抱（已用闭式先验论证）还是过滤 2v2？目前倾向拥抱。
4. **是否等 1M 矩阵**：若等不到，是否接受用"短训练 + 明确 caveat"投稿？
5. **IIGC 的引用边界**：如何措辞才能既借势又避免"与 AAAI2027 重复"的质疑。
6. **负结果的呈现**：reveal 平坦是"发现"还是"训练不足"？
   需要 1M + 更强对手才能定论——是否愿意把它作为核心贡献赌一把。

---

## 7. 讨论前必读：数据/代码中的已知问题

1. **`aggregate.py` 会把 reveal 实验并入 DYNAMIC/mlp**：它按
   `(mode, policy, opponent)` 聚合，未区分 `reveal` 参数，导致 dynamic/mlp 被
   统计成 **12 seeds（0.842±0.020）**，把 r0.25/0.5/0.75 混了进去。
   论文表格用的是未混淆的 3 seeds（0.847±0.026）。**修表前务必修正聚合键。**
2. **若干 eval 出现完全相同的数值**（疑点）：
   - `ppo_lstm_rule_dynamic_s{0,1,2}` 全部 `0.85 / 74.65 / 19.28`；
   - `ppo_mlp_rule_obvious_s0` 与 `s2` 全同；
   - `ppo_lstm_rule_single_s0` 与 `s1` 全同。
   若非确定性评估/共享 checkpoint 所致，属实现问题，**会影响 §3 结论可信度**，
   建议在讨论前先排查 eval 是否 seeding / 是否加载了正确 checkpoint。
3. **`aggregate.py` 只输出 `opponent=rule`**，random-bot 结果无从进入表格。
4. **`runs/*.zip` checkpoint 未在仓库中**（只有 eval json），合作者换机器复现
   需按 `README.md` 重跑。
5. 理论 Prop C 的 `δ(p)` 与可学习性界的联系仍 open（`notes/theory.md` 末节）。

---

## 8. 建议的下一步（按 ROI）

1. **（最高）** 修正 `aggregate.py` 聚合键 + 排查 eval 重复值问题 → 现有结论才站得住。
2. 在干净机器上启动 **1M × 5 seeds × MLP（4 modes）**，并加 **random-bot eval**
   （`README.md` 命令已备好，可 crash-tolerant 续跑）。
3. 补 **finish position / 自身分数贡献** 指标（便宜、且直接支撑 Prop D 的论证）。
4. 视时间决定是否补 LSTM/IPPO 的 static/obvious 与 MAPPO/QMIX。
5. 定稿 §5 缺口表 → 明确"哪些写进正文、哪些降级为 future work"。
