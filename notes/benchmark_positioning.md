# 510K 环境对标分析（benchmark positioning）

> 目标：让 510K 环境论文尽量不被 reviewer 挑刺，成功向学术界/业界推荐。
> 本文档回答两个问题：
> 1. 需要和哪些成熟环境/工作对比/对标？
> 2. 每个对标工作要"对齐"到哪条标准、差异化卖点在哪？
>
> 最后更新：2026-09-03

---

## 0. 一句话定位（全文的锚）

> 510K is a card-game testbed in which **immediate scoring, terminal competitive objectives, partial observability, and dynamic (hidden) cooperation** are **coupled in a single game**, and each axis is **independently ablatable**.

这个定位必须经得起两个检验：
- **唯一性**：没有任何现有环境同时具备这四个挑战 + 可控消融（见 §6 矩阵）。
- **可验证**：每个挑战都有代码开关（mode 枚举 + scorer 分支 + obs 构建），不是营销话术。

---

## 1. 对标分层总览

| 层级 | 作用 | 对标对象 |
|---|---|---|
| Tier 1 棋牌类环境 | 直接竞争者，证明"组合的唯一性" | RLCard, OpenSpiel, PyTAG, DouZero/斗地主, Hanabi, Suphx/麻将, Big2, Tichu, Hearts/Spades |
| Tier 2 MARL 套件 | 基础设施/接口标准，证明"工程达标" | PettingZoo, SMAC, Overcooked-AI, Melting Pot, MOSMAC |
| Tier 3 奖励/多目标 | 支撑"记分×胜负"多目标与 shaping 故事 | Ng 1999 (PBRS), Skalse 2022 (reward hacking), MOSMAC, Reward Hacking Benchmark |
| Tier 4 隐藏角色/社会推理 | 支撑"动态合作推理"故事 | Hanabi (ToM), Diplomacy/Cicero, 隐藏角色游戏 (Avalon/Werewolf), Overcooked |

---

## 2. Tier 1：棋牌类环境（直接竞争者）

### 2.1 RLCard（Zha et al., NeurIPS 2019 workshop / AAAI 2020；github.com/datamllab/rlcard）
- **覆盖**：Blackjack, Leduc/Texas Hold'em, UNO, 斗地主, 麻将等。
- **优点**：最成熟的棋牌 RL 工具包，接口规范、支持多种算法、被广泛引用。
- **与 510K 差异**：每个游戏孤立单一挑战（斗地主=竞争+静态合作；麻将=记分+部分可观测，无队伍；UNO=纯出牌）。
- **对齐标准**：接口规范度、预训练模型/规则 agent 基线、可视化。
- **差异化卖点**：RLCard 是"一套工具包"，不是"一个可控消融的基准"；510K 提供 RLCard 里没有任何游戏能做到的**隐藏动态队伍**。

### 2.2 OpenSpiel（Lanctot et al., 2019；DeepMind）
- **覆盖**：50+ 博弈论游戏，强调算法中立、game theory 严谨。
- **对齐标准**：术语规范（extensive-form games, information sets）、评估工具。
- **差异化**：OpenSpiel 偏完美/不完美信息博弈的"解算"研究；510K 偏"稀疏奖励 + 多目标 + 社会关系"的 RL/agent 研究。

### 2.3 PyTAG（2023, IEEE CoG）
- **覆盖**：20+ 桌面棋牌游戏，统一 API。
- **对齐标准**：现代 API、规则可配置。
- **差异化**：PyTAG 覆盖广但每个游戏浅；510K 追求单游戏深 + 可控消融。

### 2.4 斗地主 / DouZero（Zha et al., ICML 2021；DouZero+ 2022）
- **覆盖**：竞争 + 静态合作（地主 vs 两农民）+ 部分可观测 + 稀疏终局奖励。
- **优点**：学术地位高（ICML），有 SOTA 级 self-play baseline，社区认可。
- **与 510K 差异**：
  - 斗地主队伍**固定**（地主身份开局确定）；510K 队伍**隐藏、由红A分布决定** → 关系是 latent variable。
  - 斗地主无"随牌记分"与"先出完"的**权衡**；510K 两者并重。
- **对齐标准**：DouZero 证明了"竞技棋牌 + self-play + 稀疏奖励"能成为严肃 benchmark——510K 应引用这一点作合理性论证（"long horizons and sparse reward, the only time a nonzero reward is incurred is at the end of a game"）。
- **差异化卖点**：510K = 斗地主的"隐藏队伍 + 记分权衡"升级版。

### 2.5 Hanabi Challenge（Bard et al., AIJ 2020）
- **覆盖**：纯合作 + 部分可观测（看不到自己的牌）+ theory-of-mind。
- **优点**：合作不完全信息的标杆，有官方 baseline 和长期挑战。
- **与 510K 差异**：Hanabi **无竞争、无记分、无队伍推断**（关系是"所有人都是队友"）。
- **对齐标准**：官方 baseline、评估协议、社区 adoption。
- **差异化卖点**：510K 是"带竞争和隐藏队友的 Hanabi"——比 Hanabi 多一个"我该和谁合作"的推断层。

### 2.6 麻将 / Suphx（Li et al., 2020；以及后续 Sparrow Mahjong 等）
- **覆盖**：部分可观测 + 稀疏奖励 + 大状态空间 + 全局奖励（global reward）+ 记分（番数）。
- **与 510K 差异**：麻将**无队伍合作**（自摸/放炮，本质单人竞争）；510K 有动态队伍。
- **对齐标准**：Suphx 处理"稀疏奖励/全局奖励"的方法（global reward predictor, oracle training）值得引用——这正好是 510K shaping 故事的对标文献。
- **差异化卖点**：510K = "有队伍的麻将"。

### 2.7 Big Two（2026 论文, arXiv）
- **覆盖**：四人大老二，部分可观测，出牌组合，强调"短期 vs 长期策略权衡"。
- **与 510K 差异**：无记分、无队伍、无红A特殊规则。
- **对齐标准**：其对"long-term strategic action vs locally optimal action"的表述可直接引用，佐证 510K 的长期权衡。

### 2.8 Tichu / Hearts / Spades（记分 + 队伍 + 吃墩）
- **Tichu**：4 人 2v2 固定队伍 + 分值牌（5/10/K）+ 出牌 → **与 510K 结构最接近的西方游戏**。
- **Hearts/Spades**：吃墩 + 记分（避免分数）+ 搭档（Spades 有固定搭档）。
- **关键**：这些游戏队伍**固定且已知**，无人研究"隐藏动态队伍"。这是 reviewer 问"为什么不用 Tichu"时的标准回答：*510K 是 Tichu 的结构 + 斗地主的竞争 + 麻将的部分可观测 + 隐藏队伍推断*。
- 注：Tichu 在 RL 社区**不是成熟 benchmark**（无公认 baseline/评估），反而降低竞争压力。

---

## 3. Tier 2：MARL benchmark 套件（基础设施对标）

| 工作 | 特点 | 需对齐的标准 | 与 510K 关系 |
|---|---|---|---|
| **PettingZoo**（Terry et al., 2021） | AEC/Parallel 双 API，`agent_iter` 循环，社区标准 | **510K 已用 AEC API**；应补 `ParallelEnv` 包装 + `gymnasium.register` + 官方 API 测试 | 直接依赖，须达标 |
| **SMAC**（Samvelyan et al., 2019） | 最常用 MARL 基准，CTDE 评估协议 | 多 seed、评估协议、报告均值和 CI | 510K 应提供同规格评估协议 |
| **Overcooked-AI**（Carroll et al., 2019/2020） | 人类-AI 协作、ad-hoc 合作 | 人类研究接口、合作度量 | 510K 的"合作推断"可对标；IIGC 已在 Overcooked 上做过压力测试 |
| **Melting Pot**（Agapiou et al., 2022） | DeepMind 社会交互基准（合作/竞争/混合动机） | substrate 概念、混合动机划分 | 510K 是"混合动机 + 隐藏关系"的卡片版 Melting Pot |
| **MOSMAC**（AAMAS 2025） | 多目标 MARL 基准（顺序子任务） | 多目标效用、长期时域信用分配 | **510K 多目标（记分×胜负）的最直接对标**；论文可引用其"long horizon + multi-objective MARL 缺基准"的论证 |

---

## 4. Tier 3：奖励/多目标/规范（支撑 shaping 故事）

- **Ng, Harada & Russell (ICML 1999)** — PBRS 策略不变性："non-potential-based shaping 会出 bug"。这是"trick 记分 shaping 不稳定"的理论锚点。
- **Skalse et al. (NeurIPS 2022)** — reward hacking 形式化定义。510K 可实证展示"优化代理奖励（记分）反而降低真实目标（胜负）"。
- **Pan et al. (ICLR 2022)** — reward misspecification 映射与缓解。提供实验范式（故意注入 misspecification 观察行为）。
- **RUDDER (NeurIPS 2019)** — 返回分解/奖励再分配。支撑"可行性"（terminal-only 能否学会、如何学会）。
- **Reward Hacking Benchmark (2026)** — 已把"shaping 鲁棒性"做成基准主题，证明市场存在。
- **MOSMAC** — 多目标 MARL 的对标基准（见 Tier 2）。

> 若论文主题取"stability and feasibility of reward shaping"，Tier 3 是主要 related work；若取"dynamic cooperation"，Tier 4 为主。

---

## 5. Tier 4：隐藏角色 / 动态合作 / 社会推理

- **Hanabi（AIJ 2020）** — theory-of-mind、信念推断（§2.5）。
- **Diplomacy / Cicero（Silver et al., Science 2022）** — 谈判、联盟形成、隐藏意图。**是"动态合作关系推断"在更大尺度的标杆**；510K 是它的低成本卡片版。
- **隐藏角色游戏（Avalon / 抵抗组织 / 狼人杀）** — 隐藏身份推理，但**目前主要活跃在 LLM 智能体研究（2023+），尚无成熟 RL 基准**。→ 这是 510K 的**空白机会**：把"隐藏角色推理"做成有 ground-truth、可训练 RL 的基准。
- **Overcooked-AI** — ad-hoc 协作、人类配对（§3）。

> **业界/LLM 智能体角度**：Diplomacy/Cicero 与隐藏角色 LLM 研究证明"关系推断"是行业关注点；510K 可以成为该方向的**轻量、可控、可复现**测试床（与 AgentBench/GameBench/SmartPlay 的"LLM 打游戏"评估对接）。

---

## 6. 核心对标矩阵（论文的 centerpiece）

| 挑战 \ 环境 | 510K | 斗地主 | Hanabi | 麻将 | Big2 | Tichu | Hearts | Overcooked | SMAC | Melting Pot |
|---|---|---|---|---|---|---|---|---|---|---|
| 即时记分（随牌） | ● | ○ | ○ | ● | ○ | ● | ● | ○ | ○ | ◐ |
| 终局/先出完竞争 | ● | ● | ○ | ● | ● | ● | ● | ○ | ● | ◐ |
| 部分可观测 | ● | ● | ● | ● | ● | ◐ | ◐ | ○ | ◐ | ○ |
| 队伍合作 | ●(隐藏) | ●(固定) | ●(全员) | ○ | ○ | ●(固定) | ●(固定) | ●(显式) | ◐ | ● |
| **隐藏/动态队伍** | **●** | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ◐ |
| 可控消融（mode 开关） | ● | ○ | ○ | ○ | ○ | ○ | ○ | ◐ | ◐ | ◐ |
| 组合耦合（多挑战同一局） | ● | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ |

● 强/有　◐ 部分/可变　○ 无

**唯一单元格**：只有 510K 同时满足"即时记分 ● + 终局竞争 ● + 部分可观测 ● + **隐藏动态队伍 ●** + 可控消融 ●"。

---

## 7. 每个对标工作要"对齐"的标准（发布 checklist）

对 Tier 1 每个环境，论文必须做到：
1. **给出与它的 taxonomy 对比表**（§6），证明组合唯一性。
2. **解释为何不直接用该环境**（如 Tichu/斗地主队伍固定、Hanabi 无竞争、麻将无队伍）。
3. **引用其标杆结果**（DouZero 的 sparse+long-horizon 论证、Suphx 的 global reward、Hanabi 的评估协议）。

对 Tier 2/3/4：
4. **接口对齐**：Gymnasium + PettingZoo AEC/Parallel + `gymnasium.register`，跑官方 API 一致性测试。
5. **评估协议对齐**：≥5 seeds、报告 mean±CI、固定评估对手（规则 bot / 预训练 bot / self-play）。
6. **可复现**：`pip install`、随机种子、Docker、config 仓库化、轨迹数据发布。
7. **基线套件**：Independent PPO、MAPPO/QMIX（CTDE）、recurrent（history-aware）、search/planning baseline——覆盖四种典型能力。
8. **ground-truth 工具**：规则 agent、牌面可解释特征（现有 features.py 7 维）、胜率/分数/先出完统计。

---

## 8. Reviewer 可能挑的刺 + 对策

| 可能的质疑 | 对策 |
|---|---|
| "Why 510K? 为什么不用 Tichu/斗地主/桥牌/麻将？" | §6 矩阵证明组合唯一性；明确"我们比较的是挑战组合，不是单个难度" |
| "这个游戏太小众（地区性）" | 斗地主同样地区性但 ICML 认可；规则比桥牌简单（无叫牌/无拍卖），学习成本低是优点 |
| "一局才 ~20 步，算什么 long-horizon？" | 引用 DouZero："long horizons and sparse reward... only time a nonzero reward is incurred is at the end of a game"；且内部 ~100 玩家动作、17 trick |
| "奖励只有终局，太稀疏/无法学习" | 这正是论文的研究对象（reward shaping 稳定性/可行性）；同时提供稠密变体接口 |
| "没有强 baseline" | 移植兄弟仓库训练栈（PPO/A2C/SAC/DQN/REINFORCE 已有 κ 数据），补 MAPPO/QMIX/recurrent |
| "与 IIGC/AAAI2027 论文重复？" | 第一页明确：IIGC 研究**现象**，本文表征**环境能力剖面** |
| "动作空间 300 太大/组合爆炸" | 已有 action masking；实测 4 人最大合法牌型 75 < 300，需写进论文作压力测试表 |
| "红A队伍 24% 是 1v3，合作故事不成立" | 两种处理二选一并写进论文：(a) 2v2 过滤；(b) 拥抱 1v3 作为"关系是 latent variable"的又一证据 |

---

## 9. 行动清单（发布前，按优先级）

**P0（必须）**
- [ ] taxonomy 对比表 + 引用 DouZero/Hanabi/Suphx/RLCard 标杆
- [ ] baseline 套件（Independent PPO / MAPPO / recurrent / search）在 SINGLE/STATIC/DYNAMIC/OBVIOUS + 3p
- [ ] ≥5 seeds、mean±CI 评估协议、固定评估对手
- [ ] PettingZoo Parallel + gymnasium.register + API 一致性测试

**P1（强烈建议）**
- [ ] 信息消融轴（复用兄弟仓库 reveal 实验：0/25/50/75/100%）
- [ ] reward shaping 稳定性实验（trick 记分 shaping vs potential-based shaping → reward hacking 实证）
- [ ] 1v3 队伍处理决策并写进设计
- [ ] 轨迹数据发布 + 复现脚本

**P2（加分项）**
- [ ] 人类研究接口（如 Overcooked 的人类-AI 配对）
- [ ] LLM 智能体评估对接（GameBench/AgentBench 风格）
- [ ] 可视化/规则 agent 基线发布

---

## 10. 结论

- 必须对标的是 **Tier 1 六类棋牌工作**（证明组合唯一性）+ **Tier 2 五个 MARL 套件**（证明工程达标）+ **Tier 3/4 支撑各自故事的理论/基准**。
- 510K 的护城河 = **唯一同时拥有 记分×胜负×部分可观测×隐藏动态队伍×可控消融 的环境**。
- 最需要小心的两个坑：**与 IIGC 论文的区分**（叙事要明确）、**1v3 队伍与"合作"叙事的一致性**（必须显式设计）。
- 从"能用"到"推荐得出去"，差的是 **baseline 套件 + 评估协议 + 数据发布 + 一份漂亮的 taxonomy 表**，而不是规则本身。