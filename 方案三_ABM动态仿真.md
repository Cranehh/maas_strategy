# 方案三：面向MaaS运营策略的仿真优化框架

## 标题

**中文:** 面向北京MaaS运营策略的仿真优化：融合多目标进化算法、选择模型代理与基于活动出行需求的智能体仿真

**English:** Simulation-Based Multi-Objective Optimization of MaaS Operating Strategies for Beijing: Coupling Evolutionary Algorithms, Choice-Model-Derived Surrogates, and Activity-Based Agent Simulation

---

## 核心研究问题

给定北京人口异质性出行需求和MaaS采纳意愿，**最优的MaaS运营策略**（套餐定价、动态定价规则、区域营销分配）是什么？在采纳率、运营收入、空间公平性和碳减排间存在怎样的帕累托权衡？

---

## 整体架构

本文提出 **Simulation-Based Multi-Objective Optimization (SBO)** 框架，核心架构为：NSGA-II 在外层搜索运营商策略参数空间，200万全量ABM 作为黑箱适应度评估器，Ch1–Ch3 选择模型作为解析代理模型加速优化。

### 三阶段架构总览

```
Phase 1: 数据准备
  ┌──────────────────────────────────────────────────┐
  │  Ch4合成人口(200万) + Ch5活动计划                    │
  │  → 月度出行消费档案 + Ch1-Ch3意愿参数                │
  │  → 预计算: cost_alacarte、碳排放baseline            │
  └──────────────────────────────────────────────────┘

Phase 2: Surrogate-Assisted NSGA-II Optimization
  ┌──────────────────────────────────────────────────┐
  │  LHS初始化: 100个θ → Full ABM → 训练GP             │
  │  外层: NSGA-II搜索运营商策略参数θ (17维)             │
  │  快速评估: 解析代理(Ch1-Ch3) + GP残差 (毫秒级)       │
  │  精确评估: Top-10候选 → Full 200万ABM (~5秒/个)     │
  │  → 输出: 4D Pareto前沿                             │
  └──────────────────────────────────────────────────┘

Phase 3: 政策情景分析
  ┌──────────────────────────────────────────────────┐
  │  5个情景各自运行完整优化 → Pareto前沿位移对比         │
  └──────────────────────────────────────────────────┘
```

---

## 4种MaaS套餐（来自Ch3 Hao et al. 2024）

所有套餐共享固定特色：**无限公交 + 无限共享单车**

| 套餐 | 地铁次数/月 | 电动车次数/月 | 出租车公里/月 | 价格系数 | 特色定位 |
|------|-----------|-------------|-------------|---------|---------|
| **Bus First** | 10 (固定) | 基准水平 (固定) | [0-175] (可调) | [0.8-1.5] (可调) | 最便宜，公交导向 |
| **Metro Access** | 60 (固定) | 基准水平 (固定) | [0-175] (可调) | [0.8-1.5] (可调) | 地铁导向 |
| **Value Taxi** | 30 (固定) | 基准水平 (固定) | 基准水平 (固定) | [0.8-1.5] (可调) | 打车导向 |
| **Ultra Access** | 90 (固定) | 基准水平 (固定) | 基准水平 (固定) | [0.8-1.5] (可调) | 全能高端 |

### Ch3 ICLV效用函数中的套餐属性变量

基于Ch3 Table 7的估计结果，ICLV模型中仅以下3个套餐属性变量显著：

| 变量 | 适用套餐 | 估计值 | t值 |
|------|---------|--------|-----|
| 出租车/网约车资源 | Bus First, Metro Access | -0.281 | -6.51 |
| 套餐价格(100元) | 全部4种 | -0.047 | -3.23 |
| 价格系数 | 全部4种 | -0.318 | -1.39 |

地铁次数和电动车次数未进入最终效用函数，因此改变它们不会影响Ch3预测的附加值(AV)。

---

## 策略参数空间θ (17维)

策略参数θ由2类参数组成：

### 1. 套餐定价与配置

| 参数 | 维度 | 范围 | 说明 |
|------|------|------|------|
| $taxi_{BF}$ | 1 | [0, 175] km | Bus First出租车公里 |
| $ps_{BF}$ | 1 | [0.8, 1.5] | Bus First价格系数 |
| $taxi_{MA}$ | 1 | [0, 175] km | Metro Access出租车公里 |
| $ps_{MA}$ | 1 | [0.8, 1.5] | Metro Access价格系数 |
| $ps_{VT}$ | 1 | [0.8, 1.5] | Value Taxi价格系数 |
| $ps_{UA}$ | 1 | [0.8, 1.5] | Ultra Access价格系数 |

**小计:** 6维

### 2. 定价适应规则

| 参数 | 维度 | 范围 | 说明 |
|------|------|------|------|
| $\tau_{high}$ | 1 | [0.7, 1.0] | 利用率上阈值 (触发涨价) |
| $\tau_{low}$ | 1 | [0.2, 0.5] | 利用率下阈值 (触发降价) |
| $\delta_{up}$ | 1 | [0, 0.15] | 涨价幅度 |
| $\delta_{down}$ | 1 | [0, 0.15] | 降价幅度 |
| $freq_{adj}$ | 1 | [1, 4] | 调价频率 (每N季度) |

**小计:** 5维

### 3. 营销分配规则

| 参数 | 维度 | 范围 | 说明 |
|------|------|------|------|
| $B_{total}$ | 1 | 待定 | 总营销预算 (元/季度) |
| $\gamma_{potential}$ | 1 | [0, 1] | 转化潜力权重 |
| $\gamma_{gap}$ | 1 | [0, 1] | 采纳缺口权重 |
| $c_{conc}$ | 1 | [0, 5] | 集中度参数 (Softmax温度) |

**小计:** 4维

### 4. 服务质量参数

| 参数 | 维度 | 范围 | 说明 |
|------|------|------|------|
| $time\_improvement$ | 1 | [0, 0.3] | MaaS出行时间改善比例 (θ[15]) |
| $price\_discount$ | 1 | [0, 0.3] | MaaS出行价格折扣比例 (θ[16]) |

**小计:** 2维

**总维度:** 17维

**约束:**
- $ps_{BF} \leq ps_{MA}$（Bus First始终最便宜之一）
- $\tau_{low} < \tau_{high}$
- $\delta_{up}, \delta_{down} \geq 0$

---

## 核心创新：解析代理 + GP残差校正 (Choice-Model-Derived Surrogate)

### 关键叙事

Ch1–Ch3选择模型参数在本框架中扮演 **双重角色**：
- **ABM内部:** 驱动个体Agent的试用/订阅/选择行为（行为引擎）
- **ABM外部:** 作为优化的解析代理模型（Analytical Surrogate）

这是本文区别于现有 surrogate-assisted SBO 文献（通常用 GP/RBF/NN 作为纯黑箱代理）的核心创新——我们利用离散选择理论的结构化知识构建有物理意义的代理模型。

### 机制

```
predicted_fitness(θ) = analytical_surrogate(θ; Ch1, Ch2, Ch3)   # 静态近似, 毫秒级
                     + GP.predict(θ)                             # 动态残差校正
```

### 解析代理具体公式

对每个agent $i$ 和策略参数 $\theta$:

**静态采纳概率:**

$$P_{adopt}^{static}(i, \theta) = \underbrace{P_{try}^{Ch1}(i)}_{\text{月度试用概率}} \times \underbrace{P_{sub}^{Ch2}(i)}_{\text{订阅转移概率}} \times \underbrace{\mathbb{I}\left[\max_k AV^{Ch3}(i, B_k(\theta)) > 0\right]}_{\text{存在正附加值套餐}}$$

其中:
- $P_{try}^{Ch1}(i) = 1 - (1 - p_{try,single}(i))^{N_{trips}(i)}$，即月度内至少一次出行选择MaaS的概率
- $P_{sub}^{Ch2}(i)$ = Ch2潜在类别模型给出的订阅转移概率
- $AV^{Ch3}(i, B_k(\theta))$ = Ch3 ICLV Nested Logit 模型计算的附加值，效用函数包含 ASC + price + price_scale + taxi_km(仅BF/MA) + 潜变量 + 个体变量

**套餐选择:** $k^*(i) = \arg\max_{k} AV^{Ch3}(i, B_k(\theta))$

**四目标静态近似:**

$$\hat{f}_1(\theta) = -\frac{1}{N} \sum_i P_{adopt}^{static}(i, \theta) \quad \text{(采纳率)}$$

$$\hat{f}_2(\theta) = -\sum_i P_{adopt}^{static}(i, \theta) \times price_{k^*(i)} \quad \text{(收入)}$$

$$\hat{f}_3(\theta) = Gini\left(\left\{\frac{\sum_{i \in d} P_{adopt}^{static}(i)}{N_d}\right\}_{d \in districts}\right) \quad \text{(公平性)}$$

$$\hat{f}_4(\theta) = -\sum_i P_{adopt}^{static}(i, \theta) \times \Delta e_{car}(i) \quad \text{(碳减排)}$$

**GP残差** = ABM动态效应 − 静态近似 = {Bass扩散时滞 + 满意度/流失累积 + 动态调价反馈}

### 优化算法

```
Phase A: 初始训练集构建
├── Latin Hypercube Sampling (LHS) 采样100个θ
├── 每个θ运行 Full ABM (200万agent) → 获得真实适应度
├── 计算残差 = ABM结果 - 解析代理预测
└── 训练4个GP (每个目标一个, Matérn 5/2 + ARD核)

Phase B: 300代NSGA-II迭代优化
├── 每代:
│   ├── 遗传操作 → 200子代
│   ├── 解析代理 + GP预评估 (200×4目标, 毫秒级)
│   ├── 选出Top-10 (高不确定性/高潜力) 候选
│   ├── Top-10 → Full ABM精确评估 (~5秒/个)
│   ├── 更新GP (infill learning)
│   └── 非支配排序 + 拥挤距离选择
└── 输出: Pareto前沿近似集

Phase C: 验证
├── Pareto前沿Top-50解 → 各运行5次取均值
└── 输出: 经验证的4D Pareto前沿
```

### 计算量估计

| 阶段 | ABM调用次数 | 单次时间 | 小计 |
|------|-----------|---------|------|
| Phase A: LHS初始化 | 100 | ~5秒 | ~8分钟 |
| Phase B: 每代Top-10 × 300代 | 3,000 | ~5秒 | ~4.2小时 |
| Phase C: Pareto验证 | 250 | ~5秒 | ~21分钟 |
| **单情景总计** | **~3,350** | | **~4.5小时** |
| **5情景总计** | | | **~22.5小时** |

---

## ABM仿真器设计

### 仿真规格

| 项目 | 设计 |
|------|------|
| **Agent粒度** | 200万个体 (来自Ch4合成人口, 不做聚类) |
| **时间跨度** | 156周 (3年), 以周为步长 |
| **实现方式** | 向量化Python (NumPy), 不用Mesa框架 |
| **单次运行时间** | ~3-5秒 (向量化200万agent × 156周) |

### Agent初始化 (Ch4 + Ch5)

- **属性:** 社会经济属性（来自Ch4合成人口）、日活动-出行计划（来自Ch5）、居住/工作TAZ
- **月度出行消费档案:** 从Ch5活动计划聚合每月各方式出行次数和费用
- **Ch1-Ch3参数:** 每个Agent携带Ch1试用概率函数、Ch2订阅转移概率、Ch3 ICLV参数
- **预计算:** 按次付费总成本 $cost_{alacarte}(i)$、碳排放baseline $e_{baseline}(i)$

### 状态变量

| 变量 | 类型 | 说明 |
|------|------|------|
| `awareness` | $\in [0, 1]$ | MaaS认知水平 |
| `status` | $\in \{unaware, aware, trial, subscriber, churned\}$ | 采纳状态 |
| `satisfaction` | $\in [0, 1]$ | 使用满意度（影响留存/流失） |
| `bundle_choice` | $\in \{none, BF, MA, VT, UA\}$ | 当前订阅套餐 |
| `churn_weeks` | $\in \mathbb{N}$ | 连续低满意度周数 |
| `cooldown_weeks` | $\in \mathbb{N}$ | 流失后冷却期剩余周数 |

### 行为引擎

#### 认知扩散（TAZ聚合Bass模型）

$$\Delta awareness(i, t) = p_{innov} \times marketing(TAZ_i, t) + p_{imit} \times \left[\beta_{local} \cdot \frac{n_{sub}(TAZ_i, t)}{n_{total}(TAZ_i)} + \beta_{remote} \cdot \frac{N_{sub}(t)}{N_{total}}\right]$$

- $p_{innov}$: 外部营销驱动的创新系数
- $p_{imit}$: 社交影响驱动的模仿系数
- $\beta_{local} = 0.8$: 本地（同TAZ）邻居影响权重
- $\beta_{remote} = 0.2$: 全市远程影响权重
- 使用TAZ聚合代替显式社交网络（200万×200万邻接矩阵不可行），完全可向量化

#### 营销投入强度

$$B_{TAZ}(t) = B_{total} \times \frac{\exp(c_{conc} \cdot score(TAZ, t))}{\sum_{TAZ'} \exp(c_{conc} \cdot score(TAZ', t))}$$

$$score(TAZ, t) = \gamma_{potential} \times potential(TAZ) + \gamma_{gap} \times (target\_rate - current\_rate(TAZ, t))$$

$$marketing(TAZ, t) = \frac{B_{TAZ}(t)}{B_{uniform}}, \quad B_{uniform} = \frac{B_{total}}{n_{TAZ}}$$

#### 试用决策（Ch1参数）

当 $awareness(i) > threshold_{aware}$:
- 用Ch1的DCM参数计算月度至少一次选择MaaS的概率: $P_{try}(i) = 1 - (1 - p_{try,single}(i))^{N_{trips}(i)}$
- 若随机数 < $P_{try}$ → $status = trial$

#### 订阅决策（Ch2参数）

试用累计次数达到阈值后:
- 用Ch2的潜在类别模型计算订阅转移概率
- 若转移 → $status = subscriber$，进入套餐选择

#### 套餐选择（Ch3参数）

- 用Ch3的ICLV Nested Logit模型计算4个套餐的附加值 (AV)
- 选择 $\arg\max_k AV(i, B_k)$，若 $\max AV > 0$

#### 满意度更新

$$satisfaction(i, t+1) = \alpha \cdot satisfaction(i, t) + (1-\alpha) \cdot s_{current}(i, t)$$

$$s_{current}(i, t) = w_1 \cdot \underbrace{\min\left(\frac{cost_{alacarte}(i) - price_k}{cost_{alacarte}(i)}, 1\right)}_{\text{省钱比例}} + w_2 \cdot \underbrace{\left(1 - \max\left(0, \frac{usage(i,t) - quota_k}{quota_k}\right)\right)}_{\text{配额匹配度}} + w_3 \cdot \underbrace{\frac{AV(i, B_k)}{AV_{max}}}_{\text{归一化附加值}}$$

**参数:**
- $\alpha = 0.7$（记忆衰减，7:3的历史/当期权重）
- $w_1 = 0.4, w_2 = 0.35, w_3 = 0.25$（省钱最重要，配额匹配次之）
- 初始 $satisfaction = 0.5$（中性）
- 这些参数列入敏感性分析

#### 流失与再订阅

- **流失条件:** $satisfaction < 0.3$ 连续 4 周 → $status = churned$
- **再订阅机制:** churned → 冷却期8周 → 回到 $aware$ 状态，$satisfaction$ 重置为0.5
- **状态转移:** $unaware \to aware \to trial \to subscriber \rightleftharpoons churned$
- 运营商调价后，churned用户可能被新价格重新吸引

### 运营商逻辑

运营商 **不是** 独立Agent（非启发式、非RL），而是按策略参数θ **参数化执行**：
- 每季度（每13个仿真周）根据θ中的适应规则自动调整
- 定价调整: 根据利用率阈值 $\tau_{high}$, $\tau_{low}$ 和调整幅度 $\delta_{up}$, $\delta_{down}$
- 营销分配: 根据各TAZ转化潜力和采纳缺口，按Softmax规则分配预算

### 仿真循环

```python
# 向量化伪代码 (NumPy数组操作, 非逐agent循环, 200万agent)
for t in range(156):  # 156周 = 3年
    # 认知扩散 (TAZ聚合Bass模型, 向量化)
    taz_sub_ratio = compute_taz_subscription_ratio(status, agent_taz)
    global_sub_ratio = np.sum(status == SUBSCRIBER) / n_agents
    awareness += p_innov * marketing_intensity[agent_taz] \
               + p_imit * (0.8 * taz_sub_ratio[agent_taz] + 0.2 * global_sub_ratio)
    awareness = np.clip(awareness, 0, 1)

    # 试用决策 (Ch1 DCM, 向量化)
    trial_prob = ch1_trial_probability(agents, bundles, awareness)
    new_trials = (np.random.random(n_agents) < trial_prob) & (status == AWARE)
    status[new_trials] = TRIAL

    # 订阅决策 (Ch2 LC, 向量化)
    subscribe_prob = ch2_subscribe_probability(agents, trial_count)
    new_subs = (np.random.random(n_agents) < subscribe_prob) & (status == TRIAL)
    status[new_subs] = SUBSCRIBER

    # 套餐选择 (Ch3 ICLV, 向量化)
    bundle_choice[new_subs] = ch3_bundle_choice(agents[new_subs], bundles)

    # 满意度更新
    s_current = (0.4 * cost_saving + 0.35 * quota_fit + 0.25 * av_normalized)
    satisfaction = 0.7 * satisfaction + 0.3 * s_current

    # 流失判定
    low_sat = satisfaction < 0.3
    churn_weeks[low_sat] += 1
    churn_weeks[~low_sat] = 0
    newly_churned = (churn_weeks >= 4) & (status == SUBSCRIBER)
    status[newly_churned] = CHURNED
    cooldown_weeks[newly_churned] = 8

    # 再订阅 (冷却期结束后回到aware)
    cooldown_weeks[status == CHURNED] -= 1
    reactivate = (cooldown_weeks <= 0) & (status == CHURNED)
    status[reactivate] = AWARE
    satisfaction[reactivate] = 0.5

    # 运营商季度调整 (参数化执行)
    if t % 13 == 0:
        bundles = adjust_pricing(bundles, utilization, theta)
        marketing_intensity = allocate_marketing(theta, adoption_by_taz)

    record_metrics(t)
```

---

## 适应度函数 (4目标, 全最小化)

NSGA-II使用4个目标函数，统一为最小化问题：

$$f_1(\theta) = -\text{adoption\_rate}_{year3}$$

最大化第3年末MaaS总采纳率

$$f_2(\theta) = -\text{total\_revenue}_{3yr}$$

最大化3年累计运营收入

$$f_3(\theta) = +\text{Gini}(\text{adoption\_by\_district})$$

最小化采纳率的空间Gini系数（提升空间公平性）

$$f_4(\theta) = -\text{carbon\_reduction}_{3yr}$$

最大化3年累计碳减排量

### 碳减排计算

$$\text{carbon\_reduction} = \sum_{i} \sum_{t} \left[ e_{car}(i)_{baseline} - e_{car}(i,t)_{MaaS} \right]$$

- Baseline = Ch5活动计划中的原始私家车出行碳排放（预计算，不运行仿真）
- MaaS减排 = 订阅者的私家车出行被公交替代的部分
- 替代比例 = $\min(\text{套餐公交配额} / N_{car}(i), 1)$

---

## 情景设计 (5个)

每个情景独立运行完整的NSGA-II优化流程，得到各自的Pareto前沿：

| 情景 | 描述 | 关键修改 |
|------|------|----------|
| **S0: 基准** | 当前票价, 无额外补贴 | 默认17维参数空间 |
| **S1: 碳普惠** | 碳积分折扣纳入套餐价值 | Ch3 AV函数增加碳收益项 |
| **S2: 拥堵收费** | 五环内实施拥堵收费 | 私车出行成本 +15-25元/天 |
| **S3: 低收入补贴** | 月收入<5000元居民享半价套餐 | 低收入群体价格参数×0.5 |
| **S4: 空间差异化** | 允许按5大区调整价格折扣 | +5维区域折扣系数 (总22维) |

### 核心对比

跨情景Pareto前沿叠加图 → 分析哪个政策最有效推动前沿外移（即同时改善多个目标）

---

## 验证策略

### 1. Pareto极端点直觉一致性

- 最大采纳率解：应对应高营销预算 + 低价格策略
- 最大收入解：应对应高价格
- 最优公平性解：应对应空间均衡营销分配
- 验证方法：检查极端点的策略参数是否符合经济直觉

### 2. 基准情景方式分担率校验

- S0基准情景的方式分担率 vs 北京交通年报数据
- 期望偏差 < 5%

### 3. MaaS采纳率文献对标

- MaaS采纳率对标: Tang et al. (2025) SABM 的北京MaaS采纳预测
- 验证ABM产出的合理范围

### 4. 多次运行方差检验

- 每个Pareto最优解运行5次，报告均值±标准差
- 验证结果稳定性

### 5. 敏感性分析

| 参数 | 扰动范围 | 关注指标 |
|------|---------|---------|
| ICLV参数 (Ch3) | ±10% | Pareto前沿形状变化 |
| Bass参数 $p_{innov}$, $p_{imit}$ | ±20% | 采纳S曲线 |
| 满意度权重 $w_1, w_2, w_3$ | ±15% | 3年采纳率、流失率 |
| 满意度衰减 $\alpha$ | ±15% | 流失动态 |
| GP残差精度 | — | 代理 vs ABM相关系数 |

---

## 创新点

### 1. 首个MaaS运营策略的仿真优化框架

NSGA-II + 200万全量ABM用于MaaS策略优化。现有文献中：
- 静态SP设计优化仅考虑一次性套餐设计（如Reck et al., 2020）
- 简化ABM仅用启发式规则测试少量策略（如Tang et al., 2025）
- ML推荐系统无法处理动态策略（如Xi et al., 2024）

本文首次将仿真优化 (SBO) 范式引入MaaS运营策略设计。

### 2. 选择模型驱动的解析代理（新的surrogate-assisted SBO范式）

Ch1–Ch3参数的双重角色：ABM行为引擎 + 优化代理模型。区别于现有SBO文献中纯黑箱代理（GP/RBF/NN），本文利用离散选择理论的结构化知识构建有物理意义的代理模型，GP仅学习残差。这一范式可推广到所有嵌入行为模型的交通仿真优化问题。

### 3. 动态策略 > 静态设计

优化"如何随市场演化调整"（定价适应规则 + 营销分配规则）而非仅"初始卖什么套餐"。将运营策略从一次性设计问题转化为自适应策略优化问题。

### 4. 4D Pareto前沿

Adoption–Revenue–Equity–Carbon四维权衡，为决策者提供完整的策略菜单而非单一最优解。支持交互式探索不同目标优先级下的最优策略。

---

## 与方案一的关系

方案一是本方案的 **特例和基准线**。

当策略参数θ中的所有适应规则关闭时（$\delta_{up} = \delta_{down} = 0$, $c_{conc} = 0$ 即营销均匀分配），本框架退化为方案一的静态优化问题。

**核心对比:** "动态定价适应 + 智能营销分配 比 静态最优 好多少?"

| 维度 | 方案一（静态） | 方案三（动态） |
|------|-------------|-------------|
| 决策变量 | 套餐定价设计 | 定价 + 适应规则 + 营销规则 |
| 目标函数 | 社会福利/收入 | 4D Pareto前沿 |
| 需求模型 | Ch3静态AV | Ch1+Ch2+Ch3 动态 + Bass扩散 |
| 时间维度 | 单期快照 | 156周动态 |
| 用户异质性 | 分层加权 | 200万个体Agent |

---

## 预期产出

### 1. 4D Pareto前沿可视化

- X轴: adoption rate, Y轴: revenue
- 颜色: equity (Gini), 大小: carbon reduction
- 可交互选择不同策略查看详情

### 2. 最优策略菜单

5-10个代表性Pareto最优策略的详细参数：各套餐价格系数、出租车配额、定价适应规则、营销分配规则

### 3. TAZ级热力图

各策略下的空间采纳分布（北京分区），对比不同策略的空间公平性差异

### 4. 3年S曲线

各策略的订阅者增长轨迹，标注关键里程碑（100万/300万/600万用户）

### 5. 政策情景前沿位移图

5个情景的Pareto前沿叠加对比，定量分析每个政策对前沿的外移贡献

### 6. 自适应价值量化

Dynamic vs Static策略的改善百分比：
- $\Delta adoption = \frac{adoption_{dynamic} - adoption_{static}}{adoption_{static}} \times 100\%$
- 同理计算revenue/equity/carbon的改善

---

## 技术实现

| 项目 | 设计 |
|------|------|
| **编程语言** | Python |
| **ABM实现** | 向量化NumPy (不用Mesa), 200万agent |
| **优化框架** | pymoo (NSGA-II) |
| **代理模型** | scikit-learn GP (Matérn 5/2 + ARD核) + 自定义解析代理 |
| **并行化** | joblib / Ray (ABM评估并行) |
| **可视化** | matplotlib + plotly (交互式Pareto) |

---

## 发表目标

- **首选:** Transportation Research Part C — SimOpt方法论创新 + 代理模型创新
- **备选:** Transportation Research Part A — 应用导向 / Computers, Environment and Urban Systems — 计算方法+城市系统

---

## 关键参考文献

### 方法论

- Tay, S. M. M., & Osorio, C. (2022). Bayesian optimization techniques for high-dimensional simulation-based transportation problems. *Transportation Research Part B*, 164, 210-243.
- Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182-197.
- Abkarian, H., & Mahmassani, H. S. (2024). Simulation-based optimization of transportation system operations. *Transportation Research Record*.

### MaaS文献

- Tang, B., et al. (2025). Beijing SABM-MaaS. *Transportation Research Part A*.
- Ren, X., et al. (2024). Equity-aware mobility service design, New York.
- Xi, H., et al. (2024). MaaSformer-MMoE: ML-based bundle customization.
- Reck, D., et al. (2020). MaaS bundle design dimensions. *Transportation Research Part A*.
- Ho, C., et al. (2021). Sydney MaaS trial bundle design. *Transportation Research Part A*.
- Zhu, Z., et al. (2023). MaaS transitions in 41 Chinese cities.
- Li, Y., et al. (2024). Personalized MaaS bundles & carbon reduction, Beijing.
- Caiati, V., Rasouli, S., & Timmermans, H. (2020). Sequential portfolio choice experiment.
- Feneri, A., Rasouli, S., & Timmermans, H. (2023). MaaS modelling review.
- Nunez, H., & Antoniou, C. (2025). MaaSPI: MaaS potential index.
- Alonso-Gonzalez, M., et al. (2020). Drivers and barriers in MaaS adoption.
- Liljamo, T., et al. (2024). ABM willingness to pay for MaaS.
- Ferretti, F., et al. (2024). Spatio-temporal network MaaS simulation.

### 前序章节

- Yao, X., et al. (2025). First-time MaaS adoption willingness (DCM). [Ch1]
- Hao, H., et al. (2025). Subscription transition willingness (LC/ICLV). [Ch2]
- Hao, H., et al. (2024). Added value estimation of MaaS bundles (ICLV). *Transportation*. [Ch3]
