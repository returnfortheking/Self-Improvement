# Day 29: Assignment 5 - Alignment & RLHF - 面试题

> **题目难度**: ⭐⭐ ~ ⭐⭐⭐⭐⭐
> **考察重点**: RLHF流程、PPO算法、DPO原理、奖励模型、对齐方法
> **建议时间**: 40分钟

---

## Part 1: 对齐基础

### Q1: 什么是对齐问题(Alignment Problem)？⭐⭐⭐

**参考答案**:

**定义**: 确保AI系统的行为与人类价值观、意图和期望保持一致

**三个层面**:
1. **意图对齐**: AI理解并执行用户想要的操作
2. **价值观对齐**: AI的行为符合社会道德标准
3. **安全性对齐**: AI不会产生有害内容

**重要性**:
```
未对齐的风险:
- 生成有害内容（暴力、歧视、虚假信息）
- 目标错误优化（"回形针最大化器"）
- 对抗性攻击（提示词注入）
- 不可控行为

对齐的价值:
- 用户信任和体验
- 商业应用必要条件
- 监管合规要求
- AGI发展的安全保障
```

**解决方法**:
- RLHF (Reinforcement Learning from Human Feedback)
- RLAIF (Reinforcement Learning from AI Feedback)
- Constitutional AI
- DPO (Direct Preference Optimization)

---

### Q2: RLHF vs 传统监督学习的区别？⭐⭐⭐

**参考答案**:

| 维度 | 监督学习 (SFT) | RLHF |
|------|---------------|------|
| **数据** | 单个正确答案 | 成对偏好 (chosen vs rejected) |
| **目标** | 最大化似然 | 最大化奖励信号 |
| **训练** | 单阶段 | 三阶段（SFT → 奖励模型 → PPO） |
| **反馈** | 离散标签 | 连续奖励值 |
| **效果** | 基础对齐 | 更好的对齐和安全性 |

**RLHF优势**:
- 捕捉细微的人类偏好
- 处理多个正确答案的场景
- 提升响应质量和安全性

**RLHF挑战**:
- 训练复杂度高（三阶段）
- 需要大量高质量人类反馈
- PPO训练不稳定
- 计算资源消耗大

---

## Part 2: RLHF流程

### Q3: RLHF的三个阶段是什么？⭐⭐⭐⭐

**参考答案**:

**阶段1: 监督微调(SFT)**
```
目标: 训练基础模型理解指令

数据格式:
- Prompt: "解释量子计算"
- Response: "量子计算利用量子比特的叠加和纠缠..."

训练:
- 使用高质量指令-响应对
- 通常10K-100K样本
- 标准语言模型损失
- 结果: 指令遵循模型
```

**阶段2: 奖励模型(Reward Model)训练**
```
目标: 学习人类偏好函数

数据格式:
Prompt              |  Response A       |  Response B       | Preference
"解释量子计算"      | "量子计算是..."   | "量子比特是..."   | A > B

训练:
- 输入: (prompt, response)对
- 输出: 标量奖励值
- 损失: Bradley-Terry成对排序损失
- 结果: 奖励模型R(x, y)
```

**阶段3: PPO强化学习**
```
目标: 优化策略以最大化奖励

流程:
1. 使用当前策略π_θ生成response
2. 奖励模型R对response打分
3. 计算PPO损失（包含KL惩罚）
4. 更新策略参数θ
5. 重复1-4

约束:
- KL散度惩罚（防止偏离SFT模型）
- 价值函数裁剪
- 信任区域优化
```

**关键要点**:
- SFT提供初始化
- 奖励模型捕获人类偏好
- PPO在信任区域内优化
- 三阶段缺一不可

---

### Q4: 如何训练奖励模型？⭐⭐⭐⭐

**参考答案**:

**模型结构**:
```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model  # 共享基础模型
        self.reward_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids, attention_mask)
        last_hidden = outputs.last_hidden_state[:, -1, :]
        reward = self.reward_head(last_hidden)
        return reward
```

**训练数据**:
```
格式: (prompt, chosen, rejected)三元组

示例:
{
    "prompt": "什么是机器学习？",
    "chosen": "机器学习是AI的分支，让计算机从数据学习...",
    "rejected": "我不知道，可能是关于电脑的..."
}
```

**损失函数（Bradley-Terry模型）**:
```python
def reward_loss(reward_chosen, reward_rejected):
    """
    目标: P(chosen > rejected) = sigmoid(r_chosen - r_rejected)
    """
    # Log概率
    log_prob = F.logsigmoid(reward_chosen - reward_rejected)

    # 损失（最大化chosen的概率）
    loss = -log_prob.mean()

    # 准确率
    accuracy = (reward_chosen > reward_rejected).float().mean()

    return loss, accuracy
```

**训练技巧**:
1. **预训练初始化**: 从SFT模型开始
2. **数据质量**: 人工标注优于自动生成
3. **Early Stopping**: 防止过拟合
4. **Batch Size**: 64-256
5. **Learning Rate**: 1e-5 ~ 5e-5（较小）

---

## Part 3: PPO算法

### Q5: PPO算法的核心思想是什么？⭐⭐⭐⭐⭐

**参考答案**:

**核心问题**: 传统的策略梯度方法可能因为一次策略更新过大而崩溃

**PPO解决方案**: 在信任区域内优化策略，限制每次更新的幅度

**目标函数**:
```python
L(θ) = E[ min( r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A ) ]

其中:
- r(θ) = π_θ(a|s) / π_θ_old(a|s) （概率比率）
- A: 优势函数（Advantage）
- ε: 裁剪参数（通常0.2）
```

**裁剪机制**:
```
情况1: A > 0（好动作）
  - 如果r ∈ [1-ε, 1+ε]: L = r * A（正常更新）
  - 如果r > 1+ε: L = (1+ε) * A（裁剪，防止过度增加）

情况2: A < 0（坏动作）
  - 如果r ∈ [1-ε, 1+ε]: L = r * A（正常更新）
  - 如果r < 1-ε: L = (1-ε) * A（裁剪，防止过度减少）
```

**为什么有效**:
1. 限制策略更新幅度，防止崩溃
2. 保留单调性（好动作增加概率，坏动作减少）
3. 简单实现，高效训练

**KL散度惩罚**:
```python
# 额外约束：防止偏离参考策略太远
kl_div = π_θ(a|s) / (π_θ(a|s) + π_ref(a|s))

# 总损失
total_loss = ppo_loss + β * kl_penalty
```

---

### Q6: 如何计算优势函数(Advantage)？⭐⭐⭐⭐

**参考答案**:

**定义**: 优势函数衡量一个动作比平均好多少

**方法1: 蒙特卡洛估计**
```python
# 简单但高方差
A(s, a) = R(s, a) - V(s)

其中:
- R(s, a): 实际回报（从状态s采取动作a后的累积奖励）
- V(s): 价值函数（平均回报）
```

**方法2: TD (Temporal Difference) 误差**
```python
# 低方差但有偏
A(s, a) = r + γ * V(s') - V(s)

其中:
- r: 即时奖励
- γ: 折扣因子
- s': 下一状态
```

**方法3: GAE (Generalized Advantage Estimation) - 推荐**
```python
def compute_gae(rewards, values, gamma=0.99, lambda_gae=0.95):
    """
    平衡方差和偏差
    """
    advantages = []
    gae = 0

    # 从后向前计算
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]

        # TD残差
        delta = rewards[t] + gamma * next_value - values[t]

        # GAE
        gae = delta + gamma * lambda_gae * gae
        advantages.insert(0, gae)

    return advantages
```

**GAE参数**:
- **lambda_gae = 0**: 纯TD（低方差，高偏差）
- **lambda_gae = 1**: 纯蒙特卡洛（高方差，无偏）
- **推荐值**: 0.95（平衡）

---

## Part 4: DPO

### Q7: DPO和RLHF的区别？⭐⭐⭐⭐⭐

**参考答案**:

| 维度 | RLHF | DPO |
|------|------|-----|
| **训练阶段** | 三阶段（SFT → RM → PPO） | 单阶段 |
| **奖励模型** | 需要单独训练 | 无需 |
| **强化学习** | 需要PPO等RL算法 | 直接优化 |
| **参考模型** | 用于KL惩罚 | 用于计算参考log概率 |
| **训练稳定性** | 较难调参（KL系数、裁剪） | 更稳定 |
| **计算效率** | 较低（需要rollout、价值函数） | 较高 |
| **效果** | SOTA | 接近RLHF |
| **实现复杂度** | 高 | 低 |

**DPO核心思想**:
```
传统RLHF:
1. 训练奖励模型 R(x,y) = E[人类评分]
2. 用RL优化: max E[R(x,y)]

DPO:
直接优化策略，使得:
log(π(y_chosen|x) / π(y_rejected|x)) ∝ R(x,y_chosen) - R(x,y_rejected)

即: 最大化 log(π_chosen) - log(π_rejected)
```

**DPO损失**:
```python
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """
    无需奖励模型，直接优化偏好
    """
    # 策略模型的log概率差
    policy_logratios = policy_chosen_logps - policy_rejected_logps

    # 参考模型的log概率差
    ref_logratios = ref_chosen_logps - ref_rejected_logps

    # DPO损失
    loss = -F.logsigmoid(beta * (policy_logratios - ref_logratios)).mean()

    return loss
```

**何时使用DPO**:
- ✅ 快速实验和原型
- ✅ 计算资源有限
- ✅ 不熟悉RL调参
- ❌ 需要极致性能（RLHF仍略优）

---

### Q8: DPO的数学原理？⭐⭐⭐⭐⭐

**参考答案**:

**推导**:

**步骤1: 奖励模型的解析解**

根据Bradley-Terry模型，奖励模型的最优解满足:
```
R(x, y) = log(π*(y|x) / π_ref(y|x)) + Z(x)

其中:
- π*: 最优策略
- π_ref: 参考策略
- Z(x): 归一化常数
```

**步骤2: 代入奖励差**

对于偏好数据 (y_chosen, y_rejected):
```
R(x, y_chosen) - R(x, y_rejected)
= log(π*(y_chosen|x) / π_ref(y_chosen|x)) - log(π*(y_rejected|x) / π_ref(y_rejected|x))
= log(π*(y_chosen|x) / π*(y_rejected|x)) - log(π_ref(y_chosen|x) / π_ref(y_rejected|x))
```

**步骤3: DPO目标**

最大化该差值，即DPO损失:
```
L_DPO = -E[ log(σ(β * (log_ratio - log_ratio_ref))) ]

其中:
- log_ratio = log(π_θ(y_chosen|x) / π_θ(y_rejected|x))
- log_ratio_ref = log(π_ref(y_chosen|x) / π_ref(y_rejected|x))
- β: 温度参数（控制优化强度）
```

**关键洞察**:
1. **无需显式奖励模型**: 通过参考模型隐式定义奖励
2. **直接优化策略**: 避免RL的复杂性
3. **参考模型的作用**: 提供优化约束，防止过度优化

**实践建议**:
- β ∈ [0.1, 0.5]: 太小优化不足，太大不稳定
- 参考模型冻结: 不训练，仅用于计算约束
- 学习率: 1e-5 ~ 1e-4（较小，防止破坏SFT权重）

---

## Part 5: 偏好数据

### Q9: 如何收集高质量的偏好数据？⭐⭐⭐⭐

**参考答案**:

**数据质量标准**:

**好的偏好对**:
```python
# Example: 好的偏好数据
prompt = "解释气候变化的原因"

chosen = """
气候变化主要由温室气体排放引起:
1. 二氧化碳: 化石燃料燃烧（煤炭、石油、天然气）
2. 甲烷: 畜牧业、垃圾填埋、天然气泄漏
3. 氧化亚氮: 农业化肥使用

这些气体在大气中形成温室效应，导致全球气温上升。
"""

rejected = """
天气变热是因为太阳活动频繁。还有人说这是自然现象，
不需要担心。我觉得大家太夸张了，夏天热很正常。
"""

# 特点:
# chosen: 事实准确、结构清晰、有依据
# rejected: 信息错误、观点混乱、缺乏逻辑
```

**收集方法**:

**1. 人工标注（高质量但昂贵）**
```python
# 标注指南
guidelines = {
    "准确性": "信息必须事实准确",
    "完整性": "回答应该全面",
    "安全性": "不包含有害内容",
    "有用性": "直接回答用户问题",
}

# 标注流程
for prompt in dataset:
    responses = model.generate(prompt, num_samples=2)
    ranking = human_annotator.rank(responses)  # A > B
    save_pair(prompt, ranking)
```

**2. 模型对比（中等质量）**
```python
# 使用多个模型生成
models = [gpt4, claude, llama]

for prompt in dataset:
    responses = [m.generate(prompt) for m in models]
    # 用强模型（如GPT-4）评分
    scores = [gpt4.score(r) for r in responses]
    chosen = responses[np.argmax(scores)]
    rejected = responses[np.argmin(scores)]
```

**3. 用户反馈（真实场景）**
```python
# A/B测试
for user_request in logs:
    response_a = model_v1.generate(user_request)
    response_b = model_v2.generate(user_request)

    # 用户选择
    if user.clicked(response_a):
        chosen, rejected = response_a, response_b
    else:
        chosen, rejected = response_b, response_a
```

**数据平衡**:
```python
# 难度分布
difficulty_levels = {
    "easy": 0.3,    # 简单问题
    "medium": 0.5,  # 中等难度
    "hard": 0.2,    # 困难问题
}

# 任务类型
task_types = [
    "question_answering",  # 问答
    "summarization",       # 摘要
    "creative_writing",    # 创意写作
    "coding",              # 编程
    "math",                # 数学
]
```

**数据验证**:
```python
def validate_preference_pair(chosen, rejected):
    """验证偏好对质量"""
    # 1. 不能完全相同
    if chosen == rejected:
        return False

    # 2. 长度合理
    if len(chosen) < 10 or len(rejected) < 10:
        return False

    # 3. chosen应该更好（用启发式规则）
    if count_keywords(chosen) < count_keywords(rejected):
        return False

    return True
```

---

### Q10: RLHF训练不稳定怎么办？⭐⭐⭐⭐⭐

**参考答案**:

**常见问题及解决方案**:

**问题1: 奖励黑客(Reward Hacking)**
```python
症状: 模型生成高奖励但无意义的回复

原因: 策略学会了欺骗奖励模型

解决:
1. 增加KL散度惩罚
   kl_coeff = 0.02 → 0.05

2. 定期更新奖励模型
   reward_model.train_on_new_data()

3. 使用人类监督
   人工检查高风险输出
```

**问题2: 训练崩溃**
```python
症状: 损失爆炸，生成重复或乱码

解决:
1. 降低学习率
   lr = 1e-5

2. 梯度裁剪
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

3. 减小batch size
   batch_size = 256 → 64

4. 调整PPO参数
   clip_param = 0.2 → 0.1
```

**问题3: 性能退化**
```python
症状: 对齐提升但生成能力下降

原因: 过度优化奖励，遗忘预训练知识

解决:
1. 混合训练数据
   data = preference_data + pretraining_data

2. 使用更小的KL系数
   kl_coeff = 0.01

3. 定期评估
   在多个任务上验证
```

**问题4: 训练不收敛**
```python
症状: 损失振荡，奖励不稳定

解决:
1. 调整GAE参数
   lambda_gae = 0.95 → 0.9

2. 增加样本数量
   num_ppo_epochs = 4 → 10

3. 使用学习率调度
   scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

4. 归一化优势函数
   advantages = (advantages - mean) / (std + 1e-8)
```

**最佳实践**:
```python
# 渐进式训练
for epoch in range(num_epochs):
    # 早期: 更保守
    if epoch < 5:
        kl_coeff = 0.1
        lr = 1e-6
    # 后期: 更激进
    else:
        kl_coeff = 0.02
        lr = 1e-5

    # 定期评估
    if epoch % 5 == 0:
        metrics = evaluate(model, val_set)
        print(f"Epoch {epoch}: {metrics}")
```

---

## 总结

**必会题目** (面试高频):
- Q1: 对齐问题的定义和重要性
- Q3: RLHF三阶段流程
- Q5: PPO核心思想和裁剪机制
- Q7: DPO vs RLHF区别

**加分题目** (深入理解):
- Q4: 奖励模型训练方法
- Q6: 优势函数计算（GAE）
- Q8: DPO数学推导
- Q9: 偏好数据收集策略
- Q10: RLHF训练稳定性

**建议**:
1. 理解完整的RLHF流程（SFT → RM → PPO）
2. 掌握PPO的核心机制（裁剪、KL惩罚）
3. 理解DPO的原理和优势
4. 了解实际训练中的挑战和解决方案
5. 关注最新进展（Constitutional AI, ORPO等）

---

**下一步**: [Week 5: Agent架构](../../Week5_Agents/README.md)
