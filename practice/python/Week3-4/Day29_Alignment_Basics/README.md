# Day 29: Assignment 5 - Alignment & RLHF

> **学习目标**: 理解对齐(Alignment)问题，掌握RLHF和DPO方法，实现人类反馈训练
> **时间分配**: 6小时（理论2.5h + 实践3.5h）
> **难度**: ⭐⭐⭐⭐⭐
> **重要性**: ⭐⭐⭐⭐⭐ (LLM安全性和可控性的核心技术)
> **来源**: CS336 Assignment 5 - Alignment

---

## 📚 核心概念

### 1. 对齐问题(Alignment Problem)

#### 1.1 什么是对齐？

**定义**: 确保AI系统的行为与人类价值观、意图和期望保持一致

**三个层面**:
1. **意图对齐**: AI理解并执行用户想要的操作
2. **价值观对齐**: AI的行为符合社会道德标准
3. **安全性对齐**: AI不会产生有害内容

**为什么重要**:
```
未对齐的风险:
- 生成有害内容（暴力、歧视、虚假信息）
- 目标错误优化（"回形针最大化器"思想实验）
- 对抗性攻击（提示词注入）
- 不可控的行为（欺骗性、逃避监管）

对齐的价值:
- 提升用户体验和信任
- 满足监管要求（AI法案）
- 商业应用的必要条件
- AGI发展的安全保障
```

#### 1.2 对齐方法演进

**历史脉络**:
```
2017: RLHF雏形（OpenAI）
   ↓
2020: GPT-3应用InstructGPT技术
   ↓
2022: ChatGPT成功（RLHF规模化）
   ↓
2023: DPO等替代方法出现
   ↓
2024: Constitutional AI, ORPO等新方向
```

**主流方法对比**:

| 方法 | 原理 | 优点 | 缺点 | 适用场景 |
|------|------|------|------|----------|
| **Supervised Fine-tuning** | 人类标注数据直接训练 | 简单高效 | 依赖标注质量 | 基础对齐 |
| **RLHF** | 基于人类反馈的强化学习 | 效果显著 | 训练复杂 | 生产环境 |
| **DPO** | 直接偏好优化 | 无需奖励模型 | 需要成对数据 | 研究实验 |
| **RAG** | 检索增强生成 | 事实准确 | 需要外部知识库 | 知识密集型 |
| **Constitutional AI** | 基于原则的自我修正 | 可解释性强 | 计算开销大 | 安全关键 |

---

### 2. RLHF (Reinforcement Learning from Human Feedback)

#### 2.1 RLHF三阶段流程

**阶段1: 监督微调(SFT)**
```
目标: 训练基础模型理解指令

数据:
- Prompt: "解释量子计算"
- Response: "量子计算利用量子比特..."

训练:
- 使用高质量指令数据
- 通常10K-100K样本
- 标准的语言模型损失函数
```

**阶段2: 奖励模型(Reward Model)训练**
```
目标: 学习人类偏好

数据收集:
Prompt              |  Response A       |  Response B       | Preference
"解释量子计算"      | "量子计算是..."   | "量子比特是..."   | A > B
"写一首诗"         | "春天来了..."     | "诗歌是..."       | B > A

训练:
- 输入: (prompt, response)对
- 输出: 标量奖励值（打分）
- 损失: 成对排序损失
```

**阶段3: PPO强化学习**
```
目标: 优化策略以最大化奖励

流程:
1. 使用当前策略生成response
2. 奖励模型对response打分
3. 计算PPO损失
4. 更新策略参数
5. 重复1-4（多个epoch）

约束:
- KL散度惩罚（防止偏离SFT模型太远）
- 价值函数裁剪
- 信任区域优化
```

#### 2.2 奖励模型(Reward Model)

**模型结构**:
```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model  # 共享基础模型
        self.reward_head = nn.Linear(hidden_size, 1)  # 奖励头

    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids, attention_mask)
        last_hidden = outputs.last_hidden_state[:, -1, :]
        reward = self.reward_head(last_hidden)
        return reward
```

**训练损失（Bradley-Terry模型）**:
```python
def reward_loss(reward_chosen, reward_rejected):
    """
    计算成对排序损失

    目标: 使chosen的奖励 > rejected的奖励
    """
    # Log概率
    prob_chosen = torch.logsigmoid(reward_chosen - reward_rejected)

    # 损失（最大化chosen的奖励）
    loss = -prob_chosen.mean()

    # 准确率
    accuracy = (reward_chosen > reward_rejected).float().mean()

    return loss, accuracy
```

**关键超参数**:
- **学习率**: 1e-5 ~ 5e-5（较小，避免破坏预训练权重）
- **Batch size**: 64 ~ 256
- **Epoch**: 1 ~ 3（过拟合风险）
- **温度参数**: 0.1 ~ 1.0（控制奖励分布）

#### 2.3 PPO算法 (Proximal Policy Optimization)

**核心思想**: 在信任区域内优化策略，避免策略更新过大

**目标函数**:
```
L(θ) = E[ min( r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A ) ]

其中:
- r(θ) = π_θ(a|s) / π_θ_old(a|s) （概率比率）
- A: 优势函数（Advantage）
- ε: 裁剪参数（通常0.2）
```

**优势函数计算**:
```python
def compute_advantages(rewards, values, gamma=0.99, lambda_gae=0.95):
    """
    使用GAE (Generalized Advantage Estimation)计算优势
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

    return torch.tensor(advantages)
```

**PPO损失函数**:
```python
def ppo_loss(policy_log_probs, old_policy_log_probs, advantages,
             value_pred, returns, clip_param=0.2):
    """
    计算PPO损失
    """
    # 1. 策略损失（裁剪）
    ratio = torch.exp(policy_log_probs - old_policy_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # 2. 价值函数损失
    value_loss = F.mse_loss(value_pred, returns)

    # 3. 熵奖励（鼓励探索）
    entropy_bonus = -policy_log_probs.mean()

    # 总损失
    total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus

    return total_loss
```

**KL散度惩罚**:
```python
def kl_penalty(log_probs, ref_log_probs):
    """
    计算KL散度惩罚，防止策略偏离参考策略太远
    """
    kl_div = log_probs - ref_log_probs
    return kl_div.mean()

# 在总损失中添加
total_loss = ppo_loss + kl_coeff * kl_penalty
```

---

### 3. DPO (Direct Preference Optimization)

#### 3.1 DPO原理

**核心思想**: 直接优化偏好数据，无需显式的奖励模型

**推导**:
```
传统RLHF:
1. 训练奖励模型 R(x,y)
2. 用RL优化 max E[R(x,y)]

DPO:
直接优化策略，使得:
π(y_chosen|x) / π(y_rejected|x) ∝ exp(R(x,y_chosen) - R(x,y_rejected))

即: 最大化 log(π(y_chosen|x)) - log(π(y_rejected|x))
```

**DPO损失函数**:
```python
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """
    DPO损失函数

    目标: 提高chosen的log_prob，降低rejected的log_prob
    """
    # 策略模型的log概率差
    policy_logratios = policy_chosen_logps - policy_rejected_logps

    # 参考模型的log概率差
    ref_logratios = ref_chosen_logps - ref_rejected_logps

    # DPO损失
    losses = -F.logsigmoid(beta * (policy_logratios - ref_logratios))

    # 标签（chosen > rejected）
    labels = torch.zeros(losses.size())

    # 交叉熵损失
    loss = F.binary_cross_entropy_with_logits(
        -beta * (policy_logratios - ref_logratios),
        labels
    )

    # 准确率
    acc = (policy_chosen_logps > policy_rejected_logps).float().mean()

    return loss, acc
```

**DPO vs RLHF对比**:

| 维度 | RLHF | DPO |
|------|------|-----|
| **奖励模型** | 需要单独训练 | 无需 |
| **强化学习** | 需要PPO等RL算法 | 直接优化 |
| **训练稳定性** | 较难调参 | 更稳定 |
| **计算效率** | 较低（多阶段） | 较高 |
| **效果** | SOTA | 接近RLHF |
| **适用场景** | 生产环境 | 研究/快速迭代 |

#### 3.2 DPO实现

**完整训练循环**:
```python
def dpo_train_step(model, ref_model, batch, beta=0.1):
    """
    DPO单步训练
    """
    chosen_input_ids = batch['chosen_input_ids']
    rejected_input_ids = batch['rejected_input_ids']

    # 策略模型前向传播
    policy_chosen_logits = model(chosen_input_ids)
    policy_rejected_logits = model(rejected_input_ids)

    # 参考模型前向传播（不计算梯度）
    with torch.no_grad():
        ref_chosen_logits = ref_model(chosen_input_ids)
        ref_rejected_logits = ref_model(rejected_input_ids)

    # 计算log概率
    policy_chosen_logps = F.log_softmax(policy_chosen_logits, dim=-1)
    policy_rejected_logps = F.log_softmax(policy_rejected_logits, dim=-1)
    ref_chosen_logps = F.log_softmax(ref_chosen_logits, dim=-1)
    ref_rejected_logps = F.log_softmax(ref_rejected_logits, dim=-1)

    # 计算DPO损失
    loss, accuracy = dpo_loss(
        policy_chosen_logps, policy_rejected_logps,
        ref_chosen_logps, ref_rejected_logps,
        beta=beta
    )

    return loss, accuracy
```

---

### 4. 其他对齐方法

#### 4.1 ORPO (Odds Ratio Preference Optimization)

**原理**: 在SFT损失基础上添加偏好项

**损失函数**:
```python
def orpo_loss(policy_chosen_logps, policy_rejected_logps, beta=0.1):
    """
    ORPO损失: SFT损失 + 偏好损失
    """
    # SFT损失（标准语言建模损失）
    sft_loss = -(policy_chosen_logps.mean())

    # 偏好损失（odds ratio）
    log_odds = (policy_chosen_logps - policy_rejected_logps).exp()
    preference_loss = -log_odds.log().mean() * beta

    # 总损失
    total_loss = sft_loss + preference_loss

    return total_loss
```

#### 4.2 Constitutional AI (CAI)

**原理**: 基于原则的自我批评和修正

**两阶段**:
```python
# 阶段1: 批评
critique_prompt = f"""
根据以下原则批评回复:
原则: {constitution}
回复: {response}

批评:
"""

# 阶段2: 修正
revision_prompt = f"""
根据批评修改回复:
原始回复: {response}
批评: {critique}

修改后的回复:
"""

# 迭代优化
for _ in range(num_rounds):
    critique = model.generate(critique_prompt)
    revised_response = model.generate(revision_prompt)
```

---

## 🔧 实战案例

### 案例1: 完整RLHF训练流程

```python
class RLHFTrainer:
    """完整的RLHF训练器"""

    def __init__(self, policy_model, ref_model, reward_model):
        self.policy = policy_model
        self.ref_model = ref_model
        self.reward_model = reward_model

        # PPO超参数
        self.clip_param = 0.2
        self.kl_coeff = 0.02

    def generate_responses(self, prompts):
        """生成响应"""
        responses = []
        for prompt in prompts:
            response = self.policy.generate(prompt, max_length=256)
            responses.append(response)
        return responses

    def compute_rewards(self, prompts, responses):
        """计算奖励"""
        inputs = [f"{p}{r}" for p, r in zip(prompts, responses)]
        rewards = self.reward_model(inputs)
        return rewards

    def ppo_step(self, prompts, old_responses, old_log_probs):
        """PPO单步更新"""
        # 生成新响应
        new_responses = self.generate_responses(prompts)
        new_log_probs = self.policy.get_log_probs(prompts, new_responses)

        # 计算奖励
        rewards = self.compute_rewards(prompts, new_responses)

        # 计算优势
        advantages = compute_advantages(rewards, values)

        # 计算KL散度
        ref_log_probs = self.ref_model.get_log_probs(prompts, new_responses)
        kl_div = kl_penalty(new_log_probs, ref_log_probs)

        # PPO损失
        policy_loss = ppo_loss(
            new_log_probs, old_log_probs,
            advantages, self.clip_param
        )

        # 总损失
        total_loss = policy_loss + self.kl_coeff * kl_div

        return total_loss

    def train(self, dataset, num_epochs=3):
        """完整训练循环"""
        for epoch in range(num_epochs):
            for batch in dataset:
                # PPO更新
                loss = self.ppo_step(batch['prompts'], batch['responses'],
                                    batch['log_probs'])

                # 反向传播
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### 案例2: DPO训练器

```python
class DPOTrainer:
    """DPO训练器"""

    def __init__(self, policy_model, ref_model, beta=0.1):
        self.policy = policy_model
        self.ref_model = ref_model
        self.beta = beta

    def train_step(self, batch):
        """DPO单步训练"""
        # 提取chosen和rejected
        chosen = batch['chosen']
        rejected = batch['rejected']

        # 策略模型log概率
        policy_chosen_logps = self.policy.get_log_probs(chosen)
        policy_rejected_logps = self.policy.get_log_probs(rejected)

        # 参考模型log概率（无梯度）
        with torch.no_grad():
            ref_chosen_logps = self.ref_model.get_log_probs(chosen)
            ref_rejected_logps = self.ref_model.get_log_probs(rejected)

        # DPO损失
        loss, accuracy = dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps, ref_rejected_logps,
            self.beta
        )

        return loss, accuracy

    def train(self, dataset, num_epochs=3):
        """完整训练循环"""
        optimizer = torch.optim.AdamW(self.policy.parameters(), lr=1e-5)

        for epoch in range(num_epochs):
            total_loss = 0
            total_acc = 0

            for batch in dataset:
                loss, acc = self.train_step(batch)

                # 反向传播
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                total_acc += acc.item()

            avg_loss = total_loss / len(dataset)
            avg_acc = total_acc / len(dataset)

            print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Acc={avg_acc:.2%}")
```

---

## 💡 实现技巧

### 1. 偏好数据收集

**质量优于数量**:
```python
# 好的偏好数据示例
prompt = "解释气候变化的原因"

chosen = """
气候变化主要由温室气体排放引起。包括:
1. 二氧化碳：化石燃料燃烧
2. 甲烷：畜牧业、垃圾填埋
3. 氧化亚氮：农业活动
这些气体在大气中形成温室效应，导致全球变暖。
"""

rejected = """
天气变热是因为太阳活动频繁。还有人说这是自然现象，
不需要担心。我觉得大家太夸张了，夏天热很正常。
"""

# 特点:
# - chosen: 结构清晰、事实准确、有逻辑
# - rejected: 信息错误、观点混乱、缺乏依据
```

**数据收集策略**:
```python
# 1. 从多个来源收集
sources = [
    "human_annotations",      # 人工标注（高质量）
    "model_comparison",       # 模型生成对比
    "user_feedback",          # 用户反馈（真实场景）
]

# 2. 平衡难度分布
difficulty_levels = ["easy", "medium", "hard"]
for level in difficulty_levels:
    collect_data(difficulty=level)

# 3. 覆盖多种任务
task_types = [
    "question_answering",
    "summarization",
    "creative_writing",
    "coding",
]
```

### 2. 奖励模型训练技巧

**技巧1: 预训练初始化**
```python
# 从SFT模型初始化奖励模型
reward_model = AutoModel.from_pretrained("sft-model-checkpoint")
reward_model.reward_head = nn.Linear(hidden_size, 1)
```

**技巧2: 数据增强**
```python
def augment_pair(chosen, rejected):
    """数据增强"""
    # 同义词替换
    augmented_chosen = synonym_replace(chosen)

    # 回译
    augmented_chosen = back_translate(chosen)

    # 添加噪声
    augmented_rejected = add_noise(rejected)

    return augmented_chosen, augmented_rejected
```

**技巧3: 损失加权**
```python
def weighted_reward_loss(reward_chosen, reward_rejected, margin):
    """
    加权损失，强调高置信度样本
    """
    diff = reward_chosen - reward_rejected
    weight = torch.sigmoid(diff / margin)
    loss = -weight * F.logsigmoid(diff).mean()
    return loss
```

### 3. PPO训练稳定性

**技巧1: 梯度裁剪**
```python
# 防止梯度爆炸
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**技巧2: 学习率调度**
```python
# 余弦退火
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs, eta_min=1e-6
)
```

**技巧3: 值函数归一化**
```python
# 归一化优势函数
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

---

## 🎯 学习检验

### 关键问题

1. **对齐基础**:
   - 什么是对齐问题？为什么重要？
   - RLHF vs DPO的区别？
   - 如何收集高质量的偏好数据？

2. **RLHF算法**:
   - RLHF的三个阶段？
   - 奖励模型如何训练？
   - PPO的目标函数？

3. **DPO算法**:
   - DPO的核心思想？
   - 如何直接优化偏好？
   - DPO的优势和局限？

4. **实践应用**:
   - 如何设计完整的RLHF训练流程？
   - 如何提高训练稳定性？
   - 如何评估对齐效果？

### 代码练习

完成 [examples.py](examples.py) 中的练习题。

---

## 📖 延伸阅读

**论文**:
- "Training Language Models to Follow Instructions with Human Feedback" (InstructGPT)
- "Constitutional AI: Harmlessness from AI Feedback" (Anthropic)
- "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
- "Learning to Summarize with Human Feedback" (RLHF开创性工作)

**代码参考**:
- [CS336 Assignment 5](references/github/assignment5-alignment/)
- [Transformer Reinforcement Learning](https://github.com/lucidrains/trans-rlhf)
- [trl (Hugging Face)](https://github.com/huggingface/trl)

---

## ⚠️ 常见陷阱

1. **偏好数据质量**:
   - ❌ 使用低质量自动生成的偏好对
   - ❌ 标注不一致（不同标注员标准不同）
   - ✅ 严格的人工标注流程
   - ✅ 定期验证标注质量

2. **奖励模型过拟合**:
   - ❌ 在少量数据上训练太多epoch
   - ❌ 奖励值分布异常（过大或过小）
   - ✅ 早停（Early Stopping）
   - ✅ 在验证集上监控

3. **PPO训练不稳定**:
   - ❌ KL系数太大（策略更新太保守）
   - ❌ 学习率太大（策略崩溃）
   - ✅ 渐进式增加KL系数
   - ✅ 使用参考策略约束

4. **DPO实现错误**:
   - ❌ 参考模型没有冻结
   - ❌ beta参数设置不当
   - ✅ 参考模型eval()模式
   - ✅ beta在[0.1, 0.5]范围

---

## 🚀 下一步

完成Assignment 5后，你应该掌握：
- ✅ 理解对齐问题的核心挑战
- ✅ 掌握RLHF完整训练流程
- ✅ 理解DPO的原理和实现
- ✅ 能够独立训练对齐模型

**推荐项目**:
1. 在小模型上实现完整RLHF流程
2. 对比RLHF和DPO的效果
3. 研究Constitutional AI在中文场景的应用

**下一步**: [Week 5: Agent架构](../../Week5_Agents/README.md)
