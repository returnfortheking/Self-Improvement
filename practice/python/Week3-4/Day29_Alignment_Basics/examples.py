"""
Day 29: Assignment 5 - Alignment & RLHF - 代码示例
涵盖：奖励模型、PPO算法、DPO、偏好数据处理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import List, Dict, Tuple
import numpy as np


# ============================================================================
# Part 1: 奖励模型 (Reward Model)
# ============================================================================

def example_1_reward_model():
    """示例1: 奖励模型结构"""
    print("=" * 60)
    print("示例1: 奖励模型结构")
    print("=" * 60)

    class RewardModel(nn.Module):
        """简单的奖励模型"""

        def __init__(self, vocab_size=1000, hidden_size=128):
            super().__init__()
            # 嵌入层
            self.embedding = nn.Embedding(vocab_size, hidden_size)

            # 编码器（简化版Transformer）
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4),
                num_layers=2
            )

            # 奖励头（输出标量）
            self.reward_head = nn.Linear(hidden_size, 1)

        def forward(self, input_ids):
            # 嵌入
            x = self.embedding(input_ids)  # [batch, seq, hidden]

            # 编码
            x = x.transpose(0, 1)  # [seq, batch, hidden]
            x = self.encoder(x)
            x = x.transpose(0, 1)  # [batch, seq, hidden]

            # 取最后一个token的hidden state
            last_hidden = x[:, -1, :]  # [batch, hidden]

            # 计算奖励
            reward = self.reward_head(last_hidden)  # [batch, 1]

            return reward.squeeze(-1)  # [batch]

    # 创建模型
    model = RewardModel()

    # 模拟输入
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))

    # 前向传播
    rewards = model(input_ids)

    print(f"输入形状: {input_ids.shape}")
    print(f"奖励形状: {rewards.shape}")
    print(f"奖励值: {rewards.detach().numpy()}")


def example_2_reward_training():
    """示例2: 奖励模型训练"""
    print("\n" + "=" * 60)
    print("示例2: 奖励模型训练")
    print("=" * 60)

    def reward_loss(reward_chosen: torch.Tensor,
                   reward_rejected: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        计算奖励模型损失（Bradley-Terry模型）

        目标: 使chosen的奖励 > rejected的奖励
        """
        # Log概率: chosen更优的概率
        log_prob = F.logsigmoid(reward_chosen - reward_rejected)

        # 损失（最大化log_prob）
        loss = -log_prob.mean()

        # 准确率
        accuracy = (reward_chosen > reward_rejected).float().mean()

        return loss, accuracy.item()

    # 模拟数据
    batch_size = 4

    # 模型预测的奖励（训练前）
    reward_chosen = torch.randn(batch_size) * 0.1  # chosen奖励
    reward_rejected = torch.randn(batch_size) * 0.1  # rejected奖励

    print("训练前:")
    print(f"Chosen奖励:  {reward_chosen.detach().numpy()}")
    print(f"Rejected奖励: {reward_rejected.detach().numpy()}")

    # 计算损失
    loss, accuracy = reward_loss(reward_chosen, reward_rejected)
    print(f"\n初始损失: {loss.item():.4f}")
    print(f"初始准确率: {accuracy:.2%}")

    # 模拟训练（手动调整奖励值）
    optimizer = torch.optim.SGD([reward_chosen, reward_rejected], lr=0.1)

    for step in range(10):
        optimizer.zero_grad()

        loss, acc = reward_loss(reward_chosen, reward_rejected)

        loss.backward()
        optimizer.step()

        if (step + 1) % 5 == 0:
            print(f"\nStep {step + 1}:")
            print(f"损失: {loss.item():.4f}, 准确率: {acc:.2%}")
            print(f"Chosen奖励:  {reward_chosen.detach().numpy()}")
            print(f"Rejected奖励: {reward_rejected.detach().numpy()}")


# ============================================================================
# Part 2: PPO算法
# ============================================================================

def example_3_ppo_algorithm():
    """示例3: PPO核心算法"""
    print("\n" + "=" * 60)
    print("示例3: PPO核心算法")
    print("=" * 60)

    def ppo_loss(policy_log_probs: torch.Tensor,
                 old_policy_log_probs: torch.Tensor,
                 advantages: torch.Tensor,
                 clip_param: float = 0.2) -> torch.Tensor:
        """
        计算PPO裁剪损失

        Args:
            policy_log_probs: 新策略的log概率
            old_policy_log_probs: 旧策略的log概率
            advantages: 优势函数
            clip_param: 裁剪参数
        """
        # 概率比率
        ratio = torch.exp(policy_log_probs - old_policy_log_probs)

        # 裁剪前的目标
        surr1 = ratio * advantages

        # 裁剪后的目标
        surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * advantages

        # PPO损失（取最小值，因为我们在最大化）
        policy_loss = -torch.min(surr1, surr2).mean()

        return policy_loss

    # 模拟数据
    batch_size = 8
    policy_log_probs = torch.randn(batch_size) * 0.1
    old_policy_log_probs = torch.randn(batch_size) * 0.1
    advantages = torch.randn(batch_size)

    # 计算PPO损失
    loss = ppo_loss(policy_log_probs, old_policy_log_probs, advantages)

    print(f"PPO损失: {loss.item():.4f}")
    print(f"\n优势函数统计:")
    print(f"均值: {advantages.mean():.4f}")
    print(f"标准差: {advantages.std():.4f}")


def example_4_advantage_computation():
    """示例4: 优势函数计算（GAE）"""
    print("\n" + "=" * 60)
    print("示例4: 优势函数计算（GAE）")
    print("=" * 60)

    def compute_gae(rewards: List[float],
                   values: List[float],
                   gamma: float = 0.99,
                   lambda_gae: float = 0.95) -> np.ndarray:
        """
        使用GAE (Generalized Advantage Estimation)计算优势

        Args:
            rewards: 奖励序列
            values: 价值函数估计
            gamma: 折扣因子
            lambda_gae: GAE参数
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

            # GAE累积
            gae = delta + gamma * lambda_gae * gae
            advantages.insert(0, gae)

        return np.array(advantages)

    # 模拟轨迹
    rewards = [1.0, 2.0, 1.5, -0.5, 3.0]
    values = [0.8, 1.5, 1.2, 0.5, 2.0]

    # 计算优势
    advantages = compute_gae(rewards, values)

    print("奖励序列:", rewards)
    print("价值估计:", values)
    print("优势函数:", advantages)
    print(f"优势均值: {advantages.mean():.4f}")


# ============================================================================
# Part 3: DPO (Direct Preference Optimization)
# ============================================================================

def example_5_dpo_loss():
    """示例5: DPO损失函数"""
    print("\n" + "=" * 60)
    print("示例5: DPO损失函数")
    print("=" * 60)

    def dpo_loss(policy_chosen_logps: torch.Tensor,
                policy_rejected_logps: torch.Tensor,
                ref_chosen_logps: torch.Tensor,
                ref_rejected_logps: torch.Tensor,
                beta: float = 0.1) -> Tuple[torch.Tensor, float]:
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

        # 平均损失
        loss = losses.mean()

        # 准确率
        accuracy = (policy_chosen_logps > policy_rejected_logps).float().mean()

        return loss, accuracy.item()

    # 模拟数据
    batch_size = 4
    policy_chosen_logps = torch.randn(batch_size) * 0.5 + 2.0  # 较高
    policy_rejected_logps = torch.randn(batch_size) * 0.5 + 1.0  # 较低
    ref_chosen_logps = torch.randn(batch_size) * 0.5 + 1.8
    ref_rejected_logps = torch.randn(batch_size) * 0.5 + 1.2

    # 计算DPO损失
    loss, accuracy = dpo_loss(
        policy_chosen_logps, policy_rejected_logps,
        ref_chosen_logps, ref_rejected_logps,
        beta=0.1
    )

    print(f"DPO损失: {loss.item():.4f}")
    print(f"准确率: {accuracy:.2%}")
    print(f"\n策略模型log概率差: {(policy_chosen_logps - policy_rejected_logps).mean():.4f}")
    print(f"参考模型log概率差: {(ref_chosen_logps - ref_rejected_logps).mean():.4f}")


def example_6_dpo_training_step():
    """示例6: DPO训练步骤"""
    print("\n" + "=" * 60)
    print("示例6: DPO训练步骤")
    print("=" * 60)

    class SimpleDPOTrainer:
        """简化的DPO训练器"""

        def __init__(self, beta=0.1):
            self.beta = beta

            # 模拟模型参数
            self.policy_weights = torch.randn(100, requires_grad=True)
            self.ref_weights = torch.randn(100, requires_grad=False)

        def forward(self, x, weights):
            """简化的前向传播"""
            return (x * weights).sum(dim=-1)

        def train_step(self, chosen_input, rejected_input):
            """单步训练"""
            # 策略模型前向
            policy_chosen = self.forward(chosen_input, self.policy_weights)
            policy_rejected = self.forward(rejected_input, self.policy_weights)

            # 参考模型前向（无梯度）
            with torch.no_grad():
                ref_chosen = self.forward(chosen_input, self.ref_weights)
                ref_rejected = self.forward(rejected_input, self.ref_weights)

            # DPO损失
            policy_logratios = policy_chosen - policy_rejected
            ref_logratios = ref_chosen - ref_rejected

            loss = -F.logsigmoid(self.beta * (policy_logratios - ref_logratios)).mean()

            # 准确率
            accuracy = (policy_chosen > policy_rejected).float().mean()

            return loss, accuracy

    # 创建训练器
    trainer = SimpleDPOTrainer(beta=0.1)

    # 模拟数据
    batch_size = 4
    chosen_input = torch.randn(batch_size, 100)
    rejected_input = torch.randn(batch_size, 100)

    # 训练
    optimizer = torch.optim.AdamW([trainer.policy_weights], lr=1e-3)

    print("训练过程:")
    for step in range(5):
        optimizer.zero_grad()

        loss, accuracy = trainer.train_step(chosen_input, rejected_input)

        loss.backward()
        optimizer.step()

        print(f"Step {step + 1}: Loss={loss.item():.4f}, Acc={accuracy:.2%}")


# ============================================================================
# Part 4: 偏好数据处理
# ============================================================================

def example_7_preference_dataset():
    """示例7: 偏好数据集处理"""
    print("\n" + "=" * 60)
    print("示例7: 偏好数据集处理")
    print("=" * 60)

    class PreferenceDataset:
        """偏好数据集"""

        def __init__(self, data: List[Dict]):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

        def collate_fn(self, batch: List[Dict]):
            """批量处理函数"""
            # 提取字段
            prompts = [item['prompt'] for item in batch]
            chosen = [item['chosen'] for item in batch]
            rejected = [item['rejected'] for item in batch]

            # 模拟tokenization（实际需要真实的tokenizer）
            chosen_input_ids = [self._tokenize(text) for text in chosen]
            rejected_input_ids = [self._tokenize(text) for text in rejected]

            return {
                'prompts': prompts,
                'chosen_input_ids': chosen_input_ids,
                'rejected_input_ids': rejected_input_ids,
            }

        def _tokenize(self, text: str) -> List[int]:
            """简化的tokenization"""
            # 实际使用tokenizer(text)['input_ids']
            return [hash(word) % 1000 for word in text.split()][:50]

    # 创建示例数据
    data = [
        {
            'prompt': '什么是机器学习？',
            'chosen': '机器学习是人工智能的一个分支，它使计算机能够从数据中学习并改进。',
            'rejected': '机器学习就是电脑自己学东西，我也不太清楚。'
        },
        {
            'prompt': '解释深度学习',
            'chosen': '深度学习是机器学习的一种方法，使用多层神经网络来学习数据的层次表示。',
            'rejected': '深度学习就是很深的学习，要学很久。'
        },
    ]

    # 创建数据集
    dataset = PreferenceDataset(data)

    print(f"数据集大小: {len(dataset)}")
    print(f"\n示例数据:")
    print(f"Prompt: {data[0]['prompt']}")
    print(f"Chosen: {data[0]['chosen']}")
    print(f"Rejected: {data[0]['rejected']}")

    # 模拟批量处理
    batch = [dataset[0], dataset[1]]
    collated = dataset.collate_fn(batch)

    print(f"\n批处理后:")
    print(f"Prompts: {collated['prompts']}")


# ============================================================================
# Part 5: 完整RLHF训练循环
# ============================================================================

def example_8_rlhf_training_loop():
    """示例8: 简化的RLHF训练循环"""
    print("\n" + "=" * 60)
    print("示例8: 简化的RLHF训练循环")
    print("=" * 60)

    class SimpleRLHFTrainer:
        """简化的RLHF训练器"""

        def __init__(self):
            # 模拟模型
            self.policy_model = nn.Linear(10, 10)
            self.ref_model = nn.Linear(10, 10)
            self.reward_model = nn.Linear(10, 1)

            # 复制参考模型
            self.ref_model.load_state_dict(self.policy_model.state_dict())
            self.ref_model.eval()

            # 优化器
            self.policy_optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=1e-3)
            self.reward_optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=1e-3)

        def generate_response(self, prompt):
            """生成响应（简化版）"""
            with torch.no_grad():
                response = self.policy_model(prompt)
            return response

        def compute_reward(self, prompt_response):
            """计算奖励"""
            reward = self.reward_model(prompt_response)
            return reward

        def train_reward_model(self, batch):
            """训练奖励模型"""
            # chosen和rejected的奖励
            reward_chosen = self.compute_reward(batch['chosen'])
            reward_rejected = self.compute_reward(batch['rejected'])

            # Bradley-Terry损失
            loss = -F.logsigmoid(reward_chosen - reward_rejected).mean()

            # 反向传播
            self.reward_optimizer.zero_grad()
            loss.backward()
            self.reward_optimizer.step()

            return loss.item()

        def train_policy(self, prompts):
            """训练策略（PPO简化版）"""
            # 生成响应
            responses = self.generate_response(prompts)

            # 计算奖励
            rewards = self.compute_reward(responses)

            # 简化的策略损失（最大化奖励）
            loss = -rewards.mean()

            # 反向传播
            self.policy_optimizer.zero_grad()
            loss.backward()
            self.policy_optimizer.step()

            return loss.item(), rewards.mean().item()

    # 创建训练器
    trainer = SimpleRLHFTrainer()

    # 模拟数据
    prompts = torch.randn(4, 10)
    chosen = torch.randn(4, 10)
    rejected = torch.randn(4, 10)

    batch = {'chosen': chosen, 'rejected': rejected}

    # 训练循环
    print("RLHF训练:")
    for epoch in range(3):
        # 阶段1: 训练奖励模型
        reward_loss = trainer.train_reward_model(batch)

        # 阶段2: 训练策略
        policy_loss, avg_reward = trainer.train_policy(prompts)

        print(f"Epoch {epoch + 1}:")
        print(f"  奖励模型损失: {reward_loss:.4f}")
        print(f"  策略损失: {policy_loss:.4f}")
        print(f"  平均奖励: {avg_reward:.4f}")


# ============================================================================
# 运行所有示例
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Assignment 5: Alignment & RLHF - 代码示例")
    print("=" * 60)

    example_1_reward_model()
    example_2_reward_training()
    example_3_ppo_algorithm()
    example_4_advantage_computation()
    example_5_dpo_loss()
    example_6_dpo_training_step()
    example_7_preference_dataset()
    example_8_rlhf_training_loop()

    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)

    print("\n关键要点:")
    print("1. 奖励模型: 学习人类偏好，输出标量奖励")
    print("2. PPO算法: 在信任区域内优化策略")
    print("3. 优势函数: 使用GAE计算，衡量动作价值")
    print("4. DPO: 直接优化偏好，无需奖励模型")
    print("5. 偏好数据: 需要高质量的chosen-rejected对")
    print("6. RLHF流程: SFT → 奖励模型 → PPO训练")
