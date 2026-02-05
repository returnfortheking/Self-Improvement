"""
Day 29: Assignment 5 - Alignment & RLHF - 练习题
难度：⭐⭐ ~ ⭐⭐⭐⭐⭐
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
import numpy as np


# ============================================================================
# 练习1: 实现奖励模型（⭐⭐）
# ============================================================================

def exercise_1_reward_model():
    """
    练习1: 实现简单的奖励模型

    任务:
    1. 完成RewardModel类的__init__方法
    2. 完成forward方法，输入文本序列，输出奖励值
    3. 实现compute_reward_loss函数（Bradley-Terry模型）

    提示:
    - 奖励模型基于Transformer编码器
    - 输出是标量奖励值（单个数值）
    - 损失函数: -log(sigmoid(r_chosen - r_rejected))
    """
    class RewardModel(nn.Module):
        def __init__(self, vocab_size: int, hidden_size: int):
            super().__init__()
            # TODO: 初始化模型层
            # - embedding层
            # - encoder层（使用nn.TransformerEncoder）
            # - reward_head（输出层，输出1个值）
            raise NotImplementedError

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
            """
            前向传播

            Args:
                input_ids: [batch, seq_len]
                attention_mask: [batch, seq_len]

            Returns:
                rewards: [batch]
            """
            # TODO: 实现前向传播
            # 1. 嵌入
            # 2. 编码
            # 3. 取最后一个token的hidden state
            # 4. 通过reward_head输出奖励
            raise NotImplementedError

    def compute_reward_loss(reward_chosen: torch.Tensor,
                          reward_rejected: torch.Tensor) -> torch.Tensor:
        """
        计算奖励模型损失

        Args:
            reward_chosen: [batch] chosen样本的奖励
            reward_rejected: [batch] rejected样本的奖励

        Returns:
            loss: 标量损失
        """
        # TODO: 实现Bradley-Terry损失
        # 目标: 使chosen的奖励 > rejected的奖励
        raise NotImplementedError

    # 测试
    model = RewardModel(vocab_size=1000, hidden_size=128)
    reward_chosen = torch.randn(4)
    reward_rejected = torch.randn(4)

    loss = compute_reward_loss(reward_chosen, reward_rejected)
    print(f"练习1 - 奖励模型损失: {loss.item():.4f}")


# ============================================================================
# 练习2: 实现PPO算法（⭐⭐⭐⭐）
# ============================================================================

def exercise_2_ppo_algorithm():
    """
    练习2: 实现PPO核心算法

    任务:
    1. 实现compute_advantages函数（GAE算法）
    2. 实现ppo_loss函数（裁剪目标函数）
    3. 理解PPO的约束机制（为什么需要裁剪）

    提示:
    - GAE: A_t = δ_t + γλδ_{t+1} + ...，其中δ_t = r_t + γV(s_{t+1}) - V(s_t)
    - PPO裁剪: L = min(r*A, clip(r, 1-ε, 1+ε)*A)，其中r = π_new/π_old
    - ε通常设置为0.2
    """
    def compute_advantages(rewards: List[float],
                         values: List[float],
                         gamma: float = 0.99,
                         lambda_gae: float = 0.95) -> np.ndarray:
        """
        使用GAE计算优势函数

        Args:
            rewards: 奖励序列 [r_0, r_1, ..., r_T]
            values: 价值函数估计 [V_0, V_1, ..., V_T]
            gamma: 折扣因子
            lambda_gae: GAE参数

        Returns:
            advantages: 优势函数 [A_0, A_1, ..., A_T]
        """
        # TODO: 实现GAE算法
        # 1. 从后向前遍历
        # 2. 计算TD残差: δ_t = r_t + γ*V_{t+1} - V_t
        # 3. 累积GAE: A_t = δ_t + γ*λ*A_{t+1}
        raise NotImplementedError

    def ppo_loss(policy_log_probs: torch.Tensor,
                old_policy_log_probs: torch.Tensor,
                advantages: torch.Tensor,
                clip_param: float = 0.2) -> torch.Tensor:
        """
        计算PPO裁剪损失

        Args:
            policy_log_probs: 新策略的log概率 [batch]
            old_policy_log_probs: 旧策略的log概率 [batch]
            advantages: 优势函数 [batch]
            clip_param: 裁剪参数

        Returns:
            loss: PPO损失
        """
        # TODO: 实现PPO损失
        # 1. 计算概率比率: r = exp(log_p_new - log_p_old)
        # 2. 计算裁剪前的目标: surr1 = r * A
        # 3. 计算裁剪后的目标: surr2 = clip(r, 1-ε, 1+ε) * A
        # 4. 返回最小值的负平均
        raise NotImplementedError

    # 测试
    rewards = [1.0, 2.0, 1.5, -0.5, 3.0]
    values = [0.8, 1.5, 1.2, 0.5, 2.0]

    advantages = compute_advantages(rewards, values)
    print(f"练习2 - 优势函数: {advantages}")

    policy_log_probs = torch.randn(8)
    old_policy_log_probs = torch.randn(8)
    adv_tensor = torch.randn(8)

    loss = ppo_loss(policy_log_probs, old_policy_log_probs, adv_tensor)
    print(f"练习2 - PPO损失: {loss.item():.4f}")


# ============================================================================
# 练习3: 实现DPO训练（⭐⭐⭐⭐）
# ============================================================================

def exercise_3_dpo_training():
    """
    练习3: 实现DPO (Direct Preference Optimization)训练

    任务:
    1. 实现dpo_loss函数
    2. 理解DPO如何避免显式奖励模型
    3. 实现简单的DPO训练循环

    提示:
    - DPO直接优化: log(π_chosen) - log(π_rejected)
    - 参考模型提供约束: log(π_ref_chosen) - log(π_ref_rejected)
    - 损失: -log(σ(β*(log_ratio_chosen - log_ratio_rejected)))
    - β通常在[0.1, 0.5]范围
    """
    def dpo_loss(policy_chosen_logps: torch.Tensor,
                policy_rejected_logps: torch.Tensor,
                ref_chosen_logps: torch.Tensor,
                ref_rejected_logps: torch.Tensor,
                beta: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算DPO损失

        Args:
            policy_chosen_logps: 策略模型对chosen的log概率
            policy_rejected_logps: 策略模型对rejected的log概率
            ref_chosen_logps: 参考模型对chosen的log概率
            ref_rejected_logps: 参考模型对rejected的log概率
            beta: 温度参数

        Returns:
            loss: DPO损失
            accuracy: 准确率（chosen奖励 > rejected奖励的比例）
        """
        # TODO: 实现DPO损失
        # 1. 计算策略模型的log概率差
        # 2. 计算参考模型的log概率差
        # 3. 计算损失: -log(sigmoid(β*(diff_policy - diff_ref)))
        # 4. 计算准确率
        raise NotImplementedError

    class SimpleDPOTrainer:
        """简化的DPO训练器"""

        def __init__(self, beta: float = 0.1):
            self.beta = beta
            # 模拟模型权重
            self.policy_weights = nn.Parameter(torch.randn(100))
            self.ref_weights = torch.randn(100)  # 冻结的参考模型

        def forward(self, inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            """简化的前向传播"""
            return (inputs * weights).sum(dim=-1)

        def train_step(self, chosen_input: torch.Tensor, rejected_input: torch.Tensor):
            """单步训练"""
            # TODO: 实现训练步骤
            # 1. 策略模型前向传播
            # 2. 参考模型前向传播（无梯度）
            # 3. 计算DPO损失
            # 4. 返回损失和准确率
            raise NotImplementedError

    # 测试
    policy_chosen_logps = torch.randn(4) * 0.5 + 2.0
    policy_rejected_logps = torch.randn(4) * 0.5 + 1.0
    ref_chosen_logps = torch.randn(4) * 0.5 + 1.8
    ref_rejected_logps = torch.randn(4) * 0.5 + 1.2

    loss, accuracy = dpo_loss(
        policy_chosen_logps, policy_rejected_logps,
        ref_chosen_logps, ref_rejected_logps,
        beta=0.1
    )

    print(f"练习3 - DPO损失: {loss.item():.4f}, 准确率: {accuracy:.2%}")

    # 测试训练器
    trainer = SimpleDPOTrainer(beta=0.1)
    chosen_input = torch.randn(4, 100)
    rejected_input = torch.randn(4, 100)

    loss, acc = trainer.train_step(chosen_input, rejected_input)
    print(f"练习3 - 训练步骤损失: {loss.item():.4f}, 准确率: {acc:.2%}")


# ============================================================================
# 练习4: 偏好数据处理（⭐⭐⭐）
# ============================================================================

def exercise_4_preference_dataset():
    """
    练习4: 实现偏好数据集处理

    任务:
    1. 完成PreferenceDataset类的collate_fn方法
    2. 实现数据验证函数（检查数据质量）
    3. 实现数据增强函数（可选）

    提示:
    - 偏好数据格式: {prompt, chosen, rejected}
    - 需要对齐长度（padding/truncation）
    - 数据验证: 检查chosen和rejected是否有意义
    """
    class PreferenceDataset:
        """偏好数据集"""

        def __init__(self, data: List[Dict], max_length: int = 512):
            self.data = data
            self.max_length = max_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

        def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
            """
            批量处理函数

            Args:
                batch: 批量数据，每项包含 {prompt, chosen, rejected}

            Returns:
                collated_batch: 处理后的批量数据
            """
            # TODO: 实现批量处理
            # 1. 提取prompts, chosen, rejected
            # 2. 模拟tokenization（这里简化处理）
            # 3. Padding到相同长度
            # 4. 返回字典格式的数据
            raise NotImplementedError

        def validate_data(self) -> Tuple[int, int]:
            """
            验证数据质量

            Returns:
                (valid_count, invalid_count): 有效和无效样本数量
            """
            # TODO: 实现数据验证
            # 检查:
            # - chosen和rejected不能相同
            # - 文本长度合理（不为空，不太长）
            # - 包含有效内容
            raise NotImplementedError

    # 测试数据
    data = [
        {
            'prompt': '什么是机器学习？',
            'chosen': '机器学习是人工智能的一个分支。',
            'rejected': '我不知道。'
        },
        {
            'prompt': '解释深度学习',
            'chosen': '深度学习使用多层神经网络。',
            'rejected': '深度学习使用多层神经网络。'  # 无效：与chosen相同
        },
    ]

    dataset = PreferenceDataset(data)
    valid, invalid = dataset.validate_data()

    print(f"练习4 - 有效样本: {valid}, 无效样本: {invalid}")

    # 测试批量处理
    batch = [dataset[0]]
    collated = dataset.collate_fn(batch)
    print(f"练习4 - 批量处理完成，keys: {list(collated.keys())}")


# ============================================================================
# 练习5: 完整RLHF训练流程（⭐⭐⭐⭐⭐）
# ============================================================================

def exercise_5_rlhf_pipeline():
    """
    练习5: 实现完整的RLHF训练流程

    任务:
    1. 实现三阶段训练流程
       - 阶段1: 监督微调（SFT）
       - 阶段2: 奖励模型训练
       - 阶段3: PPO强化学习
    2. 理解每个阶段的作用和数据需求
    3. 实现评估函数（生成质量评估）

    提示:
    - SFT: 标准的语言模型训练，使用指令数据
    - 奖励模型: 成对排序损失，需要偏好数据
    - PPO: 生成→打分→更新，需要奖励模型
    - KL惩罚: 防止策略偏离SFT模型太远
    """
    class SimpleRLHFPipeline:
        """简化的RLHF训练流程"""

        def __init__(self):
            # 三个模型
            self.sft_model = nn.Linear(10, 10)
            self.reward_model = nn.Linear(10, 1)
            self.policy_model = nn.Linear(10, 10)
            self.ref_model = nn.Linear(10, 10)

            # 复制SFT模型到策略和参考模型
            self.policy_model.load_state_dict(self.sft_model.state_dict())
            self.ref_model.load_state_dict(self.sft_model.state_dict())
            self.ref_model.eval()

            # 优化器
            self.reward_optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=1e-3)
            self.policy_optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=1e-3)

        def stage1_sft(self, sft_data: List[Dict], num_epochs: int = 2):
            """
            阶段1: 监督微调

            Args:
                sft_data: SFT数据，每项包含 {prompt, response}
                num_epochs: 训练轮数
            """
            # TODO: 实现SFT训练
            # 标准的语言建模损失: -log P(response|prompt)
            raise NotImplementedError

        def stage2_reward_model(self, preference_data: List[Dict], num_epochs: int = 3):
            """
            阶段2: 训练奖励模型

            Args:
                preference_data: 偏好数据，每项包含 {prompt, chosen, rejected}
                num_epochs: 训练轮数
            """
            # TODO: 实现奖励模型训练
            # Bradley-Terry损失
            raise NotImplementedError

        def stage3_ppo(self, prompts: List[str], num_iterations: int = 5):
            """
            阶段3: PPO强化学习

            Args:
                prompts: 提示词列表
                num_iterations: PPO迭代次数
            """
            # TODO: 实现PPO训练
            # 1. 使用当前策略生成响应
            # 2. 使用奖励模型打分
            # 3. 计算PPO损失（包含KL惩罚）
            # 4. 更新策略
            raise NotImplementedError

        def evaluate(self, test_prompts: List[str]) -> Dict[str, float]:
            """
            评估生成质量

            Args:
                test_prompts: 测试提示词

            Returns:
                metrics: 评估指标
            """
            # TODO: 实现评估
            # 可以评估:
            # - 平均奖励
            # - 响应长度
            # - 多样性
            raise NotImplementedError

        def train_full_pipeline(self, sft_data, preference_data, prompts):
            """
            完整训练流程

            Args:
                sft_data: SFT数据
                preference_data: 偏好数据
                prompts: PPO阶段的提示词
            """
            print("开始完整RLHF训练流程...")

            # 阶段1: SFT
            print("\n阶段1: 监督微调（SFT）")
            self.stage1_sft(sft_data, num_epochs=2)

            # 阶段2: 奖励模型
            print("\n阶段2: 训练奖励模型")
            self.stage2_reward_model(preference_data, num_epochs=3)

            # 阶段3: PPO
            print("\n阶段3: PPO强化学习")
            self.stage3_ppo(prompts, num_iterations=5)

            # 评估
            print("\n评估")
            metrics = self.evaluate(prompts[:10])  # 在部分数据上评估
            print(f"评估指标: {metrics}")

    # 测试数据
    sft_data = [
        {'prompt': '什么是AI？', 'response': '人工智能是计算机科学的一个分支。'},
        {'prompt': '解释机器学习', 'response': '机器学习让计算机从数据中学习。'},
    ]

    preference_data = [
        {
            'prompt': '什么是AI？',
            'chosen': '人工智能是计算机科学的一个分支，致力于创造智能系统。',
            'rejected': 'AI就是电脑。'
        },
    ]

    prompts = ['什么是AI？', '解释机器学习', '深度学习是什么']

    # 运行训练流程
    pipeline = SimpleRLHFPipeline()
    # TODO: 取消注释以运行完整流程
    # pipeline.train_full_pipeline(sft_data, preference_data, prompts)

    print("练习5 - RLHF流程实现完成")


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    print("Assignment 5: Alignment & RLHF - 练习题")
    print("=" * 60)

    print("\n练习1: 奖励模型实现（⭐⭐）")
    # exercise_1_reward_model()

    print("\n练习2: PPO算法实现（⭐⭐⭐⭐）")
    # exercise_2_ppo_algorithm()

    print("\n练习3: DPO训练实现（⭐⭐⭐⭐）")
    # exercise_3_dpo_training()

    print("\n练习4: 偏好数据处理（⭐⭐⭐）")
    # exercise_4_preference_dataset()

    print("\n练习5: 完整RLHF流程（⭐⭐⭐⭐⭐）")
    # exercise_5_rlhf_pipeline()

    print("\n" + "=" * 60)
    print("提示: 逐个取消注释练习函数，完成实现后运行测试")
    print("=" * 60)
