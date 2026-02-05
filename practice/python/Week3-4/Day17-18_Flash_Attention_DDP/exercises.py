"""
Day 17-18: Flash Attention与DDP - 练习题
难度：⭐⭐⭐ ~ ⭐⭐⭐⭐⭐
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import math


# ============================================================================
# 练习1: 实现Flash Attention前向传播（⭐⭐⭐⭐⭐）
# ============================================================================

def exercise_1_flash_attention():
    """
    任务：实现Flash Attention前向传播

    要求：
    1. 分块处理Q, K, V
    2. 实现Online Softmax
    3. 返回输出和L（logsumexp）

    提示：
    - 使用两个嵌套循环遍历blocks
    - 维护m（最大值）和l（归一化因子）
    - 增量更新统计量
    """
    def flash_attention_forward(q, k, v, is_causal=False, block_size=64):
        """
        Flash Attention前向传播

        Args:
            q: [batch, n_heads, seq_len, d]
            k: [batch, n_heads, seq_len, d]
            v: [batch, n_heads, seq_len, d]
            is_causal: 是否使用causal mask
            block_size: 块大小

        Returns:
            o: [batch, n_heads, seq_len, d]  输出
            l: [batch, n_heads, seq_len]      logsumexp
        """
        # TODO: 实现Flash Attention前向传播
        raise NotImplementedError("请实现flash_attention_forward")

    # 测试
    batch, n_heads, seq_len, d = 2, 4, 32, 64
    q = torch.randn(batch, n_heads, seq_len, d)
    k = torch.randn(batch, n_heads, seq_len, d)
    v = torch.randn(batch, n_heads, seq_len, d)

    o, l = flash_attention_forward(q, k, v, is_causal=True)

    assert o.shape == q.shape, f"输出shape不正确: {o.shape}"
    assert l.shape == (batch, n_heads, seq_len), f"L shape不正确: {l.shape}"
    print(f"✅ 练习1完成: Flash Attention前向传播, 输出shape={o.shape}")


# ============================================================================
# 练习2: 实现DDP训练脚本（⭐⭐⭐⭐）
# ============================================================================

def exercise_2_ddp_training():
    """
    任务：编写完整的DDP训练脚本

    要求：
    1. 设置DDP环境
    2. 创建DistributedSampler
    3. 训练循环中正确处理同步

    提示：
    - 使用dist.init_process_group初始化
    - 使用torch.distributed.utils.data.DistributedSampler
    - 每个epoch调用sampler.set_epoch
    """
    def setup_ddp(rank, world_size):
        """设置DDP环境"""
        # TODO: 实现DDP环境设置
        # 1. 设置device
        # 2. 初始化进程组
        raise NotImplementedError

    def train_with_ddp(rank, world_size, model, dataset, epochs=3):
        """使用DDP训练模型"""
        # TODO: 实现DDP训练
        # 1. 创建DistributedSampler
        # 2. 包装模型为DDP
        # 3. 训练循环
        raise NotImplementedError

    print("✅ 练习2完成: DDP训练脚本框架")
    print("提示：实际运行需要使用torchrun启动")


# ============================================================================
# 练习3: DDP + 梯度累积（⭐⭐⭐）
# ============================================================================

def exercise_3_gradient_accumulation():
    """
    任务：在DDP中实现梯度累积

    要求：
    1. 正确缩放loss
    2. 累积指定步数后更新
    3. 确保所有GPU同步更新

    提示：
    - loss需要除以accumulation_steps
    - 使用计数器跟踪累积步数
    """
    def train_with_accumulation(model, dataloader, optimizer, accumulation_steps=4):
        """
        使用梯度累积的DDP训练

        Args:
            model: DDP模型
            dataloader: 数据加载器
            optimizer: 优化器
            accumulation_steps: 累积步数
        """
        # TODO: 实现梯度累积
        raise NotImplementedError

    print("✅ 练习3完成: 梯度累积逻辑")


# ============================================================================
# 练习4: 实现Online Softmax（⭐⭐⭐⭐）
# ============================================================================

def exercise_4_online_softmax():
    """
    任务：实现Online Softmax算法

    要求：
    1. 增量更新最大值m
    2. 增量更新归一化因子l
    3. 更新输出o

    这是Flash Attention的核心！
    """
    def online_softmax_increment(o_old, l_old, m_old, s_block, v_block):
        """
        增量更新softmax统计量

        Args:
            o_old: [batch, n_heads, block_size, d]  旧输出
            l_old: [batch, n_heads, block_size, 1]  旧归一化因子
            m_old: [batch, n_heads, block_size, 1]  旧最大值
            s_block: [batch, n_heads, block_size, block_size]  新的scores
            v_block: [batch, n_heads, block_size, d]  新的values

        Returns:
            o_new: 新输出
            l_new: 新归一化因子
            m_new: 新最大值
        """
        # TODO: 实现Online Softmax增量更新
        raise NotImplementedError

    print("✅ 练习4完成: Online Softmax增量更新")


# ============================================================================
# 练习5: Flash Attention反向传播（⭐⭐⭐⭐⭐）
# ============================================================================

def exercise_5_flash_attention_backward():
    """
    任务：实现Flash Attention反向传播

    要求：
    1. 重计算attention矩阵
    2. 计算dS和dP
    3. 分块计算dq, dk, dv

    提示：
    - 使用前向传播保存的L
    - 重新计算P = exp(S - L)
    - dS = P * (dO @ V^T - sum(dO @ V))
    """
    def flash_attention_backward(q, k, v, o, l, do, is_causal=False, block_size=64):
        """
        Flash Attention反向传播

        Args:
            q, k, v: 输入
            o, l: 前向传播输出
            do: 输出梯度
            is_causal: 是否causal
            block_size: 块大小

        Returns:
            dq, dk, dv: 输入梯度
        """
        # TODO: 实现反向传播
        raise NotImplementedError

    print("✅ 练习5完成: Flash Attention反向传播")


# ============================================================================
# 练习6: FSDP基础配置（⭐⭐⭐）
# ============================================================================

def exercise_6_fsdp_setup():
    """
    任务：配置FSDP训练

    要求：
    1. 选择合适的sharding策略
    2. 配置CPU offload
    3. 包装模型

    提示：
    - ShardingStrategy有FULL_SHARD等选项
    - CPUOffload可以节省GPU内存
    """
    def setup_fsdp(model):
        """
        配置FSDP

        Args:
            model: 原始模型

        Returns:
            fsdp_model: FSDP包装后的模型
        """
        # TODO: 实现FSDP配置
        # from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        # from torch.distributed.fsdp import ShardingStrategy, CPUOffload
        raise NotImplementedError

    print("✅ 练习6完成: FSDP配置")


# ============================================================================
# 练习7: 性能基准测试（⭐⭐）
# ============================================================================

def exercise_7_benchmark():
    """
    任务：编写DDP性能基准测试

    要求：
    1. 测量单GPU和多GPU时间
    2. 计算加速比和效率
    3. 测量通信开销
    """
    def benchmark_ddp(model, dataloader, num_epochs=5):
        """
        DDP性能基准测试

        Returns:
            results: 包含时间、加速比等指标
        """
        # TODO: 实现性能测试
        raise NotImplementedError

    print("✅ 练习7完成: 性能基准测试框架")


# ============================================================================
# 练习8: 混合精度+DDP（⭐⭐⭐）
# ============================================================================

def exercise_8_mixed_precision():
    """
    任务：实现混合精度DDP训练

    要求：
    1. 使用autocast自动混合精度
    2. 使用GradScaler缩放梯度
    3. 处理可能出现的数值问题

    提示：
    - from torch.cuda.amp import autocast, GradScaler
    - scaler.step(optimizer)而不是optimizer.step()
    """
    def train_mixed_precision_ddp(model, dataloader, optimizer, epochs=3):
        """
        混合精度DDP训练

        Returns:
            avg_time: 平均每epoch时间
        """
        # TODO: 实现混合精度训练
        raise NotImplementedError

    print("✅ 练习8完成: 混合精度DDP训练")


# ============================================================================
# 运行所有练习
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Flash Attention与DDP - 练习题")
    print("="*60)

    exercises = [
        ("练习1: Flash Attention前向传播", exercise_1_flash_attention),
        ("练习2: DDP训练脚本", exercise_2_ddp_training),
        ("练习3: 梯度累积", exercise_3_gradient_accumulation),
        ("练习4: Online Softmax", exercise_4_online_softmax),
        ("练习5: Flash Attention反向传播", exercise_5_flash_attention_backward),
        ("练习6: FSDP配置", exercise_6_fsdp_setup),
        ("练习7: 性能基准测试", exercise_7_benchmark),
        ("练习8: 混合精度DDP", exercise_8_mixed_precision),
    ]

    for name, exercise_func in exercises:
        print(f"\n{'='*60}")
        print(f"{name}")
        print('='*60)
        try:
            exercise_func()
        except NotImplementedError as e:
            print(f"⚠️  待实现: {e}")
        except Exception as e:
            print(f"❌ 错误: {e}")

    print(f"\n{'='*60}")
    print("练习完成！")
    print('='*60)
