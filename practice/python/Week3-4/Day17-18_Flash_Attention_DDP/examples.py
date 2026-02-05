"""
Day 17-18: Flash Attention与DDP - 代码示例
涵盖：Flash Attention实现、DDP训练、FSDP基础
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import math
from typing import Optional, Tuple


# ============================================================================
# Part 1: Flash Attention 实现
# ============================================================================

def example_1_standard_attention():
    """示例1: 标准Attention实现（对比）"""
    print("=" * 60)
    print("示例1: 标准Attention")
    print("=" * 60)

    def standard_attention(q, k, v, mask=None):
        """
        标准Attention实现

        Args:
            q: [batch, n_heads, seq_len, d]
            k: [batch, n_heads, seq_len, d]
            v: [batch, n_heads, seq_len, d]
            mask: [batch, n_heads, seq_len, seq_len] or None

        Returns:
            output: [batch, n_heads, seq_len, d]
            attn: [batch, n_heads, seq_len, seq_len]
        """
        d = q.shape[-1]
        scale = 1.0 / math.sqrt(d)

        # 计算attention scores
        scores = torch.einsum('bhqd,bhkd->bhqk', q, k) * scale

        # 应用mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax
        attn = torch.softmax(scores, dim=-1)

        # 应用到values
        output = torch.einsum('bhqk,bhkd->bhqd', attn, v)

        return output, attn

    # 测试
    batch, n_heads, seq_len, d = 2, 4, 32, 64
    q = torch.randn(batch, n_heads, seq_len, d)
    k = torch.randn(batch, n_heads, seq_len, d)
    v = torch.randn(batch, n_heads, seq_len, d)

    output, attn = standard_attention(q, k, v)

    print(f"输入shape: q={q.shape}")
    print(f"Attention matrix shape: {attn.shape}")
    print(f"输出shape: {output.shape}")
    print(f"Attention matrix内存: {attn.element_size() * attn.nelement() / 1024:.2f} KB")


def example_2_flash_attention_forward():
    """示例2: Flash Attention前向传播（简化版）"""
    print("\n" + "=" * 60)
    print("示例2: Flash Attention前向传播")
    print("=" * 60)

    def flash_attention_forward(q, k, v, is_causal=False, block_size=16):
        """
        Flash Attention前向传播（简化实现）

        核心思想：
        1. 分块处理Q, K, V
        2. 增量更新softmax统计量
        3. 避免materialize完整的N×N矩阵
        """
        batch, n_heads, seq_len, d = q.shape

        # 初始化输出和统计量
        o = torch.zeros_like(q)
        l = torch.zeros(batch, n_heads, seq_len, device=q.device)
        m = torch.full((batch, n_heads, seq_len), -float('inf'), device=q.device)

        # 分块处理
        for start_j in range(0, seq_len, block_size):
            end_j = min(start_j + block_size, seq_len)

            # 加载K, V块
            k_block = k[:, :, start_j:end_j, :]
            v_block = v[:, :, start_j:end_j, :]

            for start_i in range(0, seq_len, block_size):
                end_i = min(start_i + block_size, seq_len)

                # 加载Q块
                q_block = q[:, :, start_i:end_i, :]

                # 计算attention scores
                s_block = torch.einsum('bhqd,bhkd->bhqk', q_block, k_block) / math.sqrt(d)

                # 应用causal mask
                if is_causal:
                    mask = torch.arange(start_i, end_i, device=q.device)[:, None] >= \
                           torch.arange(start_j, end_j, device=q.device)[None, :]
                    s_block = s_block.masked_fill(~mask, -float('inf'))

                # 更新统计量（Online Softmax）
                m_new = torch.maximum(m[:, :, start_i:end_i], s_block.max(dim=-1).values)

                # 计算新的归一化因子
                l_new = torch.exp(m[:, :, start_i:end_i].unsqueeze(-1) - m_new.unsqueeze(-1)) * \
                        l[:, :, start_i:end_i].unsqueeze(-1) + \
                        torch.exp(s_block - m_new.unsqueeze(-1)).sum(dim=-1, keepdim=True)

                # 更新输出
                o[:, :, start_i:end_i, :] = (
                    torch.exp(m[:, :, start_i:end_i].unsqueeze(-1) - m_new.unsqueeze(-1)) *
                    o[:, :, start_i:end_i, :] *
                    l[:, :, start_i:end_i].unsqueeze(-1) +
                    torch.einsum('bhqk,bhkd->bhqd',
                                torch.exp(s_block - m_new.unsqueeze(-1)),
                                v_block)
                ) / l_new

                # 更新统计量
                m[:, :, start_i:end_i] = m_new
                l[:, :, start_i:end_i] = l_new.squeeze(-1)

        return o, l

    # 测试
    batch, n_heads, seq_len, d = 2, 4, 32, 64
    q = torch.randn(batch, n_heads, seq_len, d)
    k = torch.randn(batch, n_heads, seq_len, d)
    v = torch.randn(batch, n_heads, seq_len, d)

    o, l = flash_attention_forward(q, k, v, is_causal=True)

    print(f"输入shape: {q.shape}")
    print(f"输出shape: {o.shape}")
    print(f"L (logsumexp) shape: {l.shape}")
    print("✅ Flash Attention前向传播完成")


def example_3_compare_attention():
    """示例3: 标准Attention vs Flash Attention对比"""
    print("\n" + "=" * 60)
    print("示例3: 标准Attention vs Flash Attention对比")
    print("=" * 60)

    def standard_attention(q, k, v):
        d = q.shape[-1]
        scores = torch.einsum('bhqd,bhkd->bhqk', q, k) / math.sqrt(d)
        attn = torch.softmax(scores, dim=-1)
        return torch.einsum('bhqk,bhkd->bhqd', attn, v)

    def flash_attention_simple(q, k, v, block_size=8):
        """简化版Flash Attention，用于对比"""
        batch, n_heads, seq_len, d = q.shape
        o = torch.zeros_like(q)

        for start_j in range(0, seq_len, block_size):
            end_j = min(start_j + block_size, seq_len)
            for start_i in range(0, seq_len, block_size):
                end_i = min(start_i + block_size, seq_len)

                q_block = q[:, :, start_i:end_i, :]
                k_block = k[:, :, start_j:end_j, :]
                v_block = v[:, :, start_j:end_j, :]

                s = torch.einsum('bhqd,bhkd->bhqk', q_block, k_block) / math.sqrt(d)
                attn = torch.softmax(s, dim=-1)
                o[:, :, start_i:end_i, :] += torch.einsum('bhqk,bhkd->bhqd', attn, v_block)

        return o

    # 测试
    seq_len = 64
    q = torch.randn(1, 2, seq_len, 32)
    k = torch.randn(1, 2, seq_len, 32)
    v = torch.randn(1, 2, seq_len, 32)

    # 标准Attention
    o_standard = standard_attention(q, k, v)

    # Flash Attention
    o_flash = flash_attention_simple(q, k, v, block_size=8)

    print(f"标准Attention输出shape: {o_standard.shape}")
    print(f"Flash Attention输出shape: {o_flash.shape}")
    print(f"输出差异: {torch.abs(o_standard - o_flash).max().item():.6f}")
    print("✅ 对比完成")


# ============================================================================
# Part 2: DDP 实现
# ============================================================================

def example_4_ddp_setup():
    """示例4: DDP环境设置"""
    print("\n" + "=" * 60)
    print("示例4: DDP环境设置")
    print("=" * 60)

    def setup_ddp():
        """
        设置DDP训练环境

        环境变量（由torchrun自动设置）:
        - RANK: 全局rank
        - WORLD_SIZE: 总进程数
        - LOCAL_RANK: 本地rank（节点内的GPU编号）
        - MASTER_ADDR: master节点地址
        - MASTER_PORT: master节点端口
        """
        import os

        # 检查是否在分布式环境中
        if 'RANK' in os.environ:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ['LOCAL_RANK'])

            print(f"Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")

            # 设置device
            torch.cuda.set_device(local_rank)

            # 初始化进程组
            dist.init_process_group(
                backend='nccl',  # GPU使用nccl
                rank=rank,
                world_size=world_size
            )

            print("✅ DDP环境初始化成功")
            return rank, world_size, local_rank
        else:
            print("⚠️  未检测到分布式环境变量")
            print("提示：使用 torchrun --nproc_per_node=4 启动")
            return None, None, None

    def cleanup_ddp():
        """清理DDP环境"""
        if dist.is_initialized():
            dist.destroy_process_group()
            print("✅ DDP环境已清理")

    # 演示（不实际初始化）
    print("DDP设置步骤:")
    print("1. 检查环境变量 (RANK, WORLD_SIZE, LOCAL_RANK)")
    print("2. 设置CUDA device")
    print("3. 初始化进程组 (dist.init_process_group)")
    print("4. 包装模型为DDP")
    print("5. 训练完成后清理 (dist.destroy_process_group)")


def example_5_ddp_model_wrapping():
    """示例5: DDP模型包装"""
    print("\n" + "=" * 60)
    print("示例5: DDP模型包装")
    print("=" * 60)

    # 创建简单模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(128, 10)

        def forward(self, x):
            return self.linear(x)

    model = SimpleModel()

    print("模型包装前:")
    print(f"  模型类型: {type(model)}")
    print(f"  参数设备: {model.linear.weight.device}")

    # 包装为DDP（演示，不实际运行）
    print("\nDDP包装步骤:")
    print("```python")
    print("# 1. 移动模型到GPU")
    print("model = model.cuda(local_rank)")
    print()
    print("# 2. 包装为DDP")
    print("model = DDP(")
    print("    model,")
    print("    device_ids=[local_rank],")
    print("    output_device=local_rank")
    print(")")
    print("```")
    print()
    print("包装后:")
    print("  - 模型类型: torch.nn.parallel.DistributedDataParallel")
    print("  - 前向传播: 自动同步梯度")
    print("  - 反向传播: 梯度在所有GPU间allreduce")


def example_6_ddp_training_loop():
    """示例6: DDP训练循环"""
    print("\n" + "=" * 60)
    print("示例6: DDP训练循环")
    print("=" * 60)

    def train_one_epoch(rank, model, dataloader, optimizer, epoch):
        """单epoch训练"""
        model.train()
        total_loss = 0.0

        for batch_idx, (data, target) in enumerate(dataloader):
            # 移动数据到GPU
            data = data.cuda(rank)
            target = target.cuda(rank)

            # 前向传播
            optimizer.zero_grad()
            output = model(data)
            loss = nn.functional.cross_entropy(output, target)

            # 反向传播（自动同步梯度）
            loss.backward()

            # 更新参数
            optimizer.step()

            total_loss += loss.item()

            # 只在rank 0打印
            if rank == 0 and batch_idx % 100 == 0:
                print(f"  Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        return total_loss / len(dataloader)

    # 演示代码
    print("DDP训练循环:")
    print("```python")
    print("# 初始化DDP环境")
    print("rank, world_size, local_rank = setup_ddp()")
    print()
    print("# 创建模型")
    print("model = create_model().cuda(local_rank)")
    print("model = DDP(model, device_ids=[local_rank])")
    print()
    print("# 创建DataLoader（重要：使用DistributedSampler）")
    print("sampler = DistributedSampler(dataset)")
    print("dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)")
    print()
    print("# 训练循环")
    print("for epoch in range(epochs):")
    print("    # 设置epoch以确保每个epoch的shuffle不同")
    print("    sampler.set_epoch(epoch)")
    print()
    print("    # 训练一个epoch")
    print("    train_one_epoch(rank, model, dataloader, optimizer, epoch)")
    print()
    print("# 清理")
    print("cleanup_ddp()")
    print("```")


def example_7_gradient_accumulation_with_ddp():
    """示例7: DDP + 梯度累积"""
    print("\n" + "=" * 60)
    print("示例7: DDP + 梯度累积")
    print("=" * 60)

    def train_with_accumulation(model, dataloader, optimizer, accumulation_steps=4):
        """使用梯度累积的训练"""
        model.train()

        for i, (data, target) in enumerate(dataloader):
            # 前向传播
            output = model(data)
            loss = nn.functional.cross_entropy(output, target)

            # 重要：缩放loss
            loss = loss / accumulation_steps

            # 反向传播（梯度累积）
            loss.backward()

            # 每accumulation_steps步更新一次
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if i % 10 == 0:
                print(f"  Step {i}, Loss: {loss.item() * accumulation_steps:.4f}")

    print("DDP + 梯度累积:")
    print("```python")
    print("accumulation_steps = 4")
    print("effective_batch_size = batch_size * accumulation_steps * world_size")
    print()
    print("for i, batch in enumerate(dataloader):")
    print("    loss = model(batch) / accumulation_steps  # 缩放loss")
    print("    loss.backward()")
    print()
    print("    if (i + 1) % accumulation_steps == 0:")
    print("        optimizer.step()  # 所有GPU同步后更新")
    print("        optimizer.zero_grad()")
    print("```")
    print()
    print("有效batch size:")
    print(f"  假设 batch_size=32, accumulation=4, 4个GPU")
    print(f"  有效batch = 32 * 4 * 4 = 512")


# ============================================================================
# Part 3: FSDP 基础
# ============================================================================

def example_8_fsdp_introduction():
    """示例8: FSDP介绍"""
    print("\n" + "=" * 60)
    print("示例8: FSDP (Fully Sharded Data Parallel)")
    print("=" * 60)

    print("FSDP vs DDP:")
    print()
    print("DDP:")
    print("  - 每个GPU: 完整模型副本")
    print("  - 内存: model_size * num_gpus")
    print("  - 通信: 只同步梯度")
    print()
    print("FSDP:")
    print("  - 每个GPU: 1/N的模型参数")
    print("  - 内存: model_size / num_gpus")
    print("  - 通信: 同步参数 + 梯度 + 优化器状态")
    print()
    print("适用场景:")
    print("  - DDP: 模型能放入单GPU")
    print("  - FSDP: 模型太大，需要分片")


def example_9_fsdp_basic_usage():
    """示例9: FSDP基础用法"""
    print("\n" + "=" * 60)
    print("示例9: FSDP基础用法")
    print("=" * 60)

    print("FSDP训练代码:")
    print("```python")
    print("from torch.distributed.fsdp import FullyShardedDataParallel as FSDP")
    print("from torch.distributed.fsdp import ShardingStrategy")
    print()
    print("# 创建模型")
    print("model = create_large_model()")
    print()
    print("# 包装为FSDP")
    print("model = FSDP(")
    print("    model,")
    print("    sharding_strategy=ShardingStrategy.FULL_SHARD,  # 完全分片")
    print("    cpu_offload=CPUOffload(offload_params=True),     # CPU offload")
    print(")")
    print()
    print("# 训练循环（与DDP相同）")
    print("for batch in dataloader:")
    print("    loss = model(batch)")
    print("    loss.backward()")
    print("    optimizer.step()")
    print("```")
    print()
    print("Sharding策略:")
    print("  - FULL_SHARD: 完全分片（最省内存）")
    print("  - SHARD_GRAD_OP: 分片梯度和优化器状态")
    print("  - NO_SHARD: 等价于DDP")
    print("  - HYBRID_SHARD: 混合分片（节点内+节点间）")


# ============================================================================
# Part 4: 性能优化技巧
# ============================================================================

def example_10_mixed_precision_ddp():
    """示例10: DDP + 混合精度训练"""
    print("\n" + "=" * 60)
    print("示例10: DDP + 混合精度训练")
    print("=" * 60)

    print("DDP + AMP（自动混合精度）:")
    print("```python")
    print("from torch.cuda.amp import autocast, GradScaler")
    print()
    print("# 创建GradScaler")
    print("scaler = GradScaler()")
    print()
    print("# 训练循环")
    print("for batch in dataloader:")
    print("    # 使用autocast自动混合精度")
    print("    with autocast():")
    print("        output = model(batch)")
    print("        loss = criterion(output, target)")
    print()
    print("    # 缩放loss并反向传播")
    print("    scaler.scale(loss).backward()")
    print()
    print("    # 更新参数")
    print("    scaler.step(optimizer)")
    print("    scaler.update()")
    print("    optimizer.zero_grad()")
    print("```")
    print()
    print("优势:")
    print("  - 速度提升: 2-3x")
    print("  - 显存节省: ~50%")


def example_11_communication_overlap():
    """示例11: 通信与计算重叠"""
    print("\n" + "=" * 60)
    print("示例11: 通信与计算重叠（梯度bucket）")
    print("=" * 60)

    print("DDP自动将梯度组织到buckets中:")
    print()
    print("原理:")
    print("  - 小梯度被合并到buckets中")
    print("  - 当bucket满时自动开始通信")
    print("  - 通信与计算可以重叠")
    print()
    print("配置:")
    print("```python")
    print("model = DDP(")
    print("    model,")
    print("    device_ids=[local_rank],")
    print("    # 调整bucket大小（默认25MB）")
    print("    bucket_cap_mb=25,")
    print()
    print("    # 查找未使用的参数")
    print("    find_unused_parameters=False,  # 性能更好")
    print()
    print("    # 禁用梯度压缩以获得更高精度")
    print("    gradient_as_bucket_view=True")
    print(")")
    print("```")


def example_12_profiling_ddp():
    """示例12: DDP性能分析"""
    print("\n" + "=" * 60)
    print("示例12: DDP性能分析")
    print("=" * 60)

    print("使用PyTorch Profiler分析DDP性能:")
    print("```python")
    print("from torch.profiler import profile, ProfilerActivity")
    print()
    print("with profile(")
    print("    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],")
    print("    record_shapes=True,")
    print("    profile_memory=True")
    print(") as prof:")
    print("    for batch in dataloader:")
    print("        loss = model(batch)")
    print("        loss.backward()")
    print("        optimizer.step()")
    print()
    print("# 打印分析结果")
    print("print(prof.key_averages().table(sort_by='cuda_time_total'))")
    print("```")
    print()
    print("关键指标:")
    print("  - cuda_time_total: GPU总时间")
    print("  - cpu_time_total: CPU总时间")
    print("  - cuda_memory_usage: 显存使用")
    print("  - 通信开销: ncclAllReduce的时间")


# ============================================================================
# 运行所有示例
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Flash Attention与DDP - 代码示例")
    print("="*60)

    # Flash Attention示例
    example_1_standard_attention()
    example_2_flash_attention_forward()
    example_3_compare_attention()

    # DDP示例
    example_4_ddp_setup()
    example_5_ddp_model_wrapping()
    example_6_ddp_training_loop()
    example_7_gradient_accumulation_with_ddp()

    # FSDP示例
    example_8_fsdp_introduction()
    example_9_fsdp_basic_usage()

    # 性能优化示例
    example_10_mixed_precision_ddp()
    example_11_communication_overlap()
    example_12_profiling_ddp()

    print("\n" + "="*60)
    print("所有示例运行完成！")
    print("="*60)
    print("\n提示:")
    print("1. Flash Attention实现需要理解online softmax")
    print("2. DDP训练需要使用torchrun启动")
    print("3. FSDP适合训练超大模型")
    print("4. 性能优化建议使用Profiler分析瓶颈")
