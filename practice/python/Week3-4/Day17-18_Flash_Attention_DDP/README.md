# Day 17-18: Flash Attention与分布式训练

> **学习目标**: 掌握Flash Attention实现，理解DDP/FSDP原理，完成CS336 Assignment 2核心内容
> **时间分配**: 6小时（理论2h + 实践4h）
> **难度**: ⭐⭐⭐⭐⭐
> **来源**: CS336 Assignment 2 - Systems
> **重要性**: ⭐⭐⭐⭐⭐ (核心技能，面试高频)

---

## 📚 核心概念

### 1. Flash Attention：解决内存瓶颈

#### 标准Attention的问题

**计算复杂度**: O(N²d)
**内存复杂度**: O(N²)  ← 瓶颈！

```python
# 标准Attention伪代码
def standard_attention(Q, K, V):
    # Q, K, V: [batch, n_heads, seq_len, d]

    S = Q @ K.T / sqrt(d)  # [batch, n_heads, seq_len, seq_len] - 巨大的矩阵!
    P = softmax(S)          # 同样大小的矩阵
    O = P @ V               # [batch, n_heads, seq_len, d]

    return O
```

**问题**:
- 序列长度N=4096时，attention matrix需要 4096×4096×4bytes = 64MB（每个head）
- 32个heads = 2GB GPU内存
- 梯度还需要额外内存！

#### Flash Attention核心思想

**Tiling（分块计算）**:
1. 将Q, K, V分成小块（blocks）
2. 逐块计算attention，只保留必要信息
3. 避免materialize完整的N×N矩阵

**Online Softmax**:
```
增量更新softmax统计量:
- m: 当前最大值
- l: 当前归一化因子

新block到来时:
m_new = max(m_old, m_block)
l_new = l_old * exp(m_old - m_new) + l_block
O_new = (O_old * l_old * exp(m_old - m_new) + O_block) / l_new
```

**优势**:
- ✅ 内存: O(N²) → O(N)
- ✅ 速度: 2-4x加速（HBM访问优化）
- ✅ 精确: 与标准attention完全一致（数学等价）

---

### 2. 分布式训练范式

#### 2.1 数据并行（Data Parallelism）

**核心思想**: 每个GPU持有完整的模型副本，处理不同的数据

```
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│  GPU 0  │  │  GPU 1  │  │  GPU 2  │  │  GPU 3  │
│ Model   │  │ Model   │  │ Model   │  │ Model   │
└────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘
     │            │            │            │
     └────────────┴────────────┴────────────┘
                   │
            AllReduce Gradient
            (梯度同步)
```

#### 2.2 PyTorch DDP (DistributedDataParallel)

**特性**:
- ✅ 高效的梯度同步（AllReduce）
- ✅ 每个进程独立运行
- ✅ 支持多机多卡
- ✅ 自动处理梯度累积

**实现**:
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化进程组
dist.init_process_group(backend="nccl")

# 包装模型
model = DDP(model, device_ids=[local_rank])

# 训练循环
for batch in dataloader:
    loss = model(batch)
    loss.backward()  # 自动同步梯度
    optimizer.step()
```

#### 2.3 FSDP (Fully Sharded Data Parallel)

**核心思想**: 分片模型参数、梯度、优化器状态

```
标准DDP:          FSDP:
┌─────────┐       ┌─────────┐
│ GPU 0   │       │ GPU 0   │
│ Model   │       │ Layer 1 │
│ (完整)  │       │  (1/4)  │
└─────────┘       └─────────┘

┌─────────┐       ┌─────────┐
│ GPU 1   │       │ GPU 1   │
│ Model   │  →    │ Layer 2 │
│ (完整)  │       │  (1/4)  │
└─────────┘       └─────────┘
```

**优势**:
- ✅ 内存节省: 可训练超大模型
- ✅ 通信优化: 减少通信量
- ✅ 灵活性: 可配置分片粒度

---

## 🔧 CS336 Assignment 2 要求详解

### Part 1: Flash Attention实现（3小时）

**文件**: `cs336_systems/attention.py`

#### 任务1.1: PyTorch实现（必做，1.5h）

**要求**: 实现Flash Attention的autograd函数

```python
class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal):
        """
        Args:
            q: [batch, n_heads, seq_len, d]
            k: [batch, n_heads, seq_len, d]
            v: [batch, n_heads, seq_len, d]
            is_causal: bool

        Returns:
            output: [batch, n_heads, seq_len, d]
        """
        # TODO: 实现前向传播
        # 1. 分块处理Q, K, V
        # 2. 计算online softmax
        # 3. 返回输出和L（用于反向传播）

        raise NotImplementedError

    @staticmethod
    def backward(ctx, do):
        """
        Args:
            do: [batch, n_heads, seq_len, d]  (输出梯度)

        Returns:
            dq, dk, dv: 输入梯度
        """
        # TODO: 实现反向传播
        # 1. 使用保存的L重新计算attention
        # 2. 计算dS, dP
        # 3. 分块计算dq, dk, dv

        raise NotImplementedError
```

**测试**:
```bash
uv run pytest tests/test_attention.py::test_flash_forward_pass_pytorch -v
uv run pytest tests/test_attention.py::test_flash_backward_pytorch -v
```

#### 任务1.2: Triton实现（可选，1.5h）

**要求**: 使用Triton编写GPU kernel

```python
import triton

@triton.jit
def flash_attention_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    stride_q, stride_k, stride_v, stride_o,
    seq_len, d,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel for flash attention
    """
    # TODO: 实现Triton kernel
    pass
```

**优势**:
- 比PyTorch实现快2-3x
- 手动优化内存访问

**测试**:
```bash
uv run pytest tests/test_attention.py::test_flash_forward_pass_triton -v
```

---

### Part 2: DDP实现（2小时）

**文件**: `cs336_systems/parallel.py`

#### 任务2.1: 基础DDP训练（1h）

**要求**: 编写多GPU训练脚本

```python
def train_with_ddp(rank, world_size):
    """
    使用DDP训练模型

    Args:
        rank: 当前进程rank
        world_size: 总进程数
    """
    # 1. 初始化进程组
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )

    # 2. 设置device
    torch.cuda.set_device(rank)

    # 3. 创建模型并包装DDP
    model = create_model().cuda(rank)
    model = DDP(model, device_ids=[rank])

    # 4. 训练循环
    for epoch in range(epochs):
        for batch in dataloader:
            # ... 训练代码

    # 5. 清理
    dist.destroy_process_group()
```

**测试**:
```bash
# 单机多卡
torchrun --nproc_per_node=4 train.py

# 多机多卡
torchrun --nnodes=2 --nproc_per_node=4 train.py
```

#### 任务2.2: 梯度累积与DDP（0.5h）

**要求**: 在DDP中实现梯度累积

```python
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**注意**:
- ✅ DDP自动同步梯度
- ✅ 梯度累积在DDP之上

---

### Part 3: FSDP实践（可选，1小时）

**文件**: `cs336_systems/parallel.py`

**要求**: 使用FSDP训练大模型

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

def train_with_fsdp():
    # 配置FSDP
    model = FSDP(
        create_model(),
        sharding_strategy="FULL_SHARD",  # 完全分片
        cpu_offload=CPUOffload(offload_params=True),  # CPU offload
    )

    # 训练循环与DDP相同
    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
```

---

## 💡 实现技巧

### 1. Flash Attention前向传播

```python
def flash_attention_forward(q, k, v, is_causal, block_size=64):
    """
    Flash Attention前向传播（简化版）
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

            # 更新统计量
            m_new = torch.maximum(m[:, :, start_i:end_i], s_block.max(dim=-1).values)
            l_new = torch.exp(m[:, :, start_i:end_i] - m_new).unsqueeze(-1) * \
                    l[:, :, start_i:end_i].unsqueeze(-1) + \
                    torch.exp(s_block - m_new.unsqueeze(-1)).sum(dim=-1)

            # 更新输出
            o[:, :, start_i:end_i, :] = (
                torch.exp(m[:, :, start_i:end_i].unsqueeze(-1) - m_new.unsqueeze(-1)) *
                o[:, :, start_i:end_i, :] * l[:, :, start_i:end_i].unsqueeze(-1) +
                torch.einsum('bhqk,bhkd->bhqd', torch.exp(s_block - m_new.unsqueeze(-1)), v_block)
            ) / l_new.unsqueeze(-1)

            # 更新统计量
            m[:, :, start_i:end_i] = m_new
            l[:, :, start_i:end_i] = l_new

    return o, l
```

### 2. DDP初始化最佳实践

```python
import os
import torch.distributed as dist

def setup_ddp():
    """DDP环境设置"""
    # 从环境变量获取分布式信息
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    # 设置device
    torch.cuda.set_device(local_rank)

    # 初始化进程组
    dist.init_process_group(
        backend='nccl',  # GPU使用nccl，CPU可用gloo
        rank=rank,
        world_size=world_size
    )

    return rank, world_size, local_rank

def cleanup_ddp():
    """清理DDP环境"""
    dist.destroy_process_group()
```

### 3. 性能监控

```python
import time

def benchmark_ddp(model, dataloader, epochs=3):
    """DDP训练性能基准测试"""
    rank = dist.get_rank()

    times = []
    for epoch in range(epochs):
        epoch_start = time.time()

        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        epoch_time = time.time() - epoch_start
        times.append(epoch_time)

        if rank == 0:
            print(f"Epoch {epoch}: {epoch_time:.2f}s")

    avg_time = sum(times) / len(times)
    if rank == 0:
        print(f"平均每epoch: {avg_time:.2f}s")

    return avg_time
```

---

## 📊 性能对比

### Flash Attention vs Standard Attention

| 序列长度 | 标准 Attention | Flash Attention | 加速比 | 内存节省 |
|---------|---------------|----------------|--------|----------|
| 1024    | 100ms         | 45ms           | 2.2x   | 50%      |
| 2048    | 450ms         | 120ms          | 3.8x   | 65%      |
| 4096    | 1800ms        | 380ms          | 4.7x   | 75%      |
| 8192    | OOM           | 1500ms         | ∞      | 80%      |

### DDP vs 单GPU

| GPU数量 | 单GPU时间 | DDP时间 | 加速比 | 效率 |
|---------|----------|---------|--------|------|
| 1       | 100s     | 100s    | 1.0x   | 100% |
| 2       | 100s     | 52s     | 1.9x   | 95%  |
| 4       | 100s     | 28s     | 3.6x   | 90%  |
| 8       | 100s     | 15s     | 6.7x   | 84%  |

### FSDP vs DDP

| 模型大小 | DDP内存 | FSDP内存 | 内存节省 |
|---------|---------|----------|----------|
| 1B      | 4GB     | 2GB      | 50%      |
| 10B     | 40GB    | 8GB      | 80%      |
| 100B    | OOM     | 32GB     | >90%     |

---

## 🎯 学习检验

### 关键问题

1. **Flash Attention**:
   - 为什么需要Online Softmax？
   - Tiling如何减少HBM访问？
   - 反向传播如何高效计算？

2. **DDP**:
   - AllReduce如何同步梯度？
   - Gradient Bucketing是什么？
   - 如何处理不同步的batch size？

3. **FSDP**:
   - 什么情况下应该用FSDP而不是DDP？
   - CPU Offload如何工作？
   - Sharding Strategy如何选择？

### 代码练习

完成 [examples.py](examples.py) 中的练习题。

---

## 📖 延伸阅读

**论文**:
- "Flash Attention: Faster Attention with Io-Awareness" (Dao et al., 2022)
- "FlashAttention-2: Faster Attention with Better Parallelism" (Dao, 2023)
- "PyTorch Distributed: Experiences on Scaling Distributed Training" (ML team)

**代码参考**:
- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Triton Flash Attention](https://github.com/openai/triton/blob/main/python/tutorials/06-fused-attention.py)
- [HuggingFace Accelerate](https://huggingface.co/docs/accelerate/)

---

## ⚠️ 常见陷阱

1. **Flash Attention**:
   - ❌ 忘记保存L用于反向传播
   - ❌ Causal mask实现错误
   - ✅ 使用unit test验证梯度

2. **DDP**:
   - ❌ 没有正确设置CUDA_VISIBLE_DEVICES
   - ❌ DataLoader没有设置sampler
   - ✅ 使用torchrun而不是手动启动进程

3. **FSDP**:
   - ❌ 不支持某些操作（如动态shape）
   - ❌ CPU Offload配置不当导致变慢
   - ✅ 逐步增大模型大小测试

---

**下一步**: [Day 19: 系统优化](../Day19_System_Optimization/README.md)
