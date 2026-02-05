# Day 15-16: PyTorch基础与Transformer语言模型

> **学习目标**: 掌握Transformer LM实现，理解BPE Tokenizer，完成CS336 Assignment 1核心内容
> **时间分配**: 6小时（理论2h + 实践4h）
> **难度**: ⭐⭐⭐⭐
> **来源**: CS336 Assignment 1 - Basics

---

## 📚 核心概念

### 1. Transformer语言模型架构

**GPT-style Decoder-only Transformer**:
```
输入文本 → Tokenizer → Token IDs
         ↓
    Input Embedding (vocab_size × d_model)
         ↓
    Positional Encoding (RoPE)
         ↓
    N × Transformer Blocks:
      - Layer Normalization (RMSNorm)
      - Multi-Head Self-Attention
      - Feed-Forward Network (GELU)
      - Residual Connections
         ↓
    Output Layer Norm
         ↓
    Linear Projection to Vocab
         ↓
    Softmax → Token Probabilities
```

### 2. BPE Tokenizer原理

**Byte-Level Byte-Pair Encoding**:

1. **初始化**: 将文本拆分为字节序列（256个基本token）
2. **迭代合并**: 统计相邻字节对频率，合并最高频对
3. **构建词表**: 重复直到达到目标词表大小（32K）
4. **编码**: 使用学到的合并规则编码新文本

**为什么使用Byte-Level BPE？**
- ✅ 无需UNKNOWN token
- ✅ 可处理任意Unicode字符
- ✅ 压缩率高（相比字符级）
- ✅ 适合多语言文本

### 3. 关键组件详解

#### 3.1 RMSNorm（Root Mean Square Layer Normalization）

```python
# 标准LayerNorm vs RMSNorm
# LayerNorm: (x - mean) / std * γ + β
# RMSNorm: x / RMS(x) * γ  (更简单，无bias)

RMS(x) = sqrt(mean(x² + ε))
output = x / RMS(x) * weight
```

#### 3.2 RoPE（Rotary Positional Encoding）

**核心思想**: 通过旋转矩阵注入位置信息到Query和Key

```python
# 旋转角度
θ = 10000^(-2i/d)  # i为维度索引

# 旋转矩阵
m: 位置索引
rot(m, i) = exp(m * θ * i)

# 应用到Q和K
q_rotated = q * cos(mθ) + rotate(q) * sin(mθ)
k_rotated = k * cos(mθ) + rotate(k) * sin(mθ)
```

#### 3.3 Multi-Head Attention

**标准公式**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**Multi-Head**:
```
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
```

---

## 🔧 CS336 Assignment 1 要求详解

### 任务清单

#### Part 1: BPE Tokenizer (2小时)

**文件**: `cs336_basics/data.py`

**需要实现的函数**:

1. **`train_bpe(training_data, vocab_size)`** (1.5h)
   ```python
   def train_bpe(
       training_data: list[str],
       vocab_size: int = 32000,
       special_tokens: list[str] = ["<pad>", "<eos>", "<bos>"]
   ) -> tuple[list[bytes], dict[tuple[bytes, bytes], bytes]]:
       """
       训练BPE tokenizer

       Args:
           training_data: 训练文本列表
           vocab_size: 目标词表大小（包括special tokens）
           special_tokens: 特殊token列表

       Returns:
           vocab: 词表（字节序列列表）
           merges: 合并规则字典 {(pair): merged_token}
       """
   ```

   **实现步骤**:
   1. 将所有文本编码为字节序列
   2. 统计字节对频率
   3. 迭代合并最高频对，直到达到vocab_size
   4. 返回词表和合并规则

2. **`encode(text, vocab, merges)`** (0.5h)
   ```python
   def encode(
       text: str,
       vocab: list[bytes],
       merges: dict[tuple[bytes, bytes], bytes]
   ) -> list[int]:
       """
       使用BPE规则编码文本

       Returns:
           token_ids: token ID列表
       """
   ```

**测试**:
```bash
uv run pytest tests/test_train_bpe.py -v
uv run pytest tests/test_tokenizer.py -v
```

#### Part 2: Transformer Components (2小时)

**文件**: `cs336_basics/model.py`

**需要实现的组件**:

1. **Linear层** (0.5h)
   ```python
   class Linear(nn.Module):
       def __init__(self, d_in: int, d_out: int):
           # 使用截断正态分布初始化
           std = sqrt(2 / (d_in + d_out))
           weight ~ trunc_normal(0, std, -3*std, 3*std)
   ```

2. **Embedding层** (0.5h)
   ```python
   class Embedding(nn.Module):
       def __init__(self, vocab_size: int, d_model: int):
           # 同样使用截断正态分布初始化
   ```

3. **RMSNorm** (0.5h)
   ```python
   class RMSNorm(nn.Module):
       def forward(self, x):
           rms = sqrt(mean(x², dim=-1, keepdim=True) + eps)
           return x / rms * self.weight
   ```

4. **Transformer Block** (0.5h)
   - Pre-normalization架构
   - Multi-Head Attention + FFN
   - 残差连接

**测试**:
```bash
uv run pytest tests/test_model.py -v
```

#### Part 3: 训练与评估 (2小时)

**文件**: `cs336_basics/optimizer.py`

**需要实现**:

1. **AdamW优化器** (1h)
   ```python
   def adamw(
       params: list[nn.Parameter],
       grad: list[Tensor],
       lr: float = 1e-3,
       betas: tuple[float, float] = (0.9, 0.999),
       eps: float = 1e-8,
       weight_decay: float = 0.01
   ) -> None:
       """
       手动实现AdamW更新规则
       """
   ```

2. **交叉熵损失** (0.5h)
   ```python
   def cross_entropy(
       logits: Float[Tensor, "batch seq_len vocab_size"],
       targets: Int[Tensor, "batch seq_len"]
   ) -> Float[Tensor, ""]:
   ```

3. **训练脚本** (0.5h)
   - 在TinyStories数据集上训练
   - 计算perplexity
   - 保存checkpoint

**测试**:
```bash
uv run pytest tests/test_optimizer.py -v
```

---

## 💡 实现技巧

### 1. BPE训练优化

**高效统计字节对**:
```python
from collections import Counter

def get_pair_frequencies(tokens_list):
    """统计所有文本中的字节对频率"""
    pair_freqs = Counter()
    for tokens in tokens_list:
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i+1])
            pair_freqs[pair] += 1
    return pair_freqs
```

### 2. RoPE实现

```python
def apply_rotary_emb(x, cos, sin):
    """
    Args:
        x: [batch, seq_len, n_heads, head_dim]
        cos, sin: [seq_len, head_dim // 2]
    """
    # 将x分成两半
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]

    # 应用旋转
    x_rotated = torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1)

    return x_rotated
```

### 3. Flash Attention基础

**问题**: 标准Attention的内存复杂度O(N²)

**解决方案**: Tiling（分块计算）
```python
# 伪代码
def flash_attention(Q, K, V, block_size=64):
    # 分块计算，减少内存占用
    for i in range(0, seq_len, block_size):
        for j in range(0, seq_len, block_size):
            Q_block = Q[:, i:i+block_size, :]
            K_block = K[:, j:j+block_size, :]
            V_block = V[:, j:j+block_size, :]

            # 计算局部attention
            S_block = Q_block @ K_block.T / sqrt(d)
            O_block = softmax(S_block) @ V_block

            # 累积结果
            O[:, i:i+block_size, :] += O_block
```

---

## 📊 性能基准

### TinyStories数据集

| 模型大小 | 参数量 | 训练时间 | 最终Perplexity | GPU要求 |
|---------|--------|---------|----------------|---------|
| Tiny    | 1M     | ~10min  | ~25            | 1 GPU   |
| Small   | 10M    | ~30min  | ~20            | 1 GPU   |
| Base    | 50M    | ~2h     | ~15            | 1-2 GPU |

### OpenWebText子集

| 模型大小 | 参数量 | 训练时间 | 最终Perplexity |
|---------|--------|---------|----------------|
| Tiny    | 1M     | ~20min  | ~35            |
| Small   | 10M    | ~1h     | ~28            |

---

## 🎯 学习检验

### 自测题

1. **BPE Tokenizer**:
   - Q: 为什么Byte-Level BPE不需要UNK token？
   - Q: vocab_size从32K降到16K会影响什么？

2. **Transformer架构**:
   - Q: RMSNorm相比LayerNorm的优势是什么？
   - Q: RoPE如何注入位置信息？

3. **训练技巧**:
   - Q: 梯度累积如何实现？
   - Q: 混合精度训练（AMP）的优缺点？

### 代码练习

完成 [examples.py](examples.py) 中的练习题。

---

## 📖 延伸阅读

**论文**:
- "Attention Is All You Need" (Vaswani et al., 2017)
- "Language Models are Few-Shot Learners" (Brown et al., 2020)
- "Byte Pair Encoding is Suboptimal for Language Model Pretraining" (Bostrom et al., 2022)

**代码参考**:
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [nanoGPT](https://github.com/karpathy/nanoGPT)

---

## ⚠️ 常见陷阱

1. **BPE训练**:
   - ❌ 忘记处理special tokens
   - ✅ 确保special tokens在词表开头

2. **RoPE实现**:
   - ❌ 维度错误（head_dim需能被2整除）
   - ✅ 使用`einops`进行张量操作

3. **训练稳定性**:
   - ❌ 学习率过大导致NaN
   - ✅ 使用warmup + weight decay

4. **内存泄漏**:
   - ❌ 没有释放中间变量
   - ✅ 使用`del`和`torch.cuda.empty_cache()`

---

**下一步**: [Day 17-18: Flash Attention与DDP](../Day17-18_Flash_Attention_DDP/README.md)
