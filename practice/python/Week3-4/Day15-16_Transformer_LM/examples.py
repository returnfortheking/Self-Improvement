"""
Day 15-16: PyTorch基础与Transformer LM - 代码示例
涵盖：BPE Tokenizer、Transformer组件、RoPE、训练技巧
"""

import torch
import torch.nn as nn
import math
from typing import List, Tuple, Dict
from collections import Counter


# ============================================================================
# Part 1: BPE Tokenizer 实现
# ============================================================================

def example_1_bpe_training():
    """示例1: BPE Tokenizer训练核心逻辑"""
    print("=" * 60)
    print("示例1: BPE训练核心逻辑")
    print("=" * 60)

    # 初始化：将文本转换为字节序列
    text = "hello hello world"
    tokens = list(text.encode('utf-8'))  # [104, 101, 108, 108, 111, 32, ...]
    print(f"初始tokens (bytes): {tokens[:20]}...")

    # 统计字节对频率
    def get_pair_freq(tokens_list):
        pair_freqs = Counter()
        for tokens in tokens_list:
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_freqs[pair] += 1
        return pair_freqs

    # 迭代合并最高频对
    vocab = set(range(256))  # 初始256个字节
    merges = {}  # {(byte1, byte2): merged_token}

    num_merges = 10
    for i in range(num_merges):
        pair_freqs = get_pair_freq([tokens])
        if not pair_freqs:
            break

        # 找到最高频对
        best_pair = max(pair_freqs.items(), key=lambda x: x[1])[0]
        new_token = max(vocab) + 1

        # 记录合并规则
        merges[best_pair] = new_token
        vocab.add(new_token)

        # 应用合并规则到tokens
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                new_tokens.append(new_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens

        print(f"Step {i+1}: 合并 {best_pair} -> {new_token} (频率: {pair_freqs[best_pair]})")

    print(f"\n最终词表大小: {len(vocab)}")
    print(f"合并规则数: {len(merges)}")


def example_2_bpe_encoding():
    """示例2: 使用训练好的BPE规则编码新文本"""
    print("\n" + "=" * 60)
    print("示例2: BPE编码")
    print("=" * 60)

    # 简化的BPE合并规则
    merges = {
        (104, 101): 256,  # 'h' + 'e' -> 256
        (108, 108): 257,  # 'l' + 'l' -> 257
        (257, 111): 258,  # 256 + 'o' -> 258 (即 "llo")
    }

    def encode(text: str, merges: dict) -> List[int]:
        """使用BPE规则编码文本"""
        tokens = list(text.encode('utf-8'))

        # 按优先级应用合并规则
        # 注意：实际实现需要按合并顺序排序
        while True:
            merged = False
            for pair, new_token in merges.items():
                i = 0
                new_tokens = []
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                        new_tokens.append(new_token)
                        i += 2
                        merged = True
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens
            if not merged:
                break

        return tokens

    text = "hello"
    encoded = encode(text, merges)
    print(f"原始文本: '{text}'")
    print(f"编码结果: {encoded}")
    print(f"Token数: {len(encoded)} (原始: {len(text)})")


def example_3_bpe_special_tokens():
    """示例3: 处理Special Tokens"""
    print("\n" + "=" * 60)
    print("示例3: Special Tokens处理")
    print("=" * 60)

    special_tokens = ["<pad>", "<eos>", "<bos>"]
    vocab_size = 32000

    # Special tokens应该在词表开头
    special_tokens_ids = {token: i for i, token in enumerate(special_tokens)}

    print("Special Tokens映射:")
    for token, idx in special_tokens_ids.items():
        print(f"  '{token}' -> {idx}")

    # 实际使用示例
    def add_special_tokens(token_ids, bos=True, eos=True):
        """添加special tokens到序列"""
        result = []
        if bos:
            result.append(special_tokens_ids["<bos>"])
        result.extend(token_ids)
        if eos:
            result.append(special_tokens_ids["<eos>"])
        return result

    tokens = [256, 257, 258]  # 假设的token IDs
    with_special = add_special_tokens(tokens)
    print(f"\n添加后: {with_special}")


# ============================================================================
# Part 2: Transformer 组件
# ============================================================================

def example_4_linear_layer():
    """示例4: 自定义Linear层（截断正态分布初始化）"""
    print("\n" + "=" * 60)
    print("示例4: Linear层实现")
    print("=" * 60)

    class Linear(nn.Module):
        def __init__(self, d_in: int, d_out: int):
            super().__init__()
            # 截断正态分布初始化
            std = math.sqrt(2 / (d_in + d_out))
            self.weight = nn.Parameter(
                nn.init.trunc_normal_(
                    torch.empty(d_out, d_in),
                    std=std,
                    a=-3*std,
                    b=3*std
                )
            )

        def forward(self, x):
            # x: [batch, ..., d_in]
            # weight: [d_out, d_in]
            # output: [batch, ..., d_out]
            return torch.einsum('...d_in,d_out d_in->...d_out', x, self.weight)

    # 测试
    linear = Linear(d_in=128, d_out=256)
    x = torch.randn(4, 10, 128)  # [batch, seq_len, d_in]
    output = linear(x)
    print(f"输入shape: {x.shape}")
    print(f"权重shape: {linear.weight.shape}")
    print(f"输出shape: {output.shape}")
    print(f"权重范围: [{linear.weight.min():.3f}, {linear.weight.max():.3f}]")


def example_5_embedding_layer():
    """示例5: Embedding层"""
    print("\n" + "=" * 60)
    print("示例5: Embedding层实现")
    print("=" * 60)

    class Embedding(nn.Module):
        def __init__(self, vocab_size: int, d_model: int):
            super().__init__()
            std = 1.0
            self.weight = nn.Parameter(
                nn.init.trunc_normal_(
                    torch.empty(vocab_size, d_model),
                    std=std,
                    a=-3*std,
                    b=3*std
                )
            )

        def forward(self, token_ids):
            # token_ids: [batch, seq_len]
            # output: [batch, seq_len, d_model]
            return self.weight[token_ids]

    # 测试
    vocab_size = 32000
    d_model = 512
    embedding = Embedding(vocab_size, d_model)
    token_ids = torch.randint(0, vocab_size, (4, 10))
    output = embedding(token_ids)
    print(f"Token IDs shape: {token_ids.shape}")
    print(f"Embedding weight shape: {embedding.weight.shape}")
    print(f"Output shape: {output.shape}")


def example_6_rmsnorm():
    """示例6: RMSNorm实现"""
    print("\n" + "=" * 60)
    print("示例6: RMSNorm实现")
    print("=" * 60)

    class RMSNorm(nn.Module):
        def __init__(self, hidden_size: int, eps: float = 1e-5):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps

        def forward(self, x):
            # x: [batch, seq_len, hidden_size]
            rms = torch.sqrt(
                torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps
            )
            return x / rms * self.weight

    # 测试
    rms_norm = RMSNorm(hidden_size=512)
    x = torch.randn(4, 10, 512)
    output = rms_norm(x)
    print(f"输入shape: {x.shape}")
    print(f"输出shape: {output.shape}")
    print(f"输入均值/标准差: {x.mean():.3f} / {x.std():.3f}")
    print(f"输出均值/标准差: {output.mean():.3f} / {output.std():.3f}")


def example_7_rope():
    """示例7: RoPE (Rotary Positional Encoding) 实现"""
    print("\n" + "=" * 60)
    print("示例7: RoPE实现")
    print("=" * 60)

    def apply_rotary_emb(x, cos, sin):
        """
        应用旋转位置编码

        Args:
            x: [batch, seq_len, n_heads, head_dim]
            cos: [seq_len, head_dim // 2]
            sin: [seq_len, head_dim // 2]
        """
        # 将x分成两半
        x1 = x[..., :x.shape[-1]//2]
        x2 = x[..., x.shape[-1]//2:]

        # 应用旋转: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
        x_rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)

        return x_rotated

    # 生成旋转角度
    seq_len = 128
    head_dim = 64
    dim = head_dim // 2

    # 位置索引
    positions = torch.arange(seq_len).float()

    # 频率: θ_i = 10000^(-2i/d)
    freqs = 1.0 / (10000 ** (torch.arange(dim).float() / dim))

    # 计算cos和sin
    angles = positions[:, None] * freqs[None, :]  # [seq_len, dim]
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    print(f"旋转角度shape: cos={cos.shape}, sin={sin.shape}")

    # 测试应用RoPE
    batch, n_heads = 4, 8
    x = torch.randn(batch, seq_len, n_heads, head_dim)
    x_rotated = apply_rotary_emb(x, cos, sin)

    print(f"输入shape: {x.shape}")
    print(f"输出shape: {x_rotated.shape}")
    print(f"位置信息已注入到Q和K中")


def example_8_multi_head_attention():
    """示例8: Multi-Head Self-Attention"""
    print("\n" + "=" * 60)
    print("示例8: Multi-Head Attention")
    print("=" * 60)

    class MultiHeadAttention(nn.Module):
        def __init__(self, d_model: int, n_heads: int):
            super().__init__()
            assert d_model % n_heads == 0
            self.d_model = d_model
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads

            # Q, K, V投影
            self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
            self.out = nn.Linear(d_model, d_model, bias=False)

        def forward(self, x, mask=None):
            batch, seq_len, d_model = x.shape

            # 计算Q, K, V
            qkv = self.qkv(x)  # [batch, seq_len, 3*d_model]
            qkv = qkv.reshape(batch, seq_len, 3, self.n_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, n_heads, seq_len, head_dim]
            q, k, v = qkv[0], qkv[1], qkv[2]

            # 计算attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # scores: [batch, n_heads, seq_len, seq_len]

            # 应用mask（可选）
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)

            # Softmax
            attn_weights = torch.softmax(scores, dim=-1)

            # 应用到V
            output = torch.matmul(attn_weights, v)  # [batch, n_heads, seq_len, head_dim]
            output = output.transpose(1, 2).reshape(batch, seq_len, d_model)

            # 输出投影
            return self.out(output)

    # 测试
    d_model, n_heads = 512, 8
    mha = MultiHeadAttention(d_model, n_heads)
    x = torch.randn(2, 10, d_model)  # [batch, seq_len, d_model]
    output = mha(x)
    print(f"输入shape: {x.shape}")
    print(f"输出shape: {output.shape}")


# ============================================================================
# Part 3: 训练技巧
# ============================================================================

def example_9_adamw_optimizer():
    """示例9: AdamW优化器实现"""
    print("\n" + "=" * 60)
    print("示例9: AdamW优化器")
    print("=" * 60)

    def adamw_step(param, grad, m, v, t, lr=1e-3, betas=(0.9, 0.999),
                   eps=1e-8, weight_decay=0.01):
        """
        手动实现AdamW更新

        Args:
            param: 模型参数
            grad: 梯度
            m: 一阶矩估计
            v: 二阶矩估计
            t: 时间步
        """
        beta1, beta2 = betas

        # 更新一阶矩和二阶矩
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        # 偏差修正
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # AdamW: 先应用weight decay，再更新参数
        param = param - lr * weight_decay * param
        param = param - lr * m_hat / (torch.sqrt(v_hat) + eps)

        return param, m, v

    # 测试
    param = torch.randn(10, requires_grad=True)
    grad = torch.randn_like(param)
    m = torch.zeros_like(param)
    v = torch.zeros_like(param)

    for t in range(1, 6):
        param.data, m, v = adamw_step(param.data, grad, m, v, t)
        print(f"Step {t}: param mean={param.mean():.4f}, std={param.std():.4f}")


def example_10_gradient_accumulation():
    """示例10: 梯度累积"""
    print("\n" + "=" * 60)
    print("示例10: 梯度累积")
    print("=" * 60)

    # 模拟小batch训练
    model = nn.Linear(10, 5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    accumulation_steps = 4
    target_batch_size = 32

    print(f"目标batch size: {target_batch_size}")
    print(f"实际batch size: {target_batch_size // accumulation_steps}")
    print(f"累积步数: {accumulation_steps}")

    for step in range(3):
        model.zero_grad()

        for micro_step in range(accumulation_steps):
            # 模拟小batch
            x = torch.randn(target_batch_size // accumulation_steps, 10)
            y = torch.randn(target_batch_size // accumulation_steps, 5)

            # 前向传播
            output = model(x)
            loss = nn.functional.mse_loss(output, y)

            # 反向传播（梯度累积）
            loss = loss / accumulation_steps  # 缩放loss
            loss.backward()

            print(f"  Step {step}, Micro-step {micro_step}: loss={loss.item()*accumulation_steps:.4f}")

        # 更新参数
        optimizer.step()
        print(f"  Step {step}: 参数已更新")


def example_11_mixed_precision():
    """示例11: 混合精度训练（AMP）"""
    print("\n" + "=" * 60)
    print("示例11: 混合精度训练")
    print("=" * 60)

    model = nn.Linear(100, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 创建GradScaler
    scaler = torch.cuda.amp.GradScaler()

    # 模拟训练步骤
    x = torch.randn(32, 100)
    y = torch.randint(0, 10, (32,))

    # 使用autocast自动混合精度
    with torch.cuda.amp.autocast():
        output = model(x)
        loss = nn.functional.cross_entropy(output, y)

    print(f"Loss dtype: {loss.dtype}")
    print(f"Output dtype: {output.dtype}")

    # 反向传播（自动处理缩放）
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    print("混合精度训练完成")


def example_12_cross_entropy_loss():
    """示例12: 交叉熵损失实现"""
    print("\n" + "=" * 60)
    print("示例12: 交叉熵损失")
    print("=" * 60)

    def cross_entropy(logits, targets):
        """
        计算交叉熵损失

        Args:
            logits: [batch, seq_len, vocab_size]
            targets: [batch, seq_len]
        """
        # 将logits展平
        logits = logits.reshape(-1, logits.shape[-1])  # [batch*seq_len, vocab_size]
        targets = targets.reshape(-1)  # [batch*seq_len]

        # 计算log_softmax
        log_probs = torch.log_softmax(logits, dim=-1)

        # 收集目标token的log概率
        loss = -log_probs[range(len(targets)), targets].mean()

        return loss

    # 测试
    batch, seq_len, vocab_size = 4, 10, 1000
    logits = torch.randn(batch, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch, seq_len))

    loss = cross_entropy(logits, targets)
    print(f"Cross-Entropy Loss: {loss.item():.4f}")

    # 与PyTorch实现对比
    loss_pytorch = nn.functional.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1)
    )
    print(f"PyTorch Loss: {loss_pytorch.item():.4f}")
    print(f"差异: {abs(loss - loss_pytorch).item():.6f}")


# ============================================================================
# Part 4: 完整示例
# ============================================================================

def example_13_mini_transformer():
    """示例13: 迷你Transformer模型"""
    print("\n" + "=" * 60)
    print("示例13: 迷你Transformer")
    print("=" * 60)

    class MiniTransformer(nn.Module):
        """简化的Transformer语言模型"""
        def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.layers = nn.ModuleList([
                nn.ModuleDict({
                    'norm1': nn.LayerNorm(d_model),
                    'attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                    'norm2': nn.LayerNorm(d_model),
                    'ffn': nn.Sequential(
                        nn.Linear(d_model, 4 * d_model),
                        nn.GELU(),
                        nn.Linear(4 * d_model, d_model)
                    )
                })
                for _ in range(n_layers)
            ])
            self.norm_f = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, token_ids):
            x = self.embedding(token_ids)

            for layer in self.layers:
                # Self-Attention with residual
                attn_out, _ = layer['attn'](x, x, x, is_causal=True)
                x = x + attn_out
                x = layer['norm1'](x)

                # FFN with residual
                ffn_out = layer['ffn'](x)
                x = x + ffn_out
                x = layer['norm2'](x)

            x = self.norm_f(x)
            logits = self.lm_head(x)
            return logits

    # 测试
    model = MiniTransformer(vocab_size=1000)
    token_ids = torch.randint(0, 1000, (2, 10))
    logits = model(token_ids)
    print(f"输入shape: {token_ids.shape}")
    print(f"输出shape: {logits.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")


def example_14_training_loop():
    """示例14: 完整训练循环"""
    print("\n" + "=" * 60)
    print("示例14: 训练循环")
    print("=" * 60)

    # 简化示例：训练单个线性层
    model = nn.Linear(10, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # 模拟数据
    x_train = torch.randn(100, 10)
    y_train = torch.randn(100, 1)

    epochs = 5
    print("开始训练...")

    for epoch in range(epochs):
        epoch_loss = 0.0

        # 前向传播
        outputs = model(x_train)
        loss = criterion(outputs, y_train)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss = loss.item()

        print(f"Epoch {epoch+1}/{epochs}: Loss = {epoch_loss:.4f}")

    print("训练完成")


# ============================================================================
# 运行所有示例
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("PyTorch基础与Transformer LM - 代码示例")
    print("="*60)

    # BPE Tokenizer示例
    example_1_bpe_training()
    example_2_bpe_encoding()
    example_3_bpe_special_tokens()

    # Transformer组件示例
    example_4_linear_layer()
    example_5_embedding_layer()
    example_6_rmsnorm()
    example_7_rope()
    example_8_multi_head_attention()

    # 训练技巧示例
    example_9_adamw_optimizer()
    example_10_gradient_accumulation()
    example_11_mixed_precision()
    example_12_cross_entropy_loss()

    # 完整示例
    example_13_mini_transformer()
    example_14_training_loop()

    print("\n" + "="*60)
    print("所有示例运行完成！")
    print("="*60)
