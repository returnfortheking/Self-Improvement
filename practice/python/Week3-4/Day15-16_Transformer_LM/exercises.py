"""
Day 15-16: PyTorch基础与Transformer LM - 练习题
难度：⭐⭐⭐ ~ ⭐⭐⭐⭐
"""

import torch
import torch.nn as nn
import math
from typing import List, Tuple, Dict
from collections import Counter


# ============================================================================
# 练习1: 完整BPE Tokenizer实现（⭐⭐⭐⭐）
# ============================================================================

def exercise_1_implement_bpe():
    """
    任务：实现完整的BPE Tokenizer

    要求：
    1. 实现 train_bpe 函数，训练BPE tokenizer
    2. 实现 encode 函数，使用训练好的规则编码文本
    3. 实现 decode 函数，将token IDs解码回文本

    提示：
    - 使用 bytes 类型处理文本
    - 特殊token应该在词表开头
    - 编码时按优先级应用合并规则
    """

    def train_bpe(
        training_data: List[str],
        vocab_size: int = 1000,
        special_tokens: List[str] = ["<pad>"]
    ) -> Tuple[List[bytes], Dict[Tuple[bytes, bytes], bytes]]:
        """
        训练BPE tokenizer

        Args:
            training_data: 训练文本列表
            vocab_size: 目标词表大小
            special_tokens: 特殊token列表

        Returns:
            vocab: 词表（字节序列列表）
            merges: 合并规则 {(pair): merged_token}
        """
        # TODO: 实现BPE训练逻辑
        # 1. 初始化词表（特殊tokens + 256个字节）
        # 2. 将所有文本转换为字节序列
        # 3. 迭代统计并合并最高频字节对
        # 4. 返回词表和合并规则

        special_tokens_bytes = [token.encode('utf-8') for token in special_tokens]
        vocab = special_tokens_bytes + [bytes([i]) for i in range(256)]

        # 你的代码在这里
        raise NotImplementedError("请实现train_bpe函数")

    def encode(
        text: str,
        vocab: List[bytes],
        merges: Dict[Tuple[bytes, bytes], bytes]
    ) -> List[int]:
        """
        使用BPE规则编码文本

        Returns:
            token_ids: token ID列表
        """
        # TODO: 实现编码逻辑
        # 1. 将文本转换为字节序列
        # 2. 迭代应用合并规则
        # 3. 返回token IDs

        raise NotImplementedError("请实现encode函数")

    def decode(
        token_ids: List[int],
        vocab: List[bytes]
    ) -> str:
        """
        将token IDs解码回文本

        Returns:
            text: 解码后的文本
        """
        # TODO: 实现解码逻辑
        raise NotImplementedError("请实现decode函数")

    # 测试你的实现
    training_data = ["hello world", "hello there", "world of warcraft"]
    vocab, merges = train_bpe(training_data, vocab_size=300)

    text = "hello world"
    encoded = encode(text, vocab, merges)
    decoded = decode(encoded, vocab)

    assert decoded == text, f"编码解码失败: '{decoded}' != '{text}'"
    print(f"✅ 练习1完成: 原文='{text}', 编码={encoded}, 解码='{decoded}'")


# ============================================================================
# 练习2: 实现RMSNorm（⭐⭐）
# ============================================================================

def exercise_2_rmsnorm():
    """
    任务：实现RMSNorm层

    要求：
    1. 实现 forward 方法
    2. 确保数值稳定性（添加eps）
    3. 测试输出shape和统计特性

    公式：RMS(x) = sqrt(mean(x²) + eps)
         output = x / RMS(x) * weight
    """

    class RMSNorm(nn.Module):
        def __init__(self, hidden_size: int, eps: float = 1e-5):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: [batch, seq_len, hidden_size]
            Returns:
                output: [batch, seq_len, hidden_size]
            """
            # TODO: 实现RMSNorm前向传播
            raise NotImplementedError("请实现RMSNorm的forward方法")

    # 测试
    rms_norm = RMSNorm(hidden_size=128)
    x = torch.randn(2, 10, 128)
    output = rms_norm(x)

    assert output.shape == x.shape, "输出shape不正确"
    print(f"✅ 练习2完成: 输入均值={x.mean():.4f}, 输出均值={output.mean():.4f}")


# ============================================================================
# 练习3: 实现RoPE（⭐⭐⭐）
# ============================================================================

def exercise_3_rope():
    """
    任务：实现Rotary Positional Encoding

    要求：
    1. 计算旋转角度（频率）
    2. 计算cos和sin
    3. 应用旋转到Q和K

    提示：
    - 频率公式: θ_i = 10000^(-2i/d)
    - 旋转矩阵: [cos -sin; sin cos]
    """

    def apply_rope(
        x: torch.Tensor,
        seq_len: int,
        head_dim: int
    ) -> torch.Tensor:
        """
        应用RoPE到tensor

        Args:
            x: [batch, seq_len, n_heads, head_dim]
            seq_len: 序列长度
            head_dim: 注意头维度

        Returns:
            x_rotated: [batch, seq_len, n_heads, head_dim]
        """
        # TODO: 实现RoPE
        # 1. 计算频率: freqs = 10000^(-2i/head_dim)
        # 2. 计算角度: angles = positions * freqs
        # 3. 计算cos和sin
        # 4. 应用旋转: [x1*cos - x2*sin, x1*sin + x2*cos]

        device = x.device
        dim = head_dim // 2

        # 你的代码在这里
        raise NotImplementedError("请实现RoPE函数")

    # 测试
    batch, n_heads = 2, 4
    seq_len, head_dim = 32, 64
    x = torch.randn(batch, seq_len, n_heads, head_dim)
    x_rotated = apply_rope(x, seq_len, head_dim)

    assert x_rotated.shape == x.shape, "输出shape不正确"
    print(f"✅ 练习3完成: RoPE应用成功, 输出shape={x_rotated.shape}")


# ============================================================================
# 练习4: 实现Multi-Head Attention（⭐⭐⭐⭐）
# ============================================================================

def exercise_4_multi_head_attention():
    """
    任务：实现Multi-Head Self-Attention

    要求：
    1. 实现Q, K, V投影
    2. 实现causal mask（可选）
    3. 实现attention计算
    4. 实现输出投影

    提示：
    - 使用einsum或matmul计算矩阵乘法
    - 缩放因子: 1/√d_k
    """

    class MultiHeadAttention(nn.Module):
        def __init__(self, d_model: int, n_heads: int):
            super().__init__()
            assert d_model % n_heads == 0

            self.d_model = d_model
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads

            # TODO: 定义Q, K, V投影层和输出投影层
            raise NotImplementedError("请定义投影层")

        def forward(
            self,
            x: torch.Tensor,
            is_causal: bool = True
        ) -> torch.Tensor:
            """
            Args:
                x: [batch, seq_len, d_model]
                is_causal: 是否使用causal mask
            Returns:
                output: [batch, seq_len, d_model]
            """
            batch, seq_len, d_model = x.shape

            # TODO: 实现Multi-Head Attention
            # 1. 计算Q, K, V
            # 2. reshape成multi-head形式
            # 3. 计算attention scores
            # 4. 应用causal mask（如果需要）
            # 5. softmax
            # 6. 应用到V
            # 7. reshape回原形状
            # 8. 输出投影

            raise NotImplementedError("请实现forward方法")

    # 测试
    d_model, n_heads = 128, 8
    mha = MultiHeadAttention(d_model, n_heads)
    x = torch.randn(2, 10, d_model)
    output = mha(x)

    assert output.shape == x.shape, "输出shape不正确"
    print(f"✅ 练习4完成: MHA输出shape={output.shape}")


# ============================================================================
# 练习5: 实现AdamW优化器步骤（⭐⭐⭐）
# ============================================================================

def exercise_5_adamw():
    """
    任务：手动实现AdamW的单步更新

    要求：
    1. 更新一阶矩估计（m）
    2. 更新二阶矩估计（v）
    3. 偏差修正
    4. 应用weight decay
    5. 更新参数

    提示：
    - m_t = β1 * m_{t-1} + (1-β1) * g_t
    - v_t = β2 * v_{t-1} + (1-β2) * g_t^2
    - 偏差修正: m_hat = m_t / (1-β1^t)
    """

    def adamw_update(
        param: torch.Tensor,
        grad: torch.Tensor,
        m: torch.Tensor,
        v: torch.Tensor,
        t: int,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        AdamW参数更新

        Returns:
            param: 更新后的参数
            m: 更新后的一阶矩
            v: 更新后的二阶矩
        """
        beta1, beta2 = betas

        # TODO: 实现AdamW更新逻辑
        # 1. 更新一阶矩和二阶矩
        # 2. 偏差修正
        # 3. 应用weight decay
        # 4. 更新参数

        raise NotImplementedError("请实现AdamW更新")

    # 测试
    param = torch.randn(10)
    grad = torch.randn_like(param)
    m = torch.zeros_like(param)
    v = torch.zeros_like(param)

    param_new, m_new, v_new = adamw_update(param, grad, m, v, t=1)

    assert param_new.shape == param.shape, "参数shape不正确"
    print(f"✅ 练习5完成: AdamW更新成功")


# ============================================================================
# 练习6: 实现前馈神经网络（⭐⭐）
# ============================================================================

def exercise_6_feed_forward():
    """
    任务：实现Transformer中的Feed-Forward Network

    要求：
    1. 实现两层线性层
    2. 中间层扩展为4倍维度
    3. 使用GELU激活函数
    4. 测试前向传播
    """

    class FeedForward(nn.Module):
        def __init__(self, d_model: int, d_ff: int = None):
            super().__init__()
            if d_ff is None:
                d_ff = 4 * d_model

            # TODO: 定义FFN层
            raise NotImplementedError("请定义FFN层")

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: [batch, seq_len, d_model]
            Returns:
                output: [batch, seq_len, d_model]
            """
            # TODO: 实现前向传播
            raise NotImplementedError("请实现forward方法")

    # 测试
    d_model = 128
    ffn = FeedForward(d_model)
    x = torch.randn(2, 10, d_model)
    output = ffn(x)

    assert output.shape == x.shape, "输出shape不正确"
    print(f"✅ 练习6完成: FFN输出shape={output.shape}")


# ============================================================================
# 练习7: 实现Transformer Block（⭐⭐⭐⭐）
# ============================================================================

def exercise_7_transformer_block():
    """
    任务：实现完整的Transformer Block

    要求：
    1. Pre-LN架构：先normalization再attention/FFN
    2. 残差连接
    3. 组合MultiHeadAttention和FeedForward

    提示：
    - output = x + Attention(LayerNorm(x))
    - output = output + FFN(LayerNorm(output))
    """

    class TransformerBlock(nn.Module):
        def __init__(self, d_model: int, n_heads: int):
            super().__init__()
            # TODO: 定义所有需要的层
            # - norm1, norm2
            # - attn (MultiHeadAttention)
            # - ffn (FeedForward)
            raise NotImplementedError("请定义所有层")

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: [batch, seq_len, d_model]
            Returns:
                output: [batch, seq_len, d_model]
            """
            # TODO: 实现Transformer Block前向传播
            # 1. Pre-Norm + Self-Attention + Residual
            # 2. Pre-Norm + FFN + Residual
            raise NotImplementedError("请实现forward方法")

    # 测试
    d_model, n_heads = 128, 8
    block = TransformerBlock(d_model, n_heads)
    x = torch.randn(2, 10, d_model)
    output = block(x)

    assert output.shape == x.shape, "输出shape不正确"
    print(f"✅ 练习7完成: Transformer Block输出shape={output.shape}")


# ============================================================================
# 练习8: 实现交叉熵损失（⭐⭐）
# ============================================================================

def exercise_8_cross_entropy():
    """
    任务：实现语言模型的交叉熵损失

    要求：
    1. 展平batch和seq_len维度
    2. 计算log_softmax
    3. 提取目标token的log概率
    4. 计算平均值
    """

    def cross_entropy_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int = -100
    ) -> torch.Tensor:
        """
        Args:
            logits: [batch, seq_len, vocab_size]
            targets: [batch, seq_len]
            ignore_index: 忽略的token ID（如padding）
        Returns:
            loss: scalar
        """
        # TODO: 实现交叉熵损失
        # 1. reshape logits和targets
        # 2. 计算log_softmax
        # 3. 收集目标概率
        # 4. 过滤ignore_index并计算平均

        raise NotImplementedError("请实现交叉熵损失")

    # 测试
    batch, seq_len, vocab_size = 2, 10, 100
    logits = torch.randn(batch, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch, seq_len))

    loss = cross_entropy_loss(logits, targets)

    # 与PyTorch实现对比
    loss_pytorch = nn.functional.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1)
    )

    assert abs(loss - loss_pytorch) < 1e-5, f"损失不匹配: {loss} vs {loss_pytorch}"
    print(f"✅ 练习8完成: 交叉熵损失={loss.item():.4f}")


# ============================================================================
# 练习9: 实现梯度累积（⭐⭐⭐）
# ============================================================================

def exercise_9_gradient_accumulation():
    """
    任务：实现梯度累积训练逻辑

    要求：
    1. 模拟小batch前向传播
    2. 累积梯度（不立即更新）
    3. 达到累积步数后更新参数
    4. 正确缩放loss
    """

    def train_with_accumulation(
        model: nn.Module,
        dataloader,
        accumulation_steps: int = 4
    ):
        """
        使用梯度累积训练模型

        Args:
            model: PyTorch模型
            dataloader: 数据加载器
            accumulation_steps: 累积步数
        """
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        for i, (x, y) in enumerate(dataloader):
            # TODO: 实现梯度累积逻辑
            # 1. 前向传播
            # 2. 计算缩放后的loss
            # 3. 反向传播
            # 4. 每accumulation_steps步更新一次参数

            raise NotImplementedError("请实现梯度累积")

        print("✅ 练习9完成: 梯度累积训练逻辑实现")

    # 简化测试
    model = nn.Linear(10, 5)
    dummy_loader = [(torch.randn(8, 10), torch.randn(8, 5)) for _ in range(10)]
    train_with_accumulation(model, dummy_loader, accumulation_steps=2)


# ============================================================================
# 练习10: 完整的Transformer LM（⭐⭐⭐⭐⭐）
# ============================================================================

def exercise_10_transformer_lm():
    """
    任务：实现完整的Transformer语言模型

    要求：
    1. Token Embedding
    2. 位置编码（可以是简单的sinusoidal或learned）
    3. 多个Transformer Block
    4. 输出Layer Norm
    5. LM Head（投影到vocab）
    """

    class TransformerLM(nn.Module):
        def __init__(
            self,
            vocab_size: int,
            d_model: int = 128,
            n_heads: int = 4,
            n_layers: int = 2,
            max_seq_len: int = 256
        ):
            super().__init__()

            # TODO: 定义所有组件
            # - token_embedding
            # - positional_embedding
            # - transformer_blocks (nn.ModuleList)
            # - norm_f
            # - lm_head

            raise NotImplementedError("请定义所有组件")

        def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
            """
            Args:
                token_ids: [batch, seq_len]
            Returns:
                logits: [batch, seq_len, vocab_size]
            """
            # TODO: 实现完整的前向传播
            raise NotImplementedError("请实现forward方法")

    # 测试
    vocab_size = 1000
    model = TransformerLM(vocab_size, d_model=64, n_heads=4, n_layers=2)
    token_ids = torch.randint(0, vocab_size, (2, 10))
    logits = model(token_ids)

    expected_shape = (2, 10, vocab_size)
    assert logits.shape == expected_shape, f"输出shape不正确: {logits.shape} vs {expected_shape}"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 练习10完成: Transformer LM实现成功, 参数量={n_params:,}")


# ============================================================================
# 运行所有练习
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("PyTorch基础与Transformer LM - 练习题")
    print("="*60)

    exercises = [
        ("练习1: BPE Tokenizer", exercise_1_implement_bpe),
        ("练习2: RMSNorm", exercise_2_rmsnorm),
        ("练习3: RoPE", exercise_3_rope),
        ("练习4: Multi-Head Attention", exercise_4_multi_head_attention),
        ("练习5: AdamW优化器", exercise_5_adamw),
        ("练习6: Feed-Forward Network", exercise_6_feed_forward),
        ("练习7: Transformer Block", exercise_7_transformer_block),
        ("练习8: 交叉熵损失", exercise_8_cross_entropy),
        ("练习9: 梯度累积", exercise_9_gradient_accumulation),
        ("练习10: 完整Transformer LM", exercise_10_transformer_lm),
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
