# Day 15-16: PyTorch基础与Transformer LM - 面试题

> **题目难度**: ⭐⭐⭐ ~ ⭐⭐⭐⭐⭐
> **考察重点**: BPE原理、Transformer架构、训练技巧
> **建议时间**: 30分钟

---

## Part 1: BPE Tokenizer

### Q1: 为什么Byte-Level BPE不需要UNK token？⭐⭐⭐

**参考答案**:
Byte-Level BPE将所有文本编码为字节序列（0-255），任何字符都可以表示为字节的组合。因此，不会有"未知"的字符，不需要UNK token。

**关键点**:
- 字节是字符的最小单位
- 任何Unicode字符都可以编码为1-4个字节
- 初始词表包含256个字节token

---

### Q2: BPE训练中的合并顺序重要吗？为什么？⭐⭐⭐

**参考答案**:
非常重要！BPE是贪婪算法，每次合并最高频对。不同的合并顺序会产生不同的词表。

**示例**:
```
文本: "aaab"
步骤1: 合并"aa" -> "X", 结果: "Xab"
步骤2: 合并"ab" -> "Y", 结果: "XY"

如果先合并"ab":
步骤1: 合并"ab" -> "Y", 结果: "aaY"
步骤2: 无法再合并"aa"（只有一个）
```

---

### Q3: 如何实现BPE的高效训练？⭐⭐⭐⭐

**参考答案**:
1. **使用优先队列** (堆)维护最高频对
2. **批量统计**所有文本的字节对频率
3. **使用字典**快速查找和更新频率
4. **限制vocab_size**避免过度合并

**代码示例**:
```python
from collections import Counter
import heapq

def get_top_pair(tokens_list):
    pair_freqs = Counter()
    for tokens in tokens_list:
        for i in range(len(tokens) - 1):
            pair_freqs[(tokens[i], tokens[i+1])] += 1
    return pair_freqs.most_common(1)[0][0]
```

---

## Part 2: Transformer架构

### Q4: RMSNorm相比LayerNorm有什么优势？⭐⭐⭐

**参考答案**:

**LayerNorm**:
```
output = (x - mean) / std * γ + β
```
- 需要计算mean和std
- 有可学习的bias参数β

**RMSNorm**:
```
output = x / RMS(x) * γ
RMS(x) = sqrt(mean(x²) + ε)
```
- 不需要减去mean（简化计算）
- 没有bias参数
- 计算更快，效果相近

**论文**: "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)

---

### Q5: RoPE如何注入位置信息？⭐⭐⭐⭐

**参考答案**:

RoPE通过旋转矩阵注入位置信息到Query和Key：

**核心思想**:
```
对于位置m的2D向量 (x_m, y_m):
旋转角度: m * θ
旋转后: [x_m*cos(mθ) - y_m*sin(mθ), x_m*sin(mθ) + y_m*cos(mθ)]
```

**实现**:
```python
def apply_rope(x, cos, sin):
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    return torch.cat([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)
```

**优势**:
- 相对位置编码
- 可以外推到更长的序列
- 不占用额外的embedding参数

---

### Q6: Multi-Head Attention中的"多头"有什么作用？⭐⭐⭐

**参考答案**:

**单头vs多头**:
```
单头: Attention(Q, K, V)
多头: Concat(Attention(Q1, K1, V1), ..., Attention(Qh, Kh, Vh)) W^O
```

**多头的优势**:
1. **多个表示子空间**: 每个head学习不同的注意力模式
2. **并行计算**: 可以同时关注不同位置
3. **表达能力**: 组合多个head的输出

**示例**:
- Head 1: 关注语法关系
- Head 2: 关注语义关联
- Head 3: 关注长距离依赖

---

### Q7: Pre-LN和Post-LN有什么区别？⭐⭐⭐⭐

**参考答案**:

**Post-LN** (原始Transformer):
```
x = x + Attention(LayerNorm(x))  # Attention后归一化
```
- 训练不稳定，需要warmup
- 深层网络梯度容易消失

**Pre-LN** (GPT-2/3风格):
```
x = x + LayerNorm(Attention(x))  # Attention前归一化
```
- 训练更稳定
- 可以使用更大学习率
- 深层网络训练更容易

**现代实践**: 大多使用Pre-LN

---

## Part 3: 训练技巧

### Q8: 什么是梯度累积？如何实现？⭐⭐⭐

**参考答案**:

**场景**:
- GPU显存有限，只能跑小batch
- 想要大batch的训练效果

**实现**:
```python
accumulation_steps = 4

for i, (x, y) in enumerate(dataloader):
    loss = model(x, y) / accumulation_steps  # 缩放loss
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**原理**:
- 不立即更新参数
- 累积多个小batch的梯度
- 相当于虚拟的大batch训练

---

### Q9: 混合精度训练（AMP）的优缺点？⭐⭐⭐⭐

**参考答案**:

**优点**:
1. **速度快**: FP16计算速度是FP32的2-8倍
2. **省显存**: 显存占用减半
3. **加速比**: 通常训练速度提升2-3倍

**缺点**:
1. **数值溢出**: FP16范围小（±65504）
2. **精度损失**: 可能影响收敛
3. **需要GradScaler**: 动态缩放loss避免下溢

**实现**:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for x, y in dataloader:
    with autocast():  # 自动混合精度
        loss = model(x, y)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

### Q10: AdamW相比Adam改进了什么？⭐⭐⭐

**参考答案**:

**Adam**:
```
参数更新 = weight_decay * param + lr * grad
```
- L2正则化也参与自适应学习率调整
- 权重衰减效果不明显

**AdamW**:
```
参数 = 参数 - lr * weight_decay * 参数  # 解耦
参数更新 = 参数 - lr * grad_adaptive
```
- 权重衰减与自适应学习率解耦
- 更好的正则化效果
- 大模型训练的标配

**论文**: "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019)

---

### Q11: 如何处理Transformer中的位置编码外推？⭐⭐⭐⭐⭐

**参考答案**:

**问题**: 训练长度512，推理时2048怎么办？

**方案**:

1. **ALiBi** (Attention with Linear Biases):
   - 不使用位置编码
   - 在attention score中减去偏置项

2. **RoPE插值**:
   ```python
   # 推理时扩展位置索引
   positions = torch.arange(2048)
   # 重新计算旋转角度
   ```

3. **YaRN** (Yet another RoPE extensioN):
   - 结合RoPE和温度缩放
   - 更好的外推性能

---

### Q12: 如何计算语言模型的Perplexity？⭐⭐⭐

**参考答案**:

**定义**:
```
Perplexity = exp(average_negative_log_likelihood)
```

**实现**:
```python
def compute_perplexity(model, dataloader):
    total_loss = 0.0
    total_tokens = 0

    for x, y in dataloader:
        logits = model(x)
        loss = cross_entropy(logits, y)

        total_loss += loss.item() * x.numel()
        total_tokens += x.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return perplexity
```

**解释**:
- Perplexity = 15: 模型每次预测平均在15个候选中犹豫
- 越低越好
- 相当于指数化的交叉熵

---

### Q13: Transformer中的Feed-Forward Network为什么用4倍维度？⭐⭐⭐⭐

**参考答案**:

**结构**:
```
d_model = 512
d_ff = 4 * d_model = 2048

FFN(x) = GELU(xW1) W2
```

**原因**:
1. **增加非线性表达能力**
2. **模仿MoE**: 等价于4个专家的稀疏组合
3. **实验验证**: 在Transformer中效果最好

**其他扩展因子**:
- BERT: 4x
- GPT-3: 4x
- T5: 4x
- 某些变体: 2x-8x

---

### Q14: 如何实现高效的序列打包（Sequence Packing）？⭐⭐⭐⭐⭐

**参考答案**:

**问题**: 短序列拼接浪费计算资源

**方案**:
```python
def pack_sequences(sequences, max_len):
    """
    将多个短序列打包到固定长度

    Example:
    seqs = [[1,2,3], [4,5], [6,7,8,9]]
    packed = [1,2,3, EOS, 4,5, EOS, 6,7,8,9]
    mask = [1,1,1,  0, 1,1,  0, 1,1,1,1]
    """
    packed = []
    mask = []

    for seq in sequences:
        packed.extend(seq + [EOS_TOKEN])
        mask.extend([1] * len(seq) + [0])

    return packed[:max_len], mask[:max_len]
```

**优势**:
- 提高GPU利用率
- 减少padding token
- 加速训练

---

### Q15: 如何诊断训练不收敛的问题？⭐⭐⭐⭐

**参考答案**:

**检查清单**:

1. **学习率**:
   - 太大: loss爆炸或NaN
   - 太小: 收敛慢或卡在局部最优
   - 解决: 使用learning rate finder

2. **梯度**:
   ```python
   for name, param in model.named_parameters():
       if param.grad is not None:
           print(f"{name}: grad_norm={param.grad.norm():.4f}")
   ```

3. **数值稳定性**:
   - 检查是否有NaN/Inf
   - 使用torch.autograd.detect_anomaly()

4. **数据**:
   - 检查label范围
   - 确认数据预处理正确

5. **模型**:
   - 初始化是否合理
   - LayerNorm/GELU是否正确

---

## Part 4: 进阶问题

### Q16: Flash Attention的核心思想是什么？⭐⭐⭐⭐⭐

**参考答案**:

**问题**: 标准Attention的内存复杂度O(N²)

**Flash Attention创新**:
1. **Tiling**: 分块计算attention
2. **Online softmax**: 增量更新softmax
3. **Recompute**: 反向传播时重新计算

**结果**:
- 内存: O(N) → 可处理更长序列
- 速度: 2-4x加速

**关键**: 不需要materialize完整的attention matrix

---

### Q17: 如何实现KV Cache加速生成？⭐⭐⭐⭐

**参考答案**:

**问题**: 生成时每次都要重新计算历史token的K和V

**解决方案**:
```python
class KVCache:
    def __init__(self):
        self.k_cache = []
        self.v_cache = []

    def update(self, k, v):
        self.k_cache.append(k)
        self.v_cache.append(v)

    def get(self):
        return torch.cat(self.k_cache, dim=1), torch.cat(self.v_cache, dim=1)

# 生成时
for i in range(max_len):
    if i == 0:
        k, v = model.compute_kv(input_ids)
    else:
        k, v = model.compute_kv(input_ids[:, -1:])  # 只计算最后一个

    cache.update(k, v)
    k_all, v_all = cache.get()
```

**加速**: 生成速度提升5-10x

---

### Q18: 如何理解Transformer的归纳偏置（Inductive Bias）？⭐⭐⭐⭐⭐

**参考答案**:

**归纳偏置**: 模型对数据分布的假设

**Transformer的偏置**:
1. **顺序无关**: Self-Attention是permutation invariant
2. **局部性**: 相近位置更容易attend
3. **组合性**: 复杂含义由简单部分组合

**对比CNN**:
- CNN: 强空间局部性偏置
- Transformer: 弱偏置，更flexible

**对比RNN**:
- RNN: 强顺序偏置，难并行
- Transformer: 顺序偏置弱（靠位置编码）

---

## 总结

**必会题目** (面试高频):
- Q1: BPE原理
- Q4: RMSNorm vs LayerNorm
- Q5: RoPE原理
- Q8: 梯度累积
- Q12: Perplexity计算

**加分题目** (深入理解):
- Q11: 位置编码外推
- Q16: Flash Attention
- Q17: KV Cache
- Q18: 归纳偏置

**建议**:
1. 理解公式推导
2. 能够手写核心代码
3. 知道常见trick和坑点

---

**下一步**: [Day 17-18: Flash Attention与DDP](../Day17-18_Flash_Attention_DDP/quiz.md)
