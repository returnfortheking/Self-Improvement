# Day 24-25: Scaling Laws - 面试题

> **题目难度**: ⭐⭐ ~ ⭐⭐⭐⭐
> **考察重点**: Chinchilla Scaling Laws、参数拟合、最优配置计算
> **建议时间**: 30分钟

---

## Part 1: Scaling Laws基础

### Q1: 什么是Chinchilla Scaling Laws？⭐⭐⭐

**参考答案**:

**定义**: 描述语言模型性能与模型大小(N)、训练数据量(D)之间幂律关系的经验公式

**核心公式**:
```
L(N, D) = E + A/N^α + B/D^β

其中:
- L: 最终损失（perplexity的负对数）
- N: 模型参数量
- D: 训练数据量（tokens）
- E: 收敛损失（当N,D→∞时）
- A, B: 拟合参数
- α, β: 缩放指数（通常α≈0.35, β≈0.37）
```

**关键发现**:
1. 之前的模型**数据训练不足**
2. **计算最优**: 给定计算预算，N和D应该同比例增长
3. 可以准确预测模型性能，指导训练策略

---

### Q2: Chinchilla vs Kaplan Scaling Laws的区别？⭐⭐⭐⭐

**参考答案**:

| 维度 | Kaplan (2020) | Chinchilla (2022) |
|------|---------------|-------------------|
| **公式** | L(N, D) = E(N) + A(N)/D^β(N) | L(N, D) = E + A/N^α + B/D^β |
| **参数** | E, A, β都是N的函数 | E, A, B, α, β是常数 |
| **结论** | N和D需同时增长 | **N和D应同比例增长** |
| **最优策略** | 倾向大模型 | 数据量更重要 |

**Chinchilla改进**:
1. 更简洁的公式（参数更少）
2. 更准确的预测
3. 明确的最优分配策略

---

### Q3: 什么是计算最优（Compute-Optimal）训练？⭐⭐⭐⭐⭐

**参考答案**:

**定义**: 给定固定的计算预算C，如何分配模型大小N和数据量D，使最终损失最小

**数学推导**:
```
约束条件: C = 6ND（训练FLOPs）

目标: 最小化 L(N, D) = E + A/N^α + B/D^β

求解（使用拉格朗日乘数法）:
N_opt ∝ C^(1/(α+β)) × (Aα/Bβ)^(β/(α+β))
D_opt ∝ C^(1/(α+β)) × (Bβ/Aα)^(α/(α+β))

简化（当α≈β≈0.36）:
N_opt ≈ 0.04 × C^0.5
D_opt ≈ 20 × N_opt
```

**关键含义**:
- N和D应该**同比例增长**
- 模型参数量和训练tokens数的**比值约为1:20**
- 给定预算，有唯一最优配置

---

### Q4: 如何拟合Scaling Laws参数？⭐⭐⭐⭐

**参考答案**:

**步骤1**: 收集实验数据
```python
experiments = [
    {"N": 100e6, "D": 1e9, "loss": 3.2},
    {"N": 200e6, "D": 2e9, "loss": 2.9},
    {"N": 500e6, "D": 5e9, "loss": 2.5},
    # ... 至少5个不同配置
]
```

**步骤2**: 准备数据
```python
N_data = np.array([e["N"] for e in experiments])
D_data = np.array([e["D"] for e in experiments])
loss_data = np.array([e["loss"] for e in experiments])
```

**步骤3**: 定义Scaling Law函数
```python
def scaling_law(x, E, A, B, alpha, beta):
    N, D = x
    return E + A/N**alpha + B/D**beta
```

**步骤4**: 使用scipy拟合
```python
from scipy.optimize import curve_fit

initial_guess = [1.8, 400, 400, 0.35, 0.37]
params, _ = curve_fit(
    scaling_law,
    (N_data, D_data),
    loss_data,
    p0=initial_guess,
    maxfev=10000
)

E, A, B, alpha, beta = params
```

**步骤5**: 验证拟合
- 计算R²值
- 对比预测值和实际值
- 在测试集上验证

---

## Part 2: 实际应用

### Q5: 给定计算预算，如何选择最优模型配置？⭐⭐⭐⭐

**参考答案**:

**步骤1**: 确定计算预算
```python
C = 1e23  # 例如：10^23 FLOPs
```

**步骤2**: 计算最优N和D
```python
def compute_optimal(C, A, B, alpha, beta):
    ratio = (A * alpha) / (B * beta)

    N_opt = (C / 6) ** (1/(alpha+beta)) * ratio ** (beta/(alpha+beta))
    D_opt = (C / 6) ** (1/(alpha+beta)) * (1/ratio) ** (alpha/(alpha+beta))

    return N_opt, D_opt
```

**步骤3**: 验证配置
```python
# 计算预测损失
loss = E + A/N_opt**alpha + B/D_opt**beta

# 检查是否满足约束
assert 6 * N_opt * D_opt ≈ C
```

**示例**:
```python
# 给定C = 1e23 FLOPs
N_opt = 3.16e9  # 3.16B参数
D_opt = 5.27e12  # 5.27T tokens
loss_opt = 2.05  # 预测损失
```

---

### Q6: GPT-3的训练效率如何？⭐⭐⭐⭐

**参考答案**:

**GPT-3配置**:
- N = 175B 参数
- D = 300B tokens
- C = 6 × 175e9 × 300e9 = 3.15e23 FLOPs

**Chinchilla最优配置**（相同计算预算）:
- N_opt = 70B 参数
- D_opt = 1.4T tokens
- C = 6 × 70e9 × 1.4e12 = 5.88e23 FLOPs（相近）

**损失对比**:
```
GPT-3损失: L ≈ 2.10
Chinchilla最优: L ≈ 2.00

提升: 约5%
```

**结论**:
- GPT-3**数据训练不足**（应该训练更多数据，而不是增加模型大小）
- 如果用Chinchilla最优配置，相同计算量可以获得更好的性能

---

### Q7: 什么是IsoFLOPs曲线？⭐⭐⭐⭐⭐

**参考答案**:

**定义**: 在固定计算预算C下，展示模型大小N与数据量D权衡关系的曲线

**特点**:
```
固定约束: C = 6ND（常数）
变量: N从10^7到10^10

曲线形状:
- 小N, 大D: 模型容量不足，欠拟合
- 大N, 小D: 数据不足，过拟合
- 最优点: 损失最小的配置
```

**如何绘制**:
```python
C = 1e23  # 固定预算
N_range = np.logspace(7, 10, 100)
D_range = C / (6 * N_range)

# 计算每个配置的损失
losses = E + A/N_range**alpha + B/D_range**beta

# 找到最优
optimal_idx = np.argmin(losses)
N_opt = N_range[optimal_idx]
D_opt = D_range[optimal_idx]
```

**意义**:
- 帮助理解N和D的权衡
- 验证理论计算的最优点
- 指导实验设计

---

## Part 3: 进阶话题

### Q8: 如何反向计算：给定目标损失，求所需预算？⭐⭐⭐⭐⭐

**参考答案**:

**问题**: 目标损失L_target = 2.0，求所需计算预算C

**方法**: 二分查找

```python
def find_required_budget(target_loss, E, A, B, alpha, beta):
    """
    给定目标损失，计算所需预算
    """
    low, high = 1e20, 1e26

    for _ in range(30):
        mid = (low + high) / 2

        # 计算该预算下的最优配置
        N, D = compute_optimal(mid, A, B, alpha, beta)

        # 计算预测损失
        loss_pred = E + A/N**alpha + B/D**beta

        if loss_pred > target_loss:
            low = mid
        else:
            high = mid

    return (low + high) / 2

# 使用
C_required = find_required_budget(2.0, 1.8, 400, 400, 0.35, 0.37)
print(f"所需预算: {C_required:.0e} FLOPs")
```

---

### Q9: Scaling Laws在实际训练中的应用？⭐⭐⭐⭐

**参考答案**:

**应用1**: 预算规划
```python
# 给定预算，选择模型配置
budget = 1e22  # FLOPs
N_opt, D_opt = compute_optimal(budget)

print(f"训练{N_opt/1e9:.1f}B参数模型")
print(f"需要{D_opt/1e12:.1f}T tokens数据")
```

**应用2**: 性能预测
```python
# 预测新模型的性能
N_new = 1e10  # 10B参数
D_new = 1e13  # 10T tokens
loss_pred = predict_loss(N_new, D_new)

print(f"预测损失: {loss_pred:.4f}")
```

**应用3**: 对比不同方案
```python
# 方案A: 大模型，少数据
N_a, D_a = 10e10, 5e12
loss_a = predict_loss(N_a, D_a)

# 方案B: 小模型，多数据
N_b, D_b = 5e10, 10e13
loss_b = predict_loss(N_b, D_b)

print(f"方案A损失: {loss_a:.4f}")
print(f"方案B损失: {loss_b:.4f}")
```

---

## 总结

**必会题目** (面试高频):
- Q1: Chinchilla Scaling Laws公式
- Q3: 计算最优配置
- Q5: 给定预算选择配置
- Q7: IsoFLOPs曲线

**加分题目** (深入理解):
- Q4: 拟合Scaling Laws参数
- Q8: 反向计算预算
- Q9: 实际应用场景

**建议**:
1. 记住核心公式: L(N,D) = E + A/N^α + B/D^β
2. 理解N和D同比例增长的含义
3. 掌握最优配置的计算方法
4. 能够在实际项目中应用Scaling Laws

---

**下一步**: [Day 26-27: Assignment 4 - Data Processing](../Day26-28_Data_Pipeline_RAG/README.md)
