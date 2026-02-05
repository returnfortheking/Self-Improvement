# Day 24-25: Scaling Laws - 理论与实践

> **学习目标**: 理解Scaling Laws原理，掌握模型性能预测方法，完成CS336 Assignment 3核心内容
> **时间分配**: 6小时（理论3h + 实践3h）
> **难度**: ⭐⭐⭐⭐
> **重要性**: ⭐⭐⭐⭐⭐ (大模型训练的指导原则)

---

## 📚 核心概念

### 1. Scaling Laws基础

#### 1.1 什么是Scaling Laws？

**定义**: 描述模型性能随计算资源、数据量、模型规模变化的规律

**核心发现** (Kaplan et al., 2020; Chinchilla, 2022):

```
模型性能（Loss）与以下因素幂律相关:
- 模型参数量 N (Model Size)
- 训练数据量 D (Dataset Size)
- 计算量 C (Compute)
```

**数学表达**:
```
L(N, D) = E + A/N^α + B/D^β

其中:
- L: 最终损失
- N: 模型参数量
- D: 训练数据量（tokens）
- E, A, B, α, β: 拟合参数
```

#### 1.2 Chinchilla Scaling Laws

**关键结论** (Hoffmann et al., 2022):

**计算最优**: 给定计算预算C，最优的模型大小和数据量满足

```
N_opt ∝ C^(1/(α+β))
D_opt ∝ C^(1/(α+β))
```

**重要发现**:
- 之前的模型**数据训练不足**（unders optimized）
- Chinchilla法则: **N和D应该同比例增长**

**示例对比**:
```
GPT-3: 175B参数，300B tokens  (数据不足)
Chinchilla最优: 70B参数，1.4T tokens  (计算最优)
```

---

### 2. 模型性能预测

#### 2.1 损失预测公式

**Kapler Scaling Law** (2020):
```
L(N, D) = E(N) + A(N)/D^β(N)

其中:
E(N) = E_∞ + A/N^α  (当D→∞时的损失)
```

**Chinchilla改进** (2022):
```
L(N, D) = E + A/N^α + B/D^β

更简洁: 只需6个参数（E, A, B, α, β）
```

#### 2.2 拟合Scaling Laws

**步骤1**: 收集训练数据
```python
experiments = [
    {"N": 1e8, "D": 1e9, "loss": 2.5},
    {"N": 5e8, "D": 5e9, "loss": 2.0},
    {"N": 1e9, "D": 1e10, "loss": 1.8},
    # ...
]
```

**步骤2**: 最小二乘拟合
```python
from scipy.optimize import curve_fit

def scaling_law(x, E, A, B, alpha, beta):
    N, D = x
    return E + A/N**alpha + B/D**beta

params, _ = curve_fit(
    scaling_law,
    (experiments["N"], experiments["D"]),
    experiments["loss"]
)
```

**步骤3**: 预测
```python
# 预测1B参数、10B tokens的损失
predicted_loss = scaling_law((1e9, 1e10), *params)
```

---

### 3. 计算最优训练策略

#### 3.1 计算量定义

**训练计算量** (FLOPs):
```
C ≈ 6 × N × D

其中:
- N: 模型参数量
- D: 训练tokens数
- 6: 每个参数的前向+反向计算（约数）
```

**示例**:
```
GPT-3 (175B):
N = 175e9
D = 300e9
C = 6 × 175e9 × 300e9 = 3.15e23 FLOPs
```

#### 3.2 最优分配策略

**问题**: 给定计算预算C，如何分配N和D？

**Chinchilla最优解**:
```
N_opt = (C / 6)^(1/(α+β)) × (Aα/Bβ)^(β/(α+β))
D_opt = (C / 6)^(1/(α+β)) × (Bβ/Aα)^(α/(α+β))
```

**简化** (对于典型值α≈0.35, β≈0.37):
```
N_opt ≈ 0.04 × C^0.5
D_opt ≈ 20 × N_opt
```

**实践**:
```python
def compute_optimal_nd(compute_budget, A, B, alpha, beta):
    """计算最优的N和D"""
    # 假设C = 6ND
    ratio = (A * alpha) / (B * beta)

    N_opt = (compute_budget / 6) ** (1/(alpha+beta)) * ratio ** (beta/(alpha+beta))
    D_opt = (compute_budget / 6) ** (1/(alpha+beta)) * (1/ratio) ** (alpha/(alpha+beta))

    return N_opt, D_opt
```

---

### 4. IsoFLOPs曲线

#### 4.1 什么是IsoFLOPs？

**定义**: 在固定计算量下，模型大小N与数据量D的权衡曲线

**示例**:
```
对于C = 10^22 FLOPs:
- 方案1: N=1B, D=1.67T (大模型，少数据)
- 方案2: N=100M, D=16.7T (小模型，多数据)
- 方案3: N=400M, D=4.2T (Chinchilla最优)
```

#### 4.2 绘制IsoFLOPs

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_isoflops(compute_budget, A, B, alpha, beta):
    """绘制IsoFLOPs曲线"""
    N_range = np.logspace(7, 10, 100)  # 10M - 10B

    # 计算对应的D (C = 6ND)
    D_range = compute_budget / (6 * N_range)

    # 计算损失
    losses = E + A/N_range**alpha + B/D_range**beta

    # 找到最优
    optimal_idx = np.argmin(losses)
    N_opt = N_range[optimal_idx]
    D_opt = D_range[optimal_idx]

    plt.figure(figsize=(10, 6))
    plt.loglog(N_range, losses)
    plt.scatter([N_opt], [losses[optimal_idx]], c='red', s=100, label=f'Optimal: N={N_opt:.0e}, D={D_opt:.0e}')
    plt.xlabel('Model Size (N)')
    plt.ylabel('Loss')
    plt.title(f'IsoFLOPs Curve (C={compute_budget:.0e})')
    plt.legend()
    plt.grid(True)
    plt.show()
```

---

## 🔧 实战案例

### 案例1: 拟合Scaling Laws

```python
import numpy as np
from scipy.optimize import curve_fit

class ScalingLawFitter:
    """Scaling Laws拟合器"""

    def __init__(self):
        self.params = None

    def scaling_law(self, x, E, A, B, alpha, beta):
        """Chinchilla scaling law"""
        N, D = x
        return E + A/N**alpha + B/D**beta

    def fit(self, experiments):
        """
        拟合Scaling Laws

        Args:
            experiments: [{"N": ..., "D": ..., "loss": ...}, ...]
        """
        N = np.array([e["N"] for e in experiments])
        D = np.array([e["D"] for e in experiments])
        loss = np.array([e["loss"] for e in experiments])

        # 初始猜测
        initial_guess = [1.8, 400, 400, 0.35, 0.37]

        # 拟合
        self.params, _ = curve_fit(
            self.scaling_law,
            (N, D),
            loss,
            p0=initial_guess,
            maxfev=10000
        )

        E, A, B, alpha, beta = self.params
        print(f"拟合参数:")
        print(f"  E = {E:.4f}")
        print(f"  A = {A:.4f}")
        print(f"  B = {B:.4f}")
        print(f"  α = {alpha:.4f}")
        print(f"  β = {beta:.4f}")

    def predict(self, N, D):
        """预测损失"""
        if self.params is None:
            raise ValueError("模型未拟合")

        return self.scaling_law((N, D), *self.params)

    def compute_optimal(self, compute_budget):
        """计算最优N和D"""
        if self.params is None:
            raise ValueError("模型未拟合")

        _, A, B, alpha, beta = self.params

        # 最优解
        ratio = (A * alpha) / (B * beta)

        N_opt = (compute_budget / 6) ** (1/(alpha+beta)) * ratio ** (beta/(alpha+beta))
        D_opt = (compute_budget / 6) ** (1/(alpha+beta)) * (1/ratio) ** (alpha/(alpha+beta))

        return N_opt, D_opt

# 使用
fitter = ScalingLawFitter()

# 模拟实验数据
experiments = [
    {"N": 100e6, "D": 1e9, "loss": 3.2},
    {"N": 200e6, "D": 2e9, "loss": 2.9},
    {"N": 500e6, "D": 5e9, "loss": 2.5},
    {"N": 1e9, "D": 1e10, "loss": 2.2},
]

# 拟合
fitter.fit(experiments)

# 预测
predicted_loss = fitter.predict(N=1e9, D=1e10)
print(f"\n预测损失: {predicted_loss:.4f}")

# 计算最优配置
C = 1e22  # 给定计算预算
N_opt, D_opt = fitter.compute_optimal(C)
print(f"计算预算C={C:.0e}的最优配置:")
print(f"  N_opt = {N_opt:.0e}")
print(f"  D_opt = {D_opt:.0e}")
print(f"  预测损失 = {fitter.predict(N_opt, D_opt):.4f}")
```

---

### 案例2: 分析GPT-3训练

```python
def analyze_gpt3():
    """分析GPT-3的训练效率"""

    # GPT-3配置
    N_gpt3 = 175e9  # 175B参数
    D_gpt3 = 300e9  # 300B tokens

    # 计算计算量
    C_gpt3 = 6 * N_gpt3 * D_gpt3

    # Chinchilla最优配置（假设α=0.35, β=0.37）
    # N_opt ≈ 0.04 × C^0.5
    # D_opt ≈ 20 × N_opt

    N_opt = 0.04 * (C_gpt3 ** 0.5)
    D_opt = 20 * N_opt

    print("GPT-3 vs Chinchilla最优:")
    print(f"\nGPT-3:")
    print(f"  N = {N_gpt3:.0e} (175B)")
    print(f"  D = {D_gpt3:.0e} (300B tokens)")
    print(f"  C = {C_gpt3:.0e} FLOPs")

    print(f"\nChinchilla最优:")
    print(f"  N_opt = {N_opt:.0e} ({N_opt/1e9:.1f}B)")
    print(f"  D_opt = {D_opt:.0e} ({D_opt/1e12:.1f}T tokens)")
    print(f"  C = {6 * N_opt * D_opt:.0e} FLOPs (相同)")

    # 估计性能提升
    # 假设E=1.8, A=400, B=400, α=0.35, β=0.37
    E, A, B, alpha, beta = 1.8, 400, 400, 0.35, 0.37

    loss_gpt3 = E + A/N_gpt3**alpha + B/D_gpt3**beta
    loss_opt = E + A/N_opt**alpha + B/D_opt**beta

    print(f"\n预测损失:")
    print(f"  GPT-3: {loss_gpt3:.4f}")
    print(f"  Chinchilla最优: {loss_opt:.4f}")
    print(f"  提升: {(loss_gpt3 - loss_opt)/loss_gpt3*100:.1f}%")

analyze_gpt3()
```

---

## 💡 实践技巧

### 1. 实验设计

**原则**:
1. **对数采样**: N和D按对数间隔采样
   ```python
   N_values = np.logspace(7, 9, 5)  # 10M - 1B, 5个点
   D_values = np.logspace(9, 11, 5)  # 1B - 100B, 5个点
   ```

2. **覆盖范围**: 至少2个数量级
   ```python
   # 好的设计
   N = [10M, 30M, 100M, 300M, 1B]

   # 不好的设计（范围太窄）
   N = [100M, 110M, 120M, 130M, 140M]
   ```

3. **平衡采样**: N和D的变化独立
   ```python
   experiments = []
   for N in N_values:
       for D in D_values:
           experiments.append({"N": N, "D": D})
   ```

### 2. 数据收集

**关键指标**:
```python
def train_and_log(model, dataloader, epochs):
    """训练并记录关键指标"""
    for epoch in range(epochs):
        # 训练
        train_one_epoch(model, dataloader)

        # 评估
        val_loss = evaluate(model, val_dataloader)

        # 记录
        log = {
            "epoch": epoch,
            "N": count_parameters(model),
            "D": epochs * epoch * batch_size,
            "loss": val_loss,
            "compute": 6 * count_parameters(model) * epochs * epoch * batch_size
        }

        save_log(log)
```

### 3. 预测验证

**验证方法**:
```python
# 训练集拟合
fitter.fit(train_experiments)

# 测试集验证
test_errors = []
for exp in test_experiments:
    predicted = fitter.predict(exp["N"], exp["D"])
    actual = exp["loss"]
    error = abs(predicted - actual) / actual
    test_errors.append(error)

print(f"平均预测误差: {np.mean(test_errors)*100:.1f}%")
```

---

## 📊 实际应用

### 应用1: 预算规划

**场景**: 计算预算为10^23 FLOPs，如何规划模型？

```python
def plan_model(compute_budget, target_loss=2.0):
    """规划模型训练"""

    # 假设已拟合参数
    E, A, B, alpha, beta = 1.8, 400, 400, 0.35, 0.37

    # 计算最优配置
    fitter = ScalingLawFitter()
    fitter.params = (E, A, B, alpha, beta)

    N_opt, D_opt = fitter.compute_optimal(compute_budget)

    # 预测损失
    predicted_loss = fitter.predict(N_opt, D_opt)

    print(f"计算预算: {compute_budget:.0e} FLOPs")
    print(f"最优配置:")
    print(f"  模型大小: {N_opt/1e9:.2f}B 参数")
    print(f"  训练数据: {D_opt/1e12:.2f}T tokens")
    print(f"  预测损失: {predicted_loss:.4f}")

    # 检查是否满足目标
    if predicted_loss <= target_loss:
        print(f"✅ 满足目标损失 {target_loss}")
    else:
        print(f"❌ 不满足目标损失 {target_loss}")
        print(f"   需要增加计算预算到:")

        # 反推需要的计算量
        # target_loss = E + A/N^α + B/D^β
        # 且 N和D满足最优关系
        required_C = compute_required_budget(E, A, B, alpha, beta, target_loss)
        print(f"   {required_C:.0e} FLOPs")

    return N_opt, D_opt

# 使用
plan_model(compute_budget=1e23, target_loss=2.0)
```

### 应用2: 模型选择

**对比不同配置**:
```python
def compare_models(compute_budgets):
    """对比不同预算下的最优模型"""

    budgets = [1e21, 1e22, 1e23, 1e24]

    print(f"{'预算(FLOPs)':<15} {'N(参数)':<12} {'D(tokens)':<12} {'预测损失':<10}")
    print("-" * 50)

    for C in budgets:
        N_opt, D_opt = compute_optimal_nd(C, A, B, alpha, beta)
        loss = predict_loss(N_opt, D_opt, E, A, B, alpha, beta)

        print(f"{C:<15.0e} {N_opt/1e9:<12.2f} {D_opt/1e12:<12.2f} {loss:<10.4f}")

compare_models([1e21, 1e22, 1e23, 1e24])
```

---

## 🎯 学习检验

### 关键问题

1. **Scaling Laws原理**:
   - 什么是Scaling Laws？
   - Chinchilla法则的核心结论是什么？
   - 如何拟合Scaling Laws？

2. **计算优化**:
   - 给定计算预算，如何选择最优N和D？
   - IsoFLOPs曲线的含义？
   - 如何评估训练效率？

3. **实践应用**:
   - 如何预测模型性能？
   - 如何规划训练预算？
   - GPT-3训练哪里可以改进？

### 代码练习

完成 [examples.py](examples.py) 中的练习题。

---

## 📖 延伸阅读

**论文**:
- "Scaling Laws for Neural Language Models" (Kaplan et al., 2020)
- "Training Compute-Optimal Large Language Models" (Hoffmann et al., 2022) - Chinchilla

**代码**:
- [CS336 Assignment 3: Scaling](references/github/assignment3-scaling/)

---

**下一步**: [Day 26-28: 数据处理与RAG进阶](../Day26-28_Data_Pipeline_RAG/README.md)
