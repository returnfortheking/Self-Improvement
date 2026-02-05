"""
Day 24-25: Scaling Laws - 代码示例
涵盖：Scaling Laws拟合、性能预测、最优配置计算
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import List, Dict, Tuple
import json


# ============================================================================
# Part 1: Scaling Laws 基础
# ============================================================================

def example_1_chinchilla_formula():
    """示例1: Chinchilla Scaling Laws公式"""
    print("=" * 60)
    print("示例1: Chinchilla Scaling Laws公式")
    print("=" * 60)

    def chinchilla_scaling_law(N, D, E=1.8, A=400, B=400, alpha=0.35, beta=0.37):
        """
        Chinchilla Scaling Laws

        Args:
            N: 模型参数量
            D: 训练数据量（tokens）
            E: 收敛损失（当N,D→∞时）
            A, B: 拟合参数
            alpha, beta: 缩放指数

        Returns:
            预测的损失值
        """
        return E + A / (N ** alpha) + B / (D ** beta)

    # 示例：计算不同配置的损失
    configs = [
        {"N": 1e8, "D": 1e9, "name": "100M参数, 1B tokens"},
        {"N": 1e9, "D": 1e10, "name": "1B参数, 10B tokens"},
        {"N": 7e10, "D": 1.4e12, "name": "70B参数, 1.4T tokens (Chinchilla)"}
    ]

    for config in configs:
        loss = chinchilla_scaling_law(config["N"], config["D"])
        print(f"\n{config['name']}")
        print(f"  预测损失: {loss:.4f}")

    # 对比GPT-3
    print("\n" + "-" * 40)
    print("GPT-3 vs Chinchilla最优:")
    N_gpt3, D_gpt3 = 175e9, 300e9
    N_chinchilla, D_chinchilla = 70e9, 1.4e12

    loss_gpt3 = chinchilla_scaling_law(N_gpt3, D_gpt3)
    loss_chinchilla = chinchilla_scaling_law(N_chinchilla, D_chinchilla)

    print(f"\nGPT-3 (175B参数, 300B tokens):")
    print(f"  损失: {loss_gpt3:.4f}")

    print(f"\nChinchilla最优 (70B参数, 1.4T tokens):")
    print(f"  损失: {loss_chinchilla:.4f}")
    print(f"  提升: {(loss_gpt3 - loss_chinchilla) / loss_gpt3 * 100:.1f}%")


def example_2_compute_budget():
    """示例2: 计算预算与最优配置"""
    print("\n" + "=" * 60)
    print("示例2: 计算预算分析")
    print("=" * 60)

    def calculate_compute(N, D):
        """计算训练FLOPs"""
        return 6 * N * D

    def compute_optimal_nd(C, A=400, B=400, alpha=0.35, beta=0.37):
        """
        计算给定计算预算下的最优N和D

        Args:
            C: 计算预算（FLOPs）
            A, B, alpha, beta: Scaling Laws参数

        Returns:
            (N_opt, D_opt): 最优的模型大小和数据量
        """
        ratio = (A * alpha) / (B * beta)

        N_opt = (C / 6) ** (1 / (alpha + beta)) * ratio ** (beta / (alpha + beta))
        D_opt = (C / 6) ** (1 / (alpha + beta)) * (1 / ratio) ** (alpha / (alpha + beta))

        return N_opt, D_opt

    # 不同计算预算的最优配置
    budgets = [1e21, 1e22, 1e23, 1e24]

    print(f"{'预算(FLOPs)':<15} {'N_opt(参数)':<15} {'D_opt(tokens)':<15} {'预测损失':<10}")
    print("-" * 60)

    for C in budgets:
        N_opt, D_opt = compute_optimal_nd(C)
        loss = 1.8 + 400 / (N_opt ** 0.35) + 400 / (D_opt ** 0.37)

        print(f"{C:<15.0e} {N_opt/1e9:<15.2f} {D_opt/1e12:<15.2f} {loss:<10.4f}")


# ============================================================================
# Part 2: 拟合Scaling Laws
# ============================================================================

def example_3_fit_scaling_laws():
    """示例3: 拟合Scaling Laws"""
    print("\n" + "=" * 60)
    print("示例3: 拟合Scaling Laws参数")
    print("=" * 60)

    # 模拟实验数据
    experiments = [
        {"N": 100e6, "D": 1e9, "loss": 3.2},
        {"N": 200e6, "D": 2e9, "loss": 2.9},
        {"N": 500e6, "D": 5e9, "loss": 2.5},
        {"N": 1e9, "D": 1e10, "loss": 2.2},
        {"N": 2e9, "D": 2e10, "loss": 2.0},
    ]

    print("实验数据:")
    for exp in experiments:
        print(f"  N={exp['N']:.0e}, D={exp['D']:.0e} → loss={exp['loss']:.2f}")

    # 准备数据
    N = np.array([e["N"] for e in experiments])
    D = np.array([e["D"] for e in experiments])
    loss = np.array([e["loss"] for e in experiments])

    # 定义Scaling Law函数
    def scaling_law(x, E, A, B, alpha, beta):
        N, D = x
        return E + A / (N ** alpha) + B / (D ** beta)

    # 初始猜测
    initial_guess = [1.8, 400, 400, 0.35, 0.37]

    # 拟合
    params, _ = curve_fit(
        scaling_law,
        (N, D),
        loss,
        p0=initial_guess,
        maxfev=10000
    )

    E, A, B, alpha, beta = params

    print("\n拟合参数:")
    print(f"  E = {E:.4f}")
    print(f"  A = {A:.4f}")
    print(f"  B = {B:.4f}")
    print(f"  α = {alpha:.4f}")
    print(f"  β = {beta:.4f}")

    # 验证拟合
    print("\n拟合验证:")
    for exp in experiments:
        predicted = scaling_law((exp["N"], exp["D"]), *params)
        actual = exp["loss"]
        error = abs(predicted - actual) / actual * 100
        print(f"  实际={actual:.2f}, 预测={predicted:.2f}, 误差={error:.1f}%")


# ============================================================================
# Part 3: IsoFLOPs曲线
# ============================================================================

def example_4_isoflops_curve():
    """示例4: 绘制IsoFLOPs曲线"""
    print("\n" + "=" * 60)
    print("示例4: IsoFLOPs曲线分析")
    print("=" * 60)

    E, A, B, alpha, beta = 1.8, 400, 400, 0.35, 0.37
    C = 1e23  # 固定计算预算

    N_range = np.logspace(7, 10, 100)  # 10M - 10B

    # 计算对应的D (C = 6ND)
    D_range = C / (6 * N_range)

    # 计算损失
    losses = E + A / (N_range ** alpha) + B / (D_range ** beta)

    # 找到最优
    optimal_idx = np.argmin(losses)
    N_opt = N_range[optimal_idx]
    D_opt = D_range[optimal_idx]
    loss_opt = losses[optimal_idx]

    print(f"计算预算: C = {C:.0e} FLOPs")
    print(f"最优配置:")
    print(f"  N_opt = {N_opt:.0e} ({N_opt/1e9:.2f}B 参数)")
    print(f"  D_opt = {D_opt:.0e} ({D_opt/1e12:.2f}T tokens)")
    print(f"  最优损失 = {loss_opt:.4f}")

    # 简化版打印（不实际绘图，因为可能无GUI）
    print("\nIsoFLOPs曲线关键点:")
    print(f"{'N(参数)':<15} {'D(tokens)':<15} {'损失':<10}")
    print("-" * 40)

    for i in [0, 25, 50, 75, 99]:
        N = N_range[i]
        D = D_range[i]
        loss = losses[i]
        print(f"{N/1e9:<15.2f} {D/1e12:<15.2f} {loss:<10.4f}")


# ============================================================================
# Part 4: 性能预测
# ============================================================================

def example_5_performance_prediction():
    """示例5: 模型性能预测"""
    print("\n" + "=" * 60)
    print("示例5: 模型性能预测")
    print("=" * 60)

    class ScalingLawPredictor:
        """Scaling Laws预测器"""

        def __init__(self, E=1.8, A=400, B=400, alpha=0.35, beta=0.37):
            self.E = E
            self.A = A
            self.B = B
            self.alpha = alpha
            self.beta = beta

        def predict_loss(self, N, D):
            """预测损失"""
            return self.E + self.A / (N ** self.alpha) + self.B / (D ** self.beta)

        def compute_optimal(self, C):
            """计算最优配置"""
            ratio = (self.A * self.alpha) / (self.B * self.beta)

            N_opt = (C / 6) ** (1 / (self.alpha + self.beta)) * ratio ** (
                        self.beta / (self.alpha + self.beta))
            D_opt = (C / 6) ** (1 / (self.alpha + self.beta)) * (1 / ratio) ** (
                        self.alpha / (self.alpha + self.beta))

            return N_opt, D_opt

        def predict_for_budget(self, C):
            """预测给定预算下的最优性能"""
            N_opt, D_opt = self.compute_optimal(C)
            loss_opt = self.predict_loss(N_opt, D_opt)

            return {
                "N": N_opt,
                "D": D_opt,
                "loss": loss_opt,
                "compute": C
            }

    # 创建预测器
    predictor = ScalingLawPredictor()

    # 预测不同预算下的性能
    budgets = [1e21, 1e22, 1e23, 1e24]

    print(f"{'预算(FLOPs)':<15} {'模型大小(B)':<12} {'数据量(T)':<12} {'预测损失':<10}")
    print("-" * 50)

    for C in budgets:
        result = predictor.predict_for_budget(C)
        print(f"{C:<15.0e} {result['N']/1e9:<12.2f} {result['D']/1e12:<12.2f} {result['loss']:<10.4f}")

    # 实际应用：规划模型训练
    print("\n" + "=" * 60)
    print("实际应用：给定目标损失，计算所需预算")

    target_loss = 2.0

    # 反推需要的计算量（简化版）
    # target_loss = E + A/N^α + B/D^β
    # 且 N和D满足最优关系

    # 通过二分查找
    def find_required_budget(target_loss):
        low, high = 1e20, 1e26
        for _ in range(30):
            mid = (low + high) / 2
            N, D = predictor.compute_optimal(mid)
            loss = predictor.predict_loss(N, D)

            if loss > target_loss:
                low = mid
            else:
                high = mid

        return (low + high) / 2

    required_C = find_required_budget(target_loss)
    N_req, D_req = predictor.compute_optimal(required_C)

    print(f"\n目标损失: {target_loss}")
    print(f"所需预算: {required_C:.0e} FLOPs")
    print(f"最优配置: {N_req/1e9:.2f}B参数, {D_req/1e12:.2f}T tokens")


# ============================================================================
# 运行所有示例
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Scaling Laws - 代码示例")
    print("=" * 60)

    example_1_chinchilla_formula()
    example_2_compute_budget()
    example_3_fit_scaling_laws()
    example_4_isoflops_curve()
    example_5_performance_prediction()

    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)

    print("\n关键要点:")
    print("1. Chinchilla Scaling Laws: L(N,D) = E + A/N^α + B/D^β")
    print("2. 计算最优配置: N和D应该同比例增长")
    print("3. IsoFLOPs曲线展示固定预算下N和D的权衡")
    print("4. 可以拟合Scaling Laws参数来预测模型性能")
    print("5. 给定计算预算可以找到最优的模型大小和数据量")
