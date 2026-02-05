"""
Day 24-25: Scaling Laws - 练习题
难度：⭐⭐ ~ ⭐⭐⭐⭐
"""

import numpy as np
from scipy.optimize import curve_fit
from typing import List, Dict


# ============================================================================
# 练习1: 实现Scaling Laws预测器（⭐⭐⭐）
# ============================================================================

def exercise_1_scaling_law_predictor():
    """
    任务：实现一个Scaling Laws预测器类

    要求：
    1. 实现Chinchilla Scaling Laws公式
    2. 支持损失预测
    3. 支持最优配置计算

    提示：
    - L(N,D) = E + A/N^α + B/D^β
    - 最优配置时：N和D满足一定比例关系
    """
    class ScalingLawPredictor:
        """Scaling Laws预测器"""

        def __init__(self, E=1.8, A=400, B=400, alpha=0.35, beta=0.37):
            self.E = E
            self.A = A
            self.B = B
            self.alpha = alpha
            self.beta = beta

        def predict_loss(self, N, D):
            """
            预测给定模型大小和数据量的损失

            Args:
                N: 模型参数量
                D: 训练数据量（tokens）

            Returns:
                预测的损失值
            """
            # TODO: 实现Chinchilla Scaling Laws公式
            raise NotImplementedError

        def compute_optimal(self, C):
            """
            计算给定计算预算下的最优N和D

            Args:
                C: 计算预算（FLOPs）

            Returns:
                (N_opt, D_opt): 最优的模型大小和数据量
            """
            # TODO: 实现最优配置计算
            # 提示：
            # 1. C = 6ND（训练FLOPs）
            # 2. 最优比例：N/D = (Aα/Bβ)^(1/(α+β))
            raise NotImplementedError

    # 测试你的实现
    predictor = ScalingLawPredictor()

    # 测试1: 预测损失
    N, D = 1e9, 1e10
    loss = predictor.predict_loss(N, D)
    print(f"练习1: N={N:.0e}, D={D:.0e} → 预测损失={loss:.4f}")

    # 测试2: 计算最优配置
    C = 1e23
    N_opt, D_opt = predictor.compute_optimal(C)
    print(f"练习1: C={C:.0e} → N_opt={N_opt/1e9:.2f}B, D_opt={D_opt/1e12:.2f}T")

    print("✅ 练习1完成: Scaling Laws预测器实现")


# ============================================================================
# 练习2: 拟合Scaling Laws参数（⭐⭐⭐⭐）
# ============================================================================

def exercise_2_fit_scaling_laws():
    """
    任务：给定实验数据，拟合Scaling Laws参数

    要求：
    1. 实现数据准备
    2. 使用scipy.optimize.curve_fit拟合
    3. 验证拟合效果

    提示：
    - 使用curve_fit进行非线性最小二乘拟合
    - 提供合理的初始猜测值
    """
    # 模拟实验数据
    experiments = [
        {"N": 100e6, "D": 1e9, "loss": 3.2},
        {"N": 200e6, "D": 2e9, "loss": 2.9},
        {"N": 500e6, "D": 5e9, "loss": 2.5},
        {"N": 1e9, "D": 1e10, "loss": 2.2},
        {"N": 2e9, "D": 2e10, "loss": 2.0},
    ]

    # TODO: 实现拟合逻辑
    # 1. 准备数据
    N_data = np.array([e["N"] for e in experiments])
    D_data = np.array([e["D"] for e in experiments])
    loss_data = np.array([e["loss"] for e in experiments])

    # 2. 定义Scaling Law函数
    def scaling_law(x, E, A, B, alpha, beta):
        N, D = x
        # TODO: 实现公式
        raise NotImplementedError

    # 3. 拟合参数
    initial_guess = [1.8, 400, 400, 0.35, 0.37]
    # TODO: 使用curve_fit拟合
    raise NotImplementedError

    # 4. 验证拟合
    print("拟合参数:")
    print(f"  E = {params[0]:.4f}")
    print(f"  A = {params[1]:.4f}")
    print(f"  B = {params[2]:.4f}")
    print(f"  α = {params[3]:.4f}")
    print(f"  β = {params[4]:.4f}")

    print("\n拟合验证:")
    for exp in experiments:
        predicted = scaling_law((exp["N"], exp["D"]), *params)
        error = abs(predicted - exp["loss"]) / exp["loss"]
        print(f"  实际={exp["loss"]:.2f}, 预测={predicted:.2f}, 误差={error*100:.1f}%")

    print("✅ 练习2完成: Scaling Laws参数拟合")


# ============================================================================
# 练习3: 模型训练规划（⭐⭐⭐⭐⭐）
# ============================================================================

def exercise_3_training_planner():
    """
    任务：实现一个模型训练规划器

    要求：
    1. 给定目标损失，计算所需计算预算
    2. 给定计算预算，计算可达到的损失
    3. 输出最优训练配置

    提示：
    - 需要反推Scaling Laws公式
    - 可以使用二分查找求解
    """
    class TrainingPlanner:
        """模型训练规划器"""

        def __init__(self):
            # 默认参数（来自Chinchilla论文）
            self.E = 1.8
            self.A = 400
            self.B = 400
            self.alpha = 0.35
            self.beta = 0.37

        def compute_required_budget(self, target_loss):
            """
            计算达到目标损失所需的计算预算

            Args:
                target_loss: 目标损失值

            Returns:
                C: 所需的计算预算（FLOPs）
            """
            # TODO: 实现反向求解
            # 提示：
            # 1. 使用二分查找
            # 2. 对于每个预算C，计算最优配置和损失
            # 3. 找到使损失≈target_loss的C
            raise NotImplementedError

        def plan_training(self, compute_budget):
            """
            规划最优训练配置

            Args:
                compute_budget: 可用的计算预算（FLOPs）

            Returns:
                dict: 包含N, D, 预测损失等信息
            """
            # TODO: 实现训练规划
            # 1. 计算最优N和D
            # 2. 预测损失
            # 3. 返回完整规划
            raise NotImplementedError

    # 测试你的实现
    planner = TrainingPlanner()

    # 测试1: 给定目标损失，计算预算
    target_loss = 2.0
    required_budget = planner.compute_required_budget(target_loss)
    print(f"\n练习3: 目标损失={target_loss}")
    print(f"  所需预算: {required_budget:.0e} FLOPs")

    # 测试2: 给定预算，规划训练
    budget = 1e23
    plan = planner.plan_training(budget)
    print(f"\n练习3: 预算={budget:.0e}")
    print(f"  最优配置:")
    print(f"    N = {plan['N']/1e9:.2f}B 参数")
    print(f"    D = {plan['D']/1e12:.2f}T tokens")
    print(f"  预测损失: {plan['loss']:.4f}")

    print("✅ 练习3完成: 模型训练规划器")


# ============================================================================
# 运行所有练习
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Scaling Laws - 练习题")
    print("=" * 60)

    exercises = [
        ("练习1: Scaling Laws预测器", exercise_1_scaling_law_predictor),
        ("练习2: 拟合Scaling Laws参数", exercise_2_fit_scaling_laws),
        ("练习3: 模型训练规划器", exercise_3_training_planner),
    ]

    for name, exercise_func in exercises:
        print(f"\n{'=' * 60}")
        print(f"{name}")
        print('=' * 60)
        try:
            exercise_func()
        except NotImplementedError as e:
            print(f"⚠️  待实现: {e}")
        except Exception as e:
            print(f"❌ 错误: {e}")

    print(f"\n{'=' * 60}")
    print("练习完成！")
    print('=' * 60)
