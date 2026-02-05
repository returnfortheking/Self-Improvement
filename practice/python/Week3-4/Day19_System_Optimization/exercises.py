"""
Day 19: 系统优化与性能调优 - 练习题
难度：⭐⭐ ~ ⭐⭐⭐⭐
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.profiler import profile, ProfilerActivity
from torch.cuda.amp import autocast, GradScaler
import time


# ============================================================================
# 练习1: 使用PyTorch Profiler分析训练瓶颈（⭐⭐⭐）
# ============================================================================

def exercise_1_profiler_analysis():
    """
    任务：使用PyTorch Profiler分析训练瓶颈

    要求：
    1. 创建一个简单的模型和数据
    2. 使用Profiler记录训练过程
    3. 找出最耗时的操作
    4. 提出优化建议

    提示：
    - 使用ProfilerActivity.CUDA和ProfilerActivity.CPU
    - 使用sort_by="cuda_time_total"排序
    - 查找self_cuda_time占比高的操作
    """
    def analyze_training_bottleneck():
        """分析训练瓶颈"""
        # TODO: 创建模型
        model = nn.Sequential(
            # 添加你的层
        ).cuda()

        # TODO: 创建数据
        # TODO: 配置Profiler

        # TODO: 运行训练循环并记录

        # TODO: 分析结果
        # - 哪个操作最耗时？
        # - 是否有明显的瓶颈？
        # - 如何优化？

        raise NotImplementedError

    print("✅ 练习1完成: 提交你的分析和优化建议")


# ============================================================================
# 练习2: 优化DataLoader性能（⭐⭐⭐）
# ============================================================================

def exercise_2_dataloader_optimization():
    """
    任务：优化DataLoader性能

    要求：
    1. 创建一个自定义Dataset
    2. 测试不同DataLoader配置的性能
    3. 找到最优配置

    提示：
    - 测试不同的num_workers值
    - 比较pin_memory=True和False
    - 测试不同的prefetch_factor
    """
    class SlowDataset(Dataset):
        """模拟慢速数据加载"""
        def __init__(self, size=1000):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            # 模拟耗时操作
            time.sleep(0.001)
            return torch.randn(64, 64), torch.randint(0, 10, (1,)).item()

    def benchmark_dataloader_configs(dataset):
        """测试不同配置"""
        configs = [
            # TODO: 定义不同的DataLoader配置
            # 例如:
            # {"num_workers": 0, "pin_memory": False},
            # {"num_workers": 4, "pin_memory": True},
        ]

        results = []

        for config in configs:
            # TODO: 创建DataLoader
            # TODO: 测量加载时间
            # TODO: 记录结果
            pass

        # TODO: 分析结果，找出最优配置
        raise NotImplementedError

    dataset = SlowDataset(size=500)
    benchmark_dataloader_configs(dataset)

    print("✅ 练习2完成: 找到了最优DataLoader配置")


# ============================================================================
# 练习3: 实现混合精度训练（⭐⭐⭐）
# ============================================================================

def exercise_3_mixed_precision():
    """
    任务：实现混合精度训练

    要求：
    1. 使用autocast和GradScaler
    2. 正确处理梯度裁剪
    3. 对比FP32和混合精度的性能

    提示：
    - 使用torch.cuda.amp.autocast
    - 使用torch.cuda.amp.GradScaler
    - 梯度裁剪前需要unscale
    """
    def train_with_mixed_precision(model, dataloader, epochs=3):
        """使用混合精度训练"""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # TODO: 创建GradScaler
        # scaler = GradScaler()

        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.cuda(), target.cuda()

                optimizer.zero_grad()

                # TODO: 使用autocast包装前向传播
                # with autocast(dtype=torch.bfloat16):
                #     output = model(data)
                #     loss = criterion(output, target)

                # TODO: 反向传播
                # scaler.scale(loss).backward()

                # TODO: 梯度裁剪（重要！）
                # scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # TODO: 更新参数
                # scaler.step(optimizer)
                # scaler.update()

                raise NotImplementedError

    print("✅ 练习3完成: 混合精度训练实现")


# ============================================================================
# 练习4: 实现梯度检查点（⭐⭐⭐⭐）
# ============================================================================

def exercise_4_gradient_checkpointing():
    """
    任务：实现梯度检查点优化显存

    要求：
    1. 创建一个深层模型
    2. 在部分层使用checkpoint
    3. 对比使用checkpoint前后的显存占用

    提示：
    - 使用torch.utils.checkpoint.checkpoint
    - 每N层checkpoint一次
    - 使用torch.cuda.memory_allocated监控显存
    """
    from torch.utils.checkpoint import checkpoint

    class DeepModel(nn.Module):
        def __init__(self, dim=128, n_layers=20):
            super().__init__()
            # TODO: 创建20个全连接层
            self.layers = nn.ModuleList([
                nn.Linear(dim, dim)
                for _ in range(n_layers)
            ])

        def forward(self, x, use_checkpoint=False, checkpoint_interval=5):
            # TODO: 实现前向传播
            # 如果use_checkpoint=True，每checkpoint_interval层使用checkpoint
            # checkpoint(self.layers[i], x)
            raise NotImplementedError

    def compare_memory_usage():
        """对比显存使用"""
        model = DeepModel(dim=128, n_layers=20).cuda()
        x = torch.randn(16, 128).cuda()
        y = torch.randn(16, 128).cuda()

        # 测试无checkpoint
        # TODO: 测量显存

        # 测试有checkpoint
        # TODO: 测量显存

        # 对比结果
        raise NotImplementedError

    compare_memory_usage()
    print("✅ 练习4完成: 梯度检查点节省了显存")


# ============================================================================
# 练习5: 优化自定义collate_fn（⭐⭐）
# ============================================================================

def exercise_5_collate_optimization():
    """
    任务：优化自定义collate_fn

    要求：
    1. 实现一个标准的collate_fn
    2. 优化其性能
    3. 测量性能提升

    提示：
    - 预分配tensor
    - 使用torch.stack而不是循环
    - 减少不必要的复制
    """
    class CustomDataset(Dataset):
        def __len__(self):
            return 100

        def __getitem__(self, idx):
            return {
                'image': torch.randn(3, 224, 224),
                'label': torch.tensor([idx % 10]),
                'metadata': {'id': idx, 'source': 'synthetic'}
            }

    def standard_collate(batch):
        """标准实现"""
        # TODO: 实现标准collate
        raise NotImplementedError

    def optimized_collate(batch):
        """优化实现"""
        # TODO: 优化collate
        # 提示：
        # - 预分配内存
        # - 使用列表推导式
        # - 批量操作
        raise NotImplementedError

    def benchmark_collate(collate_fn, name):
        """测试collate性能"""
        dataset = CustomDataset()
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            collate_fn=collate_fn
        )

        start = time.time()
        for batch in dataloader:
            pass
        elapsed = time.time() - start

        print(f"{name}: {elapsed*1000:.2f}ms")
        return elapsed

    standard_time = benchmark_collate(standard_collate, "标准collate")
    optimized_time = benchmark_collate(optimized_collate, "优化collate")

    speedup = standard_time / optimized_time
    print(f"加速比: {speedup:.2f}x")

    print("✅ 练习5完成: collate_fn优化完成")


# ============================================================================
# 练习6: 内存分析和优化（⭐⭐⭐）
# ============================================================================

def exercise_6_memory_optimization():
    """
    任务：分析和优化内存使用

    要求：
    1. 监控训练过程中的内存使用
    2. 找出内存占用高的操作
    3. 应用优化技巧

    提示：
    - 使用torch.cuda.memory_allocated
    - 使用torch.cuda.max_memory_allocated
    - 及时删除中间变量
    - 使用inplace操作
    """
    def memory_intensive_operation():
        """内存密集型操作"""
        # 创建大tensor
        tensors = []
        for i in range(10):
            tensors.append(torch.randn(1000, 1000).cuda())

        # 逐个处理
        results = []
        for t in tensors:
            results.append(t.sum())

        return results

    def optimized_operation():
        """优化后的操作"""
        # TODO: 优化内存使用
        # 提示：
        # - 及时删除不需要的tensor
        # - 使用inplace操作
        # - 分块处理
        raise NotImplementedError

    def compare_memory():
        """对比内存使用"""
        torch.cuda.empty_cache()

        # 测试原始版本
        torch.cuda.reset_peak_memory_stats()
        _ = memory_intensive_operation()
        original_memory = torch.cuda.max_memory_allocated() / 1e9

        torch.cuda.empty_cache()

        # 测试优化版本
        torch.cuda.reset_peak_memory_stats()
        _ = optimized_operation()
        optimized_memory = torch.cuda.max_memory_allocated() / 1e9

        print(f"原始版本显存: {original_memory:.2f} GB")
        print(f"优化版本显存: {optimized_memory:.2f} GB")
        print(f"节省: {(1 - optimized_memory/original_memory) * 100:.1f}%")

    compare_memory()
    print("✅ 练习6完成: 内存优化完成")


# ============================================================================
# 练习7: 性能基准测试（⭐⭐⭐）
# ============================================================================

def exercise_7_performance_benchmark():
    """
    任务：实现完整的性能基准测试

    要求：
    1. 测试不同配置的训练速度
    2. 计算吞吐量
    3. 生成对比报告

    提示：
    - 预热GPU
    - 使用torch.cuda.synchronize()
    - 多次运行取平均
    """
    def benchmark_configuration(config, num_iterations=100):
        """
        测试特定配置的性能

        config可能包含:
        - batch_size
        - model_size
        - use_mixed_precision
        - use_gradient_checkpointing
        """
        # TODO: 根据config创建模型和数据
        # TODO: 预热
        # TODO: 测试性能
        # TODO: 返回结果（时间、吞吐量等）
        raise NotImplementedError

    def run_comparative_benchmark():
        """运行对比测试"""
        configs = [
            {
                'name': 'baseline',
                'batch_size': 32,
                'model_size': 'small',
                'use_mixed_precision': False
            },
            {
                'name': 'mixed_precision',
                'batch_size': 32,
                'model_size': 'small',
                'use_mixed_precision': True
            },
            {
                'name': 'large_batch',
                'batch_size': 64,
                'model_size': 'small',
                'use_mixed_precision': True
            },
            # 添加更多配置
        ]

        results = []
        for config in configs:
            result = benchmark_configuration(config)
            results.append((config['name'], result))

        # TODO: 打印对比表格
        print("\n性能对比:")
        print(f"{'配置':<20} {'时间(s)':<12} {'吞吐量':<15}")
        print("-" * 50)
        for name, result in results:
            print(f"{name:<20} {result['time']:<12.3f} {result['throughput']:<15.0f}")

        raise NotImplementedError

    run_comparative_benchmark()
    print("✅ 练习7完成: 性能基准测试完成")


# ============================================================================
# 练习8: 综合优化实战（⭐⭐⭐⭐）
# ============================================================================

def exercise_8_comprehensive_optimization():
    """
    任务：综合运用各种优化技巧

    要求：
    1. 从一个未优化的训练代码开始
    2. 逐步应用各种优化
    3. 记录每个优化的效果
    4. 最终达到显著加速

    提示：
    - DataLoader优化
    - 混合精度
    - 梯度检查点（如需要）
    - 内存优化
    """
    class BaseModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 10)
            )

        def forward(self, x):
            return self.layers(x)

    def baseline_training(model, dataloader, epochs=5):
        """基线训练（未优化）"""
        # TODO: 实现基线训练
        raise NotImplementedError

    def optimized_training(model, dataloader, epochs=5):
        """优化后的训练"""
        # TODO: 应用所有优化
        # - DataLoader优化
        # - 混合精度
        # - 梯度裁剪
        # - 等
        raise NotImplementedError

    def comprehensive_optimization():
        """综合优化"""
        # 创建数据集
        dataset = torch.utils.data.TensorDataset(
            torch.randn(1000, 128),
            torch.randint(0, 10, (1000,))
        )

        # 基线
        print("运行基线训练...")
        model_baseline = BaseModel().cuda()
        dataloader_baseline = DataLoader(dataset, batch_size=32)
        baseline_time = baseline_training(model_baseline, dataloader_baseline)

        # 优化
        print("\n运行优化训练...")
        model_optimized = BaseModel().cuda()
        # TODO: 创建优化的DataLoader
        dataloader_optimized = None
        optimized_time = optimized_training(model_optimized, dataloader_optimized)

        speedup = baseline_time / optimized_time
        print(f"\n总加速比: {speedup:.2f}x")

        return speedup

    speedup = comprehensive_optimization()
    print(f"✅ 练习8完成: 综合优化达到{speedup:.2f}x加速")


# ============================================================================
# 运行所有练习
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("系统优化与性能调优 - 练习题")
    print("="*60)

    exercises = [
        ("练习1: Profiler分析", exercise_1_profiler_analysis),
        ("练习2: DataLoader优化", exercise_2_dataloader_optimization),
        ("练习3: 混合精度训练", exercise_3_mixed_precision),
        ("练习4: 梯度检查点", exercise_4_gradient_checkpointing),
        ("练习5: collate优化", exercise_5_collate_optimization),
        ("练习6: 内存优化", exercise_6_memory_optimization),
        ("练习7: 性能基准测试", exercise_7_performance_benchmark),
        ("练习8: 综合优化", exercise_8_comprehensive_optimization),
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
