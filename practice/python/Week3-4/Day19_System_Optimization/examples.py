"""
Day 19: 系统优化与性能调优 - 代码示例
涵盖：PyTorch Profiler、DataLoader优化、混合精度、分布式调试
"""

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.profiler import profile, ProfilerActivity, record_function
from torch.cuda.amp import autocast, GradScaler
import time
import numpy as np


# ============================================================================
# Part 1: PyTorch Profiler 使用
# ============================================================================

def example_1_profiler_basic():
    """示例1: PyTorch Profiler基础使用"""
    print("=" * 60)
    print("示例1: PyTorch Profiler基础使用")
    print("=" * 60)

    # 创建简单模型
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    ).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 创建数据
    x = torch.randn(32, 128).cuda()
    y = torch.randint(0, 10, (32,)).cuda()

    # 使用Profiler分析性能
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True
    ) as prof:
        for _ in range(10):
            optimizer.zero_grad()

            with record_function("forward_pass"):
                output = model(x)
                loss = criterion(output, y)

            with record_function("backward_pass"):
                loss.backward()

            with record_function("optimizer_step"):
                optimizer.step()

    # 打印分析结果
    print("\nCUDA时间排序（前10个操作）:")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=10
    ))

    print("\nCPU时间排序（前10个操作）:")
    print(prof.key_averages().table(
        sort_by="cpu_time_total",
        row_limit=10
    ))


def example_2_profiler_memory():
    """示例2: 分析内存使用"""
    print("\n" + "=" * 60)
    print("示例2: 内存使用分析")
    print("=" * 60)

    model = nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 128)
    ).cuda()

    x = torch.randn(16, 256).cuda()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True
    ) as prof:
        output = model(x)

    print("\n内存使用统计:")
    print(prof.key_averages().table(
        sort_by="self_cuda_memory_usage",
        row_limit=10
    ))


def example_3_profiler_trace():
    """示例3: 生成性能追踪文件"""
    print("\n" + "=" * 60)
    print("示例3: 生成性能追踪文件")
    print("=" * 60)

    model = nn.Linear(64, 64).cuda()
    x = torch.randn(8, 64).cuda()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True
    ) as prof:
        for _ in range(5):
            y = model(x)

    # 导出为Chrome Trace格式
    prof.export_chrome_trace("trace.json")
    print("✅ 性能追踪已保存到 trace.json")
    print("提示: 在Chrome中打开 chrome://tracing 查看详情")


# ============================================================================
# Part 2: DataLoader 优化
# ============================================================================

class DummyDataset(Dataset):
    """简单的数据集用于演示"""
    def __init__(self, size=1000, dim=128):
        self.size = size
        self.dim = dim

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 模拟耗时的数据加载
        time.sleep(0.001)
        return torch.randn(self.dim), torch.randint(0, 10, (1,)).item()


def example_4_dataloader_comparison():
    """示例4: 比较不同DataLoader配置的性能"""
    print("\n" + "=" * 60)
    print("示例4: DataLoader配置对比")
    print("=" * 60)

    dataset = DummyDataset(size=100, dim=64)

    configurations = [
        {
            "name": "默认配置",
            "kwargs": {"batch_size": 16, "shuffle": True}
        },
        {
            "name": "优化配置",
            "kwargs": {
                "batch_size": 16,
                "shuffle": True,
                "num_workers": 4,
                "pin_memory": True,
                "prefetch_factor": 2,
                "persistent_workers": True
            }
        }
    ]

    for config in configurations:
        print(f"\n{config['name']}:")
        print("-" * 40)

        dataloader = DataLoader(dataset, **config['kwargs'])

        # 测量数据加载时间
        start = time.time()
        for i, batch in enumerate(dataloader):
            if i >= 10:  # 只测试前10个batch
                break
        elapsed = time.time() - start

        print(f"10个batch加载时间: {elapsed:.3f}s")
        print(f"平均每个batch: {elapsed/10*1000:.1f}ms")


def example_5_custom_collate():
    """示例5: 自定义collate_fn优化"""
    print("\n" + "=" * 60)
    print("示例5: 自定义collate_fn")
    print("=" * 60)

    class FastDataset(Dataset):
        def __len__(self):
            return 100

        def __getitem__(self, idx):
            return {
                'image': torch.randn(3, 32, 32),
                'label': torch.randint(0, 10, (1,))
            }

    # 标准collate
    def standard_collate(batch):
        images = torch.stack([item['image'] for item in batch])
        labels = torch.stack([item['label'] for item in batch])
        return {'image': images, 'label': labels}

    # 优化的collate
    def optimized_collate(batch):
        # 预分配内存
        batch_size = len(batch)
        images = torch.zeros(batch_size, 3, 32, 32)
        labels = torch.zeros(batch_size, 1, dtype=torch.long)

        for i, item in enumerate(batch):
            images[i] = item['image']
            labels[i] = item['label']

        return {'image': images, 'label': labels}

    dataset = FastDataset()

    # 测试标准collate
    dataloader_slow = DataLoader(
        dataset,
        batch_size=32,
        collate_fn=standard_collate
    )

    start = time.time()
    for batch in dataloader_slow:
        pass
    slow_time = time.time() - start

    # 测试优化collate
    dataloader_fast = DataLoader(
        dataset,
        batch_size=32,
        collate_fn=optimized_collate
    )

    start = time.time()
    for batch in dataloader_fast:
        pass
    fast_time = time.time() - start

    print(f"标准collate: {slow_time*1000:.2f}ms")
    print(f"优化collate: {fast_time*1000:.2f}ms")
    print(f"加速比: {slow_time/fast_time:.2f}x")


# ============================================================================
# Part 3: 混合精度训练
# ============================================================================

def example_6_mixed_precision():
    """示例6: 混合精度训练"""
    print("\n" + "=" * 60)
    print("示例6: 混合精度训练")
    print("=" * 60)

    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    ).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 创建GradScaler
    scaler = GradScaler()

    x = torch.randn(64, 128).cuda()
    y = torch.randint(0, 10, (64,)).cuda()

    # 测试FP32训练
    print("\nFP32训练:")
    model_fp32 = model.clone().cuda()
    optimizer_fp32 = torch.optim.Adam(model_fp32.parameters(), lr=1e-3)

    start = time.time()
    for _ in range(100):
        optimizer_fp32.zero_grad()
        output = model_fp32(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer_fp32.step()
    fp32_time = time.time() - start
    print(f"100步训练时间: {fp32_time:.3f}s")

    # 测试混合精度训练
    print("\n混合精度训练 (BF16):")
    model_amp = model.clone().cuda()
    optimizer_amp = torch.optim.Adam(model_amp.parameters(), lr=1e-3)

    start = time.time()
    for _ in range(100):
        optimizer_amp.zero_grad()

        with autocast(dtype=torch.bfloat16):
            output = model_amp(x)
            loss = criterion(output, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer_amp)
        scaler.update()
    amp_time = time.time() - start
    print(f"100步训练时间: {amp_time:.3f}s")
    print(f"加速比: {fp32_time/amp_time:.2f}x")


def example_7_gradient_clipping_with_amp():
    """示例7: 混合精度 + 梯度裁剪"""
    print("\n" + "=" * 60)
    print("示例7: 混合精度 + 梯度裁剪")
    print("=" * 60)

    model = nn.Linear(64, 64).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler()

    x = torch.randn(32, 64).cuda()
    y = torch.randint(0, 10, (32,)).cuda()

    # 训练步骤（包含梯度裁剪）
    for step in range(10):
        optimizer.zero_grad()

        with autocast(dtype=torch.bfloat16):
            output = model(x)
            loss = nn.functional.cross_entropy(output, y)

        # 反向传播
        scaler.scale(loss).backward()

        # 取消缩放（重要！）
        scaler.unscale_(optimizer)

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 检查梯度
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))

        # 更新参数
        scaler.step(optimizer)
        scaler.update()

        print(f"Step {step}: Loss={loss.item():.4f}, Grad_norm={grad_norm:.4f}")


# ============================================================================
# Part 4: 梯度检查点
# ============================================================================

def example_8_gradient_checkpointing():
    """示例8: 梯度检查点"""
    print("\n" + "=" * 60)
    print("示例8: 梯度检查点")
    print("=" * 60)

    from torch.utils.checkpoint import checkpoint

    class DeepModel(nn.Module):
        def __init__(self, dim=64, n_layers=10):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim, dim * 2),
                    nn.ReLU(),
                    nn.Linear(dim * 2, dim)
                ) for _ in range(n_layers)
            ])

        def forward(self, x, use_checkpoint=False):
            for i, layer in enumerate(self.layers):
                if use_checkpoint and i > 0 and i % 2 == 0:
                    # 使用checkpoint
                    x = checkpoint(layer, x)
                else:
                    x = layer(x)
            return x

    model_no_ckpt = DeepModel(dim=64, n_layers=10).cuda()
    model_ckpt = DeepModel(dim=64, n_layers=10).cuda()

    # 复制权重
    model_ckpt.load_state_dict(model_no_ckpt.state_dict())

    x = torch.randn(16, 64).cuda()
    y = torch.randn(16, 64).cuda()

    # 测试无checkpoint
    torch.cuda.reset_peak_memory_stats()
    start = time.time()

    for _ in range(10):
        output = model_no_ckpt(x)
        loss = nn.functional.mse_loss(output, y)
        loss.backward()

    no_ckpt_time = time.time() - start
    no_ckpt_memory = torch.cuda.max_memory_allocated() / 1e9

    print(f"\n无Checkpoint:")
    print(f"  时间: {no_ckpt_time:.3f}s")
    print(f"  峰值显存: {no_ckpt_memory:.2f} GB")

    # 测试有checkpoint
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    start = time.time()

    for _ in range(10):
        output = model_ckpt(x, use_checkpoint=True)
        loss = nn.functional.mse_loss(output, y)
        loss.backward()

    ckpt_time = time.time() - start
    ckpt_memory = torch.cuda.max_memory_allocated() / 1e9

    print(f"\n有Checkpoint:")
    print(f"  时间: {ckpt_time:.3f}s")
    print(f"  峰值显存: {ckpt_memory:.2f} GB")
    print(f"\n显存节省: {(1 - ckpt_memory/no_ckpt_memory) * 100:.1f}%")
    print(f"时间代价: {(ckpt_time/no_ckpt_time - 1) * 100:+.1f}%")


# ============================================================================
# Part 5: 内存优化
# ============================================================================

def example_9_memory_monitoring():
    """示例9: 显存监控"""
    print("\n" + "=" * 60)
    print("示例9: 显存监控")
    print("=" * 60)

    def print_memory_usage(stage):
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"{stage:20s}: Allocated={allocated:6.2f}GB, Reserved={reserved:6.2f}GB")

    model = nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 256)
    ).cuda()

    x = torch.randn(32, 256).cuda()

    print("\n训练循环中的显存使用:")
    print_memory_usage("初始化")

    # 前向传播
    output = model(x)
    print_memory_usage("前向传播后")

    # 计算loss
    loss = output.sum()
    print_memory_usage("loss计算后")

    # 反向传播
    loss.backward()
    print_memory_usage("反向传播后")

    # 清空缓存
    del output, loss
    torch.cuda.empty_cache()
    print_memory_usage("清理后")


def example_10_inplace_operations():
    """示例10: Inplace操作优化"""
    print("\n" + "=" * 60)
    print("示例10: Inplace操作优化")
    print("=" * 60)

    # 测试非inplace操作
    x = torch.randn(1000, 1000).cuda()

    torch.cuda.reset_peak_memory_stats()
    start = time.time()

    for _ in range(100):
        y = x + 1
        y = torch.relu(y)
        y = torch.sigmoid(y)

    no_inplace_time = time.time() - start
    no_inplace_memory = torch.cuda.max_memory_allocated() / 1e9

    print(f"\n非inplace操作:")
    print(f"  时间: {no_inplace_time:.3f}s")
    print(f"  显存: {no_inplace_memory:.2f} GB")

    # 测试inplace操作
    x = torch.randn(1000, 1000).cuda()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    start = time.time()

    for _ in range(100):
        x += 1
        torch.relu_(x)
        x = torch.sigmoid(x)

    inplace_time = time.time() - start
    inplace_memory = torch.cuda.max_memory_allocated() / 1e9

    print(f"\nInplace操作:")
    print(f"  时间: {inplace_time:.3f}s")
    print(f"  显存: {inplace_memory:.2f} GB")
    print(f"\n显存节省: {(1 - inplace_memory/no_inplace_memory) * 100:.1f}%")


# ============================================================================
# Part 6: 性能优化实战
# ============================================================================

def example_11_full_optimization_pipeline():
    """示例11: 完整的优化流程"""
    print("\n" + "=" * 60)
    print("示例11: 完整优化流程")
    print("=" * 60)

    class OptimizedTrainingPipeline:
        def __init__(self, model, dataset):
            self.model = model.cuda()
            self.dataset = dataset

            # 优化器
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=1e-3,
                weight_decay=0.01
            )

            # 混合精度
            self.scaler = GradScaler()

            # 优化的DataLoader
            self.dataloader = DataLoader(
                dataset,
                batch_size=32,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True,
                drop_last=True
            )

        def train_step(self, batch):
            """优化的训练步骤"""
            x, y = batch
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

            self.optimizer.zero_grad()

            # 混合精度
            with autocast(dtype=torch.bfloat16):
                output = self.model(x)
                loss = nn.functional.cross_entropy(output, y)

            # 反向传播
            self.scaler.scale(loss).backward()

            # 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # 更新
            self.scaler.step(self.optimizer)
            self.scaler.update()

            return loss.item()

        def train_epoch(self):
            """训练一个epoch"""
            self.model.train()
            total_loss = 0.0

            for batch in self.dataloader:
                loss = self.train_step(batch)
                total_loss += loss

            return total_loss / len(self.dataloader)

    # 创建模型和数据
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )

    dataset = DummyDataset(size=1000, dim=128)

    # 创建训练pipeline
    pipeline = OptimizedTrainingPipeline(model, dataset)

    # 训练
    print("\n开始训练...")
    start = time.time()

    avg_loss = pipeline.train_epoch()

    elapsed = time.time() - start

    print(f"\n训练完成:")
    print(f"  时间: {elapsed:.3f}s")
    print(f"  平均loss: {avg_loss:.4f}")


def example_12_benchmark_comparison():
    """示例12: 性能基准测试对比"""
    print("\n" + "=" * 60)
    print("示例12: 性能基准测试")
    print("=" * 60)

    def benchmark_model(model_config, data_config, training_config):
        """基准测试函数"""
        # 创建模型
        model = nn.Sequential(
            nn.Linear(model_config['input_dim'], model_config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(model_config['hidden_dim'], model_config['output_dim'])
        ).cuda()

        # 优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=training_config['lr'])

        # 数据
        x = torch.randn(data_config['batch_size'], model_config['input_dim']).cuda()
        y = torch.randint(0, model_config['output_dim'], (data_config['batch_size'],)).cuda()

        # 预热
        for _ in range(10):
            output = model(x)
            loss = nn.functional.cross_entropy(output, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize()

        # 测试
        start = time.time()
        for _ in range(training_config['num_steps']):
            output = model(x)
            loss = nn.functional.cross_entropy(output, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        torch.cuda.synchronize()

        elapsed = time.time() - start
        throughput = training_config['num_steps'] * data_config['batch_size'] / elapsed

        return {
            'time': elapsed,
            'throughput': throughput
        }

    # 测试不同配置
    configs = [
        {
            'name': '基线',
            'model': {'input_dim': 128, 'hidden_dim': 256, 'output_dim': 10},
            'data': {'batch_size': 32},
            'training': {'lr': 1e-3, 'num_steps': 100}
        },
        {
            'name': '大batch',
            'model': {'input_dim': 128, 'hidden_dim': 256, 'output_dim': 10},
            'data': {'batch_size': 64},
            'training': {'lr': 1e-3, 'num_steps': 100}
        },
        {
            'name': '大模型',
            'model': {'input_dim': 128, 'hidden_dim': 512, 'output_dim': 10},
            'data': {'batch_size': 32},
            'training': {'lr': 1e-3, 'num_steps': 100}
        }
    ]

    print("\n配置对比:")
    print(f"{'配置':<15} {'时间(s)':<12} {'吞吐量(samples/s)':<20}")
    print("-" * 50)

    for config in configs:
        result = benchmark_model(
            config['model'],
            config['data'],
            config['training']
        )
        print(f"{config['name']:<15} {result['time']:<12.3f} {result['throughput']:<20.0f}")


# ============================================================================
# 运行所有示例
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("系统优化与性能调优 - 代码示例")
    print("="*60)

    # 注意：某些示例需要CUDA才能运行
    if not torch.cuda.is_available():
        print("⚠️  警告: 未检测到CUDA，某些示例将无法运行")
        print("建议在GPU环境下运行此脚本")

    # Profiler示例
    try:
        example_1_profiler_basic()
    except Exception as e:
        print(f"示例1运行失败: {e}")

    try:
        example_2_profiler_memory()
    except Exception as e:
        print(f"示例2运行失败: {e}")

    example_3_profiler_trace()

    # DataLoader示例
    example_4_dataloader_comparison()
    example_5_custom_collate()

    # 混合精度示例
    if torch.cuda.is_available():
        example_6_mixed_precision()
        example_7_gradient_clipping_with_amp()

    # 梯度检查点示例
    if torch.cuda.is_available():
        example_8_gradient_checkpointing()

    # 内存优化示例
    if torch.cuda.is_available():
        example_9_memory_monitoring()
        example_10_inplace_operations()

    # 实战示例
    if torch.cuda.is_available():
        example_11_full_optimization_pipeline()
        example_12_benchmark_comparison()

    print("\n" + "="*60)
    print("所有示例运行完成！")
    print("="*60)
    print("\n关键要点:")
    print("1. 使用PyTorch Profiler识别性能瓶颈")
    print("2. 优化DataLoader参数（num_workers, pin_memory等）")
    print("3. 使用混合精度训练提升速度和节省显存")
    print("4. 梯度检查点可以节省显存但会增加计算时间")
    print("5. 使用inplace操作减少显存占用")
    print("6. 始终用实际数据验证优化效果")
