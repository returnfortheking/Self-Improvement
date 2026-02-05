# Day 19: 系统优化与性能调优

> **学习目标**: 掌握PyTorch性能分析工具，学会识别和解决训练瓶颈，优化分布式训练性能
> **时间分配**: 6小时（理论2h + 实践4h）
> **难度**: ⭐⭐⭐⭐
> **重要性**: ⭐⭐⭐⭐⭐ (生产环境必备技能)

---

## 📚 核心概念

### 1. 性能分析框架

训练性能的三个维度：

```
┌─────────────────────────────────────────────────┐
│  训练性能 = 计算 + 数据加载 + 通信 (分布式)      │
├─────────────────────────────────────────────────┤
│  计算: 前向传播 + 反向传播 + 优化器更新          │
│  数据: DataLoader (CPU→GPU传输)                 │
│  通信: 梯度同步 (AllReduce)                      │
└─────────────────────────────────────────────────┘
```

**优化目标**:
- ✅ 最大化GPU利用率（>90%）
- ✅ 最小化CPU等待时间
- ✅ 减少通信开销

---

### 2. PyTorch Profiler深度解析

#### 2.1 Profiler基础

**PyTorch Profiler** 是性能分析的利器：

```python
from torch.profiler import profile, ProfilerActivity, record_function

with profile(
    activities=[
        ProfilerActivity.CPU,      # CPU活动
        ProfilerActivity.CUDA,     # GPU活动
    ],
    record_shapes=True,            # 记录tensor shapes
    profile_memory=True,           # 分析内存使用
    with_stack=True,               # 记录调用栈
) as prof:
    # 训练代码
    for batch in dataloader:
        output = model(batch)
        loss.backward()
        optimizer.step()

# 打印分析结果
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

#### 2.2 关键指标解读

**输出表格列含义**:

| 列名 | 含义 | 优化目标 |
|------|------|----------|
| **Name** | 操作名称 | - |
| **Self CUDA time** | 该操作自身GPU时间 | 越短越好 |
| **CUDA time total** | 包含子操作的总时间 | 识别瓶颈 |
| **Self CPU time** | 该操作CPU时间 | CPU利用率 |
| **CPU time total** | 总CPU时间 | - |
| **Number of calls** | 调用次数 | 减少不必要调用 |

**示例输出**:
```
-------------------------------------------------------
Name                   Self CUDA    CUDA time total
-------------------------------------------------------
aten::conv2d                  10.50ms           12.30ms
aten::convolution_backward    8.20ms           15.40ms
aten::relu                    0.05ms            0.05ms
ncclAllReduce                15.00ms           15.00ms  ← 通信瓶颈
-------------------------------------------------------
```

---

### 3. 数据加载优化

#### 3.1 DataLoader瓶颈分析

**问题**: GPU等待数据（GPU空闲）

**诊断**:
```python
# 在Profiler输出中查找
- DataLoader迭代: 应该<5ms
- CPU→GPU传输: pin_memory后应该<1ms
```

**优化参数**:

| 参数 | 默认值 | 优化建议 | 效果 |
|------|--------|----------|------|
| **num_workers** | 0 | 4-8（CPU核心数的一半） | ⭐⭐⭐⭐⭐ |
| **pin_memory** | False | True（训练时） | ⭐⭐⭐⭐ |
| **prefetch_factor** | 2 | 2-4 | ⭐⭐⭐ |
| **persistent_workers** | False | True（大数据集） | ⭐⭐ |

**最佳配置**:
```python
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=8,              # 并行数据加载
    pin_memory=True,             # 锁页内存，加速CPU→GPU传输
    prefetch_factor=4,           # 预取4个batch
    persistent_workers=True,     # 保持worker进程
    drop_last=True               # 丢弃最后不完整batch
)
```

#### 3.2 自定义Collate优化

**问题**: 默认collate_fn慢

**优化**:
```python
def custom_collate_fn(batch):
    """优化的batch collate函数"""
    # 使用torch.stack而不是循环
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])

    # 预先转移到GPU（如果使用pin_memory，这一步会自动优化）
    return images, labels

dataloader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=custom_collate_fn
)
```

---

### 4. 计算优化

#### 4.1 混合精度训练（AMP）

**原理**:
```
FP32: 精度高，计算慢，显存大
FP16: 精度较低，计算快，显存小
BF16: 平衡精度和速度（推荐）
```

**实现**:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    # 前向传播（自动混合精度）
    with autocast(dtype=torch.bfloat16):  # 或torch.float16
        output = model(batch)
        loss = criterion(output, target)

    # 反向传播（自动处理缩放）
    scaler.scale(loss).backward()

    # 梯度裁剪（重要！）
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # 更新参数
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

**性能提升**:
- 速度: 1.5-3x加速
- 显存: ~40%节省
- 精度: 几乎无损（使用BF16）

#### 4.2 梯度检查点（Gradient Checkpointing）

**问题**: 深层网络的中间激活占用大量显存

**解决方案**: 只保存部分激活，反传时重新计算

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedTransformer(nn.Module):
    def __init__(self, d_model, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model) for _ in range(n_layers)
        ])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            # 每2层checkpoint一次
            if i % 2 == 0 and i > 0:
                x = checkpoint(layer, x)  # 重新计算
            else:
                x = layer(x)
        return x
```

**效果**:
- 显存节省: 30-50%
- 速度代价: +20-30%计算时间

**适用场景**:
- ✅ 模型太大，显存不足
- ✅ 愿意用时间换空间

#### 4.3 算子融合（Operator Fusion）

**原理**: 合并多个小操作为一个kernel

```python
# 未优化（多次kernel launch）
def forward(x):
    x = layer_norm(x)
    x = activation(x)
    x = linear(x)
    return x

# 融合（一次kernel launch）
def forward_fused(x):
    return fused_layer_norm_activation_linear(x)
```

**PyTorch JIT**:
```python
@torch.jit.script
def fused_function(x, weight, bias):
    x = torch.layer_norm(x)
    x = torch.relu(x)
    x = torch.linear(x, weight, bias)
    return x
```

---

### 5. 分布式训练优化

#### 5.1 DDP通信优化

**通信开销分析**:
```python
# 在Profiler中查找
ncclAllReduce           # 梯度同步
ncclBroadcast           # 参数广播
```

**优化策略**:

1. **Gradient Bucketing调整**:
   ```python
   model = DDP(
       model,
       bucket_cap_mb=25,  # 增大bucket减少通信次数
   )
   ```

2. **跳过unused参数同步**:
   ```python
   model = DDP(
       model,
       find_unused_parameters=False,  # 确保所有参数都使用
   )
   ```

3. **通信与计算重叠**:
   ```python
   # DDP自动实现，但可以通过调整bucket大小优化
   ```

#### 5.2 多GPU性能扩展性

**扩展效率**:
```
加速比 = 单GPU时间 / N GPU时间
效率 = 加速比 / N
```

**理想vs实际**:
| GPU数量 | 理想加速比 | 实际加速比 | 效率 |
|---------|-----------|-----------|------|
| 1       | 1.0x      | 1.0x      | 100% |
| 2       | 2.0x      | 1.8x      | 90%  |
| 4       | 4.0x      | 3.4x      | 85%  |
| 8       | 8.0x      | 6.2x      | 78%  |

**效率下降原因**:
- 通信开销占比增加
- 负载不均衡
- 同步等待时间

#### 5.3 分布式训练调试

**常见问题**:

1. **训练卡住**:
   ```python
   # 添加barrier调试
   if dist.get_rank() == 0:
       print("Step 1")
   dist.barrier()  # 等待所有GPU

   if dist.get_rank() == 0:
       print("Step 2")
   ```

2. **性能不扩展**:
   ```python
   # 检查数据是否正确分片
   sampler = DistributedSampler(dataset)
   assert len(sampler) == len(dataset) // world_size
   ```

3. **梯度爆炸/消失**:
   ```python
   # 检查每个rank的梯度
   for name, param in model.named_parameters():
       if param.grad is not None:
           grad_norm = param.grad.norm()
           if dist.get_rank() == 0:
               print(f"{name}: grad_norm={grad_norm}")
   ```

---

### 6. 内存优化

#### 6.1 显存分析

**使用torch.cuda.memory**:
```python
def print_memory_usage():
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"Allocated: {allocated:.2f} GB")
    print(f"Reserved: {reserved:.2f} GB")

# 在训练循环中
for batch in dataloader:
    print("Before forward:")
    print_memory_usage()

    output = model(batch)

    print("After forward:")
    print_memory_usage()

    loss.backward()
    optimizer.step()

    print("After backward:")
    print_memory_usage()
```

#### 6.2 显存优化技巧

1. **及时清理中间变量**:
   ```python
   # 不好的做法
   for layer in layers:
       x1 = layer(x)
       x2 = process(x1)
       x3 = another_process(x2)

   # 好的做法
   for layer in layers:
       x = layer(x)
       x = process(x)
       del x  # 显式删除
       torch.cuda.empty_cache()  # 清空缓存
   ```

2. **使用inplace操作**:
   ```python
   # 不节省显存
   x = x + y

   # 节省显存
   x += y  # inplace

   # 或使用relu_
   x = torch.relu(x)    # 不节省
   torch.relu_(x)       # 节省
   ```

3. **减小batch size + 梯度累积**:
   ```python
   effective_batch = 512
   micro_batch = 32
   accumulation = effective_batch // micro_batch

   for i in range(0, len(dataloader), accumulation):
       for j in range(accumulation):
           loss = model(batch) / accumulation
           loss.backward()
       optimizer.step()
   ```

---

## 🔧 实战优化案例

### 案例1: 图像分类训练优化

**初始状态**:
```
Batch size: 32
单epoch时间: 120s
GPU利用率: 60%
```

**优化步骤**:

1. **DataLoader优化** (+30%速度):
   ```python
   num_workers: 0 → 8
   pin_memory: False → True
   prefetch_factor: 2 → 4
   ```

2. **混合精度** (+40%速度):
   ```python
   with autocast(dtype=torch.bfloat16):
   ```

3. **增大batch size** (+20%速度):
   ```python
   batch_size: 32 → 64
   ```

**最终结果**:
```
Batch size: 64
单epoch时间: 45s (2.7x加速)
GPU利用率: 92%
```

---

### 案例2: Transformer训练优化

**初始状态**:
```
模型: 1B参数
Batch size: 8 (单GPU)
单step时间: 2.5s
OOM问题: 经常
```

**优化步骤**:

1. **梯度检查点** (-40%显存):
   ```python
   model = checkpoint_sequential(model, segments=4)
   ```

2. **FSDP** (-70%显存):
   ```python
   model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD)
   ```

3. **DDP扩展** (4 GPU):
   ```python
   model = DDP(model)
   ```

**最终结果**:
```
有效batch size: 8 × 4 = 32
单step时间: 0.8s (3.1x加速)
OOM问题: 解决
```

---

## 💡 性能优化清单

### 训练前检查

- [ ] 使用PyTorch最新版本
- [ ] 确认CUDA/cuDNN版本匹配
- [ ] 启用cudnn.benchmark（固定输入尺寸）
  ```python
  torch.backends.cudnn.benchmark = True
  ```

### 数据加载

- [ ] num_workers > 0（推荐4-8）
- [ ] pin_memory=True
- [ ] prefetch_factor=2-4
- [ ] persistent_workers=True（大数据集）

### 计算优化

- [ ] 使用混合精度（BF16 > FP16）
- [ ] 启用cudnn.benchmark
- [ ] 梯度检查点（显存不足时）
- [ ] 梯度累积（模拟大batch）

### 分布式训练

- [ ] 使用DDP而非DP
- [ ] find_unused_parameters=False
- [ ] 调整bucket_cap_mb
- [ ] 使用DistributedSampler

### 内存管理

- [ ] 及时删除不需要的tensor
- [ ] 使用inplace操作
- [ ] 定期调用torch.cuda.empty_cache()
- [ ] 监控显存使用

---

## 🎯 学习检验

### 关键问题

1. **性能分析**:
   - 如何使用PyTorch Profiler？
   - 如何识别训练瓶颈？
   - cuda_time_total和self cuda time有什么区别？

2. **数据加载**:
   - num_workers如何选择？
   - pin_memory的作用是什么？
   - 如何优化自定义collate_fn？

3. **计算优化**:
   - 混合精度训练的原理和注意事项？
   - 梯度检查点何时使用？
   - 算子融合如何实现？

4. **分布式优化**:
   - 如何分析DDP性能瓶颈？
   - 如何提高多GPU扩展效率？
   - 如何调试分布式训练问题？

### 代码练习

完成 [examples.py](examples.py) 中的练习题。

---

## 📖 延伸阅读

**文档**:
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [PyTorch Profiler Tutorial](https://pytorch.org/tutorials/intermediate/profiler_tutorial.html)

**代码**:
- [PyTorch Benchmark](https://github.com/pytorch/pytorch/tree/master/benchmarks)
- [NVIDIA Deep Learning Examples](https://github.com/NVIDIA/DeepLearningExamples)

---

## ⚠️ 常见陷阱

1. **过度优化**:
   - ❌ 盲目追求高num_workers（可能导致CPU竞争）
   - ✅ 用Profiler验证优化效果

2. **过早优化**:
   - ❌ 在模型没调通前就优化性能
   - ✅ 先确保正确性，再优化性能

3. **忽略硬件差异**:
   - ❌ 不同GPU使用相同配置
   - ✅ 根据硬件特性调整参数

4. **只看速度不看精度**:
   - ❌ 混合精度导致精度下降
   - ✅ 监控训练指标，确保精度无损

---

**下一步**: [Day 22-23: 向量数据库与RAG基础](../Day22-23_Vector_DB_RAG/README.md)
