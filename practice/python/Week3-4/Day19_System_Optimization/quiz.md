# Day 19: 系统优化与性能调优 - 面试题

> **题目难度**: ⭐⭐ ~ ⭐⭐⭐⭐⭐
> **考察重点**: PyTorch Profiler、DataLoader优化、混合精度、内存管理
> **建议时间**: 45分钟

---

## Part 1: PyTorch Profiler

### Q1: PyTorch Profiler的主要功能是什么？⭐⭐⭐

**参考答案**:

PyTorch Profiler是性能分析工具，主要功能：

1. **记录操作时间**:
   - CPU时间
   - CUDA时间
   - 自身时间 vs 总时间

2. **分析内存使用**:
   - 显存分配
   - 峰值显存
   - 内存泄漏检测

3. **可视化性能**:
   - Chrome Trace格式
   - Timeline视图
   - 调用栈分析

**使用场景**:
- 识别训练瓶颈
- 优化数据加载
- 分析分布式性能

---

### Q2: self_cuda_time和cuda_time_total有什么区别？⭐⭐⭐⭐

**参考答案**:

**cuda_time_total**:
- 包含该操作及其所有子操作的总时间
- 用于识别哪个函数调用链最耗时

**self_cuda_time**:
- 该操作自身的时间（不包含子操作）
- 用于识别哪个具体操作最耗时

**示例**:
```
forward_pass (cuda_time_total: 100ms, self_cuda_time: 10ms)
  ├─ layer1 (cuda_time_total: 50ms, self_cuda_time: 45ms)
  ├─ layer2 (cuda_time_total: 40ms, self_cuda_time: 38ms)
  └─ loss (cuda_time_total: 10ms, self_cuda_time: 7ms)
```

- `forward_pass`总时间100ms，但自身只有10ms
- `layer1`最耗时（self_cuda_time: 45ms）

---

### Q3: 如何使用Profiler分析训练瓶颈？⭐⭐⭐⭐

**参考答案**:

**步骤1**: 选择合适的activities
```python
with profile(
    activities=[
        ProfilerActivity.CPU,    # CPU活动
        ProfilerActivity.CUDA,   # GPU活动
    ]
) as prof:
    # 训练代码
```

**步骤2**: 排序分析
```python
# 按CUDA时间排序
prof.key_averages().table(sort_by="cuda_time_total")

# 按CPU时间排序
prof.key_averages().table(sort_by="cpu_time_total")
```

**步骤3**: 识别瓶颈
- 查找self_cuda_time占比高的操作
- 查找频繁调用的操作（Number of calls）
- 分析DataLoader迭代时间

**步骤4**: 导出trace
```python
prof.export_chrome_trace("trace.json")
# 在chrome://tracing中查看
```

---

## Part 2: DataLoader优化

### Q4: num_workers应该如何选择？⭐⭐⭐

**参考答案**:

**原则**: num_workers = min(CPU核心数, 8)

**考虑因素**:

1. **CPU核心数**:
   ```python
   import os
   num_workers = os.cpu_count()  # 所有核心
   num_workers = os.cpu_count() // 2  # 一半核心（推荐）
   ```

2. **数据加载速度**:
   - 简单数据: 2-4 workers
   - 复杂预处理: 4-8 workers

3. **batch size**:
   - 小batch: 更多workers
   - 大batch: 可以减少workers

4. **内存**:
   - 每个worker占用内存
   - 过多workers可能导致内存不足

**测试方法**:
```python
for num_workers in [0, 1, 2, 4, 8]:
    dataloader = DataLoader(dataset, num_workers=num_workers)
    # 测量性能
```

---

### Q5: pin_memory的作用是什么？⭐⭐⭐⭐

**参考答案**:

**作用**: 将数据加载到锁页内存（page-locked memory）

**优势**:
1. **加速CPU→GPU传输**:
   - 普通内存: 可能被swap到磁盘
   - 锁页内存: 始终在RAM中

2. **异步传输**:
   ```python
   dataloader = DataLoader(dataset, pin_memory=True)

   # 异步传输
   for batch in dataloader:
       batch = batch.to(device, non_blocking=True)
   ```

**何时使用**:
- ✅ GPU训练（推荐）
- ❌ CPU训练（无效）

**性能提升**:
- 传输速度: 1.5-2x加速
- 总体训练: 5-10%加速

---

### Q6: persistent_workers参数的作用？⭐⭐⭐

**参考答案**:

**作用**: 保持worker进程在多个epoch之间存活

**默认行为** (persistent_workers=False):
- 每个epoch结束，worker进程终止
- 下个epoch重新创建worker
- 额外开销：进程创建+初始化

**启用后** (persistent_workers=True):
- Worker进程一直运行
- 多个epoch共享workers
- 避免重复初始化开销

**何时使用**:
- ✅ 大数据集，多epoch训练
- ✅ 数据初始化耗时
- ❌ 单epoch训练（无效果）

**示例**:
```python
dataloader = DataLoader(
    dataset,
    num_workers=4,
    persistent_workers=True  # 保持worker进程
)
```

---

## Part 3: 混合精度训练

### Q7: 混合精度训练的原理是什么？⭐⭐⭐⭐

**参考答案**:

**核心思想**: 使用FP16/BF16进行计算，FP32保存关键变量

**数据类型对比**:

| 类型 | 位数 | 范围 | 精度 | 速度 |
|------|------|------|------|------|
| FP32 | 32 | ±3.4×10³⁸ | 高 | 慢 |
| FP16 | 16 | ±65504 | 中 | 快 |
| BF16 | 16 | ±3.4×10³⁸ | 中 | 快 |

**训练流程**:
```
1. FP32权重 → FP16（前向传播）
2. FP16计算（快）
3. FP16梯度
4. 梯度缩放（防止下溢）
5. FP32更新权重（保持精度）
```

**优势**:
- 速度: 1.5-3x加速
- 显存: ~40%节省
- 精度: 几乎无损

---

### Q8: GradScaler的作用是什么？⭐⭐⭐⭐⭐

**参考答案**:

**问题**: FP16梯度容易下溢（值太小）

**解决方案**: 动态缩放梯度

**工作原理**:
```python
scaler = GradScaler()

# 1. 缩放loss
loss_scaled = loss * scale_factor

# 2. 反向传播（缩放的梯度）
loss_scaled.backward()

# 3. 取消缩放（用于梯度裁剪）
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 4. 更新参数（使用缩放的梯度）
scaler.step(optimizer)

# 5. 动态调整scale_factor
scaler.update()
```

**动态调整**:
- 如果梯度=inf/nan: 降低scale
- 如果梯度太小: 增大scale

**为什么需要**:
```python
# 不使用scaler（可能下溢）
loss.backward()  # 梯度可能为0

# 使用scaler（安全）
scaler.scale(loss).backward()  # 梯度被缩放，不会下溢
```

---

### Q9: 梯度裁剪在混合精度训练中如何正确使用？⭐⭐⭐⭐

**参考答案**:

**关键**: 必须在unscale_之后裁剪

**错误做法**:
```python
scaler.scale(loss).backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # ❌
scaler.step(optimizer)
```
问题：梯度还在缩放状态，裁剪阈值不对

**正确做法**:
```python
scaler.scale(loss).backward()

# 先取消缩放
scaler.unscale_(optimizer)

# 再裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # ✅

scaler.step(optimizer)
scaler.update()
```

**原因**:
- `unscale_()`将梯度恢复到原始尺度
- 这样裁剪阈值才有意义

---

## Part 4: 内存优化

### Q10: 梯度检查点的原理和代价？⭐⭐⭐⭐⭐

**参考答案**:

**原理**: 不保存中间激活，反传时重新计算

**标准反向传播**:
```
前向: 保存所有激活 → 显存O(N)
反向: 使用保存的激活 → 计算快
```

**梯度检查点**:
```
前向: 只保存部分激活 → 显存O(N/k)
反向: 重新计算激活 → 计算慢
```

**示例**:
```python
from torch.utils.checkpoint import checkpoint

class CheckpointedModel(nn.Module):
    def forward(self, x):
        # 每2层checkpoint一次
        x = self.layer1(x)
        x = checkpoint(self.layer2, x)  # 不保存激活
        x = self.layer3(x)
        x = checkpoint(self.layer4, x)
        return x
```

**权衡**:
- 显存节省: 30-50%
- 计算代价: +20-30%

**适用场景**:
- ✅ 显存不足
- ✅ 愿意用时间换空间
- ❌ 显存充足（浪费时间）

---

### Q11: 如何减少训练中的显存占用？⭐⭐⭐⭐

**参考答案**:

**方法1**: 及时删除中间变量
```python
# 不好
x1 = layer1(x)
x2 = layer2(x1)
x3 = layer3(x2)
return x3

# 好
x = layer1(x)
del x  # 如果不再需要
x = layer2(x)
del x
x = layer3(x)
```

**方法2**: 使用inplace操作
```python
# 不好
x = x + 1
x = torch.relu(x)

# 好
x += 1
torch.relu_(x)
```

**方法3**: 梯度检查点
```python
x = checkpoint(layer, x)
```

**方法4**: 减小batch size + 梯度累积
```python
accumulation_steps = 4
for i in range(0, len(dataloader), accumulation_steps):
    for j in range(accumulation_steps):
        loss = model(batch) / accumulation_steps
        loss.backward()
    optimizer.step()
```

**方法5**: 清空缓存
```python
del unnecessary_variables
torch.cuda.empty_cache()
```

---

### Q12: inplace操作有什么风险？⭐⭐⭐⭐

**参考答案**:

**优势**: 节省显存

**风险**:

1. **破坏计算图**:
   ```python
   x = torch.randn(10, requires_grad=True)
   y = x + 1  # 安全
   z = y.relu()

   vs

   x += 1  # inplace，可能破坏梯度
   x.relu_()  # inplace
   ```

2. **Autograd问题**:
   ```python
   # 错误
   x = torch.randn(10, requires_grad=True)
   x += 1  # RuntimeError: a leaf Variable that requires grad
   ```

3. **多线程问题**:
   - Inplace操作可能导致数据竞争

**何时安全**:
- ✅ 激活函数（relu_, sigmoid_等）
- ✅ 不需要梯度的中间变量
- ❌ 需要梯度的叶节点

**检查方法**:
```python
x = torch.randn(10, requires_grad=True)
x.relu_()  # 通常安全
# x += 1  # 通常不安全
```

---

## Part 5: 分布式训练优化

### Q13: 如何分析DDP训练的性能瓶颈？⭐⭐⭐⭐

**参考答案**:

**使用Profiler**:
```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
) as prof:
    # DDP训练循环

# 查找通信操作
prof.key_averages().table(sort_by="cuda_time_total")
```

**关键指标**:
1. **ncclAllReduce时间**: 梯度同步时间
   - 占比高: 通信瓶颈
   - 优化: 增大bucket_cap_mb

2. **DataLoader时间**: 数据加载时间
   - 占比高: 数据瓶颈
   - 优化: num_workers, pin_memory

3. **GPU利用率**:
   ```bash
   nvidia-smi dmon -s u
   ```
   - <80%: 数据瓶颈
   - >90%: 计算饱和

**扩展效率**:
```
加速比 = 单GPU时间 / N GPU时间
效率 = 加速比 / N

理想: 加速比=N, 效率=100%
实际: 4GPU通常效率85-90%
```

---

### Q14: bucket_cap_mb参数如何影响DDP性能？⭐⭐⭐⭐⭐

**参考答案**:

**作用**: 控制DDP梯度bucket大小

**默认值**: 25MB

**工作原理**:
```
小bucket (例如10MB):
- 更多通信次数
- 每次通信快
- 可能无法overlap

大bucket (例如50MB):
- 更少通信次数
- 每次通信时间长
- 更容易与计算overlap
```

**调优**:
```python
# 尝试不同大小
for bucket_cap_mb in [10, 25, 50, 100]:
    model = DDP(model, bucket_cap_mb=bucket_cap_mb)
    # 测试性能
```

**原则**:
- 模型大: 增大bucket（减少通信次数）
- 模型小: 减小bucket（更频繁同步）
- 通常25MB已经很好

---

### Q15: 如何调试分布式训练问题？⭐⭐⭐⭐

**参考答案**:

**问题1**: 训练卡住

**诊断**:
```python
# 在关键位置添加barrier和print
if dist.get_rank() == 0:
    print("Before forward")

dist.barrier()  # 等待所有GPU

if dist.get_rank() == 0:
    print("After forward")
```

**问题2**: 性能不扩展

**检查**:
```python
# 1. 确认数据正确分片
sampler = DistributedSampler(dataset)
print(f"Rank {dist.get_rank()}: {len(sampler)} samples")

# 2. 检查GPU利用率
# nvidia-smi dmon -s u

# 3. 分析通信开销
# 在Profiler中查找ncclAllReduce时间
```

**问题3**: 结果不一致

**检查**:
```python
# 确认随机种子
torch.manual_seed(42)
np.random.seed(42)

# 确认数据shuffle
for epoch in range(epochs):
    sampler.set_epoch(epoch)  # 重要！
```

---

## Part 6: 实战案例

### Q16: 一个训练慢的问题如何系统性优化？⭐⭐⭐⭐⭐

**参考答案**:

**步骤1**: 用Profiler分析
```python
with profile(...) as prof:
    # 运行几个epoch
prof.key_averages().table(sort_by="cuda_time_total")
```

**步骤2**: 识别瓶颈
- DataLoader时间高: 数据瓶颈
- 计算时间高: 计算瓶颈
- ncclAllReduce高: 通信瓶颈

**步骤3**: 针对性优化

**数据瓶颈**:
```python
dataloader = DataLoader(
    dataset,
    num_workers=8,        # 增加workers
    pin_memory=True,      # 启用pin_memory
    prefetch_factor=4,    # 预取
)
```

**计算瓶颈**:
```python
# 混合精度
with autocast(dtype=torch.bfloat16):
    output = model(x)

# 梯度检查点
x = checkpoint(layer, x)
```

**通信瓶颈**:
```python
# 增大bucket
model = DDP(model, bucket_cap_mb=50)

# 或使用FSDP
model = FSDP(model)
```

**步骤4**: 验证效果
```python
# 对比优化前后的时间和吞吐量
```

---

### Q17: 如何在保持精度的前提下最大化训练速度？⭐⭐⭐⭐⭐

**参考答案**:

**策略**:

1. **混合精度** (BF16 > FP16):
   - 速度: 2-3x
   - 精度: 几乎无损（BF16）

2. **增大batch size**:
   - 直到精度下降
   - 配合学习率调整

3. **优化数据加载**:
   - num_workers=4-8
   - pin_memory=True

4. **使用cudnn benchmark**:
   ```python
   torch.backends.cudnn.benchmark = True
   ```

5. **编译模型** (PyTorch 2.0+):
   ```python
   model = torch.compile(model)
   ```

6. **梯度累积**:
   - 小显存也能大batch

**验证精度**:
```python
# 对比优化前后
# - 训练曲线
# - 验证准确率
# - 最终指标
```

---

### Q18: OOM问题如何系统性解决？⭐⭐⭐⭐

**参考答案**:

**诊断**:
```python
# 监控显存
torch.cuda.memory_summary()

# 找出占用大户
for name, param in model.named_parameters():
    print(f"{name}: {param.numel() * 4 / 1e6} MB")
```

**解决方案**:

**方案1**: 减小batch size
```python
batch_size = 32  # 原来是64
```

**方案2**: 梯度检查点
```python
from torch.utils.checkpoint import checkpoint

x = checkpoint(layer, x)
```

**方案3**: 梯度累积
```python
effective_batch = 64
micro_batch = 16
accumulation = effective_batch // micro_batch
```

**方案4**: 混合精度
```python
with autocast(dtype=torch.bfloat16):
    output = model(x)
```

**方案5**: FSDP（大模型）
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD)
```

**方案6**: 清理显存
```python
del unnecessary_vars
torch.cuda.empty_cache()
```

**优先级**:
1. 减小batch size + 梯度累积
2. 混合精度
3. 梯度检查点
4. FSDP（超大模型）

---

## 总结

**必会题目** (面试高频):
- Q1: Profiler功能
- Q3: 如何分析瓶颈
- Q4: num_workers选择
- Q7: 混合精度原理
- Q10: 梯度检查点

**加分题目** (深入理解):
- Q8: GradScaler原理
- Q11: 减少显存方法
- Q14: bucket_cap_mb调优
- Q16: 系统性优化流程

**建议**:
1. 掌握Profiler使用
2. 理解各优化技术的trade-off
3. 能够系统性诊断和解决问题
4. 用实际数据验证优化效果

---

**下一步**: [Day 22-23: 向量数据库与RAG基础](../Day22-23_Vector_DB_RAG/quiz.md)
