# Day 09-10: 生成器与迭代器面试真题

## 阿里巴巴

### 1. 解释Python中的迭代器协议，并实现一个自定义迭代器

**参考答案**：
迭代器协议要求对象实现两个方法：
- `__iter__()`: 返回迭代器对象本身
- `__next__()`: 返回下一个元素，没有元素时抛出`StopIteration`

```python
class ReverseIterator:
    def __init__(self, data):
        self.data = data
        self.index = len(data) - 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < 0:
            raise StopIteration
        value = self.data[self.index]
        self.index -= 1
        return value

for item in ReverseIterator([1, 2, 3]):
    print(item)  # 3, 2, 1
```

**考察点**：迭代器协议、自定义迭代器

---

### 2. 生成器和列表推导式有什么区别？何时使用生成器？

**参考答案**：

**列表推导式**：
- 立即计算，生成完整列表
- 占用内存存储所有元素
- 可以多次迭代
- 支持索引访问和len()

**生成器表达式**：
- 惰性计算，按需生成
- 几乎不占用内存
- 只能迭代一次
- 不支持索引和len()

```python
# 列表推导式
list_comp = [x ** 2 for x in range(1000000)]  # 占用大量内存

# 生成器表达式
gen_expr = (x ** 2 for x in range(1000000))  # 几乎不占内存
```

**使用生成器的场景**：
1. 处理大量数据，内存受限
2. 表示无限序列
3. 流式处理
4. 构建数据管道

**考察点**：惰性计算、内存优化

---

## 腾讯

### 3. yield和yield from有什么区别？

**参考答案**：

**yield**：
- 生成单个值
- 控制权暂时返回给调用者
- 保存状态，下次从yield后继续

**yield from**：
- 委托给子生成器
- 自动迭代子生成器的所有值
- 更简洁的嵌套生成器写法

```python
def sub_generator():
    yield 1
    yield 2
    yield 3

def without_yield_from():
    for value in sub_generator():
        yield value

def with_yield_from():
    yield from sub_generator()

# 两者等价，但yield from更简洁
```

**yield from额外功能**：
- 传递send()和throw()调用
- 更好的性能
- 支持子生成器的返回值

**考察点**：生成器进阶、yield from机制

---

### 4. 如何实现一个可迭代的类？

**参考答案**：
可迭代类需要实现`__iter__()`方法，返回一个迭代器。

```python
class CountDown:
    def __init__(self, start):
        self.start = start
        self.current = start

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < 0:
            raise StopIteration
        value = self.current
        self.current -= 1
        return value

# 使用
for i in CountDown(5):
    print(i)  # 5, 4, 3, 2, 1, 0
```

或者分离迭代器：

```python
class Range:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __iter__(self):
        return RangeIterator(self.start, self.end)

class RangeIterator:
    def __init__(self, start, end):
        self.current = start
        self.end = end

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.end:
            raise StopIteration
        value = self.current
        self.current += 1
        return value
```

**考察点**：可迭代协议、类设计

---

## 字节跳动

### 5. 生成器的send()、throw()和close()方法有什么作用？

**参考答案**：

**send(value)**：
- 向生成器发送数据
- 首次调用需要用next()或send(None)启动
- yield表达式返回发送的值

```python
def echo_generator():
    while True:
        received = yield
        print(f"Received: {received}")

gen = echo_generator()
next(gen)  # 启动生成器
gen.send("Hello")  # Received: Hello
gen.send("World")  # Received: World
```

**throw(type, value=None)**：
- 向生成器抛出异常
- 在yield处处理异常
- 可以捕获并恢复执行

```python
def exception_handler():
    try:
        while True:
            value = yield
    except ValueError as e:
        yield f"Handled: {e}"

gen = exception_handler()
next(gen)
result = gen.throw(ValueError, "Error")
print(result)  # Handled: Error
```

**close()**：
- 关闭生成器
- 在yield处抛出GeneratorExit
- 资源清理

```python
def resource_handler():
    try:
        yield
    finally:
        print("Cleaning up...")

gen = resource_handler()
next(gen)
gen.close()  # 打印: Cleaning up...
```

**考察点**：生成器高级特性、协程基础

---

### 6. itertools模块提供了哪些有用的工具？请列举并说明用途

**参考答案**：

**无限迭代器**：
- `count(start, step)`: 无限计数
- `cycle(iterable)`: 无限循环
- `repeat(elem, n)`: 重复元素

**终止迭代器**：
- `accumulate(iterable)`: 累积
- `chain(*iterables)`: 连接迭代器
- `compress(data, selectors)`: 过滤
- `dropwhile/takewhile(pred, seq)`: 条件过滤
- `filterfalse(pred, seq)`: 反向过滤
- `groupby(iterable, key)`: 分组
- `islice(iterable, start, stop)`: 切片
- `starmap(func, seq)`: 类似map但解包
- `tee(iterable, n)`: 复制迭代器
- `zip_longest(*iterables)`: 带填充的zip

**组合迭代器**：
- `product(*iterables)`: 笛卡尔积
- `permutations(iterable, r)`: 排列
- `combinations(iterable, r)`: 组合
- `combinations_with_replacement(iterable, r)`: 可重复组合

```python
import itertools

# 笛卡尔积
list(itertools.product([1, 2], ['a', 'b']))
# [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')]

# 组合
list(itertools.combinations([1, 2, 3], 2))
# [(1, 2), (1, 3), (2, 3)]

# 分组
data = [('a', 1), ('b', 2), ('a', 3)]
for key, group in itertools.groupby(data, lambda x: x[0]):
    print(key, list(group))
```

**考察点**：itertools模块、工具函数

---

## 美团

### 7. 如何使用生成器处理大文件？

**参考答案**：
生成器适合逐行处理大文件，避免一次性加载到内存。

```python
def read_large_file(filename):
    """逐行读取大文件"""
    with open(filename, 'r') as f:
        for line in f:
            yield line.strip()

def process_line(line):
    """处理单行"""
    # 处理逻辑
    return line.upper()

def process_large_file(filename):
    """处理大文件"""
    for line in read_large_file(filename):
        processed = process_line(line)
        # 处理后的数据
        yield processed

# 使用
for processed_line in process_large_file('huge_file.txt'):
    # 逐行处理，内存占用小
    print(processed_line)
```

或者使用生成器管道：

```python
def read_lines(filename):
    with open(filename) as f:
        for line in f:
            yield line

def filter_empty(lines):
    for line in lines:
        if line.strip():
            yield line

def transform(lines):
    for line in lines:
        yield line.upper()

# 构建管道
lines = read_lines('large_file.txt')
filtered = filter_empty(lines)
transformed = transform(filtered)

for line in transformed:
    print(line)
```

**优势**：
1. 内存占用小
2. 可以处理无限大小的文件
3. 流式处理，即时响应

**考察点**：文件处理、内存优化、生成器应用

---

## 百度

### 8. 什么是惰性求值？Python中如何实现？

**参考答案**：
惰性求值是延迟计算直到真正需要结果时才执行的技术。

**Python中的惰性求值**：

1. **生成器**：
```python
def infinite_sequence():
    num = 0
    while True:
        yield num
        num += 1

gen = infinite_sequence()
print(next(gen))  # 0
print(next(gen))  # 1
# 只在需要时计算
```

2. **生成器表达式**：
```python
# 不立即计算
gen_expr = (x ** 2 for x in range(1000000))
# 只在迭代时计算
for val in gen_expr:
    print(val)
```

3. **惰性map/filter**：
```python
def lazy_map(func, iterable):
    for item in iterable:
        yield func(item)

def lazy_filter(pred, iterable):
    for item in iterable:
        if pred(item):
            yield item

# 使用
data = range(1000000)
mapped = lazy_map(lambda x: x ** 2, data)
filtered = lazy_filter(lambda x: x > 100, mapped)
```

**优势**：
1. 节省内存
2. 可以处理无限序列
3. 提高性能（避免不必要的计算）
4. 支持管道操作

**考察点**：惰性求值、性能优化

---

## 网易

### 9. 生成器只能迭代一次吗？为什么？

**参考答案**：
是的，生成器只能迭代一次。这是因为：

1. **状态消耗**：生成器维护内部状态（如指令指针、局部变量）
2. **单向流动**：数据只能从生成器流出，不能回流
3. **无回溯机制**：没有保存历史数据

```python
gen = (x for x in range(3))

# 第一次迭代
print(list(gen))  # [0, 1, 2]

# 第二次迭代
print(list(gen))  # [] 已经耗尽
```

**解决方法**：
1. 重新创建生成器
2. 转换为列表（如果数据不大）
3. 使用`itertools.tee()`复制迭代器

```python
# 方法1: 函数封装
def get_generator():
    return (x for x in range(3))

gen1 = get_generator()
print(list(gen1))  # [0, 1, 2]
gen2 = get_generator()
print(list(gen2))  # [0, 1, 2]

# 方法2: itertools.tee
import itertools
gen = (x for x in range(3))
gen1, gen2 = itertools.tee(gen, 2)
print(list(gen1))  # [0, 1, 2]
print(list(gen2))  # [0, 1, 2]
```

**注意**：`tee()`会缓存数据，如果数据量大会消耗内存。

**考察点**：生成器特性、状态管理

---

## 京东

### 10. 如何使用生成器实现协程？

**参考答案**：
Python的生成器可以用于实现简单的协程，支持双向通信。

```python
def consumer():
    """消费者协程"""
    print("Waiting for data...")
    try:
        while True:
            item = yield
            print(f"Processing: {item}")
    except GeneratorExit:
        print("Consumer closing...")

def producer(consumer_gen, items):
    """生产者"""
    next(consumer_gen)  # 启动消费者（预激）
    for item in items:
        consumer_gen.send(item)
    consumer_gen.close()  # 关闭消费者

# 使用
cons = consumer()
producer(cons, [1, 2, 3, 4, 5])
```

**更复杂的协程管道**：

```python
def coroutine(func):
    """协程装饰器，自动预激"""
    def start(*args, **kwargs):
        cr = func(*args, **kwargs)
        next(cr)
        return cr
    return start

@coroutine
def printer():
    try:
        while True:
            item = yield
            print(item)
    except GeneratorExit:
        print("Printer done")

@coroutine
def filter(target, predicate):
    try:
        while True:
            item = yield
            if predicate(item):
                target.send(item)
    except GeneratorExit:
        target.close()

# 使用
print_target = printer()
filtered = filter(print_target, lambda x: x > 5)

for item in range(10):
    filtered.send(item)
```

**注意**：Python 3.5+推荐使用`async/await`语法实现协程。

**考察点**：协程概念、生成器应用

---

## 快手

### 11. 如何实现一个无限序列的生成器？

**参考答案**：
生成器可以表示无限序列，因为它是惰性计算的。

```python
# 无限计数器
def infinite_counter(start=0, step=1):
    """无限计数"""
    num = start
    while True:
        yield num
        num += step

# 无限斐波那契数列
def fibonacci():
    """无限斐波那契"""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# 无限素数
def primes():
    """无限素数"""
    yield 2
    primes_so_far = [2]
    candidate = 3
    while True:
        is_prime = all(candidate % p != 0 for p in primes_so_far)
        if is_prime:
            primes_so_far.append(candidate)
            yield candidate
        candidate += 2

# 使用（注意：必须手动限制）
count = infinite_counter()
print([next(count) for _ in range(10)])  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

fib = fibonacci()
print([next(fib) for _ in range(10)])  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

**应用场景**：
1. 数学序列
2. 数据流处理
3. 游戏循环
4. 事件监听

**注意**：使用时必须有限制条件，否则会无限循环。

**考察点**：无限序列、惰性计算

---

## 拼多多

### 12. 生成器和列表在内存使用上的差异？请举例说明

**参考答案**：

**内存对比**：

```python
import sys

# 列表：占用大量内存
list_data = [x for x in range(1000000)]
print(sys.getsizeof(list_data))  # 约8MB（实际更多）

# 生成器：几乎不占内存
gen_data = (x for x in range(1000000))
print(sys.getsizeof(gen_data))  # 约200字节
```

**详细对比**：

| 特性 | 列表 | 生成器 |
|------|------|--------|
| 内存占用 | 存储所有元素 | 只存储当前状态 |
| 访问方式 | 随机访问（索引） | 顺序访问 |
| 迭代次数 | 多次 | 一次 |
| 时间复杂度 | O(n)创建 | O(1)创建 |
| 适用场景 | 小数据、多次访问 | 大数据、流式处理 |

**实际例子**：

```python
# 处理大文件
def read_file_list(filename):
    """列表方式：内存爆炸"""
    with open(filename) as f:
        return [line for line in f]

def read_file_gen(filename):
    """生成器方式：内存友好"""
    with open(filename) as f:
        for line in f:
            yield line

# 列表方式：10GB文件可能需要10GB+内存
# lines = read_file_list('huge.txt')

# 生成器方式：只需要几KB内存
# lines = read_file_gen('huge.txt')
# for line in lines:
#     process(line)
```

**性能测试**：

```python
import time
import tracemalloc

# 列表方式
tracemalloc.start()
start = time.time()
list_sum = sum([x for x in range(10000000)])
end = time.time()
current, peak = tracemalloc.get_traced_memory()
print(f"列表: 时间={end-start:.2f}s, 内存={peak/1024/1024:.2f}MB")

# 生成器方式
tracemalloc.clear_traces()
tracemalloc.start()
start = time.time()
gen_sum = sum((x for x in range(10000000)))
end = time.time()
current, peak = tracemalloc.get_traced_memory()
print(f"生成器: 时间={end-start:.2f}s, 内存={peak/1024/1024:.2f}MB")
```

**结论**：对于大数据量，生成器内存占用显著更小。

**考察点**：内存管理、性能优化

---

## 总结

### 高频考点：
1. **迭代器协议**：`__iter__`和`__next__`
2. **生成器基础**：`yield`关键字
3. **yield from**：委托生成器
4. **惰性计算**：节省内存
5. **生成器方法**：`send()`、`throw()`、`close()`
6. **itertools模块**：各种迭代器工具
7. **应用场景**：大文件处理、无限序列、流式处理

### 实战建议：
1. 处理大数据优先使用生成器
2. 理解生成器只能迭代一次
3. 熟练使用itertools模块
4. 掌握生成器管道模式
5. 了解协程的基础概念
6. 注意内存和性能的权衡
