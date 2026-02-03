# Day 09-10: 生成器与迭代器

## 一、迭代器基础

### 1.1 什么是迭代器

迭代器是实现了迭代器协议的对象，即实现了`__iter__()`和`__next__()`方法的对象。

```python
class MyIterator:
    def __init__(self, data):
        self.data = data
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration
        value = self.data[self.index]
        self.index += 1
        return value
```

### 1.2 迭代器协议

Python的迭代器协议定义了两个方法：
- `__iter__()`: 返回迭代器对象本身
- `__next__()`: 返回下一个元素，没有元素时抛出`StopIteration`

### 1.3 可迭代对象 vs 迭代器

- **可迭代对象**：实现了`__iter__()`方法，可以返回迭代器
- **迭代器**：实现了`__next__()`方法，可以逐个返回元素

```python
# 可迭代对象
my_list = [1, 2, 3]
iter(my_list)  # 返回迭代器

# 迭代器
it = iter(my_list)
next(it)  # 1
next(it)  # 2
```

## 二、生成器

### 2.1 生成器基础

生成器是一种特殊的迭代器，使用`yield`关键字定义，自动实现迭代器协议。

```python
def simple_generator():
    yield 1
    yield 2
    yield 3

gen = simple_generator()
next(gen)  # 1
next(gen)  # 2
next(gen)  # 3
```

### 2.2 生成器表达式

类似于列表推导式，但返回生成器对象，惰性计算。

```python
# 列表推导式（立即计算）
list_comp = [x * x for x in range(1000000)]  # 占用大量内存

# 生成器表达式（惰性计算）
gen_expr = (x * x for x in range(1000000))  # 几乎不占用内存
```

### 2.3 yield关键字

`yield`暂停函数执行，保存状态，下次调用时继续执行。

```python
def countdown(n):
    while n > 0:
        yield n
        n -= 1

for i in countdown(5):
    print(i)  # 5, 4, 3, 2, 1
```

### 2.4 yield from

`yield from`委托生成器，将子生成器的值直接传递给调用者。

```python
def sub_generator():
    yield 1
    yield 2

def main_generator():
    yield from sub_generator()
    yield 3

# 等价于：
# def main_generator():
#     for value in sub_generator():
#         yield value
#     yield 3
```

## 三、生成器高级特性

### 3.1 生成器的方法

生成器对象支持以下方法：
- `send(value)`: 向生成器发送值
- `throw()`: 向生成器抛出异常
- `close()`: 关闭生成器

```python
def echo_generator():
    while True:
        value = yield
        print(f"Received: {value}")

gen = echo_generator()
next(gen)  # 启动生成器
gen.send("Hello")  # Received: Hello
gen.send("World")  # Received: World
gen.close()
```

### 3.2 协程基础

生成器可以用于实现协程，支持双向通信。

```python
def consumer():
    while True:
        item = yield
        print(f"Consuming: {item}")

def producer(consumer_gen):
    next(consumer_gen)  # 启动消费者
    for i in range(5):
        consumer_gen.send(i)

cons = consumer()
producer(cons)
```

### 3.3 无限序列

生成器可以表示无限序列，因为它们是惰性计算的。

```python
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# 取前10个斐波那契数
fib = fibonacci()
for _ in range(10):
    print(next(fib))
```

## 四、itertools模块

### 4.1 常用迭代器工具

Python的`itertools`模块提供了许多有用的迭代器工具。

```python
import itertools

# count: 无限计数
for i in itertools.count(10):
    if i > 15:
        break
    print(i)  # 10, 11, 12, 13, 14, 15

# cycle: 无限循环
for i, item in enumerate(itertools.cycle([1, 2, 3])):
    if i >= 6:
        break
    print(item)  # 1, 2, 3, 1, 2, 3

# chain: 连接多个迭代器
list(itertools.chain([1, 2], [3, 4]))  # [1, 2, 3, 4]

# islice: 切片迭代器
list(itertools.islice(count(), 5, 10))  # [5, 6, 7, 8, 9]
```

### 4.2 无限迭代器

- `count(start, step)`: 从start开始，以step递增
- `cycle(iterable)`: 无限循环iterable
- `repeat(elem, n)`: 重复elem n次

### 4.3 终止迭代器

- `accumulate(iterable)`: 累积结果
- `chain(*iterables)`: 连接多个迭代器
- `compress(data, selectors)`: 过滤数据
- `dropwhile(pred, seq)`: 丢弃满足条件的元素
- `takewhile(pred, seq)`: 保留满足条件的元素
- `filterfalse(pred, seq)`: 过滤不满足条件的元素
- `groupby(iterable, key)`: 分组
- `islice(iterable, start, stop)`: 切片
- `starmap(func, seq)`: 类似map但解包参数
- `tee(iterable, n)`: 复制迭代器
- `zip_longest(*iterables)`: 类似zip但填充缺失值

## 五、实际应用

### 5.1 处理大文件

生成器适合处理大文件，避免一次性加载到内存。

```python
def read_large_file(filename):
    with open(filename, 'r') as f:
        for line in f:
            yield line.strip()

for line in read_large_file('large_file.txt'):
    process(line)  # 逐行处理
```

### 5.2 数据管道

使用生成器构建数据处理管道。

```python
def read_data(source):
    for item in source:
        yield item

def filter_data(data, predicate):
    for item in data:
        if predicate(item):
            yield item

def transform_data(data, func):
    for item in data:
        yield func(item)

# 使用管道
data = read_data(source)
filtered = filter_data(data, lambda x: x > 0)
transformed = transform_data(filtered, lambda x: x * 2)

for result in transformed:
    print(result)
```

### 5.3 状态机

使用生成器实现状态机。

```python
def traffic_light():
    while True:
        yield "Red"
        yield "Green"
        yield "Yellow"

lights = traffic_light()
for _ in range(10):
    print(next(lights))
```

## 六、最佳实践

### 6.1 何时使用生成器

- 处理大量数据，节省内存
- 表示无限序列
- 构建数据管道
- 实现协程
- 流式处理

### 6.2 性能考虑

**生成器优势**：
- 惰性计算，节省内存
- 可以处理无限序列
- 适合流式处理

**列表优势**：
- 可以随机访问（索引）
- 可以多次迭代
- 支持len()函数

### 6.3 注意事项

1. 生成器只能迭代一次
2. 生成器不支持切片（使用islice）
3. 生成器没有长度（使用手动计数）
4. 注意生成器的状态管理

## 七、常见陷阱

### 7.1 生成器只能遍历一次

```python
gen = (x for x in range(3))

list(gen)  # [0, 1, 2]
list(gen)  # [] 已经耗尽
```

### 7.2 修改迭代中的对象

```python
# 危险：迭代时修改列表
my_list = [1, 2, 3, 4]
for item in my_list:
    if item == 2:
        my_list.remove(item)  # 可能导致跳过元素

# 安全：创建副本或使用列表推导式
my_list = [x for x in my_list if x != 2]
```

### 7.3 生成器的懒加载陷阱

```python
def get_data():
    print("Generating data...")
    for i in range(3):
        yield i

gen = get_data()  # 不会打印
list(gen)  # 此时才打印"Generating data..."
```

## 八、总结

生成器和迭代器是Python中强大的特性：
- **迭代器**：实现了迭代协议的对象
- **生成器**：使用yield定义的迭代器
- **优势**：节省内存、惰性计算、无限序列
- **应用**：大数据处理、流式处理、协程

掌握生成器和迭代器可以编写更加高效和优雅的Python代码。

## 九、学习建议

1. 理解迭代器协议
2. 掌握yield和yield from
3. 熟练使用itertools模块
4. 在大数据处理中应用生成器
5. 理解生成器的惰性计算特性
6. 学习协程和异步编程的基础
