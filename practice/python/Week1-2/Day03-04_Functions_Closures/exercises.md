# Day 3-4: 函数与闭包 练习题

## 练习1：实现计数器闭包

```python
def make_counter():
    # TODO: 实现计数器闭包
    pass

counter = make_counter()
assert counter() == 1
assert counter() == 2
assert counter() == 3
```

## 练习2：实现幂函数工厂

```python
def power_factory(exponent):
    # TODO: 返回一个 raise_to 函数
    pass

square = power_factory(2)
cube = power_factory(3)
assert square(5) == 25
assert cube(5) == 125
```

## 练习3：修复循环闭包陷阱

```python
# 修复这段代码
functions = []
for i in range(3):
    def func():
        return i
    functions.append(func)

assert functions[0]() == 0  # 当前会返回 2
```

## 练习4：实现计时装饰器

```python
import time

def timer(func):
    # TODO: 实现计时装饰器
    pass

@timer
def slow_function():
    time.sleep(0.1)
    return "Done"

# slow_function 应该打印执行时间
```

## 练习5：实现retry装饰器

```python
def retry(times=3):
    # TODO: 实现 retry 装饰器
    pass

@retry(times=3)
def unreliable_function():
    import random
    if random.random() < 0.7:
        raise ValueError("Failed!")
    return "Success"
```

## 答案

```python
# 练习1
def make_counter():
    count = 0
    def increment():
        nonlocal count
        count += 1
        return count
    return increment

# 练习2
def power_factory(exponent):
    def raise_to(base):
        return base ** exponent
    return raise_to

# 练习3
functions = []
for i in range(3):
    def func(x=i):
        return x
    functions.append(func)

# 练习4
def timer(func):
    import time
    import functools
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} 耗时: {end - start:.4f}秒")
        return result
    return wrapper

# 练习5
def retry(times=3):
    def decorator(func):
        import functools
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(times):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == times - 1:
                        raise
                    print(f"重试 {attempt + 1}/{times}")
        return wrapper
    return decorator
```

**学习建议**：
1. 先自己尝试实现
2. 运行测试验证
3. 对比答案找出差距
4. 理解闭包和装饰器的原理
