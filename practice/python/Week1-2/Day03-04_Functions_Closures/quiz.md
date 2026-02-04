# Day03-04: 函数与闭包 - 面试题

## 一、基础概念题

### Q1: Python 的函数参数传递机制是什么？
**难度**: ⭐⭐

**答案**:
Python 使用**对象引用传递**（Call by Object Reference）：
- 不可变对象（int, str, tuple）：函数内修改不影响原对象
- 可变对象（list, dict, set）：函数内修改会影响原对象

**示例**:
```python
def modify_immutable(x):
    x = x + 1  # 创建新对象，不影响原值

def modify_mutable(lst):
    lst.append(1)  # 直接修改原对象

a = 10
modify_immutable(a)
print(a)  # 10

b = []
modify_mutable(b)
print(b)  # [1]
```

---

### Q2: 什么是闭包（Closure）？
**难度**: ⭐⭐⭐

**答案**:
闭包是一个函数对象，它记住了外部作用域中的变量，即使外部函数已经执行完毕。

**三个条件**：
1. 嵌套函数
2. 内部函数引用外部函数的变量
3. 外部函数返回内部函数

**示例**:
```python
def outer(x):
    def inner():
        print(x)  # 引用外部变量 x
    return inner

func = outer(10)
func()  # 输出 10，即使 outer 已经执行完毕
```

---

### Q3: LEGB 作用域规则是什么？
**难度**: ⭐⭐

**答案**:
Python 查找变量的顺序：
1. **L**ocal - 局部作用域
2. **E**nclosing - 闭包作用域（外部函数）
3. **G**lobal - 全局作用域
4. **B**uilt-in - 内置作用域

---

## 二、陷阱与坑点

### Q4: 可变默认参数的陷阱
**难度**: ⭐⭐⭐⭐

**问题**: 下面代码的输出是什么？
```python
def append(item, lst=[]):
    lst.append(item)
    return lst

print(append(1))
print(append(2))
print(append(3))
```

**答案**:
```
[1]
[1, 2]
[1, 2, 3]
```

**原因**:
默认参数在函数定义时只创建一次，后续调用会共享同一个列表。

**正确写法**:
```python
def append(item, lst=None):
    if lst is None:
        lst = []
    lst.append(item)
    return lst
```

---

### Q5: 闭包中的延迟绑定问题
**难度**: ⭐⭐⭐⭐⭐

**问题**: 下面代码的输出是什么？
```python
def create_multipliers():
    return [lambda x: x * i for i in range(3)]

multipliers = create_multipliers()
print([m(10) for m in multipliers])
```

**答案**:
```
[20, 20, 20]
```

**原因**:
lambda 函数中的 `i` 是延迟绑定的，所有函数都共享同一个 `i`，最终值为 2。

**解决方案**:
```python
# 方法1：使用默认参数
def create_multipliers():
    return [lambda x, i=i: x * i for i in range(3)]

# 方法2：使用闭包
def create_multipliers():
    multipliers = []
    for i in range(3):
        def multiplier(x, factor=i):
            return x * factor
        multipliers.append(multiplier)
    return multipliers
```

---

## 三、实际应用题

### Q6: 实现一个计时装饰器
**难度**: ⭐⭐⭐

**答案**:
```python
import time
from functools import wraps

def timer(func):
    @wraps(func)  # 保留原函数的元数据
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} 执行时间: {end-start:.4f}秒")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
    return "完成"
```

---

### Q7: 使用 lru_cache 优化递归
**难度**: ⭐⭐⭐

**问题**: 优化斐波那契数列计算

**答案**:
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# fibonacci(50) 瞬间返回
```

**性能对比**:
- 无缓存：O(2^n) 指数级
- 有缓存：O(n) 线性级

---

### Q8: 实现一个单例装饰器
**难度**: ⭐⭐⭐⭐

**答案**:
```python
def singleton(cls):
    instances = {}

    @wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance

@singleton
class Database:
    def __init__(self):
        print("创建数据库连接")

db1 = Database()  # 输出：创建数据库连接
db2 = Database()  # 无输出
print(db1 is db2)  # True
```

---

## 四、高级概念题

### Q9: global 和 nonlocal 的区别
**难度**: ⭐⭐⭐

**答案**:

| 关键字 | 作用域 | 用途 |
|--------|--------|------|
| `global` | 全局变量 | 在函数内修改全局变量 |
| `nonlocal` | 闭包作用域 | 在嵌套函数中修改外层变量 |

**示例**:
```python
# global
count = 0
def increment():
    global count  # 不加会报 UnboundLocalError
    count += 1

# nonlocal
def outer():
    total = 0
    def inner():
        nonlocal total  # 不加会报 UnboundLocalError
        total += 1
    return inner
```

---

### Q10: *args 和 **kwargs 的作用
**难度**: ⭐⭐

**答案**:

```python
# *args：接收任意数量的位置参数（元组）
def sum_all(*args):
    return sum(args)

print(sum_all(1, 2, 3, 4))  # 10

# **kwargs：接收任意数量的关键字参数（字典）
def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="张三", age=25)

# 组合使用
def func(*args, **kwargs):
    print(args)      # 位置参数
    print(kwargs)    # 关键字参数

func(1, 2, 3, a=4, b=5)
# (1, 2, 3)
# {'a': 4, 'b': 5}
```

---

### Q11: functools.wraps 的作用
**难度**: ⭐⭐⭐

**答案**:
`@wraps(func)` 用来保留被装饰函数的元数据（`__name__`, `__doc__`, `__annotations__` 等）。

**对比**:
```python
from functools import wraps

# 不使用 @wraps
def decorator_without_wraps(func):
    def wrapper():
        return func()
    return wrapper

@decorator_without_wraps
def my_function():
    """这是我的函数"""
    pass

print(my_function.__name__)  # 'wrapper'
print(my_function.__doc__)   # None

# 使用 @wraps
def decorator_with_wraps(func):
    @wraps(func)
    def wrapper():
        return func()
    return wrapper

@decorator_with_wraps
def my_function2():
    """这是我的函数2"""
    pass

print(my_function2.__name__)  # 'my_function2'
print(my_function2.__doc__)   # '这是我的函数2'
```

---

## 五、实战场景题

### Q12: 如何实现一个带重试机制的装饰器？
**难度**: ⭐⭐⭐⭐

**答案**:
```python
import time

def retry(max_attempts=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    print(f"重试 {attempt + 1}/{max_attempts}")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(max_attempts=3, delay=1)
def unstable_function():
    import random
    if random.random() < 0.7:
        raise Exception("随机失败")
    return "成功"
```

---

### Q13: 如何实现一个权限验证装饰器？
**难度**: ⭐⭐⭐⭐

**答案**:
```python
def require_permission(permission):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 假设从上下文获取当前用户权限
            user_permissions = get_user_permissions()
            if permission not in user_permissions:
                raise PermissionError(f"需要 {permission} 权限")
            return func(*args, **kwargs)
        return wrapper
    return decorator

@require_permission("admin")
def delete_user(user_id):
    # 只有 admin 权限才能调用
    pass
```

---

## 六、优化与性能题

### Q14: 如何使用缓存优化重复计算？
**难度**: ⭐⭐⭐

**答案**:
```python
from functools import lru_cache

# 方法1：使用 lru_cache（推荐）
@lru_cache(maxsize=128)
def expensive_function(n):
    # 昂贵的计算
    return n ** 2

# 方法2：手动实现缓存
def memoize(func):
    cache = {}

    @wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper

@memoize
def expensive_function2(n):
    return n ** 2
```

---

### Q15: Lambda 函数的适用场景
**难度**: ⭐⭐

**答案**:

**适合**:
```python
# 简单的单行函数
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x ** 2, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))
pairs.sort(key=lambda x: x[1])  # 排序
```

**不适合**:
```python
# 复杂逻辑应该用 def
# ❌ 不好
bad = lambda x: (
    x + 1 if x > 0
    else x - 1 if x < 0
    else 0
)

# ✅ 好
def good(x):
    if x > 0:
        return x + 1
    elif x < 0:
        return x - 1
    else:
        return 0
```

---

## 总结

### 核心知识点
1. ✅ 函数参数传递（对象引用）
2. ✅ 闭包原理与应用
3. ✅ 作用域规则（LEGB）
4. ✅ 装饰器模式
5. ✅ global/nonlocal 关键字
6. ✅ Lambda 与高阶函数

### 常见陷阱
1. ⚠️ 可变默认参数
2. ⚠️ 闭包延迟绑定
3. ⚠️ 忘记使用 global/nonlocal

### 实战应用
1. 🎯 计时器装饰器
2. 🎯 缓存优化（lru_cache）
3. 🎯 重试机制装饰器
4. 🎯 权限验证装饰器
