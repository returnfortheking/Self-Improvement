# Day 3-4: 函数与闭包

> **学习目标**: 掌握函数定义、参数类型、闭包原理、装饰器基础
> **预估时间**: 6小时
> **难度**: ⭐⭐

---

## 一、函数定义与调用

### 1.1 基本函数定义

```python
def greet(name):
    """函数文档字符串"""
    return f"Hello, {name}!"

# 调用
message = greet("Alice")
print(message)  # Hello, Alice!
```

### 1.2 函数文档字符串

```python
def calculate_sum(a, b):
    """
    计算两个数的和

    参数:
        a (int): 第一个数
        b (int): 第二个数

    返回:
        int: 两个数的和
    """
    return a + b

# 访问文档字符串
print(calculate_sum.__doc__)
```

---

## 二、函数参数类型

### 2.1 位置参数

```python
def func(a, b, c):
    return a + b + c

func(1, 2, 3)  # 正确
func(1, 2)      # 错误 - 缺少参数
```

### 2.2 关键字参数

```python
def func(a, b, c):
    return a + b + c

func(a=1, b=2, c=3)  # 正确
func(1, c=3, b=2)     # 正确（位置 + 关键字）
```

### 2.3 默认参数

```python
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

print(greet("Alice"))        # Hello, Alice!
print(greet("Bob", "Hi"))    # Hi, Bob!
```

**⚠️ 注意：可变默认参数陷阱**

```python
# 错误示范
def bad_append(item, target=[]):
    target.append(item)
    return target

# 正确做法
def good_append(item, target=None):
    if target is None:
        target = []
    target.append(item)
    return target
```

### 2.4 可变位置参数（*args）

```python
def sum_all(*args):
    return sum(args)

print(sum_all(1, 2, 3))       # 6
print(sum_all(1, 2, 3, 4, 5))  # 15
```

### 2.5 可变关键字参数（**kwargs）

```python
def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Alice", age=30, city="Beijing")
# name: Alice
# age: 30
# city: Beijing
```

### 2.6 参数组合规则

```python
def func(a, b, c=10, *args, **kwargs):
    print(f"a={a}, b={b}, c={c}")
    print(f"args={args}")
    print(f"kwargs={kwargs}")

func(1, 2, 3, 4, 5, x=10, y=20)
# a=1, b=2, c=3
# args=(4, 5)
# kwargs={'x': 10, 'y': 20}
```

**参数顺序**：
1. 位置参数
2. 默认参数
3. `*args`
4. `**kwargs`

### 2.7 仅位置参数（Python 3.8+）

```python
def func(a, b, /, c, d, *, e, f):
    """
    / 之前的参数：仅位置参数
    * 之后的参数：仅关键字参数
    """
    return a + b + c + d + e + f

func(1, 2, 3, 4, e=5, f=6)  # 正确
func(1, 2, c=3, d=4, e=5, f=6)  # 正确
func(1, 2, 3, 4, 5, 6)  # 错误！e 和 f 必须用关键字
func(a=1, b=2, c=3, d=4, e=5, f=6)  # 错误！a 和 b 必须用位置
```

---

## 三、作用域与命名空间

### 3.1 LEGB 规则

Python 使用 **LEGB** 规则查找变量：

```
L → Local（局部）
  ↓
E → Enclosing（闭包）
  ↓
G → Global（全局）
  ↓
B → Built-in（内置）
```

```python
# Built-in
x = "Global"  # Global

def outer():
    x = "Enclosing"  # Enclosing

    def inner():
        x = "Local"  # Local
        print(x)  # 输出: Local

    inner()

outer()
```

### 3.2 global 关键字

```python
x = 10  # 全局变量

def modify_global():
    global x
    x = 20  # 修改全局变量

modify_global()
print(x)  # 20
```

### 3.3 nonlocal 关键字

```python
def outer():
    x = 10  # 闭包变量

    def inner():
        nonlocal x
        x = 20  # 修改闭包变量

    inner()
    print(x)  # 20

outer()
```

---

## 四、闭包（Closure）

### 4.1 什么是闭包

**闭包**：函数内部定义的函数，引用了外部函数的变量。

```python
def outer(x):
    def inner(y):
        return x + y  # 引用外部变量 x
    return inner  # 返回内部函数

add_5 = outer(5)
print(add_5(3))  # 8

add_10 = outer(10)
print(add_10(3))  # 13
```

### 4.2 闭包的应用场景

#### 1. 数据隐藏

```python
def make_counter():
    count = 0  # 私有变量

    def increment():
        nonlocal count
        count += 1
        return count

    return increment

counter = make_counter()
print(counter())  # 1
print(counter())  # 2
print(counter())  # 3
```

#### 2. 函数工厂

```python
def power(exponent):
    def raise_to(base):
        return base ** exponent
    return raise_to

square = power(2)
cube = power(3)

print(square(5))  # 25
print(cube(5))    # 125
```

#### 3. 延迟计算

```python
def delay_calculation(a, b):
    def calculate():
        return a + b
    return calculate

result = delay_calculation(10, 20)
# ... 中间可以执行其他操作
print(result())  # 30
```

### 4.3 闭包的陷阱

```python
# 陷阱：循环中的闭包
functions = []
for i in range(3):
    def func():
        return i  # 引用循环变量
    functions.append(func)

print(functions[0]())  # 2（不是 0！）
print(functions[1]())  # 2
print(functions[2]())  # 2

# 解决方案 1：使用默认参数
functions = []
for i in range(3):
    def func(x=i):  # 绑定当前值
        return x
    functions.append(func)

print(functions[0]())  # 0
print(functions[1]())  # 1
print(functions[2]())  # 2

# 解决方案 2：使用闭包
functions = []
for i in range(3):
    def make_func(x):
        def func():
            return x
        return func
    functions.append(make_func(i))

print(functions[0]())  # 0
```

---

## 五、装饰器基础

### 5.1 什么是装饰器

**装饰器**：在不修改原函数代码的情况下，扩展其功能。

```python
def my_decorator(func):
    def wrapper():
        print("Before function call")
        func()
        print("After function call")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()
# Before function call
# Hello!
# After function call
```

### 5.2 装饰器的等价形式

```python
# 使用 @ 语法
@my_decorator
def func():
    pass

# 等价于
func = my_decorator(func)
```

### 5.3 保留原函数信息

```python
from functools import wraps

def my_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def greet(name):
    """Greets the person"""
    return f"Hello, {name}!"

print(greet.__name__)  # greet（而不是 wrapper）
print(greet.__doc__)   # Greets the person
```

### 5.4 带参数的装饰器

```python
def repeat(times):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    print(f"Hello, {name}!")

greet("Alice")
# Hello, Alice!
# Hello, Alice!
# Hello, Alice!
```

---

## 六、lambda 函数

### 6.1 lambda 基础

```python
# 基本语法
add = lambda x, y: x + y
print(add(3, 5))  # 8

# 多个参数
calc = lambda x, y, z: (x + y) * z
print(calc(1, 2, 3))  # 9
```

### 6.2 lambda 与高阶函数

```python
# map
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
print(squared)  # [1, 4, 9, 16, 25]

# filter
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)  # [2, 4]

# reduce
from functools import reduce
sum_result = reduce(lambda x, y: x + y, numbers)
print(sum_result)  # 15
```

### 6.3 lambda 的限制

- 只能包含单个表达式
- 不能包含语句（如 return, pass, assert）
- 不适合复杂逻辑
- 可读性较差

**何时使用 lambda**：
- 简单的、一次性的函数
- 作为高阶函数的参数
- 不需要复用的函数

---

## 七、学习检查清单

完成本主题学习后，您应该能够：

- [ ] 掌握函数的各种参数类型
- [ ] 理解 LEGB 作用域规则
- [ ] 正确使用 global 和 nonlocal
- [ ] 理解闭包的原理和应用场景
- [ ] 掌握装饰器的基本使用
- [ ] 知道 lambda 的使用场景和限制
- [ ] 能手写函数工厂和闭包
- [ ] 理解装饰器的执行顺序

---

## 八、参考资源

- **来源仓库**：
  - interview_python（闭包专题，12题）
  - Python-Interview-Customs-Collection（函数相关，15题）
  - cracking-the-python-interview（进阶章节）

- **练习文件**：
  - examples.py - 25个代码示例
  - exercises.py - 20道练习题
  - quiz.md - 8道大厂面试真题
