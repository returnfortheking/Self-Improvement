# Day 05-06: 装饰器与元类面试真题

## 阿里巴巴

### 1. 解释Python装饰器的工作原理，并实现一个带参数的装饰器

**参考答案**：
装饰器本质上是一个高阶函数，它接受一个函数作为参数，返回一个新的函数。当使用`@decorator`语法时，Python会将被装饰的函数传递给装饰器，并将返回的函数重新绑定到原函数名。

```python
def repeat(times):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator
```

**考察点**：装饰器的本质、闭包、`*args`和`**kwargs`

---

### 2. 什么是元类？请举例说明元类的应用场景

**参考答案**：
元类是创建类的"类"，Python中默认使用`type`作为元类。通过自定义元类可以在类创建时进行干预。

应用场景：
- 单例模式
- ORM框架（如Django Model）
- 自动注册
- 接口检查

```python
class SingletonMeta(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
```

**考察点**：元类的理解、Python对象模型、框架设计

---

## 腾讯

### 3. 为什么要使用@functools.wraps？如果不使用会怎样？

**参考答案**：
`@functools.wraps`用于保留被装饰函数的元信息（如`__name__`、`__doc__`、`__module__`等）。

如果不使用：
- 函数的`__name__`会变成wrapper
- 函数的`__doc__`会丢失
- 调试和日志记录会变得困难
- 可能影响其他依赖函数签名的代码

```python
# 不使用wraps
def decorator(func):
    def wrapper():
        return func()
    return wrapper

@decorator
def my_func():
    """My docstring"""
    pass

print(my_func.__name__)  # 输出: wrapper
print(my_func.__doc__)   # 输出: None
```

**考察点**：装饰器的副作用、函数元信息、调试技巧

---

### 4. 如何实现一个缓存装饰器？请考虑线程安全问题

**参考答案**：
```python
import functools
from threading import Lock

def thread_safe_cache(func):
    cache = {}
    lock = Lock()

    @functools.wraps(func)
    def wrapper(*args):
        if args not in cache:
            with lock:
                # 双重检查锁定
                if args not in cache:
                    cache[args] = func(*args)
        return cache[args]
    return wrapper
```

或者使用Python标准库的`@functools.lru_cache`。

**考察点**：缓存实现、线程安全、双重检查锁定

---

## 字节跳动

### 5. 类装饰器和函数装饰器有什么区别？分别适用于什么场景？

**参考答案**：
**类装饰器**：
- 使用类实现，通过`__call__`方法
- 可以维护状态（通过实例属性）
- 适合需要状态管理的场景

```python
class CountCalls:
    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        return self.func(*args, **kwargs)
```

**函数装饰器**：
- 使用函数实现，通过闭包
- 状态通过外部变量或nonlocal维护
- 适合简单的装饰逻辑

**选择原则**：
- 简单逻辑用函数装饰器
- 需要维护状态用类装饰器
- 装饰器本身需要配置参数用类装饰器

**考察点**：设计模式选择、闭包vs类、状态管理

---

### 6. 多个装饰器的执行顺序是怎样的？请举例说明

**参考答案**：
装饰器的执行顺序是从下到上（洋葱模型）。

```python
@decorator_a
@decorator_b
@decorator_c
def func():
    pass

# 等价于
# func = decorator_a(decorator_b(decorator_c(func)))
```

执行时：
1. `decorator_c`先包装原函数
2. `decorator_b`包装`decorator_c`的结果
3. `decorator_a`包装`decorator_b`的结果
4. 调用时从`decorator_a`开始，最后执行原函数

**考察点**：装饰器嵌套、执行顺序理解

---

## 美团

### 7. 如何实现一个单例模式？比较装饰器、元类、模块三种实现方式的优缺点

**参考答案**：

**1. 装饰器实现**：
```python
def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance
```
优点：简单易懂，不影响类定义
缺点：子类会创建新实例

**2. 元类实现**：
```python
class SingletonMeta(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
```
优点：对所有子类都有效
缺点：复杂度高，不易理解

**3. 模块实现**：
```python
# singleton.py
class Singleton:
    pass
instance = Singleton()
```
优点：Pythonic，简单可靠
缺点：不够灵活，导入即实例化

**考察点**：设计模式、多种实现方案对比、架构选择

---

## 百度

### 8. 装饰器如何处理函数的返回值和异常？

**参考答案**：
```python
def handle_exception(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result  # 必须返回原函数的返回值
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")
            raise  # 可以选择重新抛出或返回默认值
    return wrapper
```

关键点：
- 必须返回原函数的返回值，否则调用者无法获得结果
- 异常可以选择捕获处理或重新抛出
- 可以在异常处理后返回默认值

**考察点**：装饰器的完整性、异常处理

---

## 拼多多

### 9. 在Web框架中，路由装饰器（如Flask的@app.route）是如何实现的？

**参考答案**：
```python
class Flask:
    def __init__(self):
        self.routes = {}

    def route(self, path):
        def decorator(func):
            self.routes[path] = func
            return func
        return decorator

app = Flask()

@app.route('/')
def index():
    return "Hello"
```

核心思想：
1. 装饰器将路径和函数的映射关系注册到字典
2. 不修改原函数，直接返回原函数
3. 请求时查找字典找到对应的处理函数

**考察点**：装饰器在实际框架中的应用、路由原理

---

## 网易

### 10. 什么是猴子补丁（Monkey Patching）？如何用装饰器实现？

**参考答案**：
猴子补丁是在运行时动态修改类或模块的行为。

```python
# 原始类
class Dog:
    def bark(self):
        return "Woof!"

# 使用装饰器打补丁
def add_trick(cls):
    original_bark = cls.bark

    def enhanced_bark(self):
        return original_bark(self) + " (enhanced!)"

    cls.bark = enhanced_bark
    return cls

@add_trick
class Dog:
    def bark(self):
        return "Woof!"

dog = Dog()
print(dog.bark())  # "Woof! (enhanced!)"
```

应用场景：
- 测试时mock对象
- 修复第三方库的bug
- 动态添加功能

**考察点**：动态语言特性、运行时修改、应用场景

---

## 京东

### 11. 如何实现一个异步函数的装饰器？

**参考答案**：
```python
import functools
import asyncio
import time

def async_timer(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f}s")
        return result
    return wrapper

@async_timer
async def async_operation():
    await asyncio.sleep(1)
    return "Done"

asyncio.run(async_operation())
```

关键点：
- 装饰器本身必须是async函数
- 使用`await`调用被装饰的异步函数
- 返回值仍然是awaitable对象

**考察点**：异步编程、装饰器与async的结合

---

## 快手

### 12. 元类的`__new__`和`__init__`有什么区别？

**参考答案**：
- `__new__`：创建类对象之前调用，负责创建并返回类对象
- `__init__`：类对象创建之后调用，负责初始化类对象

```python
class Meta(type):
    def __new__(cls, name, bases, namespace):
        print(f"Creating class {name}")
        # 可以修改namespace
        return super().__new__(cls, name, bases, namespace)

    def __init__(cls, name, bases, namespace):
        print(f"Initializing class {name}")
        super().__init__(name, bases, namespace)
```

使用场景：
- `__new__`：修改类定义、添加方法、验证
- `__init__`：注册类、设置元数据

**考察点**：元类的完整生命周期、对象创建过程

---

## 小米

### 13. 如何使用装饰器实现权限验证？

**参考答案**：
```python
def require_permission(permission):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(user, *args, **kwargs):
            if permission not in user.get('permissions', []):
                raise PermissionError(
                    f"User {user['name']} lacks {permission} permission"
                )
            return func(user, *args, **kwargs)
        return wrapper
    return decorator

@require_permission('admin')
def delete_user(user, user_id):
    return f"Deleted user {user_id}"

admin = {'name': 'Alice', 'permissions': ['admin']}
delete_user(admin, 123)  # OK
```

可以扩展为支持多权限、角色验证等。

**考察点**：装饰器在实际业务中的应用

---

## 滴滴

### 14. 装饰器如何传递参数给被装饰的函数？

**参考答案**：
```python
def validate_int(*param_names):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 验证kwargs中的指定参数
            for name in param_names:
                if name in kwargs and not isinstance(kwargs[name], int):
                    raise TypeError(f"{name} must be int")
            return func(*args, **kwargs)
        return wrapper
    return decorator

@validate_int('x', 'y')
def add(x, y):
    return x + y

add(x=1, y=2)  # OK
add(x=1, y="2")  # TypeError
```

关键点：
- 使用`*args`和`**kwargs`透传参数
- 可以在wrapper中验证或修改参数
- 保持参数传递的透明性

**考察点**：装饰器的参数处理

---

## 总结

### 高频考点：
1. **装饰器原理**：闭包、高阶函数、`*args`和`**kwargs`
2. **`@functools.wraps`**：保留函数元信息
3. **带参数的装饰器**：三层嵌套
4. **类装饰器**：`__call__`方法、状态管理
5. **元类应用**：单例、ORM、自动注册
6. **执行顺序**：装饰器堆叠的洋葱模型

### 实战建议：
1. 熟练掌握简单装饰器的编写
2. 理解`@functools.wraps`的重要性
3. 了解元类的原理但谨慎使用
4. 在实际项目中多使用装饰器解耦代码
5. 阅读优秀框架的装饰器实现（Flask、Django）
