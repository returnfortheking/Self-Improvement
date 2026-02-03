# Day 05-06: 装饰器与元类

## 一、装饰器（Decorators）

### 1.1 装饰器基础

装饰器是Python中非常强大的特性，它允许你在不修改原函数代码的情况下，为函数添加额外的功能。装饰器本质上是一个接受函数作为参数并返回一个新函数的高阶函数。

#### 为什么需要装饰器？

- **代码复用**：避免在多个函数中编写重复的代码
- **关注点分离**：将业务逻辑与辅助功能（如日志、计时、权限验证）分离
- **代码优雅**：使用`@`语法使代码更加简洁

### 1.2 简单装饰器

```python
def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

# 等价于: say_hello = my_decorator(say_hello)
```

### 1.3 装饰器执行流程

1. Python解释器遇到`@decorator`语法
2. 将被装饰的函数作为参数传递给装饰器
3. 装饰器返回一个新函数（通常是wrapper函数）
4. 原函数名绑定到返回的新函数上

### 1.4 保留原函数信息

使用`@functools.wraps`装饰器可以保留原函数的元信息（如`__name__`、`__doc__`）：

```python
import functools

def decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
```

### 1.5 带参数的装饰器

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

@repeat(3)
def greet(name):
    print(f"Hello, {name}!")
```

### 1.6 类装饰器

类也可以作为装饰器，只需要实现`__call__`方法：

```python
class CountCalls:
    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"Call {self.count} of {self.func.__name__}")
        return self.func(*args, **kwargs)
```

### 1.7 装饰器应用场景

1. **日志记录**：记录函数调用信息
2. **性能测试**：测量函数执行时间
3. **缓存**：缓存函数结果（如`@lru_cache`）
4. **权限验证**：检查用户权限
5. **重试机制**：失败自动重试
6. **类型检查**：验证参数类型

## 二、元类（Metaclasses）

### 2.1 元类基础

元类是创建类的"类"，或者说元类是类的类。在Python中，`type`就是默认的元类。

```python
# type是元类
print(type(type))  # <class 'type'>

# 普通类的类型是type
class MyClass:
    pass
print(type(MyClass))  # <class 'type'>
```

### 2.2 type动态创建类

`type`可以动态创建类：

```python
# type(name, bases, dict)
MyClass = type('MyClass', (object,), {
    'x': 10,
    'say_hello': lambda self: print('Hello')
})
```

### 2.3 自定义元类

通过继承`type`创建自定义元类：

```python
class MyMeta(type):
    def __new__(cls, name, bases, namespace):
        # 在类创建之前进行干预
        print(f"Creating class: {name}")
        return super().__new__(cls, name, bases, namespace)

class MyClass(metaclass=MyMeta):
    pass
```

### 2.4 元类应用场景

1. **单例模式**：确保类只有一个实例
2. **自动注册**：自动将类注册到某个注册表
3. **接口检查**：强制子类实现特定方法
4. **ORM框架**：如Django的Model使用元类
5. **属性验证**：自动添加属性验证逻辑

### 2.5 元类 vs 装饰器

- **装饰器**：修改已存在的函数或类
- **元类**：在类创建时就进行干预

元类更强大但也更复杂，应优先考虑使用装饰器或类装饰器。

## 三、最佳实践

### 3.1 装饰器最佳实践

1. 始终使用`@functools.wraps`
2. 装饰器应该透明地传递参数
3. 考虑使用类装饰器来管理状态
4. 保持装饰器简单和单一职责

### 3.2 元类最佳实践

1. 元类非常强大，但非常复杂
2. 优先考虑使用装饰器或类装饰器
3. 只有在框架代码中才考虑使用元类
4. 提供清晰的文档说明元类的作用

## 四、常见陷阱

### 4.1 装饰器参数丢失

未使用`@wraps`时，原函数的元信息会丢失。

### 4.2 装饰器顺序

多个装饰器的执行顺序是从下到上：

```python
@decorator1
@decorator2
def func():
    pass
# 等价于: func = decorator1(decorator2(func))
```

### 4.3 元类继承冲突

当多个基类有不同的元类时会引发冲突，需要创建兼容的元类。

## 五、总结

- **装饰器**：在不修改原函数的情况下添加功能，优先使用
- **元类**：控制类的创建行为，复杂但强大，谨慎使用
- 掌握这两种特性可以让你的代码更加优雅和灵活
- 在实际应用中，装饰器的使用频率远高于元类

## 六、学习建议

1. 从简单的装饰器开始，逐步掌握复杂用法
2. 阅读优秀的装饰器实现（如Flask的`@route`）
3. 理解元类的原理，但在实际开发中谨慎使用
4. 多练习，理解装饰器和元类的适用场景
