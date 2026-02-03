# Day 07-08: 面向对象编程面试真题

## 阿里巴巴

### 1. 解释Python中的多重继承及其MRO机制

**参考答案**：
Python支持多重继承，当类有多个父类时，使用C3线性化算法确定方法解析顺序（MRO）。

```python
class A:
    def method(self):
        print("A")

class B(A):
    def method(self):
        print("B")

class C(A):
    def method(self):
        print("C")

class D(B, C):
    pass

print(D.mro())
# [<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>]
```

MRO保证：
1. 子类优先于父类
2. 多个父类按照声明顺序查找
3. 不会重复查找同一个类

**考察点**：多重继承、MRO、C3算法

---

### 2. 什么是Python的魔法方法？请列举常用的魔法方法及其用途

**参考答案**：
魔法方法（Magic Methods）是以双下划线开头和结尾的特殊方法，用于实现对象的特殊行为。

常用魔法方法：
- `__init__`: 对象初始化
- `__str__`/`__repr__`: 字符串表示
- `__len__`: 长度
- `__getitem__`/`__setitem__`: 索引访问
- `__call__`: 对象可调用
- `__eq__`/`__lt__`/`__gt__`: 比较运算
- `__add__`/`__sub__`: 算术运算
- `__enter__`/`__exit__`: 上下文管理器

```python
class Vector:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __str__(self):
        return f"Vector({self.x}, {self.y})"
```

**考察点**：Python对象模型、运算符重载

---

## 腾讯

### 3. @classmethod、@staticmethod和实例方法有什么区别？

**参考答案**：

**实例方法**：
- 第一个参数是`self`，指向实例
- 可以访问和修改实例属性
- 可以通过实例调用

**类方法**：
- 使用`@classmethod`装饰器
- 第一个参数是`cls`，指向类
- 可以访问和修改类属性
- 可以通过类或实例调用

**静态方法**：
- 使用`@staticmethod`装饰器
- 没有特殊的第一个参数
- 不能访问实例或类属性
- 可以通过类或实例调用
- 相当于类内部的工具函数

```python
class MyClass:
    class_var = "shared"

    def instance_method(self):
        return "instance method"

    @classmethod
    def class_method(cls):
        return f"class method, {cls.class_var}"

    @staticmethod
    def static_method():
        return "static method"
```

**考察点**：方法类型、访问权限、设计模式

---

### 4. 什么是抽象基类（ABC）？如何定义和使用？

**参考答案**：
抽象基类（Abstract Base Class）定义了子类必须实现的方法接口，不能直接实例化。

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

    @abstractmethod
    def perimeter(self):
        pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14 * self.radius ** 2

    def perimeter(self):
        return 2 * 3.14 * self.radius

# Shape()  # TypeError: Can't instantiate abstract class
circle = Circle(5)  # OK
```

应用场景：
- 定义接口规范
- 强制子类实现特定方法
- 框架设计

**考察点**：抽象类、接口设计、多态

---

## 字节跳动

### 5. 解释Python中的封装，如何实现私有属性？

**参考答案**：
封装是隐藏对象的实现细节，只暴露必要的接口。

Python通过命名约定实现"私有"属性：
- 单下划线`_var`：保护属性，约定内部使用
- 双下划线`__var`：私有属性，触发名称改写（name mangling）
- 双下划线首尾`__var__`：魔法方法或特殊属性

```python
class Person:
    def __init__(self):
        self.public = "public"
        self._protected = "protected"
        self.__private = "private"

    def get_private(self):
        return self.__private

p = Person()
print(p.public)  # OK
print(p._protected)  # OK但不推荐
print(p.__private)  # AttributeError
print(p._Person__private)  # 可以访问但不推荐
```

最佳实践：使用`@property`装饰器提供受控的属性访问。

**考察点**：封装、访问控制、Python特性

---

### 6. 什么是组合优于继承？请举例说明

**参考答案**：
组合优于继承是面向对象设计的原则，优先使用组合（has-a关系）而非继承（is-a关系）。

```python
# 不推荐：继承
class Engine:
    def start(self):
        print("Engine starting")

class Car(Engine):
    pass  # Car is-a Engine? 不太合理

# 推荐：组合
class Car:
    def __init__(self):
        self.engine = Engine()  # Car has-a Engine

    def start(self):
        self.engine.start()
```

优势：
1. 更灵活，可以在运行时改变组合对象
2. 避免深层继承层次
3. 更容易理解和维护
4. 符合单一职责原则

**考察点**：设计原则、组合vs继承、架构设计

---

## 美团

### 7. 如何实现单例模式？比较不同实现方式的优缺点

**参考答案**：

**1. 使用__new__**：
```python
class Singleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

**2. 使用装饰器**：
```python
def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance
```

**3. 使用元类**：
```python
class SingletonMeta(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
```

**4. 使用模块**（Pythonic）：
```python
# singleton.py
class Singleton:
    pass
instance = Singleton()
```

比较：
- **模块**：最简单可靠，但不够灵活
- **__new__**：常用，但子类会创建新实例
- **装饰器**：不影响类定义，但子类会创建新实例
- **元类**：最强大，所有子类都单例，但复杂

**考察点**：设计模式、多种实现对比、架构选择

---

## 百度

### 8. 什么是描述符（Descriptor）？请举例说明

**参考答案**：
描述符是实现了`__get__`、`__set__`或`__delete__`方法的类，用于控制属性的访问。

```python
class ValidatedAttribute:
    def __init__(self, name, type_):
        self.name = name
        self.type_ = type_

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__[self.name]

    def __set__(self, obj, value):
        if not isinstance(value, self.type_):
            raise TypeError(f"{self.name} must be {self.type_}")
        obj.__dict__[self.name] = value

class Person:
    name = ValidatedAttribute("name", str)
    age = ValidatedAttribute("age", int)

p = Person()
p.name = "Alice"  # OK
p.age = "20"  # TypeError
```

应用场景：
- 类型验证
- 懒加载
- 计算属性
- ORM映射

**考察点**：描述符协议、属性控制、元编程

---

## 网易

### 9. 解释Python中的迭代器协议和生成器

**参考答案**：

**迭代器协议**：实现`__iter__`和`__next__`方法。

```python
class CountDown:
    def __init__(self, start):
        self.start = start

    def __iter__(self):
        self.current = self.start
        return self

    def __next__(self):
        if self.current < 0:
            raise StopIteration
        value = self.current
        self.current -= 1
        return value

for i in CountDown(3):
    print(i)  # 3, 2, 1, 0
```

**生成器**：使用`yield`关键字的函数，自动实现迭代器协议。

```python
def countdown(n):
    while n >= 0:
        yield n
        n -= 1

for i in countdown(3):
    print(i)  # 3, 2, 1, 0
```

优势：
- 代码更简洁
- 自动实现迭代器协议
- 支持惰性计算
- 节省内存

**考察点**：迭代器、生成器、Python协议

---

## 京东

### 10. 什么是`__slots__`？它有什么作用？

**参考答案**：
`__slots__`是类属性，用于限制实例可以拥有的属性，从而节省内存。

```python
class WithoutSlots:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class WithSlots:
    __slots__ = ['x', 'y']

    def __init__(self, x, y):
        self.x = x
        self.y = y

# w = WithSlots(1, 2)
# w.z = 3  # AttributeError
```

作用：
1. **节省内存**：不使用`__dict__`，每个实例占用更少内存
2. **限制属性**：防止意外添加新属性
3. **提高访问速度**：属性访问更快

缺点：
1. 不能动态添加新属性
2. 不支持多继承中的某些特性
3. 每个子类需要重新定义`__slots__`

适用场景：
- 创建大量小对象
- 性能敏感的应用
- 需要严格限制属性的场景

**考察点**：内存优化、Python对象模型

---

## 快手

### 11. 如何使用dataclass？它相比普通类有什么优势？

**参考答案**：
`dataclass`是Python 3.7+提供的装饰器，用于自动生成数据类的方法。

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

    def distance_to(self, other):
        return ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5

# 自动生成：__init__, __repr__, __eq__
p1 = Point(3, 4)
p2 = Point(3, 4)
print(p1 == p2)  # True
print(p1)  # Point(x=3, y=4)
```

优势：
1. **减少样板代码**：自动生成`__init__`、`__repr__`、`__eq__`等
2. **类型提示**：集成类型注解
3. **不可变性**：通过`frozen=True`实现
4. **继承友好**：支持继承
5. **默认值**：支持字段默认值

高级特性：
```python
@dataclass
class Student:
    name: str
    age: int = 18
    grades: list = field(default_factory=list)

    def __post_init__(self):
        """初始化后处理"""
        if not self.name:
            raise ValueError("Name cannot be empty")
```

**考察点**：Python 3.7+特性、代码简化

---

## 拼多多

### 12. 什么是混入类（Mixin）？如何使用？

**参考答案**：
混入类是一种提供特定功能的类，通过多重继承为其他类添加功能。

```python
class Flyable:
    def fly(self):
        return "Flying..."

class Swimmable:
    def swim(self):
        return "Swimming..."

class Duck(Flyable, Swimmable):
    def quack(self):
        return "Quack!"

duck = Duck()
print(duck.fly())
print(duck.swim())
```

特点：
1. 不独立使用，只为其他类提供功能
2. 通常不定义`__init__`
3. 使用窄接口，专注单一功能
4. 命名通常以Mixin结尾

最佳实践：
1. 保持简单，单一职责
2. 避免状态，尽量使用方法
3. 注意MRO顺序
4. 明确文档说明

应用场景：
- 日志功能
- 缓存功能
- 序列化功能
- 权限验证

**考察点**：多重继承、代码复用、设计模式

---

## 小米

### 13. 如何实现上下文管理器？

**参考答案**：
实现`__enter__`和`__exit__`方法，或使用`@contextmanager`装饰器。

**方法1：类实现**：
```python
class Timer:
    def __init__(self):
        self.start = None
        self.elapsed = None

    def __enter__(self):
        import time
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.elapsed = time.time() - self.start
        return False  # 不抑制异常

with Timer() as t:
    time.sleep(0.1)

print(f"Elapsed: {t.elapsed:.2f}s")
```

**方法2：生成器实现**：
```python
from contextlib import contextmanager

@contextmanager
def timer():
    import time
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"Elapsed: {elapsed:.2f}s")

with timer():
    time.sleep(0.1)
```

**考察点**：上下文管理器协议、资源管理

---

## 滴滴

### 14. 什么是鸭子类型（Duck Typing）？

**参考答案**：
鸭子类型是Python的动态类型哲学，关注对象的行为而非类型。

"如果它走起路来像鸭子，叫起来像鸭子，那它就是鸭子"

```python
class Duck:
    def quack(self):
        print("Quack!")

class Person:
    def quack(self):
        print("I'm quacking like a duck!")

def make_it_quack(obj):
    obj.quack()  # 不关心类型，只关心是否有quack方法

duck = Duck()
person = Person()

make_it_quack(duck)  # Quack!
make_it_quack(person)  # I'm quacking like a duck!
```

优势：
1. 灵活性高
2. 代码简洁
3. 易于测试和mock
4. 支持多态

注意事项：
1. 缺少类型检查可能导致运行时错误
2. 建议使用类型注解提高可读性
3. 使用Protocol定义隐式接口

**考察点**：动态类型、多态、Python哲学

---

## 总结

### 高频考点：
1. **面向对象三大特征**：封装、继承、多态
2. **魔法方法**：`__init__`、`__str__`、`__repr__`、`__eq__`等
3. **多重继承与MRO**：C3线性化算法
4. **方法类型**：实例方法、类方法、静态方法
5. **property装饰器**：属性访问控制
6. **抽象基类**：接口定义
7. **描述符**：属性控制机制
8. **dataclass**：简化数据类定义

### 实战建议：
1. 理解并应用SOLID原则
2. 掌握常用设计模式
3. 合理使用继承和组合
4. 熟练使用魔法方法
5. 了解Python对象模型
6. 编写类型友好的代码
