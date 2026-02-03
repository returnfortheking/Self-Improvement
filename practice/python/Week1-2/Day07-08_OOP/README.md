# Day 07-08: 面向对象编程

## 一、面向对象基础

### 1.1 类与对象

类（Class）是对象的模板或蓝图，对象（Object）是类的实例。

```python
class Dog:
    species = "Canis familiaris"  # 类属性

    def __init__(self, name, age):
        self.name = name  # 实例属性
        self.age = age

    def bark(self):  # 方法
        return f"{self.name} says Woof!"
```

### 1.2 面向对象的三大特征

#### 封装（Encapsulation）
隐藏对象的实现细节，只对外暴露必要的接口。

```python
class BankAccount:
    def __init__(self, balance):
        self.__balance = balance  # 私有属性

    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount

    def get_balance(self):
        return self.__balance
```

#### 继承（Inheritance）
子类继承父类的属性和方法，实现代码复用。

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof!"
```

#### 多态（Polymorphism）
不同对象对同一消息做出不同响应。

```python
def make_sound(animal):
    print(animal.speak())

dog = Dog("Buddy")
cat = Cat("Whiskers")
make_sound(dog)  # Buddy says Woof!
make_sound(cat)  # Whiskers says Meow!
```

## 二、类的高级特性

### 2.1 魔法方法

Python通过特殊方法（魔法方法）实现对象的特殊行为。

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):  # 字符串表示
        return f"Vector({self.x}, {self.y})"

    def __add__(self, other):  # 运算符重载
        return Vector(self.x + other.x, self.y + other.y)

    def __eq__(self, other):  # 相等比较
        return self.x == other.x and self.y == other.y
```

常用魔法方法：
- `__init__`: 初始化
- `__str__`/`__repr__`: 字符串表示
- `__len__`: 长度
- `__getitem__`/`__setitem__`: 索引访问
- `__call__`: 对象可调用
- `__enter__`/`__exit__`: 上下文管理器

### 2.2 属性访问控制

```python
class Person:
    def __init__(self, name):
        self._name = name  # 保护属性

    @property
    def name(self):  # getter
        return self._name

    @name.setter
    def name(self, value):  # setter
        if not isinstance(value, str):
            raise TypeError("Name must be a string")
        self._name = value
```

### 2.3 类方法与静态方法

```python
class MyClass:
    class_var = "shared"

    def __init__(self):
        self.instance_var = "unique"

    @classmethod
    def class_method(cls):
        return cls.class_var

    @staticmethod
    def static_method():
        return "No access to self or cls"
```

区别：
- **实例方法**：访问实例属性，第一个参数是`self`
- **类方法**：访问类属性，第一个参数是`cls`，装饰器`@classmethod`
- **静态方法**：不访问实例或类，装饰器`@staticmethod`

## 三、继承与多态

### 3.1 方法重写（Override）

子类可以重写父类的方法以实现特定行为。

```python
class Animal:
    def speak(self):
        return "Some sound"

class Dog(Animal):
    def speak(self):
        return "Woof!"
```

### 3.2 super()函数

调用父类方法的方式。

```python
class Child(Parent):
    def __init__(self, name, age):
        super().__init__(name)  # 调用父类__init__
        self.age = age
```

### 3.3 多继承与MRO

Python支持多继承，使用C3线性化算法确定方法解析顺序（MRO）。

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

d = D()
d.method()  # 输出: B
print(D.mro())  # [<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>]
```

## 四、抽象类与接口

### 4.1 抽象基类（ABC）

使用`abc`模块定义抽象基类，强制子类实现特定方法。

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
```

## 五、数据类

### 5.1 使用dataclasses

Python 3.7+提供`@dataclass`装饰器，简化数据类的定义。

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

    def distance_to(self, other):
        return ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5
```

## 六、设计模式

### 6.1 单例模式

确保类只有一个实例。

```python
class Singleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

### 6.2 工厂模式

根据参数创建不同对象。

```python
class AnimalFactory:
    @staticmethod
    def create_animal(animal_type):
        if animal_type == "dog":
            return Dog()
        elif animal_type == "cat":
            return Cat()
```

### 6.3 观察者模式

对象间的一对多依赖关系。

```python
class Subject:
    def __init__(self):
        self._observers = []

    def attach(self, observer):
        self._observers.append(observer)

    def notify(self, message):
        for observer in self._observers:
            observer.update(message)
```

## 七、最佳实践

### 7.1 类设计原则

1. **单一职责原则**：类应该只有一个改变的理由
2. **开闭原则**：对扩展开放，对修改关闭
3. **里氏替换原则**：子类可以替换父类
4. **接口隔离原则**：不应该强迫实现不需要的方法
5. **依赖倒置原则**：依赖抽象而非具体实现

### 7.2 命名规范

- 类名：大驼峰（`MyClass`）
- 方法/变量：小写+下划线（`my_method`）
- 私有属性：单下划线前缀（`_private`）
- 保护属性：双下划线前缀（`__private`）

### 7.3 使用type hints

```python
from typing import List, Optional

class Student:
    def __init__(self, name: str, grades: List[int]):
        self.name = name
        self.grades = grades

    def get_average(self) -> float:
        return sum(self.grades) / len(self.grades)
```

## 八、常见陷阱

### 8.1 可变默认参数

```python
# 错误示例
class Bad:
    def __init__(self, items=[]):
        self.items = items

# 正确做法
class Good:
    def __init__(self, items=None):
        self.items = items if items is not None else []
```

### 8.2 类属性vs实例属性

```python
class Dog:
    tricks = []  # 类属性，所有实例共享

    def __init__(self, name):
        self.name = name  # 实例属性
```

## 九、总结

面向对象编程是Python的核心特性之一：
- **封装**：隐藏实现细节
- **继承**：代码复用
- **多态**：灵活性
- **抽象**：简化复杂度

掌握OOP可以编写更加模块化、可维护的代码。

## 十、学习建议

1. 从简单的类开始，逐步掌握复杂特性
2. 理解魔法方法的作用和用法
3. 学习设计模式，提高代码质量
4. 使用dataclasses简化代码
5. 遵循SOLID原则
