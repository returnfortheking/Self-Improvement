# 面向对象编程练习题

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum, auto
import copy

# ==================== 练习题 1-5: 基础类定义 ====================

# 练习 1: 创建一个Rectangle类
# 要求：有width和height属性，实现area()和perimeter()方法
class Rectangle:
    """TODO: 实现矩形类"""
    pass

# 测试: r = Rectangle(3, 4); print(r.area()) 应该输出12


# 练习 2: 创建一个Student类
# 要求：有name, age, grades属性，实现add_grade()和get_average()方法
class Student:
    """TODO: 实现学生类"""
    pass

# 测试: s = Student("Alice", 18); s.add_grade(90); print(s.get_average())


# 练习 3: 创建一个BankAccount类
# 要求：封装balance属性，提供deposit()和withdraw()方法
class BankAccount:
    """TODO: 实现银行账户类"""
    pass

# 测试: account = BankAccount(100); account.deposit(50); account.withdraw(30)


# 练习 4: 使用@property装饰器
# 要求：为Student类添加name属性的getter和setter，验证name不能为空
class StudentWithValidation:
    """TODO: 实现带验证的Student类"""
    pass

# 测试: s = StudentWithValidation("Alice"); s.name = "" 应该抛出异常


# 练习 5: 实现类方法和静态方法
# 要求：添加from_birth_year类方法和is_adult静态方法
class Person:
    """TODO: 实现Person类"""
    pass

# 测试: Person.from_birth_year("Alice", 2000); Person.is_adult(20)


# ==================== 练习题 6-10: 继承与多态 ====================

# 练习 6: 创建Animal基类和子类
# 要求：Animal有speak()抽象方法，Dog和Cat分别实现
class Animal(ABC):
    """TODO: 实现抽象基类"""
    pass

class Dog(Animal):
    """TODO: 实现Dog类"""
    pass

class Cat(Animal):
    """TODO: 实现Cat类"""
    pass

# 测试: animals = [Dog("Buddy"), Cat("Whiskers")]; for a in animals: print(a.speak())


# 练习 7: 实现多态
# 要求：创建一个函数，接受不同形状对象并计算面积
class Shape(ABC):
    """TODO: 实现Shape抽象类"""
    pass

class Circle(Shape):
    """TODO: 实现Circle类"""
    pass

class Rectangle2(Shape):
    """TODO: 实现Rectangle类"""
    pass

# 测试: shapes = [Circle(5), Rectangle2(3, 4)]; print_total_area(shapes)


# 练习 8: 使用super()调用父类方法
# 要求：ElectricCar继承Car，重写describe()方法并调用父类方法
class Car:
    """TODO: 实现Car基类"""
    pass

class ElectricCar(Car):
    """TODO: 实现ElectricCar子类"""
    pass

# 测试: tesla = ElectricCar("Tesla", "Model S", 100); print(tesla.describe())


# 练习 9: 实现多继承
# 要求：创建一个可飞、可游的鸭子类
class Flyable:
    """TODO: 实现可飞行混入类"""
    pass

class Swimmable:
    """TODO: 实现可游泳混入类"""
    pass

class Duck(Animal, Flyable, Swimmable):
    """TODO: 实现Duck类"""
    pass

# 测试: duck = Duck("Donald"); print(duck.fly(), duck.swim())


# 练习 10: 理解MRO
# 要求：创建多继承类，打印MRO顺序
class A:
    pass

class B(A):
    pass

class C(A):
    pass

class D(B, C):
    pass

# 测试: print(D.mro())


# ==================== 练习题 11-15: 魔法方法 ====================

# 练习 11: 实现Vector类
# 要求：实现__add__, __sub__, __eq__, __str__方法
class Vector:
    """TODO: 实现Vector类"""
    pass

# 测试: v1 = Vector(3, 4); v2 = Vector(1, 2); print(v1 + v2)


# 练习 12: 实现自定义序列
# 要求：实现__len__, __getitem__, __setitem__方法
class CustomList:
    """TODO: 实现自定义列表"""
    pass

# 测试: cl = CustomList(); cl.append(1); print(cl[0]); print(len(cl))


# 练习 13: 实现上下文管理器
# 要求：创建一个计时的上下文管理器
class Timer:
    """TODO: 实现Timer上下文管理器"""
    pass

# 测试: with Timer() as t: time.sleep(0.1); print(f"Elapsed: {t.elapsed}s")


# 练习 14: 实现可调用对象
# 要求：创建一个Multiplier类，可以像函数一样调用
class Multiplier:
    """TODO: 实现Multiplier类"""
    pass

# 测试: times3 = Multiplier(3); print(times3(5))


# 练习 15: 实现迭代器
# 要求：创建一个Fibonacci迭代器
class FibonacciIterator:
    """TODO: 实现斐波那契迭代器"""
    pass

# 测试: for fib in FibonacciIterator(10): print(fib)


# ==================== 练习题 16-20: 设计模式 ====================

# 练习 16: 实现单例模式
# 要求：确保类只有一个实例
class Singleton:
    """TODO: 实现单例模式"""
    pass

# 测试: s1 = Singleton(); s2 = Singleton(); print(s1 is s2)


# 练习 17: 实现工厂模式
# 要求：根据类型创建不同的形状对象
class ShapeFactory:
    """TODO: 实现形状工厂"""
    pass

# 测试: factory = ShapeFactory(); circle = factory.create_shape("circle", 5)


# 练习 18: 实现观察者模式
# 要求：Subject可以通知多个Observer
class Observer(ABC):
    """TODO: 实现Observer接口"""
    pass

class Subject:
    """TODO: 实现Subject类"""
    pass

class ConcreteObserver(Observer):
    """TODO: 实现具体观察者"""
    pass

# 测试: subject = Subject(); observer = ConcreteObserver(); subject.attach(observer)


# 练习 19: 使用dataclass
# 要求：创建Point和Circle数据类
@dataclass
class Point:
    """TODO: 实现Point数据类"""
    pass

@dataclass
class Circle2:
    """TODO: 实现Circle数据类"""
    center: Point
    radius: float

    def area(self) -> float:
        """TODO: 计算面积"""
        pass

# 测试: c = Circle2(Point(0, 0), 5); print(c.area())


# 练习 20: 使用描述符
# 要求：创建一个类型验证描述符
class TypedField:
    """TODO: 实现类型验证描述符"""
    pass

class Person:
    name = TypedField(str)
    age = TypedField(int)

    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

# 测试: p = Person("Alice", 18); p.age = "20" 应该抛出异常


# ==================== 挑战题 ====================

# 挑战 1: 实现一个完整的ORM模型基类
# 要求：支持字段定义、保存、查询等操作
class Field:
    """TODO: 实现字段类"""
    pass

class Model:
    """TODO: 实现Model基类"""
    pass

class User(Model):
    name = Field(str)
    age = Field(int)
    email = Field(str)

# 测试: user = User(name="Alice", age=18, email="alice@example.com")


# 挑战 2: 实现一个事件系统
# 要求：支持事件注册、触发、取消等功能
class EventEmitter:
    """TODO: 实现事件发射器"""
    pass

class Button(EventEmitter):
    """TODO: 实现Button类"""
    pass

# 测试: button = Button(); button.on("click", lambda: print("Clicked!"))


# 挑战 3: 实现一个简单的游戏框架
# 要求：包含GameObject, Component, Scene等类
class GameObject:
    """TODO: 实现游戏对象"""
    pass

class Component:
    """TODO: 实现组件基类"""
    pass

class Transform(Component):
    """TODO: 实现Transform组件"""
    pass

class Scene:
    """TODO: 实现场景类"""
    pass

# 测试: scene = Scene(); obj = GameObject("Player")


# 挑战 4: 实现一个类型安全的枚举类
# 要求：支持值验证、名称访问、迭代等
class SafeEnum:
    """TODO: 实现安全枚举"""
    pass

class Color(SafeEnum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"

# 测试: print(Color.RED); print(Color.RED.value)


# 挑战 5: 实现一个简单的依赖注入容器
# 要求：支持注册、解析、自动注入等功能
class Container:
    """TODO: 实现DI容器"""
    pass

class Database:
    """TODO: 实现Database类"""
    pass

class UserService:
    """TODO: 实现UserService类"""
    pass

# 测试: container = Container(); container.register(Database); container.register(UserService)


if __name__ == "__main__":
    print("面向对象编程练习题")
    print("请完成每个TODO部分的代码")
    print("运行测试验证你的实现")
