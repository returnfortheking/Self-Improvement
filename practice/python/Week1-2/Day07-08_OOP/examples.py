# 面向对象编程代码示例

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Optional
import copy

# ==================== 1. 基础类定义 ====================

class Dog:
    """简单的Dog类"""
    species = "Canis familiaris"  # 类属性

    def __init__(self, name: str, age: int):
        self.name = name  # 实例属性
        self.age = age

    def bark(self) -> str:
        """实例方法"""
        return f"{self.name} says Woof!"

    @classmethod
    def from_birth_year(cls, name: str, birth_year: int):
        """类方法：根据出生年份创建实例"""
        current_year = 2024
        age = current_year - birth_year
        return cls(name, age)

    @staticmethod
    def is_dog(adjective: str) -> bool:
        """静态方法：不访问实例或类"""
        return adjective.lower() == "dog"

# dog = Dog("Buddy", 3)
# print(dog.bark())
# print(Dog.species)


# ==================== 2. 属性访问控制 ====================

class BankAccount:
    """银行账户：演示封装"""
    def __init__(self, initial_balance: float):
        self._balance = initial_balance  # 保护属性

    def deposit(self, amount: float) -> None:
        """存款"""
        if amount > 0:
            self._balance += amount
        else:
            raise ValueError("Amount must be positive")

    def withdraw(self, amount: float) -> None:
        """取款"""
        if amount > self._balance:
            raise ValueError("Insufficient funds")
        self._balance -= amount

    def get_balance(self) -> float:
        """获取余额"""
        return self._balance

    @property
    def balance(self) -> float:
        """属性装饰器：提供getter"""
        return self._balance

    @balance.setter
    def balance(self, value: float):
        """属性装饰器：提供setter"""
        if value < 0:
            raise ValueError("Balance cannot be negative")
        self._balance = value


# ==================== 3. 继承与方法重写 ====================

class Animal:
    """动物基类"""
    def __init__(self, name: str):
        self.name = name

    def speak(self) -> str:
        """发出声音"""
        raise NotImplementedError("Subclass must implement abstract method")

    def __str__(self) -> str:
        return f"{self.__class__.__name__} named {self.name}"


class Dog(Animal):
    """狗类"""
    def speak(self) -> str:
        return f"{self.name} says Woof!"

    def fetch(self) -> str:
        """狗特有方法"""
        return f"{self.name} is fetching!"


class Cat(Animal):
    """猫类"""
    def speak(self) -> str:
        return f"{self.name} says Meow!"

    def scratch(self) -> str:
        """猫特有方法"""
        return f"{self.name} is scratching!"


# ==================== 4. 多态 ====================

def animal_sound(animal: Animal) -> None:
    """多态：处理不同类型的动物"""
    print(animal.speak())


# dog = Dog("Buddy")
# cat = Cat("Whiskers")
# animal_sound(dog)  # Buddy says Woof!
# animal_sound(cat)  # Whiskers says Meow!


# ==================== 5. 魔法方法 ====================

class Vector:
    """向量类：演示魔法方法"""
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __str__(self) -> str:
        """用户友好的字符串表示"""
        return f"Vector({self.x}, {self.y})"

    def __repr__(self) -> str:
        """开发者友好的字符串表示"""
        return f"Vector(x={self.x}, y={self.y})"

    def __add__(self, other: 'Vector') -> 'Vector':
        """向量加法：+运算符"""
        return Vector(self.x + other.x, self.y + other.y)

    def __eq__(self, other: object) -> bool:
        """相等比较：==运算符"""
        if not isinstance(other, Vector):
            return False
        return self.x == other.x and self.y == other.y

    def __len__(self) -> int:
        """长度：len()函数"""
        return int((self.x**2 + self.y**2)**0.5)

    def __bool__(self) -> bool:
        """真值测试：bool()函数"""
        return self.x != 0 or self.y != 0


# v1 = Vector(3, 4)
# v2 = Vector(1, 2)
# print(v1 + v2)  # Vector(4, 6)
# print(len(v1))  # 5


# ==================== 6. 抽象基类 ====================

class Shape(ABC):
    """形状抽象基类"""
    @abstractmethod
    def area(self) -> float:
        """计算面积"""
        pass

    @abstractmethod
    def perimeter(self) -> float:
        """计算周长"""
        pass


class Circle(Shape):
    """圆形"""
    def __init__(self, radius: float):
        self.radius = radius

    def area(self) -> float:
        return 3.14 * self.radius ** 2

    def perimeter(self) -> float:
        return 2 * 3.14 * self.radius


class Rectangle(Shape):
    """矩形"""
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

    def perimeter(self) -> float:
        return 2 * (self.width + self.height)


# ==================== 7. 多继承与super() ====================

class Flyable:
    """可飞行混入类"""
    def fly(self) -> str:
        return "Flying..."


class Swimmable:
    """可游泳混入类"""
    def swim(self) -> str:
        return "Swimming..."


class Duck(Animal, Flyable, Swimmable):
    """鸭子：多继承"""
    def speak(self) -> str:
        return f"{self.name} says Quack!"


# duck = Duck("Donald")
# print(duck.fly())
# print(duck.swim())


# ==================== 8. 迭代器协议 ====================

class CountDown:
    """倒计时迭代器"""
    def __init__(self, start: int):
        self.start = start

    def __iter__(self):
        """返回迭代器对象"""
        self.current = self.start
        return self

    def __next__(self):
        """返回下一个值"""
        if self.current < 0:
            raise StopIteration
        value = self.current
        self.current -= 1
        return value


# for i in CountDown(3):
#     print(i)  # 3, 2, 1, 0


# ==================== 9. 上下文管理器 ====================

class FileManager:
    """文件管理器：上下文管理器"""
    def __init__(self, filename: str, mode: str):
        self.filename = filename
        self.mode = mode
        self.file = None

    def __enter__(self):
        """进入上下文"""
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文"""
        if self.file:
            self.file.close()
        return False  # 不抑制异常


# with FileManager("test.txt", "w") as f:
#     f.write("Hello!")


# ==================== 10. 可调用对象 ====================

class Multiplier:
    """可调用对象：像函数一样使用"""
    def __init__(self, factor: int):
        self.factor = factor

    def __call__(self, x: int) -> int:
        """使实例可调用"""
        return x * self.factor


# times3 = Multiplier(3)
# print(times3(5))  # 15


# ==================== 11. 数据类 ====================

@dataclass
class Point:
    """点：使用dataclass"""
    x: float
    y: float

    def distance_to(self, other: 'Point') -> float:
        """计算到另一点的距离"""
        return ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5


@dataclass
class Student:
    """学生：带默认值的数据类"""
    name: str
    age: int = 18
    grades: List[int] = None

    def __post_init__(self):
        """初始化后处理"""
        if self.grades is None:
            self.grades = []

    def average(self) -> float:
        """计算平均分"""
        if not self.grades:
            return 0.0
        return sum(self.grades) / len(self.grades)


# ==================== 12. 单例模式 ====================

class Singleton:
    """单例模式"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._initialized = True
            self.data = "Singleton data"


# s1 = Singleton()
# s2 = Singleton()
# print(s1 is s2)  # True


# ==================== 13. 工厂模式 ====================

class AnimalFactory:
    """动物工厂"""
    @staticmethod
    def create_animal(animal_type: str, name: str) -> Animal:
        """根据类型创建动物"""
        animals = {
            "dog": Dog,
            "cat": Cat,
            "duck": Duck,
        }

        animal_class = animals.get(animal_type.lower())
        if animal_class is None:
            raise ValueError(f"Unknown animal type: {animal_type}")

        return animal_class(name)


# factory = AnimalFactory()
# dog = factory.create_animal("dog", "Buddy")


# ==================== 14. 观察者模式 ====================

class Observer(ABC):
    """观察者抽象基类"""
    @abstractmethod
    def update(self, message: str):
        pass


class Subject:
    """被观察者"""
    def __init__(self):
        self._observers: List[Observer] = []

    def attach(self, observer: Observer):
        """添加观察者"""
        self._observers.append(observer)

    def detach(self, observer: Observer):
        """移除观察者"""
        self._observers.remove(observer)

    def notify(self, message: str):
        """通知所有观察者"""
        for observer in self._observers:
            observer.update(message)


class EmailAlert(Observer):
    """邮件提醒"""
    def update(self, message: str):
        print(f"Email alert: {message}")


class SMSAlert(Observer):
    """短信提醒"""
    def update(self, message: str):
        print(f"SMS alert: {message}")


# ==================== 15. 描述符 ====================

class ValidatedAttribute:
    """属性验证描述符"""
    def __init__(self, name: str, type_: type):
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

    def __delete__(self, obj):
        del obj.__dict__[self.name]


class Person:
    """使用描述符的类"""
    name = ValidatedAttribute("name", str)
    age = ValidatedAttribute("age", int)

    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age


# ==================== 16. 枚举类 ====================

from enum import Enum, auto

class Color(Enum):
    """颜色枚举"""
    RED = auto()
    GREEN = auto()
    BLUE = auto()

    def __str__(self):
        return self.name


class Status(Enum):
    """状态枚举：带值"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


# print(Color.RED)  # Color.RED
# print(Status.PENDING.value)  # "pending"


# ==================== 17. 对象拷贝 ====================

class Original:
    """演示对象拷贝"""
    def __init__(self, value: int, items: list):
        self.value = value
        self.items = items

    def __eq__(self, other):
        return self.value == other.value and self.items == other.items


# 测试浅拷贝和深拷贝
# orig = Original(10, [1, 2, 3])
# shallow = copy.copy(orig)
# deep = copy.deepcopy(orig)
#
# orig.items.append(4)
# print(shallow.items)  # [1, 2, 3, 4]  受影响
# print(deep.items)     # [1, 2, 3]     不受影响


# ==================== 18. __slots__ ====================

class SlottedClass:
    """使用__slots__节省内存"""
    __slots__ = ['x', 'y']

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        # self.z = 0  # AttributeError: 不能添加新属性


# ==================== 19. 类装饰器 ====================

def class_decorator(cls):
    """类装饰器"""
    class Wrapped(cls):
        def __init__(self, *args, **kwargs):
            print(f"Creating instance of {cls.__name__}")
            super().__init__(*args, **kwargs)

        def __str__(self):
            return f"<Wrapped {cls.__name__}>"

    return Wrapped


@class_decorator
class MyClass:
    def __init__(self, value):
        self.value = value


# ==================== 20. 类型检查 ====================

class Typed:
    """运行时类型检查装饰器"""
    def __init__(self, name: str, expected_type: type):
        self.name = name
        self.expected_type = expected_type

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__[self.name]

    def __set__(self, obj, value):
        if not isinstance(value, self.expected_type):
            raise TypeError(f"{self.name} must be {self.expected_type}")
        obj.__dict__[self.name] = value


class StrictPerson:
    name = Typed("name", str)
    age = Typed("age", int)

    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age


if __name__ == "__main__":
    # 示例代码
    dog = Dog("Buddy", 3)
    print(dog.bark())
    print(dog.species)

    v1 = Vector(3, 4)
    v2 = Vector(1, 2)
    print(f"v1 + v2 = {v1 + v2}")

    circle = Circle(5)
    print(f"Circle area: {circle.area()}")

    student = Student("Alice", 18, [90, 85, 95])
    print(f"{student.name}'s average: {student.average()}")
