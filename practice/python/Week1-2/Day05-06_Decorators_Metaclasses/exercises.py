# 装饰器与元类练习题

import functools
import time

# ==================== 练习题 1-5: 基础装饰器 ====================

# 练习 1: 创建一个日志装饰器
# 要求：在函数调用前后打印日志，显示函数名和参数
def log_decorator(func):
    """TODO: 实现日志装饰器"""
    pass

@log_decorator
def add(a, b):
    return a + b

# 测试: add(3, 5) 应该打印调用信息


# 练习 2: 创建一个计数装饰器
# 要求：统计函数被调用的次数
def count_calls(func):
    """TODO: 实现计数装饰器"""
    pass

@count_calls
def greet():
    print("Hello!")

# 测试: 调用3次后，应该显示调用3次


# 练习 3: 创建一个只执行一次的装饰器
# 要求：函数只会在第一次调用时执行，后续调用直接返回缓存结果
def once(func):
    """TODO: 实现只执行一次的装饰器"""
    pass

@once
def initialize():
    print("Initializing...")
    return "Initialized"

# 测试: 第一次调用应该打印，第二次不应该


# 练习 4: 创建一个测量执行时间的装饰器
# 要求：精确测量函数执行时间并打印
def measure_time(func):
    """TODO: 实现时间测量装饰器"""
    pass

@measure_time
def slow_operation():
    time.sleep(0.1)
    return "Done"

# 测试: 应该打印执行时间


# 练习 5: 创建一个异常处理装饰器
# 要求：捕获函数中的异常并打印错误信息
def handle_errors(func):
    """TODO: 实现异常处理装饰器"""
    pass

@handle_errors
def divide(a, b):
    return a / b

# 测试: divide(10, 0) 不应该抛出异常


# ==================== 练习题 6-10: 高级装饰器 ====================

# 练习 6: 创建一个带参数的重复执行装饰器
# 要求：可以指定重复次数
def repeat(times):
    """TODO: 实现重复执行装饰器"""
    pass

@repeat(3)
def say_hi(name):
    print(f"Hi, {name}!")

# 测试: say_hi("Alice") 应该打印3次


# 练习 7: 创建一个缓存装饰器
# 要求：缓存函数结果，相同参数直接返回缓存
def memoize(func):
    """TODO: 实现缓存装饰器"""
    pass

@memoize
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# 测试: fibonacci(50) 应该快速计算


# 练习 8: 创建一个类型检查装饰器
# 要求：检查参数类型是否正确
def type_check(**types):
    """TODO: 实现类型检查装饰器"""
    pass

@type_check(x=int, y=str)
def process(x, y):
    return f"{x}: {y}"

# 测试: process(10, "hello") 正常，process("10", 20) 抛出异常


# 练习 9: 创建一个超时装饰器
# 要求：如果函数执行超时则抛出异常
import signal

def timeout(seconds):
    """TODO: 实现超时装饰器（使用signal.alarm）"""
    pass

@timeout(2)
def long_running_task():
    time.sleep(3)
    return "Done"

# 测试: long_running_task() 应该超时


# 练习 10: 创建一个验证装饰器
# 要求：验证参数是否满足条件
def validate(**conditions):
    """TODO: 实现参数验证装饰器"""
    pass

@validate(age=lambda x: x >= 0, name=lambda x: len(x) > 0)
def create_user(name, age):
    return {"name": name, "age": age}

# 测试: create_user("", -1) 应该抛出异常


# ==================== 练习题 11-15: 类装饰器 ====================

# 练习 11: 创建一个计数类装饰器
# 要求：统计类实例被创建的次数
class CountInstances:
    """TODO: 实现实例计数类装饰器"""
    pass

@CountInstances
class User:
    def __init__(self, name):
        self.name = name

# 测试: 创建多个实例，应该显示创建次数


# 练习 12: 创建一个单例类装饰器
# 要求：确保类只有一个实例
class Singleton:
    """TODO: 实现单例类装饰器"""
    pass

@Singleton
class Database:
    def __init__(self):
        print("Connecting to database...")

# 测试: db1 and db2 应该是同一个实例


# 练习 13: 创建一个属性验证类装饰器
# 要求：为类属性添加getter和setter
class ValidateAttributes:
    """TODO: 实现属性验证类装饰器"""
    pass

@ValidateAttributes
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

# 测试: 应该自动生成property


# 练习 14: 创建一个字符串表示类装饰器
# 要求：自动添加__str__和__repr__方法
class StringRepresentation:
    """TODO: 实现字符串表示类装饰器"""
    pass

@StringRepresentation
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# 测试: print(Point(1, 2)) 应该显示友好信息


# 练习 15: 创建一个比较方法类装饰器
# 要求：自动添加比较方法（__lt__, __gt__等）
class Comparable:
    """TODO: 实现比较方法类装饰器"""
    pass

@Comparable
class Student:
    def __init__(self, name, score):
        self.name = name
        self.score = score

# 测试: 应该可以使用 <, >, == 等比较运算符


# ==================== 练习题 16-20: 元类 ====================

# 练习 16: 创建一个单例元类
# 要求：使用元类实现单例模式
class SingletonMeta(type):
    """TODO: 实现单例元类"""
    pass

class Database(metaclass=SingletonMeta):
    def __init__(self):
        print("Initializing database")

# 测试: 应该只创建一个实例


# 练习 17: 创建一个自动注册元类
# 要求：自动将所有子类注册到字典中
class RegistryMeta(type):
    """TODO: 实现自动注册元类"""
    registry = {}

class Plugin(metaclass=RegistryMeta):
    pass

class AudioPlugin(Plugin):
    pass

class VideoPlugin(Plugin):
    pass

# 测试: RegistryMeta.registry 应该包含所有插件


# 练习 18: 创建一个抽象方法元类
# 要求：强制子类实现特定方法
class AbstractMeta(type):
    """TODO: 实现抽象方法元类"""
    pass

class Shape(metaclass=AbstractMeta):
    @abstractmethod
    def area(self):
        pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    # TODO: 应该强制实现area方法


# 练习 19: 创建一个属性追踪元类
# 要求：追踪类的所有属性访问
class TrackedMeta(type):
    """TODO: 实现属性追踪元类"""
    pass

class MyClass(metaclass=TrackedMeta):
    def __init__(self):
        self.x = 10
        self.y = 20

# 测试: 访问属性时应该打印日志


# 练习 20: 创建一个接口检查元类
# 要求：确保类实现了指定的接口
class InterfaceMeta(type):
    """TODO: 实现接口检查元类"""
    pass

class Drawable(metaclass=InterfaceMeta):
    _required_methods = ['draw', 'resize']

class Circle(Drawable):
    def draw(self):
        pass

    def resize(self, factor):
        pass

# 测试: 如果没有实现所有方法应该报错


# ==================== 挑战题 ====================

# 挑战 1: 实现一个完整的ORM元类
# 要求：自动创建数据库表的映射关系
class ModelMeta(type):
    """TODO: 实现ORM元类"""
    pass

class Model(metaclass=ModelMeta):
    _fields = {}

    @classmethod
    def create_table(cls):
        """生成CREATE TABLE SQL语句"""
        pass

class User(Model):
    id = int
    name = str
    email = str

# 测试: User.create_table() 应该生成SQL


# 挑战 2: 实现一个依赖注入装饰器
# 要求：自动注入依赖到函数参数
def inject(*dependencies):
    """TODO: 实现依赖注入装饰器"""
    pass

class Database:
    def query(self):
        return "Data"

@inject('database')
def get_user(database):
    return database.query()

# 测试: 应该自动注入Database实例


# 挑战 3: 实现一个事件系统装饰器
# 要求：在函数调用前后触发事件
def event_handler(event_name):
    """TODO: 实现事件处理装饰器"""
    pass

@event_handler('user.login')
def login(username, password):
    return "Logged in"

# 测试: 应该在登录前后触发事件


# 挑战 4: 实现一个权限装饰器链
# 要求：支持多个权限检查，按顺序执行
def require_all(*permissions):
    """TODO: 实现多权限检查装饰器"""
    pass

@require_all('admin', 'write')
def delete_user(user_id):
    return f"User {user_id} deleted"

# 测试: 应该检查所有权限


# 挑战 5: 实现一个延迟执行装饰器
# 要求：延迟执行函数，可以取消执行
def delay(seconds):
    """TODO: 实现延迟执行装饰器"""
    pass

@delay(2)
def send_email(to, subject):
    print(f"Sending email to {to}")

# 测试: 应该在2秒后发送邮件，可以取消


if __name__ == "__main__":
    print("装饰器与元类练习题")
    print("请完成每个TODO部分的代码")
    print("运行测试验证你的实现")
