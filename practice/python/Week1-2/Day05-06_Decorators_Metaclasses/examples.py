# 装饰器与元类代码示例

import functools
import time
from typing import Callable, Any

# ==================== 1. 基础装饰器 ====================

def simple_decorator(func):
    """最简单的装饰器"""
    def wrapper():
        print("Before function call")
        func()
        print("After function call")
    return wrapper

@simple_decorator
def greet():
    print("Hello, World!")

# greet()  # Before function call -> Hello, World! -> After function call

# ==================== 2. 保留函数元信息 ====================

def preserve_info_decorator(func):
    """使用functools.wraps保留原函数信息"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@preserve_info_decorator
def add(a, b):
    """Add two numbers"""
    return a + b

# print(add.__name__)  # add (而不是wrapper)
# print(add.__doc__)   # Add two numbers

# ==================== 3. 计时装饰器 ====================

def timer(func):
    """测量函数执行时间"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(0.1)
    return "Done"

# slow_function()  # slow_function took 0.1001 seconds

# ==================== 4. 调试装饰器 ====================

def debug(func):
    """打印函数调用信息"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        print(f"Calling {func.__name__}({signature})")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned {result!r}")
        return result
    return wrapper

@debug
def calculate(x, y=10):
    return x * y

# calculate(5, y=2)
# Calling calculate(5, y=2)
# calculate returned 10

# ==================== 5. 重试装饰器 ====================

def retry(max_attempts=3, delay=1):
    """失败时自动重试"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        raise
                    print(f"Attempt {attempts} failed, retrying...")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.1)
def unreliable_function():
    import random
    if random.random() < 0.7:
        raise ValueError("Random failure")
    return "Success"

# unreliable_function()

# ==================== 6. 缓存装饰器 ====================

def cache(func):
    """简单的缓存装饰器"""
    cache_dict = {}

    @functools.wraps(func)
    def wrapper(*args):
        if args not in cache_dict:
            cache_dict[args] = func(*args)
        return cache_dict[args]
    return wrapper

@cache
def fibonacci(n):
    """计算斐波那契数列（带缓存）"""
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# print(fibonacci(100))  # 快速计算

# ==================== 7. 单例装饰器 ====================

def singleton(cls):
    """单例模式装饰器"""
    instances = {}

    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class Database:
    def __init__(self):
        print("Connecting to database...")

# db1 = Database()
# db2 = Database()
# print(db1 is db2)  # True

# ==================== 8. 验证装饰器 ====================

def validate_types(**type_map):
    """验证参数类型"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 检查参数类型
            for param_name, expected_type in type_map.items():
                if param_name in kwargs:
                    value = kwargs[param_name]
                    if not isinstance(value, expected_type):
                        raise TypeError(
                            f"{param_name} must be {expected_type}, "
                            f"got {type(value)}"
                        )
            return func(*args, **kwargs)
        return wrapper
    return decorator

@validate_types(x=int, y=str)
def process_data(x, y):
    return f"{x}: {y}"

# process_data(x=10, y="hello")  # OK
# process_data(x="10", y=20)    # TypeError

# ==================== 9. 权限检查装饰器 ====================

def require_permission(permission):
    """检查用户权限"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(user, *args, **kwargs):
            if permission not in user.get('permissions', []):
                raise PermissionError(
                    f"User {user.get('name')} lacks {permission} permission"
                )
            return func(user, *args, **kwargs)
        return wrapper
    return decorator

@require_permission('admin')
def delete_user(user, user_id):
    return f"User {user_id} deleted"

# admin = {'name': 'Alice', 'permissions': ['admin']}
# delete_user(admin, 123)  # OK

# guest = {'name': 'Bob', 'permissions': ['read']}
# delete_user(guest, 123)  # PermissionError

# ==================== 10. 类装饰器 ====================

class CountCalls:
    """统计函数调用次数"""
    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"{self.func.__name__} called {self.count} times")
        return self.func(*args, **kwargs)

@CountCalls
def say_hello():
    print("Hello!")

# say_hello()
# say_hello()
# say_hello()  # called 3 times

# ==================== 11. 带参数的类装饰器 ====================

class Repeat:
    """重复执行函数"""
    def __init__(self, times):
        self.times = times

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            results = []
            for _ in range(self.times):
                results.append(func(*args, **kwargs))
            return results
        return wrapper

@Repeat(3)
def greet(name):
    return f"Hello, {name}!"

# print(greet("Alice"))  # ['Hello, Alice!', 'Hello, Alice!', 'Hello, Alice!']

# ==================== 12. 元类基础 ====================

class Meta(type):
    """简单的元类"""
    def __new__(cls, name, bases, namespace):
        print(f"Creating class: {name}")
        return super().__new__(cls, name, bases, namespace)

class MyClass(metaclass=Meta):
    pass
# 输出: Creating class: MyClass

# ==================== 13. 单例元类 ====================

class SingletonMeta(type):
    """单例元类"""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=SingletonMeta):
    def __init__(self):
        print("Initializing database")

# db1 = Database()
# db2 = Database()
# print(db1 is db2)  # True

# ==================== 14. 自动注册元类 ====================

class RegisteredMeta(type):
    """自动注册类到注册表"""
    registry = {}

    def __new__(cls, name, bases, namespace):
        new_cls = super().__new__(cls, name, bases, namespace)
        cls.registry[name] = new_cls
        return new_cls

class Base(metaclass=RegisteredMeta):
    pass

class User(Base):
    pass

class Product(Base):
    pass

# print(RegisteredMeta.registry)  # {'Base': Base, 'User': User, 'Product': Product}

# ==================== 15. 强制实现方法元类 ====================

class MustImplementMeta(type):
    """强制子类实现特定方法"""
    def __new__(cls, name, bases, namespace):
        if name != 'Base':
            required_methods = getattr(bases[0], '_required_methods', [])
            for method in required_methods:
                if method not in namespace:
                    raise NotImplementedError(
                        f"{name} must implement {method} method"
                    )
        return super().__new__(cls, name, bases, namespace)

class Base(metaclass=MustImplementMeta):
    _required_methods = ['save', 'delete']

    def save(self):
        raise NotImplementedError

    def delete(self):
        raise NotImplementedError

class User(Base):
    def save(self):
        print("Saving user...")

    def delete(self):
        print("Deleting user...")

# user = User()  # OK

# ==================== 16. 属性验证元类 ====================

class ValidatedMeta(type):
    """自动验证类属性"""
    def __new__(cls, name, bases, namespace):
        # 为所有属性添加验证
        for key, value in namespace.items():
            if not key.startswith('_') and not callable(value):
                private_key = f'_{key}'
                if private_key not in namespace:
                    namespace[private_key] = value

                    def make_property(key):
                        def getter(self):
                            return getattr(self, f'_{key}')
                        def setter(self, value):
                            setattr(self, f'_{key}', value)
                        return property(getter, setter)

                    namespace[key] = make_property(key)

        return super().__new__(cls, name, bases, namespace)

class Person(metaclass=ValidatedMeta):
    name = "Unknown"
    age = 0

# p = Person()
# p.name = "Alice"
# p.age = 30
# print(p.name, p.age)

# ==================== 17. 装饰器堆叠 ====================

@timer
@debug
@cache
def expensive_operation(n):
    """昂贵的操作"""
    time.sleep(0.01)
    return n ** 2

# expensive_operation(5)
# 执行顺序从下到上: cache -> debug -> timer

# ==================== 18. 可选参数装饰器 ====================

def smart_decorator(func=None, *, option="default"):
    """可以带参数或不带参数的装饰器"""
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            print(f"Option: {option}")
            return f(*args, **kwargs)
        return wrapper

    if func is None:
        # 带参数调用: @smart_decorator(option="value")
        return decorator
    else:
        # 不带参数调用: @smart_decorator
        return decorator(func)

@smart_decorator
def func1():
    pass

@smart_decorator(option="custom")
def func2():
    pass

# ==================== 19. 上下文管理装饰器 ====================

def context_manager(func):
    """将函数转换为上下文管理器"""
    class ContextManager:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def __enter__(self):
            print("Entering context")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            print("Exiting context")
            return False

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return ContextManager(*args, **kwargs)

    return wrapper

@context_manager
def my_context():
    pass

# ==================== 20. 异步函数装饰器 ====================

def async_timer(func):
    """异步函数计时装饰器"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        import asyncio
        start = time.time()
        result = await func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

# @async_timer
# async def async_operation():
#     import asyncio
#     await asyncio.sleep(0.1)
#     return "Done"
