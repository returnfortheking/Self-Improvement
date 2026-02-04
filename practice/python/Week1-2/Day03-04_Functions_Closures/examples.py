"""
Python 函数与闭包 - 代码示例
涵盖：函数参数传递、闭包、作用域、lambda、装饰器基础
"""


# ========== 1. 函数参数传递机制 ==========

def demonstrate_parameter_passing():
    """演示 Python 的参数传递机制（对象引用传递）"""

    # 可变对象作为参数
    def modify_list(lst):
        lst.append(4)
        print(f"函数内: {lst}")

    my_list = [1, 2, 3]
    modify_list(my_list)
    print(f"函数外: {my_list}")  # [1, 2, 3, 4] - 原列表被修改！

    # 不可变对象作为参数
    def modify_int(n):
        n = n + 1
        print(f"函数内: {n}")

    x = 10
    modify_int(x)
    print(f"函数外: {x}")  # 10 - 原值不变


# ========== 2. 默认参数的陷阱 ==========

def demonstrate_default_args():
    """演示可变默认参数的问题"""

    # ❌ 错误示例：使用可变对象作为默认参数
    def append_bad(item, lst=[]):
        lst.append(item)
        return lst

    print(append_bad(1))  # [1]
    print(append_bad(2))  # [1, 2] - 默认列表被累积修改！
    print(append_bad(3))  # [1, 2, 3]

    # ✅ 正确示例：使用 None 作为默认值
    def append_good(item, lst=None):
        if lst is None:
            lst = []
        lst.append(item)
        return lst

    print(append_good(1))  # [1]
    print(append_good(2))  # [2] - 每次都是新列表


# ========== 3. 闭包（Closure）==========

def demonstrate_closures():
    """演示闭包的概念和应用"""

    # 闭包基础：内部函数引用外部函数的变量
    def outer_func(x):
        def inner_func(y):
            return x + y  # 访问外部变量 x
        return inner_func

    add_5 = outer_func(5)
    print(add_5(3))  # 8
    print(add_5(10))  # 15

    # 闭包陷阱：循环中的延迟绑定
    def create_multipliers_bad():
        multipliers = []
        for i in range(3):
            def multiplier(x):
                return x * i  # i 会被绑定到最后的值 2
            multipliers.append(multiplier)
        return multipliers

    mul_bad = create_multipliers_bad()
    print([m(10) for m in mul_bad])  # [20, 20, 20] - 全是 10*2！

    # ✅ 使用默认参数解决
    def create_multipliers_good():
        multipliers = []
        for i in range(3):
            def multiplier(x, factor=i):  # 立即绑定 i 的值
                return x * factor
            multipliers.append(multiplier)
        return multipliers

    mul_good = create_multipliers_good()
    print([m(10) for m in mul_good])  # [0, 10, 20] - 正确！


# ========== 4. 闭包的应用：函数工厂 ==========

def create_counter():
    """使用闭包创建计数器"""
    count = 0

    def increment():
        nonlocal count  # 声明使用外部变量
        count += 1
        return count

    return increment


counter1 = create_counter()
print(counter1())  # 1
print(counter1())  # 2
print(counter1())  # 3

counter2 = create_counter()
print(counter2())  # 1 - 独立的计数器


# ========== 5. 作用域规则（LEGB）==========

def demonstrate_scope():
    """演示 Python 的作用域查找顺序：Local -> Enclosing -> Global -> Built-in"""

    global_var = "global"

    def outer_function():
        enclosing_var = "enclosing"

        def inner_function():
            local_var = "local"

            # LEGB 查找顺序
            print(local_var)      # Local
            print(enclosing_var)  # Enclosing
            print(global_var)     # Global
            # print(undefined_var)  # 会报错：NameError

        inner_function()

    outer_function()


# ========== 6. global 和 nonlocal 关键字 ==========

def demonstrate_global_nonlocal():
    """演示 global 和 nonlocal 的使用"""

    # global：修改全局变量
    count = 0

    def increment_global():
        global count
        count += 1  # 不加 global 会报 UnboundLocalError

    increment_global()
    print(count)  # 1

    # nonlocal：修改外层（非全局）变量
    def outer():
        total = 0

        def add():
            nonlocal total
            total += 1  # 不加 nonlocal 会报 UnboundLocalError
            return total

        return add

    counter = outer()
    print(counter())  # 1
    print(counter())  # 2


# ========== 7. Lambda 函数 ==========

def demonstrate_lambda():
    """演示 lambda 函数的使用"""

    # 基础语法
    square = lambda x: x ** 2
    print(square(5))  # 25

    # 多参数
    add = lambda x, y: x + y
    print(add(3, 4))  # 7

    # 与常用函数结合
    numbers = [1, 2, 3, 4, 5]
    squared = list(map(lambda x: x ** 2, numbers))
    print(squared)  # [1, 4, 9, 16, 25]

    evens = list(filter(lambda x: x % 2 == 0, numbers))
    print(evens)  # [2, 4]

    # 排序
    pairs = [(1, 'one'), (3, 'three'), (2, 'two'), (4, 'four')]
    pairs.sort(key=lambda x: len(x[1]))  # 按字符串长度排序
    print(pairs)  # [(1, 'one'), (3, 'three'), (2, 'two'), (4, 'four')]


# ========== 8. 高阶函数 ==========

def demonstrate_higher_order_functions():
    """演示高阶函数：接受或返回函数的函数"""

    # 接受函数作为参数
    def apply_operation(x, y, operation):
        return operation(x, y)

    print(apply_operation(10, 5, lambda a, b: a + b))  # 15
    print(apply_operation(10, 5, lambda a, b: a * b))  # 50

    # 返回函数
    def get_power_function(exponent):
        def power(base):
            return base ** exponent
        return power

    square = get_power_function(2)
    cube = get_power_function(3)

    print(square(4))  # 16
    print(cube(4))    # 64


# ========== 9. 函数装饰器（基础）==========

def simple_decorator(func):
    """最简单的装饰器"""
    def wrapper():
        print("装饰器：调用前")
        result = func()
        print("装饰器：调用后")
        return result
    return wrapper


@simple_decorator
def say_hello():
    print("Hello!")


# say_hello() 会输出：
# 装饰器：调用前
# Hello!
# 装饰器：调用后


# ========== 10. 带参数的装饰器 ==========

def repeat(times):
    """重复执行函数多次的装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            results = []
            for _ in range(times):
                result = func(*args, **kwargs)
                results.append(result)
            return results
        return wrapper
    return decorator


@repeat(3)
def greet(name):
    return f"Hello, {name}!"


# greet("Alice") 会返回 ['Hello, Alice!', 'Hello, Alice!', 'Hello, Alice!']


# ========== 11. functools.wraps 保留元数据 ==========

from functools import wraps


def logged(func):
    """保留原函数元数据的装饰器"""
    @wraps(func)  # 保留 __name__, __doc__ 等属性
    def wrapper(*args, **kwargs):
        print(f"调用 {func.__name__}")
        return func(*args, **kwargs)
    return wrapper


@logged
def calculate(x, y):
    """计算两个数的和"""
    return x + y


# calculate.__name__ 会返回 'calculate' 而不是 'wrapper'


# ========== 12. 实际应用：缓存装饰器 ==========

from functools import lru_cache


@lru_cache(maxsize=128)
def fibonacci(n):
    """带缓存的斐波那契数列"""
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)


# fibonacci(50) 会瞬间返回，因为有缓存


# ========== 13. 实际应用：计时装饰器 ==========

import time


def timer(func):
    """计算函数执行时间的装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} 执行时间: {end - start:.4f}秒")
        return result
    return wrapper


@timer
def slow_function():
    time.sleep(1)
    return "完成"


# ========== 14. 函数注解（Type Hints）==========

def demonstrate_annotations():
    """演示函数注解的使用"""

    def greet(name: str, age: int = 18) -> str:
        """带类型注解的函数"""
        return f"你好，我是 {name}，今年 {age} 岁"

    print(greet("张三"))
    print(greet("李四", 25))

    # 访问注解
    print(greet.__annotations__)  # {'name': <class 'str'>, 'age': <class 'int'>, 'return': <class 'str'>}


# ========== 主程序 ==========

if __name__ == "__main__":
    print("=" * 60)
    print("1. 参数传递机制")
    print("=" * 60)
    demonstrate_parameter_passing()

    print("\n" + "=" * 60)
    print("2. 默认参数陷阱")
    print("=" * 60)
    demonstrate_default_args()

    print("\n" + "=" * 60)
    print("3. 闭包基础")
    print("=" * 60)
    demonstrate_closures()

    print("\n" + "=" * 60)
    print("4. 作用域规则")
    print("=" * 60)
    demonstrate_scope()

    print("\n" + "=" * 60)
    print("5. global 和 nonlocal")
    print("=" * 60)
    demonstrate_global_nonlocal()

    print("\n" + "=" * 60)
    print("6. Lambda 函数")
    print("=" * 60)
    demonstrate_lambda()

    print("\n" + "=" * 60)
    print("7. 高阶函数")
    print("=" * 60)
    demonstrate_higher_order_functions()

    print("\n" + "=" * 60)
    print("8. 函数注解")
    print("=" * 60)
    demonstrate_annotations()
