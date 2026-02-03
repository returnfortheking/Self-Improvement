"""
Day 1-2: Python基础语法示例代码
运行方法：python examples.py
"""

# ============================================================
# 示例 1-5：变量与对象模型
# ============================================================

def example_1_everything_is_object():
    """演示：Python 中一切都是对象"""
    print("=== 一切皆对象 ===")
    # 数字是对象
    num = 10
    print(f"数字类型: {type(num)}")
    print(f"数字方法: {num.bit_length()}")

    # 字符串是对象
    text = "hello"
    print(f"字符串类型: {type(text)}")
    print(f"字符串方法: {text.upper()}")

    # 函数也是对象
    def my_func():
        pass
    print(f"函数类型: {type(my_func)}")

    # 类也是对象
    print(f"int 类型: {type(int)}")


def example_2_variables_are_references():
    """演示：变量是引用"""
    print("\n=== 变量是引用 ===")
    a = [1, 2, 3]
    b = a  # b 和 a 指向同一个对象
    b.append(4)

    print(f"a: {a}")  # [1, 2, 3, 4]
    print(f"b: {b}")  # [1, 2, 3, 4]
    print(f"a is b: {a is b}")  # True


def example_3_mutable_vs_immutable():
    """演示：可变 vs 不可变对象"""
    print("\n=== 可变 vs 不可变 ===")
    # 可变对象：列表
    my_list = [1, 2, 3]
    list_id = id(my_list)
    my_list.append(4)
    print(f"列表修改后 ID 相同: {id(my_list) == list_id}")

    # 不可变对象：元组
    my_tuple = (1, 2, 3)
    tuple_id = id(my_tuple)
    my_tuple += (4,)
    print(f"元组修改后 ID 不同: {id(my_tuple) != tuple_id}")


def example_4_is_vs_equals():
    """演示：is vs == 的区别"""
    print("\n=== is vs == ===")
    # 小整数缓存
    a = 256
    b = 256
    print(f"256 is 256: {a is b}")  # True

    c = 257
    d = 257
    print(f"257 is 257: {c is d}")  # False
    print(f"257 == 257: {c == d}")  # True

    # 列表比较
    list1 = [1, 2, 3]
    list2 = [1, 2, 3]
    print(f"[1,2,3] == [1,2,3]: {list1 == list2}")  # True
    print(f"[1,2,3] is [1,2,3]: {list1 is list2}")  # False


def example_5_reference_counting():
    """演示：引用计数"""
    print("\n=== 引用计数 ===")
    import sys

    a = [1, 2, 3]
    print(f"初始引用计数: {sys.getrefcount(a)}")

    b = a
    print(f"b = a 后: {sys.getrefcount(a)}")

    del b
    print(f"del b 后: {sys.getrefcount(a)}")


# ============================================================
# 示例 6-10：基本数据类型
# ============================================================

def example_6_numbers():
    """演示：数字类型"""
    print("\n=== 数字类型 ===")
    # 整数
    x = 10
    y = 0b1010  # 二进制
    z = 0o12    # 八进制
    w = 0xa     # 十六进制
    print(f"整数的各种进制: {x}, {y}, {z}, {w}")

    # 浮点数
    pi = 3.14159
    scientific = 1.23e-4
    print(f"浮点数: {pi}, 科学计数法: {scientific}")

    # 复数
    c = 3 + 4j
    print(f"复数: {c}, 实部: {c.real}, 虚部: {c.imag}")

    # 布尔
    print(f"True + True: {True + True}")
    print(f"False * 10: {False * 10}")


def example_7_strings():
    """演示：字符串操作"""
    print("\n=== 字符串操作 ===")
    s = "Python Programming"

    # 索引和切片
    print(f"s[0]: {s[0]}")
    print(f"s[-1]: {s[-1]}")
    print(f"s[0:6]: {s[0:6]}")
    print(f"s[7:]: {s[7:]}")
    print(f"s[::-1]: {s[::-1]}")

    # 方法
    print(f"s.lower(): {s.lower()}")
    print(f"s.upper(): {s.upper()}")
    print(f"s.split(): {s.split()}")
    print(f"'-'.join(['a', 'b', 'c']): {'-'.join(['a', 'b', 'c'])}")


def example_8_lists():
    """演示：列表操作"""
    print("\n=== 列表操作 ===")
    lst = [1, 2, 3, 4, 5]

    # 基本操作
    lst.append(6)
    lst.insert(0, 0)
    lst.extend([7, 8])
    print(f"列表操作: {lst}")

    # 切片
    lst = [0, 1, 2, 3, 4, 5]
    print(f"lst[1:4]: {lst[1:4]}")
    print(f"lst[::2]: {lst[::2]}")
    print(f"lst[::-1]: {lst[::-1]}")

    # 列表推导式
    squares = [x**2 for x in range(10)]
    evens = [x for x in range(20) if x % 2 == 0]
    print(f"平方: {squares}")
    print(f"偶数: {evens}")


def example_9_tuples():
    """演示：元组操作"""
    print("\n=== 元组操作 ===")
    t = (1, 2, 3)

    # 索引和切片
    print(f"t[0]: {t[0]}")
    print(f"t[1:3]: {t[1:3]}")
    print(f"t * 2: {t * 2}")

    # 解包
    a, b, c = t
    print(f"解包: a={a}, b={b}, c={c}")


def example_10_sets_and_dicts():
    """演示：集合和字典"""
    print("\n=== 集合和字典 ===")

    # 集合
    s1 = {1, 2, 3}
    s2 = {3, 4, 5}
    print(f"并集: {s1 | s2}")
    print(f"交集: {s1 & s2}")
    print(f"差集: {s1 - s2}")

    # 字典
    d = {'name': 'Alice', 'age': 30}
    d['city'] = 'Beijing'
    print(f"字典: {d}")
    print(f"键: {list(d.keys())}")
    print(f"值: {list(d.values())}")

    # 字典推导式
    squares = {x: x**2 for x in range(6)}
    print(f"字典推导式: {squares}")


# ============================================================
# 示例 11-15：控制流
# ============================================================

def example_11_conditionals():
    """演示：条件语句"""
    print("\n=== 条件语句 ===")
    x = 10

    # if-else
    if x > 0:
        print("x 是正数")
    elif x < 0:
        print("x 是负数")
    else:
        print("x 是零")

    # 三元表达式
    result = "Positive" if x > 0 else "Non-positive"
    print(f"三元表达式: {result}")

    # 逻辑运算
    if x > 0 and x < 100:
        print("x 在 0 和 100 之间")


def example_12_loops():
    """演示：循环"""
    print("\n=== 循环 ===")

    # for 循环
    print("for 循环:")
    for i in range(5):
        print(f"  {i}")

    # enumerate
    print("enumerate:")
    for i, value in enumerate(['a', 'b', 'c']):
        print(f"  {i}: {value}")

    # 字典遍历
    print("字典遍历:")
    for key, value in {'a': 1, 'b': 2}.items():
        print(f"  {key}: {value}")

    # while 循环
    print("while 循环:")
    count = 0
    while count < 3:
        print(f"  {count}")
        count += 1


def example_13_loop_control():
    """演示：循环控制"""
    print("\n=== 循环控制 ===")

    # continue 和 break
    for i in range(10):
        if i == 3:
            continue
        if i == 7:
            break
        print(f"  {i}")

    # else 子句
    for i in range(3):
        print(f"  {i}")
    else:
        print("循环正常结束")


def example_14_exceptions():
    """演示：异常处理"""
    print("\n=== 异常处理 ===")

    # 基本异常处理
    try:
        result = 10 / 0
    except ZeroDivisionError:
        print("捕获到除零错误")
    except Exception as e:
        print(f"其他错误: {e}")
    else:
        print("没有错误")
    finally:
        print("总是执行")

    # 抛出异常
    def divide(a, b):
        if b == 0:
            raise ValueError("不能除以零")
        return a / b


def example_15_lambda():
    """演示：lambda 函数"""
    print("\n=== lambda 函数 ===")

    # 基本 lambda
    add = lambda x, y: x + y
    print(f"add(3, 5): {add(3, 5)}")

    # 与高阶函数配合
    numbers = [1, 2, 3, 4, 5]
    squared = list(map(lambda x: x**2, numbers))
    evens = list(filter(lambda x: x % 2 == 0, numbers))
    print(f"平方: {squared}")
    print(f"偶数: {evens}")


# ============================================================
# 示例 16-20：高级操作
# ============================================================

def example_16_list_comprehensions():
    """演示：列表推导式高级用法"""
    print("\n=== 列表推导式高级 ===")

    # 嵌套列表推导式
    matrix = [[i*j for j in range(3)] for i in range(3)]
    print(f"矩阵: {matrix}")

    # 带条件的推导式
    even_squares = [x**2 for x in range(10) if x % 2 == 0]
    print(f"偶数的平方: {even_squares}")


def example_17_dict_methods():
    """演示：字典方法"""
    print("\n=== 字典方法 ===")

    d = {'a': 1, 'b': 2, 'c': 3}

    # get 方法
    print(f"d.get('a'): {d.get('a')}")
    print(f"d.get('x', 0): {d.get('x', 0)}")

    # pop 方法
    value = d.pop('a')
    print(f"弹出的值: {value}")
    print(f"弹出后: {d}")

    # update 方法
    d.update({'d': 4, 'e': 5})
    print(f"更新后: {d}")


def example_18_string_methods():
    """演示：字符串方法"""
    print("\n=== 字符串方法 ===")

    s = "  Hello, World!  "

    # 去除空格
    print(f"strip(): '{s.strip()}'")

    # 查找和替换
    print(f"find('World'): {s.find('World')}")
    print(f"replace('World', 'Python'): {s.replace('World', 'Python')}")

    # 大小写
    print(f"upper(): {s.upper()}")
    print(f"lower(): {s.lower()}")
    print(f"title(): {'hello world'.title()}")


def example_19_set_operations():
    """演示：集合操作"""
    print("\n=== 集合操作 ===")

    s1 = {1, 2, 3, 4}
    s2 = {3, 4, 5, 6}

    print(f"并集: {s1 | s2}")
    print(f"交集: {s1 & s2}")
    print(f"差集: {s1 - s2}")
    print(f"对称差: {s1 ^ s2}")

    # 集合方法
    s = set()
    s.add(1)
    s.add(2)
    s.add(3)
    print(f"集合: {s}")
    print(f"长度: {len(s)}")
    print(f"是否包含 2: {2 in s}")


def example_20_type_conversion():
    """演示：类型转换"""
    print("\n=== 类型转换 ===")

    # 数字转换
    print(f"int('123'): {int('123')}")
    print(f"float('3.14'): {float('3.14')}")
    print(f"str(123): {str(123)}")

    # 序列转换
    print(f"list('abc'): {list('abc')}")
    print(f"tuple([1, 2, 3]): {tuple([1, 2, 3])}")
    print(f"set([1, 2, 2, 3]): {set([1, 2, 2, 3])}")

    # 字典转换
    print(f"dict([('a', 1), ('b', 2)]): {dict([('a', 1), ('b', 2)])}")


# ============================================================
# 主函数
# ============================================================

def main():
    """运行所有示例"""
    print("=" * 60)
    print("Python 基础语法示例")
    print("=" * 60)

    examples = [
        ("1. 一切皆对象", example_1_everything_is_object),
        ("2. 变量是引用", example_2_variables_are_references),
        ("3. 可变 vs 不可变", example_3_mutable_vs_immutable),
        ("4. is vs ==", example_4_is_vs_equals),
        ("5. 引用计数", example_5_reference_counting),
        ("6. 数字类型", example_6_numbers),
        ("7. 字符串操作", example_7_strings),
        ("8. 列表操作", example_8_lists),
        ("9. 元组操作", example_9_tuples),
        ("10. 集合和字典", example_10_sets_and_dicts),
        ("11. 条件语句", example_11_conditionals),
        ("12. 循环", example_12_loops),
        ("13. 循环控制", example_13_loop_control),
        ("14. 异常处理", example_14_exceptions),
        ("15. lambda 函数", example_15_lambda),
        ("16. 列表推导式高级", example_16_list_comprehensions),
        ("17. 字典方法", example_17_dict_methods),
        ("18. 字符串方法", example_18_string_methods),
        ("19. 集合操作", example_19_set_operations),
        ("20. 类型转换", example_20_type_conversion),
    ]

    for title, func in examples:
        print("-" * 60)
        func()

    print("\n" + "=" * 60)
    print("所有示例运行完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
