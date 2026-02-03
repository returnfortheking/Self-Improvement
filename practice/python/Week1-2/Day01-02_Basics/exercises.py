"""
Day 1-2: Python基础语法练习题
主题：变量、数据类型、控制流
难度：基础 → 进阶

运行方法：python exercises.py
"""

# ============================================================
# 基础题（⭐）
# ============================================================

def exercise_1_swap_variables():
    """
    题目：交换两个变量的值
    要求：不使用第三个变量

    示例：
        a = 5
        b = 10
        # 交换后
        a = 10
        b = 5
    """
    a = 5
    b = 10

    # TODO: 实现交换
    # 提示：Python 可以使用 a, b = b, a

    # 测试
    assert a == 10 and b == 5
    print("✓ 练习1 通过")


def exercise_2_fahrenheit_to_celsius():
    """
    题目：华氏温度转摄氏温度
    公式：C = (F - 32) * 5/9

    示例：
        fahrenheit_to_celsius(32) → 0.0
        fahrenheit_to_celsius(212) → 100.0
    """
    def fahrenheit_to_celsius(f):
        # TODO: 实现转换
        pass

    # 测试
    assert abs(fahrenheit_to_celsius(32) - 0.0) < 0.01
    assert abs(fahrenheit_to_celsius(212) - 100.0) < 0.01
    print("✓ 练习2 通过")


def exercise_3_check_palindrome():
    """
    题目：判断字符串是否为回文
    要求：忽略大小写和非字母字符

    示例：
        "A man, a plan, a canal: Panama" → True
        "race a car" → False
    """
    def is_palindrome(s):
        # TODO: 实现回文判断
        pass

    # 测试
    assert is_palindrome("A man, a plan, a canal: Panama") == True
    assert is_palindrome("race a car") == False
    print("✓ 练习3 通过")


def exercise_4_count_characters():
    """
    题目：统计字符串中各字符出现的次数
    要求：返回字典，键为字符，值为出现次数

    示例：
        "hello" → {'h': 1, 'e': 1, 'l': 2, 'o': 1}
    """
    def count_chars(s):
        # TODO: 实现字符统计
        pass

    # 测试
    result = count_chars("hello")
    assert result == {'h': 1, 'e': 1, 'l': 2, 'o': 1}
    print("✓ 练习4 通过")


def exercise_5_remove_duplicates():
    """
    题目：去除列表中的重复元素，保持顺序
    要求：不使用 set()

    示例：
        [1, 2, 2, 3, 1, 4] → [1, 2, 3, 4]
    """
    def remove_duplicates(lst):
        # TODO: 实现去重逻辑
        pass

    # 测试
    assert remove_duplicates([1, 2, 2, 3, 1, 4]) == [1, 2, 3, 4]
    print("✓ 练习5 通过")


def exercise_6_list_flatten():
    """
    题目：展平嵌套列表
    要求：将多层嵌套列表展平为一维列表

    示例：
        [1, [2, [3, 4], 5], 6] → [1, 2, 3, 4, 5, 6]
    """
    def flatten_list(nested_list):
        # TODO: 实现展平逻辑
        pass

    # 测试
    assert flatten_list([1, [2, [3, 4], 5], 6]) == [1, 2, 3, 4, 5, 6]
    print("✓ 练习6 通过")


def exercise_7_find_second_largest():
    """
    题目：找出列表中第二大的数
    要求：不使用排序

    示例：
        [1, 5, 2, 8, 3] → 5
    """
    def second_largest(lst):
        # TODO: 实现查找逻辑
        pass

    # 测试
    assert second_largest([1, 5, 2, 8, 3]) == 5
    print("✓ 练习7 通过")


def exercise_8_merge_dicts():
    """
    题目：合并两个字典
    要求：如果有相同的键，后面的字典覆盖前面的

    示例：
        {'a': 1, 'b': 2}, {'b': 3, 'c': 4} → {'a': 1, 'b': 3, 'c': 4}
    """
    def merge_dicts(dict1, dict2):
        # TODO: 实现合并逻辑
        pass

    # 测试
    result = merge_dicts({'a': 1, 'b': 2}, {'b': 3, 'c': 4})
    assert result == {'a': 1, 'b': 3, 'c': 4}
    print("✓ 练习8 通过")


def exercise_9_common_elements():
    """
    题目：找出两个列表的公共元素
    要求：结果中没有重复元素

    示例：
        [1, 2, 2, 3], [2, 3, 3, 4] → [2, 3]
    """
    def common_elements(list1, list2):
        # TODO: 实现查找逻辑
        pass

    # 测试
    assert set(common_elements([1, 2, 2, 3], [2, 3, 3, 4])) == {2, 3}
    print("✓ 练习9 通过")


def exercise_10_is_anagram():
    """
    题目：判断两个字符串是否为变位词
    变位词：包含相同的字母，顺序不同

    示例：
        "listen", "silent" → True
        "hello", "world" → False
    """
    def is_anagram(s1, s2):
        # TODO: 实现判断逻辑
        pass

    # 测试
    assert is_anagram("listen", "silent") == True
    assert is_anagram("hello", "world") == False
    print("✓ 练习10 通过")


# ============================================================
# 进阶题（⭐⭐）
# ============================================================

def exercise_11_deep_copy():
    """
    题目：实现深拷贝函数
    要求：
    1. 支持列表、字典
    2. 处理嵌套结构

    面试重点：理解深拷贝和浅拷贝的区别
    """
    def deep_copy(obj):
        # TODO: 实现深拷贝逻辑
        pass

    # 测试
    original = [1, 2, [3, 4]]
    copy = deep_copy(original)
    original[2].append(5)

    assert copy == [1, 2, [3, 4]]
    print("✓ 练习11 通过")


def exercise_12_group_anagrams():
    """
    题目：将变位词分组
    要求：给定字符串列表，将变位词分组

    示例：
        ["eat", "tea", "tan", "ate", "nat", "bat"]
        → [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]
    """
    def group_anagrams(words):
        # TODO: 实现分组逻辑
        pass

    # 测试
    result = group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
    assert len(result) == 3
    print("✓ 练习12 通过")


def exercise_13_two_sum():
    """
    题目：找出数组中两个数之和等于目标值
    要求：返回索引

    示例：
        [2, 7, 11, 15], 9 → [0, 1]
    """
    def two_sum(nums, target):
        # TODO: 实现查找逻辑
        pass

    # 测试
    assert two_sum([2, 7, 11, 15], 9) == [0, 1]
    print("✓ 练习13 通过")


def exercise_14_valid_parentheses():
    """
    题目：验证括号是否有效
    要求：括号必须正确闭合

    示例：
        "()" → True
        "()[]{}" → True
        "(]" → False
        "([)]" → False
    """
    def is_valid_parentheses(s):
        # TODO: 实现验证逻辑
        pass

    # 测试
    assert is_valid_parentheses("()") == True
    assert is_valid_parentheses("()[]{}") == True
    assert is_valid_parentheses("(]") == False
    assert is_valid_parentheses("([)]") == False
    print("✓ 练习14 通过")


def exercise_15_reverse_integer():
    """
    题目：反转整数
    要求：如果反转后溢出32位整数范围，返回0

    示例：
        123 → 321
        -123 → -321
        120 → 21
    """
    def reverse_integer(x):
        # TODO: 实现反转逻辑
        pass

    # 测试
    assert reverse_integer(123) == 321
    assert reverse_integer(-123) == -321
    assert reverse_integer(120) == 21
    print("✓ 练习15 通过")


# ============================================================
# 大厂面试真题（⭐⭐⭐）
# ============================================================

def exercise_16_is_vs_equals():
    """
    题目：解释以下代码的输出（字节跳动 2025 真题）

    频率：90% 面试遇到

    请解释：
        a = 256
        b = 256
        print(a is b)  # ?

        c = 257
        d = 257
        print(c is d)  # ?

    答案要求：
    1. 解释每行输出
    2. 解释原因（小整数缓存机制）
    3. 说明缓存范围是多少
    """
    # TODO: 写出你的答案和解释
    answer = """
    你的答案：
    """

    print(f"你的答案：\n{answer}")


def exercise_17_mutable_default_arg():
    """
    题目：解释以下代码的问题（阿里巴巴 2024 真题）

    频率：80% 面试遇到

    def append_to(element, target=[]):
        target.append(element)
        return target

    调用 3 次 append_to(1) 后，返回什么？

    要求：
    1. 解释为什么会这样
    2. 给出正确的实现方式
    3. 说明 Python 默认参数的创建时机
    """
    # TODO: 写出你的答案和解释
    answer = """
    你的答案：
    """

    print(f"你的答案：\n{answer}")


def exercise_18_reference_counting():
    """
    题目：引用计数分析（腾讯 2025 真题）

    频率：70% 面试遇到

    import sys
    a = [1, 2, 3]
    b = a
    c = b
    print(sys.getrefcount(a))

    请解释输出，并说明为什么不是 3。

    要求：
    1. 解释输出结果
    2. 说明为什么比预期多
    3. 如何正确查看引用计数
    """
    # TODO: 写出你的答案和解释
    answer = """
    你的答案：
    """

    print(f"你的答案：\n{answer}")


def exercise_19_list_copy_methods():
    """
    题目：列表复制方法对比（美团 2024 真题）

    频率：75% 面试遇到

    original = [1, 2, [3, 4]]
    copy1 = original.copy()
    copy2 = original[:]
    copy3 = list(original)
    copy4 = copy.deepcopy(original)

    original[2].append(5)

    哪些复制方式会修改嵌套列表？

    要求：
    1. 解释每种复制方式
    2. 说明哪些是浅拷贝，哪些是深拷贝
    3. 何时使用深拷贝
    """
    # TODO: 写出你的答案和解释
    answer = """
    你的答案：
    """

    print(f"你的答案：\n{answer}")


def exercise_20_memory_optimization():
    """
    题目：内存优化（拼多多 2025 真题）

    频率：65% 面试遇到

    # 创建包含 1000 万个整数的列表
    big_list = list(range(10000000))

    如何优化内存使用？

    要求：
    1. 分析问题
    2. 提供优化方案
    3. 对比内存占用
    """
    # TODO: 写出你的答案和解释
    answer = """
    你的答案：
    """

    print(f"你的答案：\n{answer}")


# ============================================================
# 主函数：运行所有练习
# ============================================================

def main():
    """运行所有练习"""
    print("=" * 60)
    print("Python 基础语法练习题")
    print("=" * 60)

    exercises = [
        ("基础题1: 交换变量", exercise_1_swap_variables),
        ("基础题2: 温度转换", exercise_2_fahrenheit_to_celsius),
        ("基础题3: 回文判断", exercise_3_check_palindrome),
        ("基础题4: 字符统计", exercise_4_count_characters),
        ("基础题5: 去重", exercise_5_remove_duplicates),
        ("基础题6: 展平列表", exercise_6_list_flatten),
        ("基础题7: 第二大数", exercise_7_find_second_largest),
        ("基础题8: 合并字典", exercise_8_merge_dicts),
        ("基础题9: 公共元素", exercise_9_common_elements),
        ("基础题10: 变位词", exercise_10_is_anagram),
        ("进阶题11: 深拷贝", exercise_11_deep_copy),
        ("进阶题12: 变位词分组", exercise_12_group_anagrams),
        ("进阶题13: 两数之和", exercise_13_two_sum),
        ("进阶题14: 有效括号", exercise_14_valid_parentheses),
        ("进阶题15: 反转整数", exercise_15_reverse_integer),
        ("真题16: is vs ==", exercise_16_is_vs_equals),
        ("真题17: 可变默认参数", exercise_17_mutable_default_arg),
        ("真题18: 引用计数", exercise_18_reference_counting),
        ("真题19: 列表复制", exercise_19_list_copy_methods),
        ("真题20: 内存优化", exercise_20_memory_optimization),
    ]

    completed = 0
    failed = 0

    for title, func in exercises:
        print("\n" + "-" * 60)
        print(title)
        print("-" * 60)
        try:
            func()
            completed += 1
        except NotImplementedError:
            print("⚠ 待实现")
        except AssertionError as e:
            print(f"✗ 测试失败: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ 错误: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"完成：{completed}/{len(exercises)}")
    print(f"失败：{failed}/{len(exercises)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
