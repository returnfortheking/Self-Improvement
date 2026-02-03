# 生成器与迭代器练习题

from typing import Generator, Iterator, Iterable
import itertools

# ==================== 练习题 1-5: 基础迭代器 ====================

# 练习 1: 实现一个Range类
# 要求：模仿内置range()实现自定义Range类
class Range:
    """TODO: 实现自定义Range类"""
    pass

# 测试: for i in Range(0, 10, 2): print(i)


# 练习 2: 实现斐波那契数列迭代器
# 要求：使用迭代器实现斐波那契数列
class FibonacciIterator:
    """TODO: 实现斐波那契迭代器"""
    pass

# 测试: fib = FibonacciIterator(10); print(list(fib))


# 练习 3: 创建一个倒计时迭代器
# 要求：从start倒数到0
class CountDown:
    """TODO: 实现倒计时迭代器"""
    pass

# 测试: for i in CountDown(5): print(i)


# 练习 4: 实现一个循环迭代器
# 要求：无限循环迭代给定列表
class CycleIterator:
    """TODO: 实现循环迭代器"""
    pass

# 测试: cyc = CycleIterator([1, 2, 3]); print([next(cyc) for _ in range(10)])


# 练习 5: 实现一个跳过重复元素的迭代器
# 要求：跳过连续重复的元素
class UniqueIterator:
    """TODO: 实现去重迭代器"""
    pass

# 测试: list(UniqueIterator([1, 1, 2, 2, 2, 3, 3])) == [1, 2, 3]


# ==================== 练习题 6-10: 基础生成器 ====================

# 练习 6: 创建平方数生成器
# 要求：生成0到n的平方数
def squares(n: int) -> Generator[int, None, None]:
    """TODO: 实现平方数生成器"""
    pass

# 测试: print(list(squares(5)))  # [0, 1, 4, 9, 16]


# 练习 7: 创建素数生成器
# 要求：生成无限素数序列
def primes() -> Generator[int, None, None]:
    """TODO: 实现素数生成器"""
    pass

# 测试: gen = primes(); print([next(gen) for _ in range(10)])


# 练习 8: 创建随机数生成器
# 要求：生成指定范围内的随机数
import random

def random_numbers(count: int, min_val: int, max_val: int) -> Generator[int, None, None]:
    """TODO: 实现随机数生成器"""
    pass

# 测试: print(list(random_numbers(5, 1, 100)))


# 练习 9: 创建文件读取生成器
# 要求：逐行读取文件
def read_file_lines(filename: str) -> Generator[str, None, None]:
    """TODO: 实现文件读取生成器"""
    pass

# 测试: for line in read_file_lines('test.txt'): print(line)


# 练习 10: 创建笛卡尔积生成器
# 要求：生成两个列表的笛卡尔积
def cartesian_product(list1: list, list2: list) -> Generator[tuple, None, None]:
    """TODO: 实现笛卡尔积生成器"""
    pass

# 测试: print(list(cartesian_product([1, 2], ['a', 'b'])))


# ==================== 练习题 11-15: 高级生成器 ====================

# 练习 11: 使用yield from
# 要求：使用yield from实现嵌套列表扁平化
def flatten(nested_list: list) -> Generator:
    """TODO: 实现扁平化生成器"""
    pass

# 测试: print(list(flatten([1, [2, [3, 4], 5], 6])))


# 练习 12: 实现生成器的send方法
# 要求：创建一个可接收值的生成器
def accumulator() -> Generator[int, int, None]:
    """TODO: 实现累加器生成器"""
    pass

# 测试: gen = accumulator(); next(gen); print(gen.send(10)); print(gen.send(20))


# 练习 13: 实现管道处理
# 要求：创建filter、map、reduce管道
def pipeline(data, *filters):
    """TODO: 实现管道处理"""
    pass

# 测试: result = pipeline(range(10), lambda x: x > 5, lambda x: x * 2)


# 练习 14: 创建批处理生成器
# 要求：将数据分成固定大小的批次
def batch(data: Iterable, batch_size: int):
    """TODO: 实现批处理生成器"""
    pass

# 测试: for b in batch(range(100), 10): print(b)


# 练习 15: 实现滑动窗口生成器
# 要求：生成指定大小的滑动窗口
def sliding_window(data: Iterable, window_size: int):
    """TODO: 实现滑动窗口生成器"""
    pass

# 测试: print(list(sliding_window([1, 2, 3, 4, 5], 3)))


# ==================== 练习题 16-20: itertools应用 ====================

# 练习 16: 使用itertools.chain
# 要求：连接多个迭代器
def chain_iterables(*iterables):
    """TODO: 实现chain功能"""
    pass

# 测试: print(list(chain_iterables([1, 2], [3, 4], [5, 6])))


# 练习 17: 使用itertools.combinations
# 要求：生成所有组合
def all_combinations(iterable, r):
    """TODO: 实现组合生成器"""
    pass

# 测试: print(list(all_combinations([1, 2, 3], 2)))


# 练习 18: 使用itertools.groupby
# 要求：按key分组
def group_by_key(iterable, key_func):
    """TODO: 实现分组功能"""
    pass

# 测试: data = [('a', 1), ('b', 2), ('a', 3)]; print(dict(group_by_key(data, lambda x: x[0])))


# 练习 19: 使用itertools.islice
# 要求：对生成器进行切片
def slice_iterable(iterable, start=None, stop=None, step=None):
    """TODO: 实现切片功能"""
    pass

# 测试: gen = (x for x in range(100)); print(list(slice_iterable(gen, 5, 10)))


# 练习 20: 使用itertools.tee
# 要求：复制迭代器
def duplicate_iterator(iterator, n):
    """TODO: 实现迭代器复制"""
    pass

# 测试: gen = (x for x in range(5)); it1, it2 = duplicate_iterator(gen, 2)


# ==================== 练习题 21-25: 实际应用 ====================

# 练习 21: 实现树的遍历生成器
# 要求：中序遍历二叉树
class TreeNode:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def inorder_traversal(root: TreeNode) -> Generator:
    """TODO: 实现中序遍历"""
    pass

# 测试: root = TreeNode(1, TreeNode(2), TreeNode(3)); print(list(inorder_traversal(root)))


# 练习 22: 实现CSV读取生成器
# 要求：逐行解析CSV
def read_csv(filename: str) -> Generator[dict, None, None]:
    """TODO: 实现CSV读取"""
    pass

# 测试: for row in read_csv('data.csv'): print(row)


# 练习 23: 实现进度报告生成器
# 要求：在处理过程中报告进度
def progress_reporter(tasks):
    """TODO: 实现进度报告"""
    pass

# 测试: for progress in progress_reporter(range(100)): print(progress)


# 练习 24: 实现分页生成器
# 要求：支持分页的数据生成器
def paginate(data: Iterable, page_size: int, page_number: int):
    """TODO: 实现分页功能"""
    pass

# 测试: print(list(paginate(range(100), 10, 2)))


# 练习 25: 实现惰性求值
# 要求：实现惰性map和filter
def lazy_map(func, iterable):
    """TODO: 实现惰性map"""
    pass

def lazy_filter(predicate, iterable):
    """TODO: 实现惰性filter"""
    pass

# 测试: result = lazy_filter(lambda x: x > 5, lazy_map(lambda x: x**2, range(10)))


# ==================== 挑战题 ====================

# 挑战 1: 实现无限回文生成器
# 要求：生成所有回文数字
def palindrome_numbers():
    """TODO: 生成所有回文数字"""
    pass

# 测试: gen = palindrome_numbers(); print([next(gen) for _ in range(10)])


# 挑战 2: 实现博弈树的DFS生成器
# 要求：深度优先搜索博弈树
class GameNode:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children or []

def dfs_game_tree(root: GameNode) -> Generator:
    """TODO: 实现DFS遍历"""
    pass

# 测试: tree = GameNode(1, [GameNode(2), GameNode(3)]); print(list(dfs_game_tree(tree)))


# 挑战 3: 实现协程调度器
# 要求：管理多个协程的执行
class Scheduler:
    """TODO: 实现协程调度器"""
    pass

# 测试: scheduler = Scheduler(); scheduler.add_task(coroutine1()); scheduler.run()


# 挑战 4: 实现流式JOIN
# 要求：对两个已排序的流进行合并
def stream_join(stream1: Iterable, stream2: Iterable, key=None):
    """TODO: 实现流式JOIN"""
    pass

# 测试: s1 = [1, 3, 5, 7]; s2 = [2, 3, 4, 5]; print(list(stream_join(s1, s2)))


# 挑战 5: 实现自定义的生成器装饰器
# 要求：添加缓存、限流等功能
def generator_decorator(func):
    """TODO: 实现生成器装饰器"""
    pass

@generator_decorator
def my_generator():
    for i in range(10):
        yield i

# 测试: print(list(my_generator()))


if __name__ == "__main__":
    print("生成器与迭代器练习题")
    print("请完成每个TODO部分的代码")
    print("运行测试验证你的实现")
