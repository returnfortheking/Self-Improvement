# 生成器与迭代器代码示例

import itertools
from typing import Iterator, Iterable, Generator

# ==================== 1. 基础迭代器 ====================

class CountDown:
    """倒计时迭代器"""
    def __init__(self, start: int):
        self.start = start
        self.current = start

    def __iter__(self) -> Iterator[int]:
        return self

    def __next__(self) -> int:
        if self.current < 0:
            raise StopIteration
        value = self.current
        self.current -= 1
        return value


# for i in CountDown(5):
#     print(i)  # 5, 4, 3, 2, 1, 0


# ==================== 2. 简单生成器 ====================

def simple_generator() -> Generator[int, None, None]:
    """简单的生成器"""
    yield 1
    yield 2
    yield 3


# gen = simple_generator()
# print(next(gen))  # 1
# print(next(gen))  # 2
# print(next(gen))  # 3


# ==================== 3. 生成器表达式 ====================

# 列表推导式
list_comp = [x ** 2 for x in range(10)]  # 立即计算

# 生成器表达式
gen_expr = (x ** 2 for x in range(10))  # 惰性计算

# print(list(list_comp))   # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
# print(list(gen_expr))    # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]


# ==================== 4. 斐波那契数列生成器 ====================

def fibonacci(n: int) -> Generator[int, None, None]:
    """生成前n个斐波那契数"""
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b


# print(list(fibonacci(10)))  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]


# ==================== 5. 无限序列生成器 ====================

def infinite_fibonacci() -> Generator[int, None, None]:
    """无限斐波那契数列"""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b


# fib = infinite_fibonacci()
# for _ in range(10):
#     print(next(fib))


# ==================== 6. yield from 委托生成器 ====================

def sub_generator() -> Generator[str, None, None]:
    """子生成器"""
    yield "A"
    yield "B"
    yield "C"


def main_generator() -> Generator[str, None, None]:
    """主生成器"""
    yield "Start"
    yield from sub_generator()
    yield "End"


# print(list(main_generator()))  # ['Start', 'A', 'B', 'C', 'End']


# ==================== 7. 生成器管道 ====================

def read_lines(filename: str) -> Generator[str, None, None]:
    """读取文件行"""
    with open(filename, 'r') as f:
        for line in f:
            yield line.strip()


def filter_empty(lines: Iterable[str]) -> Generator[str, None, None]:
    """过滤空行"""
    for line in lines:
        if line:
            yield line


def transform_upper(lines: Iterable[str]) -> Generator[str, None, None]:
    """转换为大写"""
    for line in lines:
        yield line.upper()


# 使用管道（示例）
# lines = read_lines('data.txt')
# filtered = filter_empty(lines)
# upper = transform_upper(filtered)
# for line in upper:
#     print(line)


# ==================== 8. 生成器的send方法 ====================

def echo_generator() -> Generator[str, str, None]:
    """回显生成器"""
    while True:
        received = yield
        yield f"Echo: {received}"


# gen = echo_generator()
# next(gen)  # 启动生成器
# print(gen.send("Hello"))  # Echo: Hello
# print(gen.send("World"))  # Echo: World


# ==================== 9. 生成器的throw方法 ====================

def exception_handling_generator() -> Generator[int, None, None]:
    """处理异常的生成器"""
    try:
        while True:
            value = yield
            print(f"Received: {value}")
    except ValueError as e:
        print(f"Caught ValueError: {e}")
        yield "Error handled"
    except GeneratorExit:
        print("Generator closing")


# gen = exception_handling_generator()
# next(gen)
# gen.send(10)
# gen.throw(ValueError, "Something went wrong")


# ==================== 10. 使用itertools.count ====================

# 无限计数
for i in itertools.count(10, 2):
    if i > 20:
        break
    print(i)  # 10, 12, 14, 16, 18, 20


# ==================== 11. 使用itertools.cycle ====================

# 无限循环
colors = itertools.cycle(['red', 'green', 'blue'])
for _ in range(5):
    print(next(colors))  # red, green, blue, red, green


# ==================== 12. 使用itertools.repeat ====================

# 重复元素
for i in itertools.repeat(10, 3):
    print(i)  # 10, 10, 10


# ==================== 13. 使用itertools.chain ====================

# 连接多个可迭代对象
list1 = [1, 2, 3]
list2 = [4, 5, 6]
list3 = [7, 8, 9]

chained = itertools.chain(list1, list2, list3)
# print(list(chained))  # [1, 2, 3, 4, 5, 6, 7, 8, 9]


# ==================== 14. 使用itertools.islice ====================

# 切片迭代器
gen = (x ** 2 for x in range(100))
sliced = itertools.islice(gen, 5, 10)
# print(list(sliced))  # [25, 36, 49, 64, 81]


# ==================== 15. 使用itertools.accumulate ====================

# 累积运算
data = [1, 2, 3, 4, 5]
acc = itertools.accumulate(data)
# print(list(acc))  # [1, 3, 6, 10, 15]


# ==================== 16. 使用itertools.dropwhile和takewhile ====================

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 丢弃满足条件的元素
dropped = itertools.dropwhile(lambda x: x < 5, data)
# print(list(dropped))  # [5, 6, 7, 8, 9, 10]

# 保留满足条件的元素
taken = itertools.takewhile(lambda x: x < 5, data)
# print(list(taken))  # [1, 2, 3, 4]


# ==================== 17. 使用itertools.groupby ====================

data = [('a', 1), ('b', 2), ('a', 3), ('b', 4), ('a', 5)]

# 分组
sorted_data = sorted(data, key=lambda x: x[0])  # 必须先排序
for key, group in itertools.groupby(sorted_data, key=lambda x: x[0]):
    print(f"{key}: {[item[1] for item in group]}")


# ==================== 18. 使用itertools.tee ====================

# 复制迭代器
gen = (x for x in range(5))
gen1, gen2, gen3 = itertools.tee(gen, 3)

# print(list(gen1))  # [0, 1, 2, 3, 4]
# print(list(gen2))  # [0, 1, 2, 3, 4]
# print(list(gen3))  # [0, 1, 2, 3, 4]


# ==================== 19. 使用itertools.combinations和permutations ====================

items = ['a', 'b', 'c']

# 组合（不考虑顺序）
comb = itertools.combinations(items, 2)
# print(list(comb))  # [('a', 'b'), ('a', 'c'), ('b', 'c')]

# 排列（考虑顺序）
perm = itertools.permutations(items, 2)
# print(list(perm))  # [('a', 'b'), ('a', 'c'), ('b', 'a'), ('b', 'c'), ('c', 'a'), ('c', 'b')]


# ==================== 20. 使用itertools.product ====================

# 笛卡尔积
colors = ['red', 'blue']
sizes = ['S', 'M', 'L']
products = itertools.product(colors, sizes)
# print(list(products))  # [('red', 'S'), ('red', 'M'), ('red', 'L'),
                        #  ('blue', 'S'), ('blue', 'M'), ('blue', 'L')]


# ==================== 21. 自定义可迭代对象 ====================

class Range:
    """自定义range类"""
    def __init__(self, start: int, stop: int, step: int = 1):
        self.start = start
        self.stop = stop
        self.step = step

    def __iter__(self) -> Iterator[int]:
        current = self.start
        while current < self.stop:
            yield current
            current += self.step


# for i in Range(0, 10, 2):
#     print(i)  # 0, 2, 4, 6, 8


# ==================== 22. 生成器状态管理 ====================

def state_machine_generator():
    """状态机生成器"""
    while True:
        state = yield
        if state == 'start':
            yield "Started"
        elif state == 'stop':
            yield "Stopped"
        elif state == 'pause':
            yield "Paused"
        else:
            yield "Unknown state"


# ==================== 23. 协程消费者 ====================

def consumer():
    """消费者协程"""
    while True:
        item = yield
        print(f"Consuming: {item}")


def producer(consumer_gen, items):
    """生产者"""
    next(consumer_gen)  # 启动消费者
    for item in items:
        consumer_gen.send(item)


# cons = consumer()
# producer(cons, [1, 2, 3, 4, 5])


# ==================== 24. 树的遍历生成器 ====================

class TreeNode:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right


def inorder_traversal(root: TreeNode) -> Generator[int, None, None]:
    """中序遍历二叉树"""
    if root:
        yield from inorder_traversal(root.left)
        yield root.value
        yield from inorder_traversal(root.right)


# 构建树
#       1
#      / \
#     2   3
#    / \
#   4   5
# root = TreeNode(1,
#                 TreeNode(2, TreeNode(4), TreeNode(5)),
#                 TreeNode(3))
# print(list(inorder_traversal(root)))  # [4, 2, 5, 1, 3]


# ==================== 25. 扁平化嵌套列表 ====================

def flatten(nested_list):
    """扁平化嵌套列表"""
    for item in nested_list:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item


# nested = [1, [2, [3, 4], 5], 6, [7, [8, [9]]]]
# print(list(flatten(nested)))  # [1, 2, 3, 4, 5, 6, 7, 8, 9]


# ==================== 26. 批处理生成器 ====================

def batch(iterable: Iterable, batch_size: int):
    """将可迭代对象分批处理"""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:  # 处理剩余元素
        yield batch


# data = range(100)
# for batch_data in batch(data, 10):
#     print(f"Processing batch: {batch_data[0]} to {batch_data[-1]}")


# ==================== 27. 滑动窗口生成器 ====================

def sliding_window(iterable: Iterable, window_size: int):
    """滑动窗口"""
    it = iter(iterable)
    window = []
    for _ in range(window_size):
        window.append(next(it))
    yield window.copy()

    for item in it:
        window.pop(0)
        window.append(item)
        yield window.copy()


# data = [1, 2, 3, 4, 5]
# for window in sliding_window(data, 3):
#     print(window)  # [1, 2, 3], [2, 3, 4], [3, 4, 5]


# ==================== 28. 回文检测器 ====================

def palindrome_generator(s: str) -> Generator[str, None, None]:
    """生成所有回文子串"""
    n = len(s)
    for i in range(n):
        for j in range(i + 1, n + 1):
            substr = s[i:j]
            if substr == substr[::-1]:
                yield substr


# print(list(palindrome_generator("ababa")))  # ['a', 'b', 'a', 'b', 'a', 'aba', 'bab', 'aba', 'ababa']


# ==================== 29. 素数生成器 ====================

def primes():
    """生成素数"""
    yield 2
    primes_so_far = [2]
    candidate = 3
    while True:
        is_prime = True
        for p in primes_so_far:
            if p * p > candidate:
                break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes_so_far.append(candidate)
            yield candidate
        candidate += 2


# prime_gen = primes()
# for _ in range(10):
#     print(next(prime_gen))  # 前10个素数


# ==================== 30. 惰性求值示例 ====================

def lazy_map(func, iterable):
    """惰性map"""
    for item in iterable:
        yield func(item)


def lazy_filter(predicate, iterable):
    """惰性filter"""
    for item in iterable:
        if predicate(item):
            yield item


# 组合使用
data = range(10)
mapped = lazy_map(lambda x: x ** 2, data)
filtered = lazy_filter(lambda x: x > 10, mapped)
# print(list(filtered))  # [16, 25, 36, 49, 64, 81]


if __name__ == "__main__":
    # 示例：斐波那契数列
    print("前10个斐波那契数:")
    print(list(fibonacci(10)))

    # 示例：扁平化嵌套列表
    print("\n扁平化嵌套列表:")
    nested = [1, [2, [3, 4], 5], 6]
    print(list(flatten(nested)))

    # 示例：滑动窗口
    print("\n滑动窗口:")
    for window in sliding_window([1, 2, 3, 4, 5], 3):
        print(window)
