# Day 1-2: Python基础语法面试题

> **主题**: 变量、数据类型、内存管理
> **来源**: 字节跳动、阿里巴巴、腾讯、美团、拼多多
> **更新**: 2026-02-03

---

## Q1: is vs == 的区别（字节跳动 2025 真题）

**难度**: ⭐⭐
**频率**: 90% 面试遇到

### 题目

请解释以下代码的输出：

```python
a = 256
b = 256
print(a is b)  # 输出？

c = 257
d = 257
print(c is d)  # 输出？
```

### 答案

```
a = 256
b = 256
print(a is b)  # True

c = 257
d = 257
print(c is d)  # False
```

### 解析

**核心原因**：Python 的**小整数缓存机制**

1. **小整数缓存**：
   - Python 预先创建并缓存了 -5 到 256 范围内的整数对象
   - 在这个范围内，所有变量引用同一个对象
   - 超出这个范围，每次创建新对象

2. **is vs ==**：
   - `is` 比较对象的**身份**（id，内存地址）
   - `==` 比较对象的**值**（内容）

3. **验证**：
   ```python
   print(a == b)  # True（值相等）
   print(a is b)  # True（256 在缓存范围内，同一对象）

   print(c == d)  # True（值相等）
   print(c is d)  # False（257 超出缓存范围，不同对象）
   ```

### 面试加分点

- 了解 Python 的对象缓存优化策略
- 知道字符串也有类似的 intern 机制
- 理解 `is` 和 `==` 的使用场景

---

## Q2: 可变默认参数的陷阱（阿里巴巴 2024 真题）

**难度**: ⭐⭐
**频率**: 80% 面试遇到

### 题目

以下代码有什么问题？

```python
def append_to(element, target=[]):
    target.append(element)
    return target

# 调用
print(append_to(1))  # [1]
print(append_to(2))  # [1, 2] ？？
print(append_to(3))  # [1, 2, 3] ？？
```

### 答案

**问题**：默认参数 `target=[]` 只在函数定义时创建一次，后续调用会复用同一个列表！

### 原因

1. Python 默认参数在**函数定义时**创建，不是每次调用时创建
2. 可变对象（列表、字典）作为默认参数会被多次调用共享

### 正确做法

```python
def append_to(element, target=None):
    if target is None:
        target = []
    target.append(element)
    return target
```

### 面试加分点

- 理解 Python 函数定义的执行时机
- 知道如何避免可变默认参数的问题
- 使用 `None` 作为默认值，然后在函数内部创建新对象

---

## Q3: 深拷贝如何处理嵌套列表？（腾讯 2025 真题）

**难度**: ⭐⭐⭐
**频率**: 70% 面试遇到

### 题目

以下代码的输出是什么？

```python
original = [1, 2, [3, 4]]
shallow = original.copy()
deep = original.copy()
original[2].append(5)

print(original)  # ?
print(shallow)   # ?
```

### 答案

```
original = [1, 2, [3, 4, 5]]
shallow = [1, 2, [3, 4, 5]]  # 嵌套列表被修改！
```

### 解析

**浅拷贝 vs 深拷贝**：

1. **浅拷贝**（`.copy()` 或 `[:]`）：
   - 只复制第一层
   - 嵌套对象仍然是引用
   - 修改嵌套对象会影响原列表

2. **深拷贝**（`copy.deepcopy()`）：
   - 递归复制所有层级
   - 完全独立的新对象
   - 修改不影响原列表

### 正确做法

```python
import copy

original = [1, 2, [3, 4]]
deep = copy.deepcopy(original)
original[2].append(5)

print(deep)  # [1, 2, [3, 4]] - 未受影响
```

### 面试加分点

- 理解浅拷贝和深拷贝的区别
- 知道何时使用深拷贝
- 能说出深拷贝的时间复杂度（O(n)）

---

## Q4: 列表的各种复制方法（美团 2024 真题）

**难度**: ⭐⭐
**频率**: 75% 面试遇到

### 题目

以下哪些复制方式是深拷贝？

```python
original = [1, 2, [3, 4]]
copy1 = original.copy()
copy2 = original[:]
copy3 = list(original)
copy4 = copy.deepcopy(original)

original[2].append(5)
```

### 答案

**只有 `copy4` 是深拷贝**，其他都是浅拷贝。

### 解析

| 方法 | 类型 | 嵌套列表是否独立 |
|------|------|----------------|
| `original.copy()` | 浅拷贝 | ❌ |
| `original[:]` | 浅拷贝 | ❌ |
| `list(original)` | 浅拷贝 | ❌ |
| `copy.deepcopy(original)` | 深拷贝 | ✅ |

### 验证

```python
import copy

original = [1, 2, [3, 4]]

# 浅拷贝
copy1 = original.copy()
copy2 = original[:]
copy3 = list(original)

# 深拷贝
copy4 = copy.deepcopy(original)

original[2].append(5)

print(copy1)  # [1, 2, [3, 4, 5]] - 被修改
print(copy2)  # [1, 2, [3, 4, 5]] - 被修改
print(copy3)  # [1, 2, [3, 4, 5]] - 被修改
print(copy4)  # [1, 2, [3, 4]] - 未被修改
```

### 面试加分点

- 知道所有列表复制方法
- 理解浅拷贝和深拷贝的区别
- 能根据场景选择合适的复制方式

---

## Q5: 字符串的 intern 机制（字节跳动 2024 真题）

**难度**: ⭐⭐⭐
**频率**: 60% 面试遇到（高级岗位）

### 题目

解释以下代码的输出：

```python
a = "hello"
b = "hello"
print(a is b)  # True

c = "hello world"
d = "hello world"
print(c is d)  # True

e = "hello!"
f = "hello!"
print(e is f)  # ???
```

### 答案

```
a is b  # True - 短字符串自动 intern
c is d  # True - 字符串字面量自动 intern
e is f  # 取决于实现（通常是 True）
```

### 解析

**字符串 Intern 机制**：

1. **什么是 Intern**：
   - Python 会缓存某些字符串
   - 相同内容的字符串共享同一个对象
   - 节省内存，提高比较效率

2. **自动 Intern 的字符串**：
   - 短字符串（通常长度 < 20）
   - 字符串字面量
   - 标识符（变量名、函数名等）

3. **手动 Intern**：
   ```python
   import sys
   s1 = sys.intern("a very long string")
   s2 = sys.intern("a very long string")
   print(s1 is s2)  # True
   ```

### 面试加分点

- 理解字符串 intern 机制
- 知道 `sys.intern()` 的用法
- 能解释 intern 的优缺点

---

## Q6: 引用计数的陷阱（腾讯 2024 真题）

**难度**: ⭐⭐
**频率**: 70% 面试遇到

### 题目

以下代码的输出是什么？

```python
import sys

a = [1, 2, 3]
b = a
c = b
print(sys.getrefcount(a))
```

### 答案

```
输出：4（而不是 3）
```

### 解析

**为什么是 4？**

1. **引用来源**：
   - `a` 引用：1
   - `b` 引用：1
   - `c` 引用：1
   - `getrefcount()` 自身：1

2. **临时引用**：
   - 函数调用时会创建临时引用
   - 参数传递也会增加引用计数

3. **正确查看方式**：
   ```python
   import sys

   a = [1, 2, 3]
   b = a
   c = a

   # 不使用 getrefcount
   ref_count = sys.getrefcount(a) - 1  # 减去函数自身的引用
   print(ref_count)  # 3
   ```

### 面试加分点

- 理解引用计数的工作原理
- 知道 `getrefcount()` 的特殊性
- 能解释循环引用导致的内存泄漏

---

## Q7: 元组的不可变性（拼多多 2025 真题）

**难度**: ⭐⭐
**频率**: 65% 面试遇到

### 题目

元组是不可变的，以下代码会报错吗？

```python
t = (1, 2, [3, 4])
t[2].append(5)
print(t)  # ?
```

### 答案

```
不会报错
输出：(1, 2, [3, 4, 5])
```

### 解析

**元组的"不可变"是相对的**：

1. **元组本身不可变**：
   - 不能添加、删除元素
   - 不能修改元素指向

2. **但如果元素是可变对象**：
   - 可以修改可变对象的内容
   - 元组只是存储了对象的引用

3. **验证**：
   ```python
   t = (1, 2, [3, 4])
   # t[0] = 10  # 报错 - 不能修改引用
   t[2].append(5)  # 正常 - 可以修改可变对象
   print(t)  # (1, 2, [3, 4, 5])
   ```

### 面试加分点

- 理解元组的不可变性
- 知道元组存储的是引用
- 能解释"不可变"的真正含义

---

## Q8: 集合和字典的性能（美团 2025 真题）

**难度**: ⭐⭐
**频率**: 80% 面试遇到

### 题目

以下代码的时间复杂度是多少？

```python
# 1
s = set()
s.add(1)
if 1 in s:
    print("Found")

# 2
d = {}
d['key'] = 'value'
if 'key' in d:
    print("Found")

# 3
lst = [1, 2, 3]
if 1 in lst:
    print("Found")
```

### 答案

```python
# 1 - 集合
s.add(1)     # O(1)
1 in s       # O(1)

# 2 - 字典
d['key'] = 'value'  # O(1)
'key' in d          # O(1)

# 3 - 列表
1 in lst      # O(n) - 需要遍历
```

### 解析

**哈希表 vs 线性查找**：

| 操作 | 集合/字典 | 列表 |
|------|----------|------|
| 查找 | O(1) | O(n) |
| 插入 | O(1) | O(1) |
| 删除 | O(1) | O(n) |

**原因**：
- 集合和字典使用**哈希表**
- 列表使用**数组**，需要遍历

### 面试加分点

- 理解哈希表的工作原理
- 知道何时使用集合/字典优化查找
- 能说出哈希冲突的处理方法

---

## Q9: 字典推导式 vs 列表推导式（字节跳动 2024 真题）

**难度**: ⭐
**频率**: 50% 面试遇到

### 题目

使用推导式将以下列表转换为字典：

```python
lst = ['a', 'b', 'c']
# 目标：{'a': 0, 'b': 1, 'c': 2}
```

### 答案

```python
# 方法 1：使用 enumerate
d = {k: v for v, k in enumerate(lst)}

# 方法 2：使用 dict 和 enumerate
d = dict(enumerate(lst))
d = {v: k for k, v in d.items()}

# 方法 3：使用 range
d = {lst[i]: i for i in range(len(lst))}
```

### 解析

**字典推导式的优势**：
- 代码简洁
- 性能好（比循环构建快）
- 可读性强

**其他示例**：
```python
# 平方字典
squares = {x: x**2 for x in range(6)}
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# 过滤字典
d = {'a': 1, 'b': 2, 'c': 3}
filtered = {k: v for k, v in d.items() if v > 1}
# {'b': 2, 'c': 3}
```

### 面试加分点

- 熟练掌握字典推导式
- 知道 enumerate 的用法
- 能根据场景选择合适的推导式

---

## Q10: 列表切片的高级用法（阿里巴巴 2025 真题）

**难度**: ⭐⭐
**频率**: 70% 面试遇到

### 题目

解释以下切片操作的结果：

```python
lst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

print(lst[::2])     # ?
print(lst[1::2])    # ?
print(lst[::-1])    # ?
print(lst[5::-1])   # ?
print(lst[:5:-1])   # ?
```

### 答案

```python
lst[::2]     # [0, 2, 4, 6, 8] - 从头到尾，步长2
lst[1::2]    # [1, 3, 5, 7, 9] - 从索引1到尾，步长2
lst[::-1]    # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0] - 反转
lst[5::-1]   # [5, 4, 3, 2, 1, 0] - 从索引5到头，反向
lst[:5:-1]   # [9, 8, 7, 6] - 从尾到索引5（不含），反向
```

### 解析

**切片语法**：`[start:stop:step]`

- **start**：起始位置（默认 0）
- **stop**：结束位置（默认 len）
- **step**：步长（默认 1，负数表示反向）

**特殊情况**：
- 省略 start：从开头开始
- 省略 stop：到结尾
- step 为负：反向切片

### 面试加分点

- 熟练掌握切片语法
- 能使用切片实现字符串反转
- 知道切片不越界（自动调整）

---

## 总结

### 必背知识点

1. **小整数缓存**：-5 到 256
2. **is vs ==**：身份 vs 值
3. **可变默认参数**：使用 `None`
4. **浅拷贝 vs 深拷贝**：`.copy()` vs `copy.deepcopy()`
5. **字符串 intern**：短字符串自动缓存
6. **引用计数**：理解内存管理
7. **元组的不可变性**：引用不可变，内容可变
8. **哈希表性能**：O(1) 查找
9. **推导式**：简洁高效
10. **切片语法**：`[start:stop:step]`

### 学习建议

- **理解原理**：不要死记硬背，理解背后的设计
- **动手实验**：自己运行代码验证
- **关联对比**：对比相似概念（深拷贝 vs 浅拷贝）
- **实战应用**：在实际项目中应用这些知识

---

**更新时间**: 2026-02-03
**题目来源**: 字节跳动、阿里巴巴、腾讯、美团、拼多多 2024-2025 面试题
