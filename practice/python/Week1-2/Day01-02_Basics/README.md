# Day 1-2: Python基础语法

> **学习目标**: 理解Python变量模型、内存管理、基本数据类型
> **预估时间**: 6小时
> **难度**: ⭐→⭐⭐

---

## 一、Python变量模型：一切皆对象

### 1.1 变量是对象的引用

**核心概念**：
- Python中一切都是对象（everything is an object）
- 变量是对象的**引用**，不是容器
- 对象有类型，变量没有
- 对象有身份（id），通过 `id()` 查看

```python
# 示例
a = [1, 2, 3]  # 创建列表对象，a 引用它
b = a          # b 和 a 指向同一个对象
b.append(4)

print(a)  # [1, 2, 3, 4] - a 也被修改了！
print(id(a) == id(b))  # True - 同一个对象
```

### 1.2 可变 vs 不可变对象

| 类型 | 可变/不可变 | 示例 |
|------|------------|------|
| 不可变 | int, float, str, tuple, bool, frozenset | `x = 10` |
| 可变 | list, dict, set, bytearray | `lst = [1, 2, 3]` |

**关键区别**：
- 不可变对象：修改会创建新对象，id 改变
- 可变对象：修改不创建新对象，id 不变

```python
# 可变对象示例
lst = [1, 2, 3]
print(id(lst))  # 140234567890000
lst.append(4)
print(id(lst))  # 140234567890000 - id 不变

# 不可变对象示例
s = "hello"
print(id(s))  # 140234567890100
s += " world"
print(id(s))  # 140234567890200 - id 改变了！
```

### 1.3 is vs == 的区别

| 运算符 | 作用 | 比较内容 |
|--------|------|---------|
| `==` | 值相等 | 对象的内容 |
| `is` | 身份相同 | 对象的id（内存地址）|

```python
# 小整数缓存（-5 到 256）
a = 256
b = 256
print(a is b)  # True - 在缓存范围内

c = 257
d = 257
print(c is d)  # False - 超出缓存范围
print(c == d)  # True - 值相等

# 列表比较
list1 = [1, 2, 3]
list2 = [1, 2, 3]
print(list1 == list2)  # True - 内容相同
print(list1 is list2)  # False - 不同对象
```

### 1.4 内存管理：引用计数

Python使用**引用计数**进行内存管理：
- 每个对象有一个引用计数
- 引用计数为 0 时，对象被回收
- `sys.getrefcount()` 查看引用计数

```python
import sys

a = [1, 2, 3]  # 引用计数 = 1
b = a          # 引用计数 = 2
print(sys.getrefcount(a))  # 3（包括getrefcount自身的引用）

del a          # 引用计数 = 1
print(b)       # [1, 2, 3] - 对象仍在
```

---

## 二、基本数据类型

### 2.1 数字类型（Numbers）

```python
# 整数（int）
x = 10
y = 0b1010  # 二进制
z = 0o12    # 八进制
w = 0xa     # 十六进制

# 浮点数（float）
pi = 3.14159
scientific = 1.23e-4  # 科学计数法

# 复数（complex）
c = 3 + 4j
print(c.real)  # 3.0
print(c.imag)  # 4.0

# 布尔（bool）
print(True + True)  # 2
print(False * 10)   # 0
```

### 2.2 序列类型（Sequence）

#### 字符串（str）

```python
# 字符串创建
s1 = 'hello'
s2 = "world"
s3 = '''multi
line
string'''

# 字符串操作
s = "Python Programming"
print(s[0])        # 'P'
print(s[-1])       # 'g'
print(s[0:6])      # 'Python'
print(s[7:])       # 'Programming'
print(s[::-1])     # 反转字符串

# 字符串方法
print(s.lower())   # 转小写
print(s.upper())   # 转大写
print(s.split())   # 分割成列表
print('-'.join(['a', 'b', 'c']))  # 'a-b-c'
```

#### 列表（list）

```python
# 列表创建
lst = [1, 2, 3, 4, 5]

# 列表操作
lst.append(6)        # 末尾添加
lst.insert(0, 0)     # 指定位置插入
lst.extend([7, 8])   # 扩展列表
lst.remove(0)        # 删除指定值
popped = lst.pop()   # 弹出最后一个元素
lst[0] = 100         # 修改元素

# 列表切片
lst = [0, 1, 2, 3, 4, 5]
print(lst[1:4])      # [1, 2, 3]
print(lst[::2])      # [0, 2, 4] - 步长为2
print(lst[::-1])     # [5, 4, 3, 2, 1, 0] - 反转

# 列表推导式
squares = [x**2 for x in range(10)]
evens = [x for x in range(20) if x % 2 == 0]
```

#### 元组（tuple）

```python
# 元组创建
t = (1, 2, 3)
t = 1, 2, 3  # 简写
single = (1,)  # 单元素元组需要逗号

# 元组操作（不可变）
print(t[0])     # 1
print(t[1:3])   # (2, 3)
print(t * 2)    # (1, 2, 3, 1, 2, 3)

# 元组解包
a, b, c = t
```

### 2.3 集合类型（Set）

```python
# 集合创建（无序、唯一）
s = {1, 2, 3, 4, 4, 4}  # {1, 2, 3, 4}
s = set([1, 2, 3])

# 集合操作
s.add(5)           # 添加元素
s.remove(3)        # 删除元素
s.discard(100)     # 安全删除（不存在不报错）

# 集合运算
s1 = {1, 2, 3}
s2 = {3, 4, 5}
print(s1 | s2)     # 并集: {1, 2, 3, 4, 5}
print(s1 & s2)     # 交集: {3}
print(s1 - s2)     # 差集: {1, 2}
print(s1 ^ s2)     # 对称差: {1, 2, 4, 5}
```

### 2.4 映射类型（Dictionary）

```python
# 字典创建
d = {'name': 'Alice', 'age': 30}
d = dict(name='Alice', age=30)
d = dict([('name', 'Alice'), ('age', 30)])

# 字典操作
d['city'] = 'Beijing'         # 添加键值对
d['age'] = 31                  # 修改值
value = d.get('name')          # 获取值（不存在返回None）
value = d.get('salary', 0)     # 获取值（不存在返回默认值）
keys = d.keys()                # 所有键
values = d.values()            # 所有值
items = d.items()              # 所有键值对

# 字典方法
print('name' in d)             # 检查键是否存在
d.update({'salary': 50000})    # 更新字典
d.pop('age')                   # 删除键值对
d.pop('nonexistent', None)     # 安全删除

# 字典推导式
squares = {x: x**2 for x in range(6)}
```

---

## 三、控制流

### 3.1 条件语句

```python
# if-else
x = 10
if x > 0:
    print("Positive")
elif x < 0:
    print("Negative")
else:
    print("Zero")

# 三元表达式
result = "Positive" if x > 0 else "Non-positive"

# and, or, not
if x > 0 and x < 100:
    print("x is between 0 and 100")

if x == 10 or x == 20:
    print("x is 10 or 20")

if not x < 0:
    print("x is not negative")
```

### 3.2 循环语句

```python
# for 循环
for i in range(5):
    print(i)

for i, value in enumerate(['a', 'b', 'c']):
    print(f"{i}: {value}")

for key, value in {'a': 1, 'b': 2}.items():
    print(f"{key}: {value}")

# while 循环
count = 0
while count < 5:
    print(count)
    count += 1

# 循环控制
for i in range(10):
    if i == 3:
        continue  # 跳过本次迭代
    if i == 7:
        break     # 退出循环
    print(i)

# else 子句（循环正常结束时执行）
for i in range(5):
    print(i)
else:
    print("Loop completed")
```

### 3.3 异常处理

```python
# 基本异常处理
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero")
except Exception as e:
    print(f"Other error: {e}")
else:
    print("No error occurred")
finally:
    print("Always executed")

# 抛出异常
def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

# 自定义异常
class CustomError(Exception):
    pass

raise CustomError("This is a custom error")
```

---

## 四、函数基础

### 4.1 函数定义

```python
# 基本函数
def greet(name):
    return f"Hello, {name}!"

# 默认参数
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

# 可变位置参数
def sum_all(*args):
    return sum(args)

# 可变关键字参数
def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

# 混合使用
def func(a, b, c=10, *args, **kwargs):
    print(f"a={a}, b={b}, c={c}")
    print(f"args={args}")
    print(f"kwargs={kwargs}")
```

### 4.2 lambda 函数

```python
# lambda 表达式
add = lambda x, y: x + y
print(add(3, 5))  # 8

# 与高阶函数配合
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))
```

---

## 五、学习检查清单

完成本主题学习后，您应该能够：

- [ ] 解释 Python 变量是对象引用的概念
- [ ] 区分可变对象和不可变对象
- [ ] 理解 `is` vs `==` 的区别
- [ ] 了解 Python 的内存管理（引用计数）
- [ ] 熟练使用基本数据类型（str, list, tuple, dict, set）
- [ ] 掌握列表推导式和字典推导式
- [ ] 熟练使用条件语句和循环
- [ ] 理解异常处理机制
- [ ] 掌握函数的定义和各种参数类型

---

## 六、参考资源

- **来源仓库**：
  - cracking-the-python-interview（Python基础.md）
  - interview_python（基础语法部分）
  - matacoder/senior（基础章节）

- **练习文件**：
  - examples.py - 20个代码示例
  - exercises.py - 20道练习题
  - quiz.md - 10道大厂面试真题
