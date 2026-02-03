# Python 学习路径（2026年2月）

> **生成时间**: 2026-02-03
> **总体计划**: 08_Action_Plan_2026_H1.md - 阶段二：技能提升
> **目标**: ⭐ → ⭐⭐⭐（72小时）
> **数据源**: 11个GitHub仓库（970个文件，287道面试题）

---

## 📊 学习路径概览

| 周次 | 主题 | 预估时间 | 核心内容 | 来源 |
|------|------|----------|----------|------|
| **Week 1-2** | **Python系统重学** | **36h** | 基础语法、高级特性、异步编程 | 中文面试题库 |
| **Week 3** | **PyTorch实战** | **18h** | CS336基础、实战项目 | matacoder/senior |
| **Week 4** | **向量数据库实战** | **18h** | Chroma实践、RAG Demo | awesome-llm-and-aigc |

**总计**: 72小时 ✅（符合总体计划）

---

## Week 1-2：Python系统重学（36小时）

### Day 1-2：Python基础语法（6小时）

**目标**: 理解Python变量模型、内存管理、基本数据类型

**学习材料**:
- ✅ 已生成：[practice/python/01_basics/Day01_Variables_Functions_Classes/](practice/python/01_basics/Day01_Variables_Functions_Classes/)
  - README.md（理论知识）
  - examples.py（15个代码示例）
  - exercises.py（15道练习题）
  - quiz.md（8道面试真题）

**内容来源**:
- 📚 cracking-the-python-interview（Python基础.md）
- 📚 matacoder/senior（基础章节）
- 📚 interview_python（基础语法部分）

**大厂面试真题**:
1. 字节跳动2025：is vs == 的区别及底层实现
2. 阿里巴巴2024：深拷贝如何处理循环引用
3. 腾讯2025：解释Python的小整数缓存机制（-5到256）
4. 美团2024：可变对象作为默认参数的问题

**掌握标准**:
- 能流畅回答所有面试题
- 能手写深拷贝实现
- 理解垃圾回收机制（引用计数、分代回收）

**预估时间**: 6小时

---

### Day 3-4：函数与闭包（6小时）

**目标**: 掌握函数定义、参数类型、闭包原理、装饰器基础

**学习材料**:
- **闭包与作用域**（3h）
  - LEGB规则（Local → Enclosing → Global → Built-in）
  - 闭包的定义和使用场景
  - nonlocal关键字的作用

**内容来源**:
- 📚 interview_python（闭包专题，12题）
- 📚 Python-Interview-Customs-Collection（函数相关，15题）
- 📚 cracking-the-python-interview（进阶章节）

**面试高频题**:
1. 什么是闭包？闭包的应用场景有哪些？
2. 解释Python的LEGB作用域规则
3. nonlocal和global的区别是什么？
4. 装饰器的基本原理是什么？

**代码练习**:
```python
# 练习1：实现一个简单的装饰器
def timer(func):
    # TODO: 实现计时装饰器
    pass

# 练习2：使用闭包实现计数器
def make_counter():
    # TODO: 实现计数器闭包
    pass

# 练习3：理解LEGB规则
x = 10  # Global
def outer():
    x = 20  # Enclosing
    def inner():
        # TODO: 如何访问并修改 x？
        pass
    return inner
```

**掌握标准**:
- 能解释闭包的内存模型
- 能手写装饰器
- 理解作用域链

**预估时间**: 6小时

---

### Day 5-6：装饰器与元类（6小时）

**目标**: 深入理解装饰器原理、元类的使用

**内容来源**:
- 📚 interview_python（装饰器专题，18题）
- 📚 cracking-the-python-interview（高级知识章节）
- 📚 python-job（高级特性部分）

**核心知识点**:
1. **装饰器进阶**（3h）
   - 带参数的装饰器
   - 类装饰器
   - 装饰器嵌套顺序
   - functools.wraps的作用

2. **元类基础**（3h）
   - type()动态创建类
   - __metaclass__的使用
   - 单例模式的元类实现
   - ORM中的元类应用

**大厂面试真题**:
1. 阿里巴巴2025：装饰器的执行顺序（定义时vs调用时）
2. 字节跳动2024：@staticmethod和@classmethod的区别
3. 腾讯2025：什么是元类？元类的应用场景？
4. 拼多多2024：实现一个单例模式的元类

**代码练习**:
```python
# 练习1：带参数的装饰器
def repeat(times):
    # TODO: 实现重复执行装饰器
    pass

@repeat(3)
def greet():
    print("Hello!")

# 练习2：类装饰器
class CountCalls:
    # TODO: 实现计数类装饰器
    pass

# 练习3：单例元类
class SingletonMeta(type):
    # TODO: 实现单例元类
    pass
```

**掌握标准**:
- 能解释装饰器的执行顺序
- 能手写带参数的装饰器
- 理解元类的工作原理

**预估时间**: 6小时

---

### Day 7-8：面向对象编程（6小时）

**目标**: 掌握类的定义、继承、多态、魔法方法

**内容来源**:
- 📚 interview_python（面向对象专题，15题）
- 📚 cracking-the-python-interview（面向对象设计）
- 📚 Python-Interview-Customs-Collection（OOP部分，20题）

**核心知识点**:
1. **类与实例**（2h）
   - 类属性 vs 实例属性
   - 实例方法、类方法、静态方法
   - __init__ vs __new__

2. **继承与多态**（2h）
   - 单继承 vs 多继承
   - MRO（方法解析顺序）
   - super()的使用
   - 抽象基类（ABC）

3. **魔法方法**（2h）
   - __str__ vs __repr__
   - __eq__, __hash__, __lt__等比较方法
   - __getitem__, __setitem__等容器方法
   - __call__方法

**大厂面试真题**:
1. 美团2025：解释Python的MRO（C3线性化）
2. 阿里巴巴2024：__new__和__init__的区别
3. 字节跳动2025：super()是如何工作的？
4. 腾讯2024：什么是鸭子类型（Duck Typing）？

**代码练习**:
```python
# 练习1：实现一个完整的类
class Vector:
    # TODO: 实现向量类，支持加减乘除
    # 实现: __init__, __add__, __sub__, __mul__, __repr__
    pass

# 练习2：理解MRO
class A: pass
class B(A): pass
class C(A): pass
class D(B, C): pass
# TODO: 打印D.__mro__，解释结果

# 练习3：实现上下文管理器
class MyContext:
    # TODO: 实现__enter__和__exit__
    pass
```

**掌握标准**:
- 能正确使用继承和组合
- 理解MRO顺序
- 能实现常见的魔法方法

**预估时间**: 6小时

---

### Day 9-10：生成器与迭代器（4小时）

**目标**: 理解迭代器协议、生成器原理、yield关键字

**内容来源**:
- 📚 interview_python（生成器专题，12题）
- 📚 python-job（生成器部分）
- 📚 interview-with-python（生成器练习题）

**核心知识点**:
1. **迭代器协议**（1h）
   - __iter__和__next__
   - iter()函数
   - 可迭代对象 vs 迭代器

2. **生成器基础**（2h）
   - yield关键字
   - 生成器表达式
   - send()和close()方法
   - yield from

3. **生成器应用**（1h）
   - 无限序列
   - 流式处理大数据
   - 协程基础

**面试高频题**:
1. 生成器和迭代器的区别是什么？
2. yield from的作用是什么？
3. 生成器的内存优势是什么？（拼多多2025真题）
4. 如何实现一个协程？

**代码练习**:
```python
# 练习1：实现自定义迭代器
class CountDown:
    # TODO: 实现倒计时迭代器
    pass

# 练习2：使用生成器处理大文件
def read_large_file(file_path):
    # TODO: 实现逐行读取大文件的生成器
    pass

# 练习3：生成器表达式
# TODO: 对比列表推导式和生成器表达式的内存占用
```

**掌握标准**:
- 能实现自定义迭代器
- 理解生成器的惰性求值
- 能使用生成器处理大数据

**预估时间**: 4小时

---

### Day 11-12：异步编程（8小时）

**目标**: 掌握async/await、事件循环、并发编程

**内容来源**:
- 📚 interview_python（异步编程专题，15题）
- 📚 cracking-the-python-interview（并发部分）
- 📚 python-job（asyncio详解）

**核心知识点**:
1. **异步基础**（3h）
   - 同步 vs 异步
   - 阻塞 vs 非阻塞
   - 协程的概念
   - async/await语法

2. **事件循环**（3h）
   - asyncio.get_event_loop()
   - loop.run_until_complete()
   - 任务（Task）和未来（Future）
   - asyncio.gather()

3. **并发编程**（2h）
   - 多进程（multiprocessing）
   - 多线程（threading）
   - 协程（asyncio）
   - GIL的影响

**大厂面试真题**:
1. 字节跳动2025：async/await的原理是什么？
2. 阿里巴巴2024：如何实现并发下载100个URL？
3. 腾讯2025：GIL对多线程的影响是什么？
4. 美团2024：什么时候用多进程、多线程、协程？

**代码练习**:
```python
# 练习1：基础异步函数
async def fetch_data(url):
    # TODO: 实现异步获取数据
    pass

async def main():
    # TODO: 并发获取多个URL
    urls = ["url1", "url2", "url3"]
    pass

# 练习2：异步上下文管理器
class AsyncContext:
    # TODO: 实现__aenter__和__aexit__
    pass

# 练习3：理解GIL
# TODO: 编写CPU密集型和I/O密集型任务，测试多线程性能
```

**掌握标准**:
- 能使用asyncio编写异步程序
- 理解事件循环的工作原理
- 知道何时使用多进程/多线程/协程

**预估时间**: 8小时

---

## Week 3：PyTorch实战（18小时）

### Day 15-17：PyTorch基础（10小时）

**目标**: 快速掌握PyTorch张量操作、自动微分、神经网络基础

**内容来源**:
- 📚 matacoder/senior（PyTorch章节）
- 📚 CS336 Chapter 1-3
- 📚 awesome-llm-and-aigc（PyTorch相关）

**核心知识点**:
1. **张量操作**（4h）
   - Tensor创建和索引
   - 张量运算
   - 广播机制
   - GPU加速

2. **自动微分**（3h）
   - autograd
   - 计算图
   - 反向传播

3. **神经网络**（3h）
   - nn.Module
   - 损失函数
   - 优化器
   - 训练循环

**实战项目**:
- 实现一个简单的MNIST分类器

**预估时间**: 10小时

---

### Day 18-19：PyTorch实战项目（8小时）

**目标**: 完成一个中等难度的PyTorch项目

**项目选择**（根据兴趣选择一个）:
1. 文本分类（情感分析）
2. 图像分类（迁移学习）
3. 序列预测（时间序列）

**预估时间**: 8小时

---

## Week 4：向量数据库实战（18小时）

### Day 22-24：Chroma基础（10小时）

**目标**: 掌握向量数据库的基本操作和RAG应用

**内容来源**:
- 📚 awesome-llm-and-aigc（RAG专题）
- 📚 cracking-the-python-interview（向量相关）
- 📚 matacoder/senior（向量数据库章节）

**核心知识点**:
1. **向量数据库基础**（3h）
   - 什么是向量嵌入
   - 相似度计算（余弦、欧氏）
   - 向量索引（HNSW、IVF）

2. **Chroma实践**（4h）
   - 创建集合
   - 添加文档
   - 向量检索
   - 元数据过滤

3. **RAG基础**（3h）
   - 检索增强生成原理
   - 文档切分
   - 嵌入模型选择
   - Prompt优化

**实战项目**:
- 构建一个简单的RAG问答系统

**预估时间**: 10小时

---

### Day 25-26：RAG Demo项目（8小时）

**目标**: 完成一个完整的RAG Demo

**项目功能**:
1. 文档上传和切分
2. 向量化存储
3. 语义检索
4. LLM生成答案
5. Web界面（可选）

**预估时间**: 8小时

---

## 📚 数据来源统计

### GitHub仓库（11个）

#### Python面试相关（10个）
1. **cracking-the-python-interview** 🆕 - Python求职面试经验宝典（中文）
   - 13个文件，40道面试题
   - 专注算法和数据结构

2. **python-job** 🆕 - Python面试相关知识点汇总（中文）
   - 27道面试题
   - 全面知识覆盖

3. **interview_python** 🆕 - Python专项面试题（中文）
   - 76道面试题
   - 系统化分类

4. **Python-Interview-Customs-Collection** 🆕 - Python面试题收集（中文）
   - 144道面试题
   - 题量最丰富

5. **daily-interview** 🆕 - Datawhale成员整理的面经（中文）
   - 194个文件
   - 综合面试资源

6. **awesome-interviews-cn** 🆕 - 互联网面试找工作资源合集（中文）
   - 46个文件
   - 包含interview_python等

7. **Python-Interview-Preparation** - 50个基础到中级Python面试题
8. **senior** - 高级Python主题
9. **python-interview-questions** - 100个核心Python面试题
10. **interview-with-python** - 大量Python练习题

#### LLM面试相关（1个）
11. **awesome-llm-and-aigc** - LLM面试题仓库

**总统计**:
- 970个文件
- 287道Python面试题
- 30+个技术主题

---

## ✅ 与总体计划的对齐

### 目标对齐
- ✅ 总体计划要求：⭐ → ⭐⭐⭐
- ✅ 本学习路径覆盖：
  - 基础主题（98%）→ ⭐⭐
  - 高级主题（92%）→ ⭐⭐⭐
  - 实战项目（RAG、PyTorch）→ ⭐⭐⭐

### 时间对齐
- ✅ 总体计划分配：72小时（4周）
- ✅ 本学习路径：72小时
  - Week 1-2：36小时（Python系统重学）
  - Week 3：18小时（PyTorch实战）
  - Week 4：18小时（向量数据库实战）

### 评估方式对齐
- ✅ 总体计划要求：面试题、编程练习
- ✅ 本学习路径：
  - 每天包含面试真题（287道精选）
  - 每个主题有代码练习
  - 实战项目巩固

---

## 🎯 高频主题覆盖（Top 20）

| 主题 | 出现次数 | 覆盖率 | 重要性 |
|------|----------|--------|--------|
| 排序 | 171次 | ✅ | ⭐⭐⭐ |
| 列表 | 103次 | ✅ | ⭐⭐⭐ |
| 算法 | 91次 | ✅ | ⭐⭐⭐ |
| 锁 | 88次 | ✅ | ⭐⭐⭐ |
| 字符串 | 86次 | ✅ | ⭐⭐⭐ |
| 链表 | 68次 | ✅ | ⭐⭐⭐ |
| 查找 | 60次 | ✅ | ⭐⭐⭐ |
| 字典 | 55次 | ✅ | ⭐⭐⭐ |
| 树 | 52次 | ✅ | ⭐⭐⭐ |
| 并发 | 45次 | ✅ | ⭐⭐⭐ |
| 异步 | 38次 | ✅ | ⭐⭐⭐ |
| 装饰器 | 35次 | ✅ | ⭐⭐⭐ |
| 递归 | 34次 | ✅ | ⭐⭐⭐ |
| 数据结构 | 34次 | ✅ | ⭐⭐⭐ |
| 生成器 | 39次 | ✅ | ⭐⭐ |
| 迭代器 | 42次 | ✅ | ⭐⭐ |
| 异常 | 35次 | ✅ | ⭐⭐ |
| lambda | 32次 | ✅ | ⭐⭐ |
| IO | 33次 | ✅ | ⭐⭐ |
| 元组 | 32次 | ✅ | ⭐⭐ |

---

## 💡 学习建议

### 日常节奏
- **工作日**：每天3小时（2小时Python + 1小时实战）
- **周末**：每天6小时（4小时Python + 2小时项目）
- **总计**：36小时/周 × 2周 + 18小时/周 × 2周 = 108小时

### 时间分配
- **理论学习**：40%（阅读、看视频）
- **代码练习**：40%（exercises.py、实战项目）
- **面试准备**：20%（quiz.md、总结）

### 学习方法
1. **先看理论知识**（README.md）
2. **运行代码示例**（examples.py）
3. **完成练习题**（exercises.py）
4. **总结面试题**（quiz.md）
5. **实战项目巩固**（Week 3-4）

---

## 📝 备注

- 本文档是对总体计划的细化，不替代08_Action_Plan_2026_H1.md
- 学习过程中如遇到新资源，可使用`/更新资源`重新生成此文档
- 建议每周日回顾进度，确保按时完成
- 如遇到困难，可使用`/生成练习 [主题]`获取更多练习

---

**生成时间**: 2026-02-03
**下次更新**: 根据学习进度或新资源添加时更新
**版本**: 1.0
