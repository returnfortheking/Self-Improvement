# Python self 与方法类型详解 (2025版)

> 本文档基于 2025 年最新面试高频问题编写
> 来源：GeeksforGeeks、CSDN、阿里/腾讯面试真题

---

## Part 1: `self` 详解

### 1.1 `self` 是什么？

**定义**：
- `self` 是**实例对象的引用**
- 它指向调用方法的那个对象本身
- `self` **不是 Python 关键字**，只是约定俗成的命名

### 1.2 为什么需要 `self`？

```python
class Person:
    def __init__(self, name):
        self.name = name  # self.name 是实例属性

    def say_hello(self):
        # self.name 访问当前对象的 name 属性
        print(f"Hello, I'm {self.name}")

p1 = Person("Alice")
p1.say_hello()  # 这里 self 就是 p1

p2 = Person("Bob")
p2.say_hello()  # 这里 self 就是 p2
```

**关键理解**：
```
p1.say_hello()
   ↓
Python 实际调用: Person.say_hello(p1)
   ↓
self = p1
```

### 1.3 `self` 的本质

```python
class Demo:
    def show_self(self):
        print(f"self 的 id: {id(self)}")

obj = Demo()
print(f"obj 的 id: {id(obj)}")
obj.show_self()
# 输出会相同！证明 self 就是 obj 本身
```

### 1.4 不用 `self` 会怎样？

```python
class Wrong:
    def __init__(self, name):  # 必须有 self
        name = name  # ❌ 这只是局部变量赋值，不会存储到对象

w = Wrong("Alice")
print(w.name)  # ❌ AttributeError: 'Wrong' object has no attribute 'name'
```

### 1.5 `self` 可以改名吗？

```python
class Person:
    def __init__(this, name):  # 用 this 代替 self
        this.name = name

    def show(me):  # 用 me 代替 self
        print(me.name)

p = Person("Alice")
p.show()  # ✅ 可以工作，但强烈不推荐！
```

**结论**：虽然可以改名，但**永远使用 `self`**，这是 Python 社区的强约定。

---

## Part 2: 三种方法类型对比

### 2.1 快速对比表

| 特性 | 实例方法 | 类方法 (@classmethod) | 静态方法 (@staticmethod) |
|-----|---------|---------------------|------------------------|
| **第一个参数** | `self` | `cls` | 无特殊参数 |
| **参数含义** | 实例对象 | 类本身 | 普通参数 |
| **访问实例属性** | ✅ | ❌ | ❌ |
| **访问类属性** | ✅ | ✅ | ❌ |
| **修改实例属性** | ✅ | ❌ | ❌ |
| **修改类属性** | ⚠️ 不推荐 | ✅ | ❌ |
| **调用方式** | `obj.method()` | `Class.method()` 或 `obj.method()` | `Class.method()` 或 `obj.method()` |
| **用途** | 操作实例数据 | 操作类数据、工厂方法 | 工具函数、与类/实例无关 |

### 2.2 实例方法（Instance Method）

**定义**：最常用的方法，操作**具体对象**的数据

```python
class Student:
    school = "Python学院"  # 类属性

    def __init__(self, name, age):
        self.name = name   # 实例属性
        self.age = age     # 实例属性

    def study(self, subject):
        """实例方法：使用 self 访问实例属性"""
        print(f"{self.name} ({self.age}岁) 正在学习{subject}")

s1 = Student("Alice", 18)
s1.study("数学")  # Alice (18岁) 正在学习数学

s2 = Student("Bob", 20)
s2.study("英语")  # Bob (20岁) 正在学习英语
```

**关键点**：
- 第一个参数必须是 `self`
- `self` 代表调用方法的具体对象
- 可以访问和修改实例属性

---

### 2.3 类方法（Class Method）

**定义**：操作**类本身**的数据，不依赖具体实例

```python
class Student:
    school = "Python学院"
    total_students = 0  # 类属性：统计总人数

    def __init__(self, name):
        self.name = name
        Student.total_students += 1  # 修改类属性

    @classmethod
    def get_school_name(cls):
        """类方法：使用 cls 访问类属性"""
        return f"学校名称: {cls.school}"

    @classmethod
    def get_total_students(cls):
        """类方法：统计类级别的数据"""
        return f"总学生数: {cls.total_students}"

    @classmethod
    def change_school(cls, new_name):
        """类方法：修改类属性"""
        cls.school = new_name

# 使用示例
print(Student.get_school_name())  # 学校名称: Python学院
print(Student.get_total_students())  # 总学生数: 0

s1 = Student("Alice")
s2 = Student("Bob")

print(Student.get_total_students())  # 总学生数: 2

Student.change_school("新学院")
print(Student.get_school_name())  # 学校名称: 新学院

print(s1.get_school_name())  # 也可以通过实例调用
```

**关键点**：
- 使用 `@classmethod` 装饰器
- 第一个参数是 `cls`（代表类本身）
- 可以访问和修改类属性
- **不能访问实例属性**（因为没有 `self`）
- 常用于：
  - 工厂方法（创建不同类型的实例）
  - 修改类级别的数据

---

### 2.4 静态方法（Static Method）

**定义**：与类和实例都无关的**工具函数**

```python
class MathUtils:
    @staticmethod
    def add(a, b):
        """静态方法：普通函数，只是放在类的命名空间里"""
        return a + b

    @staticmethod
    def is_even(n):
        """静态方法：工具函数"""
        return n % 2 == 0

# 使用示例
print(MathUtils.add(5, 3))      # 8
print(MathUtils.is_even(4))     # True

utils = MathUtils()
print(utils.add(5, 3))          # 也可以通过实例调用
```

**什么时候用静态方法？**

```python
class DateValidator:
    """日期验证工具"""

    @staticmethod
    def is_leap_year(year):
        """判断闰年：不需要访问任何类/实例属性"""
        return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

    @staticmethod
    def validate_date(day, month, year):
        """验证日期有效性"""
        if month < 1 or month > 12:
            return False
        if day < 1:
            return False
        # 各月份天数检查...
        return True

# 使用
print(DateValidator.is_leap_year(2024))  # True
print(DateValidator.validate_date(31, 2, 2024))  # False
```

**关键点**：
- 使用 `@staticmethod` 装饰器
- **没有** `self` 或 `cls` 参数
- **不能访问类属性或实例属性**
- 本质上是一个"放在类里面的普通函数"
- 用于：
  - 工具函数
  - 与类/实例无关的功能
  - 代码组织（逻辑相关的函数放在一起）

---

## Part 3: 实战对比示例

### 3.1 综合示例

```python
class Employee:
    """员工管理系统"""

    # 类属性：公司所有员工共享
    company = "Tech Corp"
    min_salary = 3000

    def __init__(self, name, salary):
        # 实例属性：每个员工不同
        self.name = name
        self.salary = salary

    # ========== 实例方法 ==========
    def get_annual_salary(self):
        """实例方法：计算某个员工的年薪"""
        return self.salary * 12

    def give_raise(self, amount):
        """实例方法：给某个员工加薪"""
        self.salary += amount
        print(f"{self.name} 的工资涨到了 {self.salary}")

    # ========== 类方法 ==========
    @classmethod
    def set_min_salary(cls, new_min):
        """类方法：修改公司最低工资标准"""
        cls.min_salary = new_min
        print(f"公司最低工资调整为: {new_min}")

    @classmethod
    def create_from_bonus(cls, name, base_salary, bonus):
        """类方法：工厂方法 - 根据基础工资+奖金创建员工"""
        total = base_salary + bonus
        if total < cls.min_salary:
            raise ValueError(f"工资不能低于最低标准 {cls.min_salary}")
        return cls(name, total)

    # ========== 静态方法 ==========
    @staticmethod
    def calculate_tax(salary):
        """静态方法：计算个税（工具函数，不依赖类或实例）"""
        if salary <= 5000:
            return 0
        elif salary <= 10000:
            return (salary - 5000) * 0.1
        else:
            return (salary - 10000) * 0.2 + 500

    @staticmethod
    def format_salary(amount):
        """静态方法：格式化工资显示"""
        return f"¥{amount:,.2f}"
```

**使用示例**：

```python
# 1. 使用实例方法
emp1 = Employee("Alice", 8000)
print(emp1.get_annual_salary())  # 96000
emp1.give_raise(2000)             # Alice 的工资涨到了 10000

# 2. 使用类方法
Employee.set_min_salary(4000)     # 公司最低工资调整为: 4000
emp2 = Employee.create_from_bonus("Bob", 5000, 3000)  # 工厂方法创建

# 3. 使用静态方法（不需要创建对象！）
tax = Employee.calculate_tax(8000)        # 300
formatted = Employee.format_salary(8000)  # ¥8,000.00
```

---

## Part 4: 如何选择方法类型？

### 决策树

```
需要访问或修改实例属性？
├─ 是 → 使用 实例方法
└─ 否 → 继续

需要访问或修改类属性？
├─ 是 → 使用 类方法
└─ 否 → 使用 静态方法
```

### 具体场景

| 场景 | 方法类型 | 示例 |
|-----|---------|------|
| 操作对象的数据 | 实例方法 | `student.get_grade()` |
| 修改对象的状态 | 实例方法 | `account.withdraw(100)` |
| 统计类级别的数据 | 类方法 | `Student.get_total_count()` |
| 工厂方法（创建对象） | 类方法 | `Date.from_string("2024-01-01")` |
| 工具函数 | 静态方法 | `MathUtils.sqrt(16)` |
| 验证函数 | 静态方法 | `EmailValidator.is_valid(email)` |

---

## Part 5: 面试高频问题

### Q1: 为什么需要类方法和静态方法？

**答案**：

1. **类方法**：
   - 操作类级别的数据（如计数器、配置）
   - 工厂模式（不同方式创建对象）
   - 不需要实例就能调用

2. **静态方法**：
   - 代码组织（相关工具函数放在一起）
   - 命名空间管理（`MathUtils.add` 比 `add` 更清晰）
   - 不需要访问类或实例的数据

### Q2: 实例方法可以访问类属性吗？

```python
class Demo:
    class_var = "类变量"

    def method(self):
        # ✅ 可以通过类名访问
        print(Demo.class_var)
        # ✅ 也可以通过 self.__class__ 访问
        print(self.__class__.class_var)
```

**答案**：可以，但不推荐修改类属性（应该用类方法）。

### Q3: 三种方法的内存占用有什么区别？

```python
class Test:
    def instance_method(self):
        pass

    @classmethod
    def class_method(cls):
        pass

    @staticmethod
    def static_method():
        pass

t = Test()
print(t.instance_method)    # <bound method Test.instance_method of <Test object>>
print(t.class_method)       # <bound method Test.class_method of <class 'Test'>>
print(t.static_method)      # <function Test.static_method>
```

**关键区别**：
- 实例方法绑定到**对象**
- 类方法绑定到**类**
- 静态方法**不绑定**（就是普通函数）

---

## Part 6: 练习题

### 练习 1：补全代码

```python
class Circle:
    """圆类"""

    # 类属性：记录创建的圆的数量
    count = 0

    def __init__(self, radius):
        self.radius = radius
        # TODO: 增加 count 计数

    # TODO: 实例方法 - 计算面积
    def area(self):
        pass

    # TODO: 类方法 - 获取圆的数量
    @classmethod
    def get_count(cls):
        pass

    # TODO: 静态方法 - 验证半径是否有效
    @staticmethod
    def is_valid_radius(radius):
        pass

# 测试
c1 = Circle(5)
c2 = Circle(10)
print(Circle.get_count())    # 应该输出: 2
print(c1.area())             # 应该输出: 78.5
print(Circle.is_valid_radius(-1))  # 应该输出: False
```

### 练习 2：工厂方法

```python
class Person:
    """人类"""

    def __init__(self, name, age):
        self.name = name
        self.age = age

    @classmethod
    def from_birth_year(cls, name, birth_year):
        """根据出生年份创建 Person 对象
        假设当前年份是 2024
        """
        # TODO: 实现这个方法
        pass

# 测试
p = Person.from_birth_year("Alice", 1990)
print(p.age)  # 应该输出: 34
```

### 练习 3：工具类

```python
class StringValidator:
    """字符串验证工具类"""

    @staticmethod
    def is_empty(s):
        """判断是否为空字符串"""
        pass

    @staticmethod
    def is_email(s):
        """简单判断是否为邮箱格式"""
        pass

    @staticmethod
    def is_phone(s):
        """简单判断是否为手机号"""
        pass

# 测试
print(StringValidator.is_empty(""))      # True
print(StringValidator.is_email("a@b.c")) # True
print(StringValidator.is_phone("13800138000"))  # True
```

---

## Part 7: 答案

### 练习 1 答案

```python
import math

class Circle:
    count = 0

    def __init__(self, radius):
        self.radius = radius
        Circle.count += 1

    def area(self):
        return 3.14 * self.radius ** 2

    @classmethod
    def get_count(cls):
        return cls.count

    @staticmethod
    def is_valid_radius(radius):
        return radius > 0
```

### 练习 2 答案

```python
from datetime import datetime

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    @classmethod
    def from_birth_year(cls, name, birth_year):
        current_year = datetime.now().year
        age = current_year - birth_year
        return cls(name, age)
```

### 练习 3 答案

```python
import re

class StringValidator:
    @staticmethod
    def is_empty(s):
        return not s or s.strip() == ""

    @staticmethod
    def is_email(s):
        return "@" in s and "." in s.split("@")[-1]

    @staticmethod
    def is_phone(s):
        return len(s) == 11 and s.isdigit() and s.startswith("1")
```

---

## 总结

| 要点 | 记忆口诀 |
|-----|---------|
| `self` | 指向调用对象本身 |
| 实例方法 | 操作对象数据，用 `self` |
| 类方法 | 操作类数据，用 `cls` |
| 静态方法 | 工具函数，无特殊参数 |
| 选择原则 | 用实例数据→实例方法；用类数据→类方法；都不用→静态方法 |

---

**下一步**：
1. 阅读本文档 2-3 遍
2. 完成练习题
3. 告诉我准备好继续评估了！
