# Python 函数进阶 - Python-100-Days 完整补充材料

## 📑 目录
- [函数的定义和调用](#函数的定义和调用)
- [函数的参数](#函数的参数)
- [作用域问题](#作用域问题)
- [函数的高级用法](#函数的高级用法)

---


## 源文件: 14.函数和模块.md

## 函数和模块

在讲解本节课的内容之前，我们先来研究一道数学题，请说出下面的方程有多少组正整数解。

$$
x_{1} + x_{2} + x_{3} + x_{4} = 8
$$

你可能已经想到了，这个问题其实等同于将 8 个苹果分成四组且每组至少一个苹果有多少种方案，也等价于在分隔 8 个苹果的 7 个间隙之间放入三个隔断将苹果分成四组有多少种方案，所以答案是 $\small{C_{7}^{3} = 35}$ ，其中 $\small{C_{7}^{3}}$ 代表 7 选 3 的组合数，其计算公式如下所示。

$$
C_m^n = \frac {m!} {n!(m-n)!}
$$

根据之前学习的知识，我们可以用循环做累乘的方式分别计算出 $\small{m!}$ 、 $\small{n!}$ 和 $\small{(m-n)!}$ ，然后再通过除法运算得到组合数 $\small{C_{m}^{n}}$ ，代码如下所示。

```python
"""
输入m和n，计算组合数C(m,n)的值

Version: 1.0
Author: 骆昊
"""

m = int(input('m = '))
n = int(input('n = '))
# 计算m的阶乘
fm = 1
for num in range(1, m + 1):
    fm *= num
# 计算n的阶乘
fn = 1
for num in range(1, n + 1):
    fn *= num
# 计算m-n的阶乘
fk = 1
for num in range(1, m - n + 1):
    fk *= num
# 计算C(M,N)的值
print(fm // fn // fk)
```

输入：

```
m = 7
n = 3
```

输出：

```
35
```

不知大家是否注意到，上面的代码中我们做了三次求阶乘的操作，虽然 $\small{m}$ 、 $\small{n}$ 、 $\small{m - n}$ 的值各不相同，但是三段代码并没有实质性的区别，属于重复代码。世界级的编程大师*Martin Fowler*曾经说过：“**代码有很多种坏味道，重复是最坏的一种！**”。要写出高质量的代码，首先就要解决重复代码的问题。对于上面的代码来说，我们可以将求阶乘的功能封装到一个称为“函数”的代码块中，在需要计算阶乘的地方，我们只需“调用函数”即可实现对求阶乘功能的复用。

### 定义函数

数学上的函数通常形如 $\small{y = f(x)}$ 或者 $\small{z = g(x, y)}$ 这样的形式，在 $\small{y = f(x)}$ 中， $\small{f}$ 是函数的名字， $\small{x}$ 是函数的自变量， $\small{y}$ 是函数的因变量；而在 $\small{z = g(x, y)}$ 中， $\small{g}$ 是函数名， $\small{x}$ 和 $\small{y}$ 是函数的自变量， $\small{z}$ 是函数的因变量。Python 中的函数跟这个结构是一致的，每个函数都有自己的名字、自变量和因变量。我们通常把 Python 函数的自变量称为函数的参数，而因变量称为函数的返回值。

Python 中可以使用`def`关键字来定义函数，和变量一样每个函数也应该有一个漂亮的名字，命名规则跟变量的命名规则是一样的（大家赶紧想想我们之前讲过的变量的命名规则）。在函数名后面的圆括号中可以设置函数的参数，也就是我们刚才说的函数的自变量，而函数执行完成后，我们会通过`return`关键字来返回函数的执行结果，这就是我们刚才说的函数的因变量。如果函数中没有`return`语句，那么函数会返回代表空值的`None`。另外，函数也可以没有自变量（参数），但是函数名后面的圆括号是必须有的。一个函数要做的事情（要执行的代码），是通过代码缩进的方式放到函数定义行之后，跟之前分支和循环结构的代码块类似，如下图所示。

<img src="res/day14/function_definition.png" style="zoom:45%;">

下面，我们将之前代码中求阶乘的操作放到一个函数中，通过这种方式来重构上面的代码。**所谓重构，是在不影响代码执行结果的前提下对代码的结构进行调整**，重构之后的代码如下所示。

```python
"""
输入m和n，计算组合数C(m,n)的值

Version: 1.1
Author: 骆昊
"""


# 通过关键字def定义求阶乘的函数
# 自变量（参数）num是一个非负整数
# 因变量（返回值）是num的阶乘
def fac(num):
    result = 1
    for n in range(2, num + 1):
        result *= n
    return result


m = int(input('m = '))
n = int(input('n = '))
# 计算阶乘的时候不需要写重复的代码而是直接调用函数
# 调用函数的语法是在函数名后面跟上圆括号并传入参数
print(fac(m) // fac(n) // fac(m - n))
```

大家可以感受下，上面的代码是不是比之前的版本更加简单优雅。更为重要的是，我们定义的求阶乘函数`fac`还可以在其他需要求阶乘的代码中重复使用。所以，**使用函数可以帮助我们将功能上相对独立且会被重复使用的代码封装起来**，当我们需要这些的代码，不是把重复的代码再编写一遍，而是**通过调用函数实现对既有代码的复用**。事实上，Python 标准库的`math`模块中，已经有一个名为`factorial`的函数实现了求阶乘的功能，我们可以直接用`import math`导入`math`模块，然后使用`math.factorial`来调用求阶乘的函数；我们也可以通过`from math import factorial`直接导入`factorial`函数来使用它，代码如下所示。

```python
"""
输入m和n，计算组合数C(m,n)的值

Version: 1.2
Author: 骆昊
"""
from math import factorial

m = int(input('m = '))
n = int(input('n = '))
print(factorial(m) // factorial(n) // factorial(m - n))
```

将来我们使用的函数，要么是自定义的函数，要么是 Python 标准库或者三方库中提供的函数，如果已经有现成的可用的函数，我们就没有必要自己去定义，“**重复发明轮子**”是一件非常糟糕的事情。对于上面的代码，如果你觉得`factorial`这个名字太长，书写代码的时候不是特别方便，我们在导入函数的时候还可以通过`as`关键字为其别名。在调用函数的时候，我们可以用函数的别名，而不再使用它之前的名字，代码如下所示。

```python
"""
输入m和n，计算组合数C(m,n)的值

Version: 1.3
Author: 骆昊
"""
from math import factorial as f

m = int(input('m = '))
n = int(input('n = '))
print(f(m) // f(n) // f(m - n))
```

### 函数的参数

#### 位置参数和关键字参数

我们再来写一个函数，根据给出的三条边的长度判断是否可以构成三角形，如果可以构成三角形则返回`True`，否则返回`False`，代码如下所示。

```python
def make_judgement(a, b, c):
    """判断三条边的长度能否构成三角形"""
    return a + b > c and b + c > a and a + c > b
```

上面`make_judgement`函数有三个参数，这种参数叫做位置参数，在调用函数时通常按照从左到右的顺序依次传入，而且传入参数的数量必须和定义函数时参数的数量相同，如下所示。

```python
print(make_judgement(1, 2, 3))  # False
print(make_judgement(4, 5, 6))  # True
```

如果不想按照从左到右的顺序依次给出`a`、`b`、`c` 三个参数的值，也可以使用关键字参数，通过“参数名=参数值”的形式为函数传入参数，如下所示。

```python
print(make_judgement(b=2, c=3, a=1))  # False
print(make_judgement(c=6, b=4, a=5))  # True
```

在定义函数时，我们可以在参数列表中用`/`设置**强制位置参数**（*positional-only arguments*），用`*`设置**命名关键字参数**。所谓强制位置参数，就是调用函数时只能按照参数位置来接收参数值的参数；而命名关键字参数只能通过“参数名=参数值”的方式来传递和接收参数，大家可以看看下面的例子。

```python
# /前面的参数是强制位置参数
def make_judgement(a, b, c, /):
    """判断三条边的长度能否构成三角形"""
    return a + b > c and b + c > a and a + c > b


# 下面的代码会产生TypeError错误，错误信息提示“强制位置参数是不允许给出参数名的”
# TypeError: make_judgement() got some positional-only arguments passed as keyword arguments
# print(make_judgement(b=2, c=3, a=1))
```

> **说明**：强制位置参数是 Python 3.8 引入的新特性，在使用低版本的 Python 解释器时需要注意。

```python
# *后面的参数是命名关键字参数
def make_judgement(*, a, b, c):
    """判断三条边的长度能否构成三角形"""
    return a + b > c and b + c > a and a + c > b


# 下面的代码会产生TypeError错误，错误信息提示“函数没有位置参数但却给了3个位置参数”
# TypeError: make_judgement() takes 0 positional arguments but 3 were given
# print(make_judgement(1, 2, 3))
```

#### 参数的默认值

Python 中允许函数的参数拥有默认值，我们可以把之前讲过的一个例子“CRAPS赌博游戏”（《第07课：分支和循环结构的应用》）中摇色子获得点数的功能封装到函数中，代码如下所示。

```python
from random import randrange


# 定义摇色子的函数
# 函数的自变量（参数）n表示色子的个数，默认值为2
# 函数的因变量（返回值）表示摇n颗色子得到的点数
def roll_dice(n=2):
    total = 0
    for _ in range(n):
        total += randrange(1, 7)
    return total


# 如果没有指定参数，那么n使用默认值2，表示摇两颗色子
print(roll_dice())
# 传入参数3，变量n被赋值为3，表示摇三颗色子获得点数
print(roll_dice(3))
```

我们再来看一个更为简单的例子。

```python
def add(a=0, b=0, c=0):
    """三个数相加求和"""
    return a + b + c


# 调用add函数，没有传入参数，那么a、b、c都使用默认值0
print(add())         # 0
# 调用add函数，传入一个参数，该参数赋值给变量a, 变量b和c使用默认值0
print(add(1))        # 1
# 调用add函数，传入两个参数，分别赋值给变量a和b，变量c使用默认值0
print(add(1, 2))     # 3
# 调用add函数，传入三个参数，分别赋值给a、b、c三个变量
print(add(1, 2, 3))  # 6
```

需要注意的是，**带默认值的参数必须放在不带默认值的参数之后**，否则将产生`SyntaxError`错误，错误消息是：`non-default argument follows default argument`，翻译成中文的意思是“没有默认值的参数放在了带默认值的参数后面”。

#### 可变参数

Python 语言中可以通过星号表达式语法让函数支持可变参数。所谓可变参数指的是在调用函数时，可以向函数传入`0`个或任意多个参数。将来我们以团队协作的方式开发商业项目时，很有可能要设计函数给其他人使用，但有的时候我们并不知道函数的调用者会向该函数传入多少个参数，这个时候可变参数就能派上用场。

下面的代码演示了如何使用可变位置参数实现对任意多个数求和的`add`函数，调用函数时传入的参数会保存到一个元组，通过对该元组的遍历，可以获取传入函数的参数。

```python
# 用星号表达式来表示args可以接收0个或任意多个参数
# 调用函数时传入的n个参数会组装成一个n元组赋给args
# 如果一个参数都没有传入，那么args会是一个空元组
def add(*args):
    total = 0
    # 对保存可变参数的元组进行循环遍历
    for val in args:
        # 对参数进行了类型检查（数值型的才能求和）
        if type(val) in (int, float):
            total += val
    return total


# 在调用add函数时可以传入0个或任意多个参数
print(add())         # 0
print(add(1))        # 1
print(add(1, 2, 3))  # 6
print(add(1, 2, 'hello', 3.45, 6))  # 12.45
```

如果我们希望通过“参数名=参数值”的形式传入若干个参数，具体有多少个参数也是不确定的，我们还可以给函数添加可变关键字参数，把传入的关键字参数组装到一个字典中，代码如下所示。

```python
# 参数列表中的**kwargs可以接收0个或任意多个关键字参数
# 调用函数时传入的关键字参数会组装成一个字典（参数名是字典中的键，参数值是字典中的值）
# 如果一个关键字参数都没有传入，那么kwargs会是一个空字典
def foo(*args, **kwargs):
    print(args)
    print(kwargs)


foo(3, 2.1, True, name='骆昊', age=43, gpa=4.95)
```

输出：

```
(3, 2.1, True)
{'name': '骆昊', 'age': 43, 'gpa': 4.95}
```

### 用模块管理函数

不管用什么样的编程语言来写代码，给变量、函数起名字都是一个让人头疼的问题，因为我们会遇到**命名冲突**这种尴尬的情况。最简单的场景就是在同一个`.py`文件中定义了两个同名的函数，如下所示。

```python
def foo():
    print('hello, world!')


def foo():
    print('goodbye, world!')

    
foo()  # 大家猜猜调用foo函数会输出什么
```

当然上面的这种情况我们很容易就能避免，但是如果项目是团队协作多人开发的时候，团队中可能有多个程序员都定义了名为`foo`的函数，这种情况下怎么解决命名冲突呢？答案其实很简单，Python 中每个文件就代表了一个模块（module），我们在不同的模块中可以有同名的函数，在使用函数的时候，我们通过`import`关键字导入指定的模块再使用**完全限定名**（`模块名.函数名`）的调用方式，就可以区分到底要使用的是哪个模块中的`foo`函数，代码如下所示。

`module1.py`

```python
def foo():
    print('hello, world!')
```

`module2.py`

```python
def foo():
    print('goodbye, world!')
```

`test.py`

```python
import module1
import module2

# 用“模块名.函数名”的方式（完全限定名）调用函数，
module1.foo()  # hello, world!
module2.foo()  # goodbye, world!
```

在导入模块时，还可以使用`as`关键字对模块进行别名，这样我们可以使用更为简短的完全限定名。

`test.py`

```python
import module1 as m1
import module2 as m2

m1.foo()  # hello, world!
m2.foo()  # goodbye, world!
```

上面两段代码，我们导入的是定义函数的模块，我们也可以使用`from...import...`语法从模块中直接导入需要使用的函数，代码如下所示。

`test.py`

```python
from module1 import foo

foo()  # hello, world!

from module2 import foo

foo()  # goodbye, world!
```

但是，如果我们如果从两个不同的模块中导入了同名的函数，后面导入的函数会替换掉之前的导入，就像下面的代码，调用`foo`会输出`goodbye, world!`，因为我们先导入了`module1`的`foo`，后导入了`module2`的`foo` 。如果两个`from...import...`反过来写，那就是另外一番光景了。

`test.py`

```python
from module1 import foo
from module2 import foo

foo()  # goodbye, world!
```

如果想在上面的代码中同时使用来自两个模块的`foo`函数还是有办法的，大家可能已经猜到了，还是用`as`关键字对导入的函数进行别名，代码如下所示。

`test.py`

```python
from module1 import foo as f1
from module2 import foo as f2

f1()  # hello, world!
f2()  # goodbye, world!
```

### 标准库中的模块和函数

Python 标准库中提供了大量的模块和函数来简化我们的开发工作，我们之前用过的`random`模块就为我们提供了生成随机数和进行随机抽样的函数；而`time`模块则提供了和时间操作相关的函数；我们之前用到过的`math`模块中还包括了计算正弦、余弦、指数、对数等一系列的数学函数。随着我们深入学习 Python 语言，我们还会用到更多的模块和函数。

Python 标准库中还有一类函数是不需要`import`就能够直接使用的，我们将其称之为**内置函数**，这些内置函数不仅有用而且还很常用，下面的表格列出了一部分的内置函数。

| 函数    | 说明                                                         |
| ------- | ------------------------------------------------------------ |
| `abs`   | 返回一个数的绝对值，例如：`abs(-1.3)`会返回`1.3`。           |
| `bin`   | 把一个整数转换成以`'0b'`开头的二进制字符串，例如：`bin(123)`会返回`'0b1111011'`。 |
| `chr`   | 将Unicode编码转换成对应的字符，例如：`chr(8364)`会返回`'€'`。 |
| `hex`   | 将一个整数转换成以`'0x'`开头的十六进制字符串，例如：`hex(123)`会返回`'0x7b'`。 |
| `input` | 从输入中读取一行，返回读到的字符串。                         |
| `len`   | 获取字符串、列表等的长度。                                   |
| `max`   | 返回多个参数或一个可迭代对象中的最大值，例如：`max(12, 95, 37)`会返回`95`。 |
| `min`   | 返回多个参数或一个可迭代对象中的最小值，例如：`min(12, 95, 37)`会返回`12`。 |
| `oct`   | 把一个整数转换成以`'0o'`开头的八进制字符串，例如：`oct(123)`会返回`'0o173'`。 |
| `open`  | 打开一个文件并返回文件对象。                                 |
| `ord`   | 将字符转换成对应的Unicode编码，例如：`ord('€')`会返回`8364`。 |
| `pow`   | 求幂运算，例如：`pow(2, 3)`会返回`8`；`pow(2, 0.5)`会返回`1.4142135623730951`。 |
| `print` | 打印输出。                                                   |
| `range` | 构造一个范围序列，例如：`range(100)`会产生`0`到`99`的整数序列。 |
| `round` | 按照指定的精度对数值进行四舍五入，例如：`round(1.23456, 4)`会返回`1.2346`。 |
| `sum`   | 对一个序列中的项从左到右进行求和运算，例如：`sum(range(1, 101))`会返回`5050`。 |
| `type`  | 返回对象的类型，例如：`type(10)`会返回`int`；而` type('hello')`会返回`str`。 |

###  总结

**函数是对功能相对独立且会重复使用的代码的封装**。学会使用定义和使用函数，就能够写出更为优质的代码。当然，Python 语言的标准库中已经为我们提供了大量的模块和常用的函数，用好这些模块和函数就能够用更少的代码做更多的事情；如果这些模块和函数不能满足我们的要求，可能就需要自定义函数，然后再通过模块的概念来管理这些自定义函数。
---


## 源文件: 16.函数使用进阶.md

## 函数使用进阶

我们继续探索定义和使用函数的相关知识。通过前面的学习，我们知道了函数有自变量（参数）和因变量（返回值），自变量可以是任意的数据类型，因变量也可以是任意的数据类型，那么这里就有一个小问题，我们能不能用函数作为函数的参数，用函数作为函数的返回值？这里我们先说结论：**Python 中的函数是“一等函数”**，所谓“一等函数”指的就是函数可以赋值给变量，函数可以作为函数的参数，函数也可以作为函数的返回值。把一个函数作为其他函数的参数或返回值的用法，我们通常称之为“高阶函数”。

### 高阶函数

我们回到之前讲过的一个例子，设计一个函数，传入任意多个参数，对其中`int`类型或`float`类型的元素实现求和操作。我们对之前的代码稍作调整，让整个代码更加紧凑一些，如下所示。

```python
def calc(*args, **kwargs):
    items = list(args) + list(kwargs.values())
    result = 0
    for item in items:
        if type(item) in (int, float):
            result += item
    return result
```

如果我们希望上面的`calc`函数不仅仅可以做多个参数的求和，还可以实现更多的甚至是自定义的二元运算，我们该怎么做呢？上面的代码只能求和是因为函数中使用了`+=`运算符，这使得函数跟加法运算形成了耦合关系，如果能解除这种耦合关系，函数的通用性和灵活性就会更好。解除耦合的办法就是将`+`运算符变成函数调用，并将其设计为函数的参数，代码如下所示。

```python
def calc(init_value, op_func, *args, **kwargs):
    items = list(args) + list(kwargs.values())
    result = init_value
    for item in items:
        if type(item) in (int, float):
            result = op_func(result, item)
    return result
```

注意，上面的函数增加了两个参数，其中`init_value`代表运算的初始值，`op_func`代表二元运算函数，为了调用修改后的函数，我们先定义做加法和乘法运算的函数，代码如下所示。

```python
def add(x, y):
    return x + y


def mul(x, y):
    return x * y
```

如果要做求和的运算，我们可以按照下面的方式调用`calc`函数。

```python
print(calc(0, add, 1, 2, 3, 4, 5))  # 15
```

如果要做求乘积运算，我们可以按照下面的方式调用`calc`函数。

```python
print(calc(1, mul, 1, 2, 3, 4, 5))  # 120 
```

上面的`calc`函数通过将运算符变成函数的参数，实现了跟加法运算耦合，这是一种非常高明和实用的编程技巧，但对于最初学者来说可能会觉得难以理解，建议大家细品一下。需要注意上面的代码中，将函数作为参数传入其他函数和直接调用函数是有显著的区别的，**调用函数需要在函数名后面跟上圆括号，而把函数作为参数时只需要函数名即可**。

如果我们没有提前定义好`add`和`mul`函数，也可以使用 Python 标准库中的`operator`模块提供的`add`和`mul`函数，它们分别代表了做加法和做乘法的二元运算，我们拿过来直接使用即可，代码如下所示。

```python
import operator

print(calc(0, operator.add, 1, 2, 3, 4, 5))  # 15
print(calc(1, operator.mul, 1, 2, 3, 4, 5))  # 120
```

Python 内置函数中有不少高阶函数，我们前面提到过的`filter`和`map`函数就是高阶函数，前者可以实现对序列中元素的过滤，后者可以实现对序列中元素的映射，例如我们要去掉一个整数列表中的奇数，并对所有的偶数求平方得到一个新的列表，就可以直接使用这两个函数来做到，具体的做法是如下所示。

```python
def is_even(num):
    """判断num是不是偶数"""
    return num % 2 == 0


def square(num):
    """求平方"""
    return num ** 2


old_nums = [35, 12, 8, 99, 60, 52]
new_nums = list(map(square, filter(is_even, old_nums)))
print(new_nums)  # [144, 64, 3600, 2704]
```

当然，要完成上面代码的功能，也可以使用列表生成式，列表生成式的做法更为简单优雅。

```python
old_nums = [35, 12, 8, 99, 60, 52]
new_nums = [num ** 2 for num in old_nums if num % 2 == 0]
print(new_nums)  # [144, 64, 3600, 2704]
```

我们再来讨论一个内置函数`sorted`，它可以实现对容器型数据类型（如：列表、字典等）元素的排序。我们之前讲过`list`类型的`sort`方法，它实现了对列表元素的排序，`sorted`函数从功能上来讲跟列表的`sort`方法没有区别，但它会返回排序后的列表对象，而不是直接修改原来的列表，这一点我们称为**函数的无副作用设计**，也就是说调用函数除了产生返回值以外，不会对程序的状态或外部环境产生任何其他的影响。使用`sorted`函数排序时，可以通过高阶函数的形式自定义排序的规则，我们通过下面的例子加以说明。

```python
old_strings = ['in', 'apple', 'zoo', 'waxberry', 'pear']
new_strings = sorted(old_strings)
print(new_strings)  # ['apple', 'in', 'pear', waxberry', 'zoo']
```

上面的代码对大家来说并不陌生，但是如果希望根据字符串的长度而不是字母表顺序对列表元素排序，我们可以向`sorted`函数传入一个名为`key`的参数，将`key`参数赋值为获取字符串长度的函数`len`，这个函数我们在之前的课程中讲到过，代码如下所示。

```python
old_strings = ['in', 'apple', 'zoo', 'waxberry', 'pear']
new_strings = sorted(old_strings, key=len)
print(new_strings)  # ['in', 'zoo', 'pear', 'apple', 'waxberry']
```

> **说明**：列表类型的`sort`方法也有同样的`key`参数，有兴趣的读者可以自行尝试。

### Lambda函数

在使用高阶函数的时候，如果作为参数或者返回值的函数本身非常简单，一行代码就能够完成，也不需要考虑对函数的复用，那么我们可以使用 lambda 函数。Python 中的 lambda 函数是没有的名字函数，所以很多人也把它叫做**匿名函数**，lambda 函数只能有一行代码，代码中的表达式产生的运算结果就是这个匿名函数的返回值。之前的代码中，我们写的`is_even`和`square`函数都只有一行代码，我们可以考虑用 lambda 函数来替换掉它们，代码如下所示。

```python
old_nums = [35, 12, 8, 99, 60, 52]
new_nums = list(map(lambda x: x ** 2, filter(lambda x: x % 2 == 0, old_nums)))
print(new_nums)  # [144, 64, 3600, 2704]
```

通过上面的代码可以看出，定义 lambda 函数的关键字是`lambda`，后面跟函数的参数，如果有多个参数用逗号进行分隔；冒号后面的部分就是函数的执行体，通常是一个表达式，表达式的运算结果就是 lambda 函数的返回值，不需要写`return` 关键字。

前面我们说过，Python 中的函数是“一等函数”，函数是可以直接赋值给变量的。在学习了 lambda 函数之后，前面我们写过的一些函数就可以用一行代码来实现它们了，大家可以看看能否理解下面的求阶乘和判断素数的函数。

```python
import functools
import operator

# 用一行代码实现计算阶乘的函数
fac = lambda n: functools.reduce(operator.mul, range(2, n + 1), 1)

# 用一行代码实现判断素数的函数
is_prime = lambda x: all(map(lambda f: x % f, range(2, int(x ** 0.5) + 1)))

# 调用Lambda函数
print(fac(6))        # 720
print(is_prime(37))  # True
```

> **提示1**：上面使用的`reduce`函数是 Python 标准库`functools`模块中的函数，它可以实现对一组数据的归约操作，类似于我们之前定义的`calc`函数，第一个参数是代表运算的函数，第二个参数是运算的数据，第三个参数是运算的初始值。很显然，`reduce`函数也是高阶函数，它和`filter`函数、`map`函数一起构成了处理数据中非常关键的三个动作：**过滤**、**映射**和**归约**。
>
> **提示2**：上面判断素数的 lambda 函数通过`range`函数构造了从 2 到 $\small{\sqrt{x}}$ 的范围，检查这个范围有没有`x`的因子。`all`函数也是 Python 内置函数，如果传入的序列中所有的布尔值都是`True`，`all`函数返回`True`，否则`all`函数返回`False`。

### 偏函数

偏函数是指固定函数的某些参数，生成一个新的函数，这样就无需在每次调用函数时都传递相同的参数。在 Python 语言中，我们可以使用`functools`模块的`partial`函数来创建偏函数。例如，`int`函数在默认情况下可以将字符串视为十进制整数进行类型转换，如果我们修修改它的`base`参数，就可以定义出三个新函数，分别用于将二进制、八进制、十六进制字符串转换为整数，代码如下所示。

```python
import functools

int2 = functools.partial(int, base=2)
int8 = functools.partial(int, base=8)
int16 = functools.partial(int, base=16)

print(int('1001'))    # 1001

print(int2('1001'))   # 9
print(int8('1001'))   # 513
print(int16('1001'))  # 4097
```

不知大家是否注意到，`partial`函数的第一个参数和返回值都是函数，它将传入的函数处理成一个新的函数返回。通过构造偏函数，我们可以结合实际的使用场景将原函数变成使用起来更为便捷的新函数，不知道大家有没有觉得这波操作很有意思。

###  总结

Python 中的函数是一等函数，可以赋值给变量，也可以作为函数的参数和返回值，这也就意味着我们可以在 Python 中使用高阶函数。高阶函数的概念对新手并不友好，但它却带来了函数设计上的灵活性。如果我们要定义的函数非常简单，只有一行代码，而且不需要函数名来复用它，我们可以使用 lambda 函数。


---


## 源文件: 17.函数高级应用.md

## 函数高级应用

在上一个章节中，我们探索了 Python 中的高阶函数，相信大家对函数的定义和应用有了更深刻的认知。本章我们继续为大家讲解函数相关的知识，一个是 Python 中的特色语法装饰器，一个是函数的递归调用。

### 装饰器

Python 语言中，装饰器是“**用一个函数装饰另外一个函数并为其提供额外的能力**”的语法现象。装饰器本身是一个函数，它的参数是被装饰的函数，它的返回值是一个带有装饰功能的函数。通过前面的描述，相信大家已经听出来了，装饰器是一个高阶函数，它的参数和返回值都是函数。但是，装饰器的概念对编程语言的初学者来说，还是让人头疼的，下面我们先通过一个简单的例子来说明装饰器的作用。假设有名为`downlaod`和`upload`的两个函数，分别用于文件的上传和下载，如下所示。

```python
import random
import time


def download(filename):
    """下载文件"""
    print(f'开始下载{filename}.')
    time.sleep(random.random() * 6)
    print(f'{filename}下载完成.')

    
def upload(filename):
    """上传文件"""
    print(f'开始上传{filename}.')
    time.sleep(random.random() * 8)
    print(f'{filename}上传完成.')

    
download('MySQL从删库到跑路.avi')
upload('Python从入门到住院.pdf')
```

> **说明**：上面的代码用休眠一段随机时间的方式模拟了下载和上传文件需要花费一定的时间，并没有真正的联网上传下载文件。用 Python 语言实现联网上传下载文件也非常简单，后面我们会讲到相关的知识。

现在有一个新的需求，我们希望知道调用`download`和`upload`函数上传下载文件到底用了多少时间，这应该如何实现呢？相信很多小伙伴已经想到了，我们可以在函数开始执行的时候记录一个时间，在函数调用结束后记录一个时间，两个时间相减就可以计算出下载或上传的时间，代码如下所示。

```python
start = time.time()
download('MySQL从删库到跑路.avi')
end = time.time()
print(f'花费时间: {end - start:.2f}秒')
start = time.time()
upload('Python从入门到住院.pdf')
end = time.time()
print(f'花费时间: {end - start:.2f}秒')
```

通过上面的代码，我们可以在下载和上传文件时记录下耗费的时间，但不知道大家是否注意到，上面记录时间、计算和显示执行时间的代码都是重复代码。有编程经验的人都知道，**重复的代码是万恶之源**，那么有没有办法在不写重复代码的前提下，用一种简单优雅的方式记录下函数的执行时间呢？在 Python 语言中，装饰器就是解决这类问题的最佳选择。通过装饰器语法，我们可以把跟原来的业务（上传和下载）没有关系计时功能的代码封装到一个函数中，如果`upload`和`download`函数需要记录时间，我们直接把装饰器作用到这两个函数上即可。既然上面提到了，装饰器是一个高阶函数，它的参数和返回值都是函数，我们将记录时间的装饰器姑且命名为`record_time`，那么它的整体结构应该如下面的代码所示。

```python
def record_time(func):
    
    def wrapper(*args, **kwargs):
        
        result = func(*args, **kwargs)
        
        return result
    
    return wrapper
```

相信大家注意到了，`record_time`函数的参数`func`代表了一个被装饰的函数，函数里面定义的`wrapper`函数是带有装饰功能的函数，它会执行被装饰的函数`func`，它还需要返回在最后产生函数执行的返回值。不知大家是否留意到，上面的代码我在第4行和第6行留下了两个空行，这意味着我们可以这些地方添加代码来实现额外的功能。`record_time`函数最终会返回这个带有装饰功能的函数`wrapper`并通过它替代原函数`func`，当原函数`func`被`record_time`函数装饰后，我们调用它时其实调用的是`wrapper`函数，所以才获得了额外的能力。`wrapper`函数的参数比较特殊，由于我们要用`wrapper`替代原函数`func`，但是我们又不清楚原函数`func`会接受哪些参数，所以我们就通过可变参数和关键字参数照单全收，然后在调用`func`的时候，原封不动的全部给它。这里还要强调一下，Python 语言支持函数的嵌套定义，就像上面，我们可以在`record_time`函数中定义`wrapper`函数，这个操作在很多编程语言中并不被支持。

看懂这个结构后，我们就可以把记录时间的功能写到这个装饰器中，代码如下所示。

```python
import time


def record_time(func):

    def wrapper(*args, **kwargs):
        # 在执行被装饰的函数之前记录开始时间
        start = time.time()
        # 执行被装饰的函数并获取返回值
        result = func(*args, **kwargs)
        # 在执行被装饰的函数之后记录结束时间
        end = time.time()
        # 计算和显示被装饰函数的执行时间
        print(f'{func.__name__}执行时间: {end - start:.2f}秒')
        # 返回被装饰函数的返回值
        return result
    
    return wrapper
```

写装饰器虽然颇费周折，但是这是个一劳永逸的骚操作，将来再有记录函数执行时间的需求时，我们只需要添加上面的装饰器即可。使用上面的装饰器函数有两种方式，第一种方式就是直接调用装饰器函数，传入被装饰的函数并获得返回值，我们可以用这个返回值直接替代原来的函数，那么在调用时就已经获得了装饰器提供的额外的能力（记录执行时间），大家试试下面的代码就明白了。

```python
download = record_time(download)
upload = record_time(upload)
download('MySQL从删库到跑路.avi')
upload('Python从入门到住院.pdf')
```

在 Python 中，使用装饰器很有更为便捷的**语法糖**（编程语言中添加的某种语法，这种语法对语言的功能没有影响，但是使用更加方法，代码的可读性也更强，我们将其称之为“语法糖”或“糖衣语法”），可以用`@装饰器函数`将装饰器函数直接放在被装饰的函数上，效果跟上面的代码相同。我们把完整的代码为大家罗列出来，大家可以再看看我们是如何定义和使用装饰器的。

```python
import random
import time


def record_time(func):

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f'{func.__name__}执行时间: {end - start:.2f}秒')
        return result

    return wrapper


@record_time
def download(filename):
    print(f'开始下载{filename}.')
    time.sleep(random.random() * 6)
    print(f'{filename}下载完成.')


@record_time
def upload(filename):
    print(f'开始上传{filename}.')
    time.sleep(random.random() * 8)
    print(f'{filename}上传完成.')


download('MySQL从删库到跑路.avi')
upload('Python从入门到住院.pdf')
```

上面的代码，我们通过装饰器语法糖为`download`和`upload`函数添加了装饰器，被装饰后的`download`和`upload`函数其实就是我们在装饰器中返回的`wrapper`函数，调用它们其实就是在调用`wrapper`函数，所以才有了记录函数执行时间的功能。

如果在代码的某些地方，我们想去掉装饰器的作用执行原函数，那么在定义装饰器函数的时候，需要做一点点额外的工作。Python 标准库`functools`模块的`wraps`函数也是一个装饰器，我们将它放在`wrapper`函数上，这个装饰器可以帮我们保留被装饰之前的函数，这样在需要取消装饰器时，可以通过被装饰函数的`__wrapped__`属性获得被装饰之前的函数。

```python
import random
import time

from functools import wraps


def record_time(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f'{func.__name__}执行时间: {end - start:.2f}秒')
        return result

    return wrapper


@record_time
def download(filename):
    print(f'开始下载{filename}.')
    time.sleep(random.random() * 6)
    print(f'{filename}下载完成.')


@record_time
def upload(filename):
    print(f'开始上传{filename}.')
    time.sleep(random.random() * 8)
    print(f'{filename}上传完成.')


# 调用装饰后的函数会记录执行时间
download('MySQL从删库到跑路.avi')
upload('Python从入门到住院.pdf')
# 取消装饰器的作用不记录执行时间
download.__wrapped__('MySQL必知必会.pdf')
upload.__wrapped__('Python从新手到大师.pdf')
```

**装饰器函数本身也可以参数化**，简单的说就是装饰器也是可以通过调用者传入的参数来进行定制的，这个知识点我们在后面用到的时候再为大家讲解。

### 递归调用

Python 中允许函数嵌套定义，也允许函数之间相互调用，而且一个函数还可以直接或间接的调用自身。函数自己调用自己称为递归调用，那么递归调用有什么用处呢？现实中，有很多问题的定义本身就是一个递归定义，例如我们之前讲到的阶乘，非负整数`N`的阶乘是`N`乘以`N-1`的阶乘，即 $\small{N! = N \times (N-1)!}$ ，定义的左边和右边都出现了阶乘的概念，所以这是一个递归定义。既然如此，我们可以使用递归调用的方式来写一个求阶乘的函数，代码如下所示。

```python
def fac(num):
    if num in (0, 1):
        return 1
    return num * fac(num - 1)
```

上面的代码中，`fac`函数中又调用了`fac`函数，这就是所谓的递归调用。代码第2行的`if`条件叫做递归的收敛条件，简单的说就是什么时候要结束函数的递归调用，在计算阶乘时，如果计算到`0`或`1`的阶乘，就停止递归调用，直接返回`1`；代码第4行的`num * fac(num - 1)`是递归公式，也就是阶乘的递归定义。下面，我们简单的分析下，如果用`fac(5)`计算`5`的阶乘，整个过程会是怎样的。

```python
# 递归调用函数入栈
# 5 * fac(4)
# 5 * (4 * fac(3))
# 5 * (4 * (3 * fac(2)))
# 5 * (4 * (3 * (2 * fac(1))))
# 停止递归函数出栈
# 5 * (4 * (3 * (2 * 1)))
# 5 * (4 * (3 * 2))
# 5 * (4 * 6)
# 5 * 24
# 120
print(fac(5))    # 120
```

注意，函数调用会通过内存中称为“栈”（stack）的数据结构来保存当前代码的执行现场，函数调用结束后会通过这个栈结构恢复之前的执行现场。栈是一种先进后出的数据结构，这也就意味着最早入栈的函数最后才会返回，而最后入栈的函数会最先返回。例如调用一个名为`a`的函数，函数`a`的执行体中又调用了函数`b`，函数`b`的执行体中又调用了函数`c`，那么最先入栈的函数是`a`，最先出栈的函数是`c`。每进入一个函数调用，栈就会增加一层栈帧（stack frame），栈帧就是我们刚才提到的保存当前代码执行现场的结构；每当函数调用结束后，栈就会减少一层栈帧。通常，内存中的栈空间很小，因此递归调用的次数如果太多，会导致栈溢出（stack overflow），所以**递归调用一定要确保能够快速收敛**。我们可以尝试执行`fac(5000)`，看看是不是会提示`RecursionError`错误，错误消息为：`maximum recursion depth exceeded in comparison`（超出最大递归深度），其实就是发生了栈溢出。

如果我们使用官方的 Python 解释器（CPython），默认将函数调用的栈结构最大深度设置为`1000`层。如果超出这个深度，就会发生上面说的`RecursionError`。当然，我们可以使用`sys`模块的`setrecursionlimit`函数来改变递归调用的最大深度，但是我们不建议这样做，因为让递归快速收敛才是我们应该做的事情，否则就应该考虑使用循环递推而不是递归。

再举一个之前讲过的生成斐波那契数列的例子，因为斐波那契数列前两个数都是`1`，从第三个数开始，每个数是前两个数相加的和，可以记为`f(n) = f(n - 1) + f(n - 2)`，很显然这又是一个递归的定义，所以我们可以用下面的递归调用函数来计算第​`n`个斐波那契数。

```python
def fib1(n):
    if n in (1, 2):
        return 1
    return fib1(n - 1) + fib1(n - 2)


for i in range(1, 21):
    print(fib1(i))
```

需要提醒大家，上面计算斐波那契数的代码虽然看起来非常简单明了，但执行性能是比较糟糕的。大家可以试一试，把上面代码`for`循环中`range`函数的第二个参数修改为`51`，即输出前50个斐波那契数，看看需要多长时间，也欢迎大家在评论区留下你的代码执行时间。至于为什么这么慢，大家可以自己思考一下原因。很显然，直接使用循环递推的方式获得斐波那契数列是更好的选择，代码如下所示。

```python
def fib2(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
```

除此以外，我们还可以使用 Python 标准库中`functools`模块的`lru_cache`函数来优化上面的递归代码。`lru_cache`函数是一个装饰器函数，我们将其置于上面的函数`fib1`之上，它可以缓存该函数的执行结果从而避免在递归调用的过程中产生大量的重复运算，这样代码的执行性能就有“飞一般”的提升。大家可以尝试输出前50个斐波那契数，看看加上装饰器以后代码需要执行多长时间，评论区见！

```python
from functools import lru_cache


@lru_cache()
def fib1(n):
    if n in (1, 2):
        return 1
    return fib1(n - 1) + fib1(n - 2)


for i in range(1, 51):
    print(i, fib1(i))
```

> **提示**：`lru_cache`函数是一个带参数的装饰器，所以上面第4行代码使用装饰器语法糖时，`lru_cache`后面要跟上圆括号。`lru_cache`函数有一个非常重要的参数叫`maxsize`，它可以用来定义缓存空间的大小，默认值是128。

###  总结

装饰器是 Python 语言中的特色语法，**可以通过装饰器来增强现有的函数**，这是一种非常有用的编程技巧。另一方面，通过函数递归调用，可以在代码层面将一些复杂的问题简单化，但是**递归调用一定要注意收敛条件和递归公式**，找到递归公式才有机会使用递归调用，而收敛条件则确保了递归调用能停下来。函数调用通过内存中的栈空间来保存现场和恢复现场，栈空间通常都很小，所以**递归如果不能迅速收敛，很可能会引发栈溢出错误，从而导致程序的崩溃**。

---

