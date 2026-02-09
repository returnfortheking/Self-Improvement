# Python 生成器和迭代器 - Python-100-Days 完整补充材料

## 📑 目录
- [生成器函数](#生成器函数)
- [生成器表达式](#生成器表达式)
- [迭代器协议](#迭代器协议)
- [实际应用](#实际应用)

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

