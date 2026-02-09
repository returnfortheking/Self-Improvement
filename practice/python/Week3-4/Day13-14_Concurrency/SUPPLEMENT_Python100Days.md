# Python-100-Days 补充材料

**主题**: Python并发编程

**描述**: 多线程、多进程、协程、并发爬虫应用

**来源**: [jackfrued/Python-100-Days](https://github.com/jackfrued/Python-100-Days)

**映射Day**: Day61-65

**生成时间**: 2026-02-09 19:57:53

---

## 📚 概述

本材料从 **Python-100-Days** 仓库中提取与 "Python并发编程" 相关的内容，作为学习计划的补充。

**包含文件数**: 4
**代码示例数**: 0

---

## 🔑 关键知识点

- ## Python中的并发编程-1 *(来源: Day61-65/63.Python中的并发编程-1.md)*
- ### 线程和进程 *(来源: Day61-65/63.Python中的并发编程-1.md)*
- ### 多线程编程 *(来源: Day61-65/63.Python中的并发编程-1.md)*
- #### 使用 Thread 类创建线程对象 *(来源: Day61-65/63.Python中的并发编程-1.md)*
- #### 继承 Thread 类自定义线程 *(来源: Day61-65/63.Python中的并发编程-1.md)*
- #### 使用线程池 *(来源: Day61-65/63.Python中的并发编程-1.md)*
- ### 守护线程 *(来源: Day61-65/63.Python中的并发编程-1.md)*
- ### 资源竞争 *(来源: Day61-65/63.Python中的并发编程-1.md)*
- ### GIL问题 *(来源: Day61-65/63.Python中的并发编程-1.md)*
- ## Python中的并发编程-2 *(来源: Day61-65/63.Python中的并发编程-2.md)*
- ###创建进程 *(来源: Day61-65/63.Python中的并发编程-2.md)*
- ### 多进程和多线程的比较 *(来源: Day61-65/63.Python中的并发编程-2.md)*
- ### 进程间通信 *(来源: Day61-65/63.Python中的并发编程-2.md)*
- ###  总结 *(来源: Day61-65/63.Python中的并发编程-2.md)*
- ## Python中的并发编程-3 *(来源: Day61-65/63.Python中的并发编程-3.md)*
- ### 基本概念 *(来源: Day61-65/63.Python中的并发编程-3.md)*
- #### 阻塞 *(来源: Day61-65/63.Python中的并发编程-3.md)*
- #### 非阻塞 *(来源: Day61-65/63.Python中的并发编程-3.md)*
- #### 同步 *(来源: Day61-65/63.Python中的并发编程-3.md)*
- #### 异步 *(来源: Day61-65/63.Python中的并发编程-3.md)*
- ### 生成器和协程 *(来源: Day61-65/63.Python中的并发编程-3.md)*
- ### 异步函数 *(来源: Day61-65/63.Python中的并发编程-3.md)*
- ### aiohttp库 *(来源: Day61-65/63.Python中的并发编程-3.md)*
- ## 并发编程在爬虫中的应用 *(来源: Day61-65/63.并发编程在爬虫中的应用.md)*
- ### 单线程版本 *(来源: Day61-65/63.并发编程在爬虫中的应用.md)*
- ### 多线程版本 *(来源: Day61-65/63.并发编程在爬虫中的应用.md)*
- ### 异步I/O版本 *(来源: Day61-65/63.并发编程在爬虫中的应用.md)*
- ### 总结 *(来源: Day61-65/63.并发编程在爬虫中的应用.md)*


---

## 💻 代码示例

以下是从相关文件中提取的典型代码示例：



---

## 📖 完整内容来源

以下是包含完整内容的源文件：


### Day61-65/63.Python中的并发编程-1.md

**标题**: ## Python中的并发编程-1

<details>
<summary>点击查看完整内容</summary>

## Python中的并发编程-1

现如今，我们使用的计算机早已是多 CPU 或多核的计算机，而我们使用的操作系统基本都支持“多任务”，这使得我们可以同时运行多个程序，也可以将一个程序分解为若干个相对独立的子任务，让多个子任务“并行”或“并发”的执行，从而缩短程序的执行时间，同时也让用户获得更好的体验。因此当下，不管用什么编程语言进行开发，实现“并行”或“并发”编程已经成为了程序员的标配技能。为了讲述如何在 Python 程序中实现“并行”或“并发”，我们需要先了解两个重要的概念：进程和线程。

### 线程和进程

我们通过操作系统运行一个程序会创建出一个或多个进程，进程是具有一定独立功能的程序关于某个数据集合上的一次运行活动。简单的说，进程是操作系统分配存储空间的基本单位，每个进程都有自己的地址空间、数据栈以及其他用于跟踪进程执行的辅助数据；操作系统管理所有进程的执行，为它们合理的分配资源。一个进程可以通过 fork 或 spawn 的方式创建新的进程来执行其他的任务，不过新的进程也有自己独立的内存空间，因此两个进程如果要共享数据，必须通过进程间通信机制来实现，具体的方式包括管道、信号、套接字等。

一个进程还可以拥有多个执行线索，简单的说就是拥有多个可以获得 CPU 调度的执行单元，这就是所谓的线程。由于线程在同一个进程下，它们可以共享相同的上下文，因此相对于进程而言，线程间的信息共享和通信更加容易。当然在单核 CPU 系统中，多个线程不可能同时执行，因为在某个时刻只有一个线程能够获得 CPU，多个线程通过共享 CPU 执行时间的方式来达到并发的效果。

在程序中使用多线程技术通常都会带来不言而喻的好处，最主要的体现在提升程序的性能和改善用户体验，今天我们使用的软件几乎都用到了多线程技术，这一点可以利用系统自带的进程监控工具（如 macOS 中的“活动监视器”、Windows 中的“任务管理器”）来证实，如下图所示。

<img src="res/20210822094243.png" width="80%">

这里，我们还需要跟大家再次强调两个概念：**并发**（concurrency）和**并行**（parallel）。**并发**通常是指同一时刻只能有一条指令执行，但是多个线程对应的指令被快速轮换地执行。比如一个处理器，它先执行线程 A 的指令一段时间，再执行线程 B 的指令一段时间，再切回到线程 A 执行一段时间。由于处理器执行指令的速度和切换的速度极快，人们完全感知不到计算机在这个过程中有多个线程切换上下文执行的操作，这就使得宏观上看起来多个线程在同时运行，但微观上其实只有一个线程在执行。**并行**是指同一时刻，有多条指令在多个处理器上同时执行，并行必须要依赖于多个处理器，不论是从宏观上还是微观上，多个线程可以在同一时刻一起执行的。很多时候，我们并不用严格区分并发和并行两个词，所以我们有时候也把 Python 中的多线程、多进程以及异步 I/O 都视为实现并发编程的手段，但实际上前面两者也可以实现并行编程，当然这里还有一个全局解释器锁（GIL）的问题，我们稍后讨论。

### 多线程编程

Python 标准库中`threading`模块的`Thread`类可以帮助我们非常轻松的实现多线程编程。我们用一个联网下载文件的例子来对比使用多线程和不使用多线程到底有什么区别，代码如下所示。

不使用多线程的下载。

```Python
import random
import time


def download(*, filename):
    start = time.time()
    print(f'开始下载 {filename}.')
    time.sleep(random.randint(3, 6))
    print(f'{filename} 下载完成.')
    end = time.time()
    print(f'下载耗时: {end - start:.3f}秒.')


def main():
    start = time.time()
    download(filename='Python从入门到住院.pdf')
    download(filename='MySQL从删库到跑路.avi')
    download(filename='Linux从精通到放弃.mp4')
    end = time.time()
    print(f'总耗时: {end - start:.3f}秒.')


if __name__ == '__main__':
    main()
```

> **说明**：上面的代码并没有真正实现联网下载的功能，而是通过`time.sleep()`休眠一段时间来模拟下载文件需要一些时间上的开销，跟实际下载的状况比较类似。

运行上面的代码，可以得到如下所示的运行结果。可以看出，当我们的程序只有一个工作线程时，每个下载任务都需要等待上一个下载任务执行结束才能开始，所以程序执行的总耗时是三个下载任务各自执行时间的总和。

```
开始下载Python从入门到住院.pdf.
Python从入门到住院.pdf下载完成.
下载耗时: 3.005秒.
开始下载MySQL从删库到跑路.avi.
MySQL从删库到跑路.avi下载完成.
下载耗时: 5.006秒.
开始下载Linux从精通到放弃.mp4.
Linux从精通到放弃.mp3下载完成.
下载耗时: 6.007秒.
总耗时: 14.018秒.
```

事实上，上面的三个下载任务之间并没有逻辑上的因果关系，三者是可以“并发”的，下一个下载任务没有必要等待上一个下载任务结束，为此，我们可以使用多线程编程来改写上面的代码。

```Python
import random
import time
from threading import Thread


def download(*, filename):
    start = time.time()
    print(f'开始下载 {filename}.')
    time.sleep(random.randint(3, 6))
    print(f'{filename} 下载完成.')
    end = time.time()
    print(f'下载耗时: {end - start:.3f}秒.')


def main():
    threads = [
        Thread(target=download, kwargs={'filename': 'Python从入门到住院.pdf'}),
        Thread(target=download, kwargs={'filename': 'MySQL从删库到跑路.avi'}),
        Thread(target=download, kwargs={'filename': 'Linux从精通到放弃.mp4'})
    ]
    start = time.time()
    # 启动三个线程
    fo
...(内容过长，已截断)

</details>

---


### Day61-65/63.Python中的并发编程-2.md

**标题**: ## Python中的并发编程-2

<details>
<summary>点击查看完整内容</summary>

## Python中的并发编程-2

在上一课中我们说过，由于 GIL 的存在，CPython 中的多线程并不能发挥 CPU 的多核优势，如果希望突破 GIL 的限制，可以考虑使用多进程。对于多进程的程序，每个进程都有一个属于自己的 GIL，所以多进程不会受到 GIL 的影响。那么，我们应该如何在 Python 程序中创建和使用多进程呢？

###创建进程

在 Python 中可以基于`Process`类来创建进程，虽然进程和线程有着本质的差别，但是`Process`类和`Thread`类的用法却非常类似。在使用`Process`类的构造器创建对象时，也是通过`target`参数传入一个函数来指定进程要执行的代码，而`args`和`kwargs`参数可以指定该函数使用的参数值。

```Python
from multiprocessing import Process, current_process
from time import sleep


def sub_task(content, nums):
    # 通过current_process函数获取当前进程对象
    # 通过进程对象的pid和name属性获取进程的ID号和名字
    print(f'PID: {current_process().pid}')
    print(f'Name: {current_process().name}')
    # 通过下面的输出不难发现，每个进程都有自己的nums列表，进程之间本就不共享内存
    # 在创建子进程时复制了父进程的数据结构，三个进程从列表中pop(0)得到的值都是20
    counter, total = 0, nums.pop(0)
    print(f'Loop count: {total}')
    sleep(0.5)
    while counter < total:
        counter += 1
        print(f'{counter}: {content}')
        sleep(0.01)


def main():
    nums = [20, 30, 40]
    # 创建并启动进程来执行指定的函数
    Process(target=sub_task, args=('Ping', nums)).start()
    Process(target=sub_task, args=('Pong', nums)).start()
    # 在主进程中执行sub_task函数
    sub_task('Good', nums)


if __name__ == '__main__':
    main()
```

> **说明**：上面的代码通过`current_process`函数获取当前进程对象，再通过进程对象的`pid`属性获取进程ID。在 Python 中，使用`os`模块的`getpid`函数也可以达到同样的效果。

如果愿意，也可以使用`os`模块的`fork`函数来创建进程，调用该函数时，操作系统自动把当前进程（父进程）复制一份（子进程），父进程的`fork`函数会返回子进程的ID，而子进程中的`fork`函数会返回`0`，也就是说这个函数调用一次会在父进程和子进程中得到两个不同的返回值。需要注意的是，Windows 系统并不支持`fork`函数，如果你使用的是 Linux 或 macOS 系统，可以试试下面的代码。

```Python
import os

print(f'PID: {os.getpid()}')
pid = os.fork()
if pid == 0:
    print(f'子进程 - PID: {os.getpid()}')
    print('Todo: 在子进程中执行的代码')
else:
    print(f'父进程 - PID: {os.getpid()}')
    print('Todo: 在父进程中执行的代码')
```

简而言之，我们还是推荐大家通过直接使用`Process`类、继承`Process`类和使用进程池（`ProcessPoolExecutor`）这三种方式来创建和使用多进程，这三种方式不同于上面的`fork`函数，能够保证代码的兼容性和可移植性。具体的做法跟之前讲过的创建和使用多线程的方式比较接近，此处不再进行赘述。

### 多进程和多线程的比较

对于爬虫这类 I/O 密集型任务来说，使用多进程并没有什么优势；但是对于计算密集型任务来说，多进程相比多线程，在效率上会有显著的提升，我们可以通过下面的代码来加以证明。下面的代码会通过多线程和多进程两种方式来判断一组大整数是不是质数，很显然这是一个计算密集型任务，我们将任务分别放到多个线程和多个进程中来加速代码的执行，让我们看看多线程和多进程的代码具体表现有何不同。

我们先实现一个多线程的版本，代码如下所示。

```Python
import concurrent.futures

PRIMES = [
    1116281,
    1297337,
    104395303,
    472882027,
    533000389,
    817504243,
    982451653,
    112272535095293,
    112582705942171,
    112272535095293,
    115280095190773,
    115797848077099,
    1099726899285419
] * 5


def is_prime(n):
    """判断素数"""
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return n != 1


def main():
    """主函数"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        for number, prime in zip(PRIMES, executor.map(is_prime, PRIMES)):
            print('%d is prime: %s' % (number, prime))


if __name__ == '__main__':
    main()
```

假设上面的代码保存在名为`example.py`的文件中，在 Linux 或 macOS 系统上，可以使用`time python example.py`命令执行程序并获得操作系统关于执行时间的统计，在我的 macOS 上，某次的运行结果的最后一行输出如下所示。

```
python example09.py  38.69s user 1.01s system 101% cpu 
...(内容过长，已截断)

</details>

---


### Day61-65/63.Python中的并发编程-3.md

**标题**: ## Python中的并发编程-3

<details>
<summary>点击查看完整内容</summary>

## Python中的并发编程-3

爬虫是典型的 I/O 密集型任务，I/O 密集型任务的特点就是程序会经常性的因为 I/O 操作而进入阻塞状态，比如我们之前使用`requests`获取页面代码或二进制内容，发出一个请求之后，程序必须要等待网站返回响应之后才能继续运行，如果目标网站不是很给力或者网络状况不是很理想，那么等待响应的时间可能会很久，而在这个过程中整个程序是一直阻塞在那里，没有做任何的事情。通过前面的课程，我们已经知道了可以通过多线程的方式为爬虫提速，使用多线程的本质就是，当一个线程阻塞的时候，程序还有其他的线程可以继续运转，因此整个程序就不会在阻塞和等待中浪费了大量的时间。

事实上，还有一种非常适合 I/O 密集型任务的并发编程方式，我们称之为异步编程，你也可以将它称为异步 I/O。这种方式并不需要启动多个线程或多个进程来实现并发，它是通过多个子程序相互协作的方式来提升 CPU 的利用率，解决了 I/O 密集型任务 CPU  利用率很低的问题，我一般将这种方式称为“协作式并发”。这里，我不打算探讨操作系统的各种 I/O 模式，因为这对很多读者来说都太过抽象；但是我们得先抛出两组概念给大家，一组叫做“阻塞”和“非阻塞”，一组叫做“同步”和“异步”。

### 基本概念

#### 阻塞

阻塞状态指程序未得到所需计算资源时被挂起的状态。程序在等待某个操作完成期间，自身无法继续处理其他的事情，则称该程序在该操作上是阻塞的。阻塞随时都可能发生，最典型的就是 I/O 中断（包括网络 I/O 、磁盘 I/O 、用户输入等）、休眠操作、等待某个线程执行结束，甚至包括在 CPU 切换上下文时，程序都无法真正的执行，这就是所谓的阻塞。

#### 非阻塞

程序在等待某操作过程中，自身不被阻塞，可以继续处理其他的事情，则称该程序在该操作上是非阻塞的。非阻塞并不是在任何程序级别、任何情况下都可以存在的。仅当程序封装的级别可以囊括独立的子程序单元时，它才可能存在非阻塞状态。显然，某个操作的阻塞可能会导程序耗时以及效率低下，所以我们会希望把它变成非阻塞的。

#### 同步

不同程序单元为了完成某个任务，在执行过程中需靠某种通信方式以协调一致，我们称这些程序单元是同步执行的。例如前面讲过的给银行账户存钱的操作，我们在代码中使用了“锁”作为通信信号，让多个存钱操作强制排队顺序执行，这就是所谓的同步。

#### 异步

不同程序单元在执行过程中无需通信协调，也能够完成一个任务，这种方式我们就称之为异步。例如，使用爬虫下载页面时，调度程序调用下载程序后，即可调度其他任务，而无需与该下载任务保持通信以协调行为。不同网页的下载、保存等操作都是不相关的，也无需相互通知协调。很显然，异步操作的完成时刻和先后顺序并不能确定。

很多人都不太能准确的把握这几个概念，这里我们简单的总结一下，同步与异步的关注点是**消息通信机制**，最终表现出来的是“有序”和“无序”的区别；阻塞和非阻塞的关注点是**程序在等待消息时状态**，最终表现出来的是程序在等待时能不能做点别的。如果想深入理解这些内容，推荐大家阅读经典著作[《UNIX网络编程》](https://item.jd.com/11880047.html)，这本书非常的赞。

### 生成器和协程

前面我们说过，异步编程是一种“协作式并发”，即通过多个子程序相互协作的方式提升 CPU 的利用率，从而减少程序在阻塞和等待中浪费的时间，最终达到并发的效果。我们可以将多个相互协作的子程序称为“协程”，它是实现异步编程的关键。在介绍协程之前，我们先通过下面的代码，看看什么是生成器。

```Python
def fib(max_count):
    a, b = 0, 1
    for _ in range(max_count):
        a, b = b, a + b
        yield a
```

上面我们编写了一个生成斐波那契数列的生成器，调用上面的`fib`函数并不是执行该函数获得返回值，因为`fib`函数中有一个特殊的关键字`yield`。这个关键字使得`fib`函数跟普通的函数有些区别，调用该函数会得到一个生成器对象，我们可以通过下面的代码来验证这一点。

```Python
gen_obj = fib(20)
print(gen_obj)
```

输出：

```
<generator object fib at 0x106daee40>
```

我们可以使用内置函数`next`从生成器对象中获取斐波那契数列的值，也可以通过`for-in`循环对生成器能够提供的值进行遍历，代码如下所示。

```Python
for value in gen_obj:
    print(value)
```

生成器经过预激活，就是一个协程，它可以跟其他子程序协作。

```Python
def calc_average():
    total, counter = 0, 0
    avg_value = None
    while True:
        curr_value = yield avg_value
        total += curr_value
        counter += 1
        avg_value = total / counter


def main():
    obj = calc_average()
    # 生成器预激活
    obj.send(None)
    for _ in range(5):
        print(obj.send(float(input())))


if __name__ == '__main__':
    main()
```

上面的`main`函数首先通过生成器对象的`send`方法发送一个`None`值来将其激活为协程，也可以通过`next(obj)`达到同样的效果。接下来，协程对象会接收`main`函数发送的数据并产出（`yield`）数据的平均值。通过上面的例子，不知道大家是否看出两段子程序是怎么“协作”的。

### 异步函数

Python 3.5版本中，引入了两个非常有意思的元素，一个叫`async`，一个叫`await`，它们在Python 3.7版本中成为了正式的关键字。通过这两个关键字，可以简化协程代码的编写，可以用更为简单的方式让多个子程序很好的协作起来。我们通过一个例子来加以说明，请大家先看看下面的代码。

```Python
import time


def display(num):
    time.sleep(1)
    print(num)


def main():
    start = time.time()
    for i in range(1, 10):
        display(i)
    end = time.time()
    print(f'{end - start:.3f}秒')


if __name__ == '__main__':
    main()
...(内容过长，已截断)

</details>

---


### Day61-65/63.并发编程在爬虫中的应用.md

**标题**: ## 并发编程在爬虫中的应用

<details>
<summary>点击查看完整内容</summary>

## 并发编程在爬虫中的应用

之前的课程，我们已经为大家介绍了 Python 中的多线程、多进程和异步编程，通过这三种手段，我们可以实现并发或并行编程，这一方面可以加速代码的执行，另一方面也可以带来更好的用户体验。爬虫程序是典型的 I/O 密集型任务，对于 I/O 密集型任务来说，多线程和异步 I/O 都是很好的选择，因为当程序的某个部分因 I/O 操作阻塞时，程序的其他部分仍然可以运转，这样我们不用在等待和阻塞中浪费大量的时间。下面我们以爬取“[360图片](https://image.so.com/)”网站的图片并保存到本地为例，为大家分别展示使用单线程、多线程和异步 I/O 编程的爬虫程序有什么区别，同时也对它们的执行效率进行简单的对比。

“360图片”网站的页面使用了 [Ajax](https://developer.mozilla.org/zh-CN/docs/Web/Guide/AJAX) 技术，这是很多网站都会使用的一种异步加载数据和局部刷新页面的技术。简单的说，页面上的图片都是通过 JavaScript 代码异步获取 JSON 数据并动态渲染生成的，而且整个页面还使用了瀑布式加载（一边向下滚动，一边加载更多的图片）。我们在浏览器的“开发者工具”中可以找到提供动态内容的数据接口，如下图所示，我们需要的图片信息就在服务器返回的 JSON 数据中。

<img src="res/20211205221352.png" style="zoom:40%;">

例如，要获取“美女”频道的图片，我们可以请求如下所示的URL，其中参数`ch`表示请求的频道，`=`后面的参数值`beauty`就代表了“美女”频道，参数`sn`相当于是页码，`0`表示第一页（共`30`张图片），`30`表示第二页，`60`表示第三页，以此类推。

```
https://image.so.com/zjl?ch=beauty&sn=0
```

### 单线程版本

通过上面的 URL 下载“美女”频道共`90`张图片。

```Python
"""
example04.py - 单线程版本爬虫
"""
import os

import requests


def download_picture(url):
    filename = url[url.rfind('/') + 1:]
    resp = requests.get(url)
    if resp.status_code == 200:
        with open(f'images/beauty/{filename}', 'wb') as file:
            file.write(resp.content)


def main():
    if not os.path.exists('images/beauty'):
        os.makedirs('images/beauty')
    for page in range(3):
        resp = requests.get(f'https://image.so.com/zjl?ch=beauty&sn={page * 30}')
        if resp.status_code == 200:
            pic_dict_list = resp.json()['list']
            for pic_dict in pic_dict_list:
                download_picture(pic_dict['qhimg_url'])

if __name__ == '__main__':
    main()
```

在 macOS 或 Linux 系统上，我们可以使用`time`命令来了解上面代码的执行时间以及 CPU 的利用率，如下所示。

```Bash
time python3 example04.py 
```

下面是单线程爬虫代码在我的电脑上执行的结果。

```
python3 example04.py  2.36s user 0.39s system 12% cpu 21.578 total
```

这里我们只需要关注代码的总耗时为`21.578`秒，CPU 利用率为`12%`。

### 多线程版本

我们使用之前讲到过的线程池技术，将上面的代码修改为多线程版本。

```Python
"""
example05.py - 多线程版本爬虫
"""
import os
from concurrent.futures import ThreadPoolExecutor

import requests


def download_picture(url):
    filename = url[url.rfind('/') + 1:]
    resp = requests.get(url)
    if resp.status_code == 200:
        with open(f'images/beauty/{filename}', 'wb') as file:
            file.write(resp.content)


def main():
    if not os.path.exists('images/beauty'):
        os.makedirs('images/beauty')
    with ThreadPoolExecutor(max_workers=16) as pool:
        for page in range(3):
            resp = requests.get(f'https://image.so.com/zjl?ch=beauty&sn={page * 30}')
            if resp.status_code == 200:
                pic_dict_list = resp.json()['list']
                for pic_dict in pic_dict_list:
                    pool.submit(download_picture, pic_dict['qhimg_url'])


if __name__ == '__main__':
    main()
```

执行如下所示的命令。

```Bash
time python3 example05.py
```

代码的执行结果如下所示：

```
python3 example05.py  2.65s user 0.40s system 95% cpu 3.193 total
```

### 异步I/O版本

我们使用`aiohttp`将上面的代码修改为异步 I/O 的版本。为了以异步 I/O 的方式实现网络资源的获取和写文件操作，我们首先得安装三方库`aiohttp`和`aiofile`，
...(内容过长，已截断)

</details>

---


---

## 🎯 学习建议

1. **先学基础**: 确保掌握 Python 基础语法后再学习本主题
2. **动手实践**: 运行上述代码示例，理解每个示例的输出
3. **对比学习**: 将本材料与现有学习材料对比，查漏补缺
4. **实战应用**: 尝试使用所学知识完成小项目

---

## 📚 扩展阅读

- [Python-100-Days GitHub](https://github.com/jackfrued/Python-100-Days)
- [Python 官方文档](https://docs.python.org/zh-cn/3/)

---

*本材料由 doc-sync skill 自动生成，最后更新: 2026-02-09 19:57:53*
