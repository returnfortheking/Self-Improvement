# Day 11-12: 异步编程

## 一、异步编程基础

### 1.1 为什么需要异步编程

传统的同步编程模式中，I/O操作会阻塞程序执行，导致资源浪费。异步编程允许程序在等待I/O时执行其他任务，提高效率。

**同步 vs 异步**：
```python
# 同步：阻塞等待
def sync_request():
    response = requests.get(url)  # 阻塞
    return response

# 异步：非阻塞
async def async_request():
    response = await aiohttp.get(url)  # 不阻塞
    return response
```

### 1.2 异步编程的优势

1. **高并发**：单线程处理大量I/O操作
2. **资源高效**：避免线程/进程切换开销
3. **响应快速**：I/O等待时处理其他任务
4. **适合场景**：网络请求、文件操作、数据库访问

### 1.3 asyncio模块

Python 3.4+引入的异步I/O库，提供事件循环、协程、Future等组件。

```python
import asyncio

async def main():
    print("Hello")
    await asyncio.sleep(1)
    print("World")

asyncio.run(main())
```

## 二、协程（Coroutines）

### 2.1 async/await语法

**async def**：定义协程函数
**await**：等待协程完成

```python
import asyncio
import time

async def say_after(delay, what):
    await asyncio.sleep(delay)
    print(what)

async def main():
    print(f"started at {time.strftime('%X')}")
    await say_after(1, 'hello')
    await say_after(2, 'world')
    print(f"finished at {time.strftime('%X')}")

asyncio.run(main())
```

### 2.2 创建协程

```python
# 方式1: 直接调用
async def coro():
    return "result"

# 方式2: 使用asyncio.create_task()
task = asyncio.create_task(coro())

# 方式3: 使用asyncio.gather()
results = await asyncio.gather(coro1(), coro2())
```

### 2.3 并发执行

使用`asyncio.gather()`或`asyncio.create_task()`实现并发：

```python
import asyncio

async def fetch_data(id):
    await asyncio.sleep(1)
    return f"Data {id}"

async def main():
    # 并发执行多个协程
    results = await asyncio.gather(
        fetch_data(1),
        fetch_data(2),
        fetch_data(3)
    )
    print(results)

asyncio.run(main())
```

## 三、事件循环

### 3.1 事件循环基础

事件循环是异步编程的核心，负责调度和执行协程。

```python
import asyncio

async def main():
    print("Hello")

# 获取事件循环
loop = asyncio.get_event_loop()
loop.run_until_complete(main())

# 或使用更简洁的方式
asyncio.run(main())
```

### 3.2 事件循环方法

```python
# 创建任务
task = loop.create_task(coro())

# 运行直到完成
loop.run_until_complete(coro())

# 运行一段时间
loop.run_until_complete(asyncio.sleep(1))

# 关闭循环
loop.close()
```

## 四、并发与并行

### 4.1 并发执行协程

```python
import asyncio

async def coro(id):
    print(f"Start {id}")
    await asyncio.sleep(1)
    print(f"End {id}")
    return id

async def main():
    # 方法1: asyncio.gather
    results = await asyncio.gather(
        coro(1),
        coro(2),
        coro(3)
    )
    print(results)

    # 方法2: asyncio.create_task
    tasks = [asyncio.create_task(coro(i)) for i in range(3)]
    for task in tasks:
        await task

asyncio.run(main())
```

### 4.2 超时控制

```python
import asyncio

async def slow_operation():
    await asyncio.sleep(5)
    return "Done"

async def main():
    try:
        # 设置超时
        result = await asyncio.wait_for(
            slow_operation(),
            timeout=2.0
        )
    except asyncio.TimeoutError:
        print("Timeout!")

asyncio.run(main())
```

## 五、同步原语

### 5.1 Lock（锁）

```python
import asyncio

async def worker(lock, worker_id):
    async with lock:
        print(f"Worker {worker_id} is working")
        await asyncio.sleep(1)
        print(f"Worker {worker_id} finished")

async def main():
    lock = asyncio.Lock()
    await asyncio.gather(
        worker(lock, 1),
        worker(lock, 2)
    )

asyncio.run(main())
```

### 5.2 Event（事件）

```python
import asyncio

async def waiter(event):
    print("waiting for event...")
    await event.wait()
    print("event triggered!")

async def setter(event):
    await asyncio.sleep(2)
    event.set()
    print("event set!")

async def main():
    event = asyncio.Event()
    await asyncio.gather(
        waiter(event),
        setter(event)
    )

asyncio.run(main())
```

### 5.3 Queue（队列）

```python
import asyncio

async def producer(queue):
    for i in range(5):
        await queue.put(i)
        print(f"Produced: {i}")
        await asyncio.sleep(0.1)

async def consumer(queue):
    while True:
        item = await queue.get()
        print(f"Consumed: {item}")
        await asyncio.sleep(0.2)
        queue.task_done()

async def main():
    queue = asyncio.Queue()
    await asyncio.gather(
        producer(queue),
        consumer(queue)
    )

asyncio.run(main())
```

## 六、异步I/O操作

### 6.1 异步文件操作

使用`aiofiles`库进行异步文件操作：

```python
import aiofiles
import asyncio

async def read_file(filename):
    async with aiofiles.open(filename, 'r') as f:
        content = await f.read()
        return content

async def write_file(filename, content):
    async with aiofiles.open(filename, 'w') as f:
        await f.write(content)

async def main():
    content = await read_file('input.txt')
    await write_file('output.txt', content)

asyncio.run(main())
```

### 6.2 异步网络请求

使用`aiohttp`进行异步HTTP请求：

```python
import aiohttp
import asyncio

async def fetch(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

async def main():
    urls = [
        'https://api.github.com',
        'https://www.python.org',
    ]
    tasks = [fetch(url) for url in urls]
    results = await asyncio.gather(*tasks)
    print(results)

asyncio.run(main())
```

## 七、错误处理

### 7.1 异步异常处理

```python
import asyncio

async def failed_coro():
    raise ValueError("Something went wrong")

async def main():
    try:
        await failed_coro()
    except ValueError as e:
        print(f"Caught: {e}")

asyncio.run(main())
```

### 7.2 gather中的异常

```python
import asyncio

async def coro1():
    raise ValueError("Error in coro1")

async def coro2():
    await asyncio.sleep(1)
    return "Success"

async def main():
    results = await asyncio.gather(
        coro1(),
        coro2(),
        return_exceptions=True  # 不抛出异常，返回异常对象
    )
    for result in results:
        if isinstance(result, Exception):
            print(f"Error: {result}")
        else:
            print(f"Result: {result}")

asyncio.run(main())
```

## 八、最佳实践

### 8.1 避免阻塞操作

异步代码中避免使用同步阻塞操作：

```python
# 错误：使用同步sleep
import time
async def bad():
    time.sleep(1)  # 阻塞整个事件循环

# 正确：使用异步sleep
async def good():
    await asyncio.sleep(1)  # 不阻塞
```

### 8.2 使用create_task而非await

如果不需要立即使用结果，使用create_task实现并发：

```python
# 串行执行
async def bad():
    result1 = await fetch1()
    result2 = await fetch2()
    return result1, result2

# 并发执行
async def good():
    task1 = asyncio.create_task(fetch1())
    task2 = asyncio.create_task(fetch2())
    return await task1, await task2
```

### 8.3 合理使用超时

为外部请求设置超时，避免无限等待：

```python
async def safe_fetch(url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=5) as response:
                return await response.text()
    except asyncio.TimeoutError:
        return None
```

### 8.4 资源清理

使用async with和try-finally确保资源释放：

```python
async def fetch(url):
    session = aiohttp.ClientSession()
    try:
        async with session.get(url) as response:
            return await response.text()
    finally:
        await session.close()
```

或使用async with：

```python
async def fetch(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()
```

## 九、常见陷阱

### 9.1 混用同步和异步代码

避免在异步代码中使用同步阻塞操作。

### 9.2 忘记await

调用协程函数时必须使用await或create_task：

```python
async def main():
    # 错误：忘记await
    coro()  # 什么都不做

    # 正确：使用await
    await coro()
```

### 9.3 事件循环已运行

不能在已有事件循环中创建新循环：

```python
# 错误
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# 正确
asyncio.run(main())
```

## 十、性能优化

### 10.1 批量处理

将多个小操作合并为一个大操作：

```python
# 不好：多次小请求
for id in ids:
    data = await fetch_item(id)

# 好：一次批量请求
data = await fetch_items(ids)
```

### 10.2 连接池

使用连接池复用连接：

```python
async with aiohttp.ClientSession() as session:
    tasks = [fetch(session, url) for url in urls]
    await asyncio.gather(*tasks)
```

### 10.3 限制并发

使用Semaphore限制并发数量：

```python
semaphore = asyncio.Semaphore(10)

async def fetch(url):
    async with semaphore:
        return await fetch_data(url)
```

## 十一、总结

异步编程是Python中处理I/O密集型任务的利器：
- **async/await**：简洁的语法
- **事件循环**：高效的调度
- **并发执行**：充分利用等待时间
- **同步原语**：安全的并发控制

掌握异步编程可以编写高效的Web服务、爬虫、数据处理程序。

## 十二、学习建议

1. 理解事件循环机制
2. 掌握async/await语法
3. 学习asyncio模块
4. 避免阻塞操作
5. 合理使用同步原语
6. 使用第三方库（aiohttp、aiofiles）
7. 实战：Web服务、爬虫、数据处理
