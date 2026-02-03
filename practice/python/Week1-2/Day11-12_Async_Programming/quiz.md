# Day 11-12: 异步编程面试真题

## 阿里巴巴

### 1. 解释Python中的async、await和协程是什么

**参考答案**：

**协程（Coroutine）**：
- 协程是一种用户态的轻量级线程
- 由程序自己控制调度，非抢占式
- 适合I/O密集型任务

**async def**：
- 定义协程函数
- 调用协程函数返回协程对象，不会立即执行
- 需要通过await或事件循环来执行

**await**：
- 暂停当前协程的执行
- 等待另一个协程完成
- 期间事件循环可以执行其他协程

```python
import asyncio

async def fetch_data():
    await asyncio.sleep(1)  # 模拟I/O操作
    return "Data"

async def main():
    result = await fetch_data()  # 等待fetch_data完成
    print(result)

asyncio.run(main())
```

**工作原理**：
1. 事件循环获取就绪的协程
2. 执行协程直到遇到await
3. 切换到其他协程执行
4. I/O完成后恢复执行

**考察点**：异步编程基础、协程原理

---

### 2. asyncio的事件循环是什么？它如何工作？

**参考答案**：

事件循环是异步编程的核心调度机制，负责管理和执行所有协程。

**主要功能**：
1. 注册和执行协程
2. 管理I/O事件
3. 调度任务
4. 处理定时器

**工作流程**：
```python
import asyncio

async def task1():
    print("Task 1 started")
    await asyncio.sleep(1)
    print("Task 1 completed")

async def task2():
    print("Task 2 started")
    await asyncio.sleep(1)
    print("Task 2 completed")

async def main():
    # 创建任务并添加到事件循环
    task1_obj = asyncio.create_task(task1())
    task2_obj = asyncio.create_task(task2())

    # 等待任务完成
    await task1_obj
    await task2_obj

asyncio.run(main())
```

**关键方法**：
- `asyncio.run()`: 运行协程
- `loop.run_until_complete()`: 运行直到完成
- `loop.create_task()`: 创建任务
- `loop.run_forever()`: 持续运行

**考察点**：事件循环原理、任务调度

---

## 腾讯

### 3. 如何实现并发执行多个协程？比较gather和create_task

**参考答案**：

**方法1: asyncio.gather()**
```python
import asyncio

async def fetch(id):
    await asyncio.sleep(1)
    return f"Data {id}"

async def main():
    # 并发执行，等待全部完成
    results = await asyncio.gather(
        fetch(1),
        fetch(2),
        fetch(3)
    )
    print(results)  # ['Data 1', 'Data 2', 'Data 3']

asyncio.run(main())
```

**方法2: asyncio.create_task()**
```python
async def main():
    # 创建任务（立即开始执行）
    task1 = asyncio.create_task(fetch(1))
    task2 = asyncio.create_task(fetch(2))
    task3 = asyncio.create_task(fetch(3))

    # 等待所有任务完成
    results = await asyncio.gather(task1, task2, task3)
    print(results)
```

**对比**：
- `gather()`: 简洁，一次性创建多个任务
- `create_task()`: 更灵活，可以分阶段创建
- `gather()`可以设置`return_exceptions=True`处理异常
- `create_task()`返回Task对象，可以单独控制

**选择建议**：
- 简单场景使用`gather()`
- 需要细粒度控制使用`create_task()`

**考察点**：并发执行、API选择

---

### 4. 如何在异步代码中处理超时？

**参考答案**：

**方法1: asyncio.wait_for()**
```python
import asyncio

async def slow_operation():
    await asyncio.sleep(5)
    return "Done"

async def main():
    try:
        result = await asyncio.wait_for(
            slow_operation(),
            timeout=2.0
        )
        print(result)
    except asyncio.TimeoutError:
        print("Operation timed out!")

asyncio.run(main())
```

**方法2: asyncio.timeout()（Python 3.11+）**
```python
async def main():
    try:
        async with asyncio.timeout(2.0):
            result = await slow_operation()
            print(result)
    except TimeoutError:
        print("Timed out!")
```

**方法3: 手动超时**
```python
async def main():
    task = asyncio.create_task(slow_operation())
    try:
        result = await asyncio.wait_for(task, timeout=2.0)
        print(result)
    except asyncio.TimeoutError:
        task.cancel()
        print("Cancelled due to timeout")
```

**注意事项**：
1. 超时后任务需要手动取消
2. 使用try-finally确保资源释放
3. 考虑使用`asyncio.shield()`保护某些任务不被取消

**考察点**：超时控制、资源管理

---

## 字节跳动

### 5. 什么是事件循环的阻塞？如何避免？

**参考答案**：

**阻塞操作**：
在异步代码中使用同步阻塞操作会阻塞整个事件循环。

```python
# 错误示例
import time

async def bad():
    time.sleep(1)  # 阻塞整个事件循环！
    print("Done")

# 正确示例
async def good():
    await asyncio.sleep(1)  # 不阻塞
    print("Done")
```

**常见的阻塞操作**：
1. `time.sleep()` → 使用`asyncio.sleep()`
2. 同步文件I/O → 使用`aiofiles`
3. 同步网络请求 → 使用`aiohttp`
4. CPU密集计算 → 使用`loop.run_in_executor()`

**处理CPU密集任务**：
```python
import asyncio
from concurrent.futures import ProcessPoolExecutor

def cpu_bound_task(n):
    """CPU密集型任务"""
    return sum(i * i for i in range(n))

async def main():
    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor() as pool:
        result = await loop.run_in_executor(
            pool,
            cpu_bound_task,
            1000000
        )
        print(result)

asyncio.run(main())
```

**最佳实践**：
1. 所有I/O操作使用异步版本
2. CPU密集任务使用线程池/进程池
3. 定期检查代码是否有阻塞操作

**考察点**：阻塞操作、性能优化

---

### 6. asyncio中的同步原语有哪些？分别用于什么场景？

**参考答案**：

**1. Lock（锁）**
```python
lock = asyncio.Lock()

async def worker():
    async with lock:
        # 临界区
        print("Working")
```
用于：保护共享资源，同一时间只有一个协程访问

**2. Event（事件）**
```python
event = asyncio.Event()

async def waiter():
    await event.wait()
    print("Event triggered!")

async def setter():
    await asyncio.sleep(1)
    event.set()
```
用于：协程间通信，一个协程通知其他协程

**3. Condition（条件变量）**
```python
condition = asyncio.Condition()

async def producer():
    async with condition:
        # 生产数据
        condition.notify()

async def consumer():
    async with condition:
        await condition.wait()
        # 消费数据
```
用于：复杂的等待/通知模式

**4. Semaphore（信号量）**
```python
sem = asyncio.Semaphore(10)  # 限制并发数

async def worker():
    async with sem:
        # 最多10个协程同时执行
        pass
```
用于：限制并发数量

**5. Queue（队列）**
```python
queue = asyncio.Queue()

async def producer():
    await queue.put(item)

async def consumer():
    item = await queue.get()
    queue.task_done()
```
用于：生产者-消费者模式

**对比同步版本**：
- 都是线程/协程安全的
- API类似但使用`async with`和`await`
- 更轻量，适合协程环境

**考察点**：同步原语、并发控制

---

## 美团

### 7. 如何实现一个异步的生成器？

**参考答案**：

异步生成器使用`async def`和`yield`：

```python
import asyncio

async def async_range(n):
    """异步生成器"""
    for i in range(n):
        await asyncio.sleep(0.1)
        yield i

async def main():
    # 使用async for迭代
    async for value in async_range(5):
        print(value)

    # 或使用异步推导式
    result = [x async for x in async_range(5)]
    print(result)

asyncio.run(main())
```

**异步生成器迭代器**：
```python
class AsyncIterator:
    def __init__(self, n):
        self.n = n
        self.current = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.current >= self.n:
            raise StopAsyncIteration
        await asyncio.sleep(0.1)
        value = self.current
        self.current += 1
        return value

async def main():
    async for value in AsyncIterator(5):
        print(value)

asyncio.run(main())
```

**异步上下文管理器**：
```python
class AsyncResource:
    async def __aenter__(self):
        print("Acquiring resource")
        await asyncio.sleep(0.1)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("Releasing resource")
        await asyncio.sleep(0.1)

async def main():
    async with AsyncResource() as resource:
        print("Using resource")

asyncio.run(main())
```

**应用场景**：
1. 流式处理大数据
2. 分批获取数据
3. 异步文件读取
4. 事件流处理

**考察点**：异步迭代器、生成器

---

## 百度

### 8. 如何在异步代码中进行错误处理？

**参考答案**：

**基本异常处理**：
```python
import asyncio

async def failed_task():
    raise ValueError("Something went wrong")

async def main():
    try:
        await failed_task()
    except ValueError as e:
        print(f"Caught: {e}")

asyncio.run(main())
```

**gather中的异常**：
```python
async def task1():
    raise ValueError("Error in task1")

async def task2():
    return "Success"

async def main():
    # 方法1: return_exceptions
    results = await asyncio.gather(
        task1(),
        task2(),
        return_exceptions=True
    )
    for result in results:
        if isinstance(result, Exception):
            print(f"Error: {result}")
        else:
            print(f"Result: {result}")

    # 方法2: 捕获异常
    try:
        results = await asyncio.gather(task1(), task2())
    except ValueError as e:
        print(f"Caught: {e}")

asyncio.run(main())
```

**wait中的异常**：
```python
async def main():
    tasks = [
        asyncio.create_task(task1()),
        asyncio.create_task(task2())
    ]

    done, pending = await asyncio.wait(
        tasks,
        return_when=asyncio.FIRST_EXCEPTION
    )

    for task in done:
        try:
            result = task.result()
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error: {e}")
```

**Task中的异常**：
```python
async def main():
    task = asyncio.create_task(failed_task())

    try:
        await task
    except ValueError as e:
        print(f"Caught: {e}")

    # 或检查task.exception()
    if task.exception():
        print(f"Task failed: {task.exception()}")

asyncio.run(main())
```

**最佳实践**：
1. 使用`return_exceptions=True`收集所有异常
2. 及时处理异常，避免任务静默失败
3. 使用try-finally确保资源释放
4. 考虑使用超时避免无限等待

**考察点**：异常处理、错误恢复

---

## 网易

### 9. 什么是协程的取消？如何实现？

**参考答案**：

协程取消是通过抛出`CancelledError`来实现的。

**基本取消**：
```python
import asyncio

async def cancellable_task():
    try:
        print("Task started")
        await asyncio.sleep(10)
        print("Task completed")
    except asyncio.CancelledError:
        print("Task was cancelled!")
        raise  # 重新抛出

async def main():
    task = asyncio.create_task(cancellable_task())

    # 等待1秒后取消
    await asyncio.sleep(1)
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        print("Main caught cancellation")

asyncio.run(main())
```

**批量取消**：
```python
async def main():
    tasks = [
        asyncio.create_task(cancellable_task())
        for _ in range(5)
    ]

    await asyncio.sleep(1)

    # 取消所有任务
    for task in tasks:
        task.cancel()

    # 等待所有任务完成取消
    await asyncio.gather(*tasks, return_exceptions=True)

asyncio.run(main())
```

**防止取消**：
```python
async def protected_task():
    try:
        # 使用shield保护任务不被取消
        result = await asyncio.shield(cancellable_task())
        print(f"Result: {result}")
    except asyncio.CancelledError:
        print("This won't be cancelled")
```

**清理资源**：
```python
async def task_with_cleanup():
    try:
        # 执行任务
        await asyncio.sleep(10)
    except asyncio.CancelledError:
        # 清理资源
        print("Cleaning up...")
        await cleanup()
        raise  # 重新抛出CancelledError
```

**注意事项**：
1. 取消是协作的，任务需要检查CancelledError
2. 取消后任务应该抛出CancelledError
3. 使用shield保护关键任务
4. 确保资源正确释放

**考察点**：任务取消、资源管理

---

## 京东

### 10. 如何实现一个异步的WebSocket客户端？

**参考答案**：

使用`websockets`库实现WebSocket客户端：

```python
import asyncio
import websockets

async def websocket_client():
    uri = "ws://localhost:8765"

    async with websockets.connect(uri) as websocket:
        # 发送消息
        await websocket.send("Hello, Server!")

        # 接收消息
        response = await websocket.recv()
        print(f"Received: {response}")

asyncio.run(websocket_client())
```

**更完整的客户端**：
```python
import asyncio
import websockets
import json

class WebSocketClient:
    def __init__(self, uri):
        self.uri = uri
        self.websocket = None

    async def connect(self):
        """连接服务器"""
        self.websocket = await websockets.connect(self.uri)
        print(f"Connected to {self.uri}")

    async def send(self, message):
        """发送消息"""
        if self.websocket:
            await self.websocket.send(json.dumps(message))
            print(f"Sent: {message}")

    async def receive(self):
        """接收消息"""
        if self.websocket:
            message = await self.websocket.recv()
            return json.loads(message)

    async def close(self):
        """关闭连接"""
        if self.websocket:
            await self.websocket.close()
            print("Connection closed")

    async def listen(self):
        """监听消息"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                print(f"Received: {data}")
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed")

async def main():
    client = WebSocketClient("ws://localhost:8765")

    await client.connect()

    # 启动监听任务
    listen_task = asyncio.create_task(client.listen())

    # 发送消息
    await client.send({"type": "greeting", "content": "Hello"})

    # 等待
    await asyncio.sleep(10)

    # 清理
    listen_task.cancel()
    await client.close()

asyncio.run(main())
```

**使用aiohttp的WebSocket**：
```python
import aiohttp
import asyncio

async def websocket_session():
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect('ws://localhost:8765') as ws:
            # 发送文本
            await ws.send_str('Hello, Server!')

            # 接收文本
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    print(f"Received: {msg.data}")
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    break

asyncio.run(websocket_session())
```

**注意事项**：
1. 处理连接断开和重连
2. 心跳保持连接
3. 消息序列化/反序列化
4. 错误处理和重试机制

**考察点**：WebSocket、异步网络编程

---

## 总结

### 高频考点：
1. **async/await**：协程定义和执行
2. **事件循环**：调度机制
3. **并发执行**：gather、create_task、wait
4. **同步原语**：Lock、Event、Semaphore、Queue
5. **错误处理**：异常捕获、return_exceptions
6. **超时控制**：wait_for、timeout
7. **任务取消**：cancel、CancelledError
8. **避免阻塞**：异步I/O、run_in_executor

### 实战建议：
1. 理解事件循环的工作原理
2. 掌握asyncio的核心API
3. 熟练使用同步原语
4. 注意避免阻塞操作
5. 做好错误处理和资源清理
6. 使用第三方库（aiohttp、websockets）
7. 实战：Web服务、爬虫、实时通信
