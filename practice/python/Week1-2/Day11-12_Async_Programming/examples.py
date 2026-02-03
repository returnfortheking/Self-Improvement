# 异步编程代码示例

import asyncio
import time
from typing import List, Tuple

# ==================== 1. 基础协程 ====================

async def simple_coroutine():
    """简单的协程"""
    print("Hello")
    await asyncio.sleep(1)
    print("World")


async def main():
    await simple_coroutine()


# asyncio.run(main())


# ==================== 2. 并发执行 ====================

async def fetch_data(id: int, delay: float) -> str:
    """模拟获取数据"""
    print(f"Fetching data {id}...")
    await asyncio.sleep(delay)
    return f"Data {id}"


async def concurrent_fetch():
    """并发获取数据"""
    start = time.time()

    # 并发执行
    results = await asyncio.gather(
        fetch_data(1, 1.0),
        fetch_data(2, 1.0),
        fetch_data(3, 1.0)
    )

    end = time.time()
    print(f"Results: {results}")
    print(f"Time: {end - start:.2f}s")  # 约1秒，而非3秒


# asyncio.run(concurrent_fetch())


# ==================== 3. 使用create_task ====================

async def process_task(task_id: int):
    """处理任务"""
    print(f"Task {task_id} started")
    await asyncio.sleep(1)
    print(f"Task {task_id} completed")
    return task_id * 2


async def task_example():
    """使用create_task"""
    # 创建任务（立即开始执行）
    tasks = [
        asyncio.create_task(process_task(i))
        for i in range(5)
    ]

    # 等待所有任务完成
    results = await asyncio.gather(*tasks)
    print(f"Results: {results}")


# asyncio.run(task_example())


# ==================== 4. 超时控制 ====================

async def slow_operation():
    """慢操作"""
    await asyncio.sleep(5)
    return "Done"


async def timeout_example():
    """超时示例"""
    try:
        result = await asyncio.wait_for(
            slow_operation(),
            timeout=2.0
        )
        print(result)
    except asyncio.TimeoutError:
        print("Operation timed out!")


# asyncio.run(timeout_example())


# ==================== 5. 使用Lock ====================

async def worker_with_lock(lock: asyncio.Lock, worker_id: int):
    """使用锁的工作者"""
    print(f"Worker {worker_id} trying to acquire lock...")
    async with lock:
        print(f"Worker {worker_id} acquired lock")
        await asyncio.sleep(1)
        print(f"Worker {worker_id} released lock")


async def lock_example():
    """锁示例"""
    lock = asyncio.Lock()
    await asyncio.gather(
        worker_with_lock(lock, 1),
        worker_with_lock(lock, 2),
        worker_with_lock(lock, 3)
    )


# asyncio.run(lock_example())


# ==================== 6. 使用Event ====================

async def waiter(event: asyncio.Event, waiter_id: int):
    """等待事件"""
    print(f"Waiter {waiter_id} is waiting...")
    await event.wait()
    print(f"Waiter {waiter_id} got the event!")


async def setter(event: asyncio.Event):
    """设置事件"""
    await asyncio.sleep(2)
    print("Setting event!")
    event.set()


async def event_example():
    """事件示例"""
    event = asyncio.Event()
    await asyncio.gather(
        waiter(event, 1),
        waiter(event, 2),
        setter(event)
    )


# asyncio.run(event_example())


# ==================== 7. 使用Queue ====================

async def producer(queue: asyncio.Queue, producer_id: int):
    """生产者"""
    for i in range(3):
        item = f"Item {producer_id}-{i}"
        await queue.put(item)
        print(f"Producer {producer_id} produced: {item}")
        await asyncio.sleep(0.1)


async def consumer(queue: asyncio.Queue, consumer_id: int):
    """消费者"""
    while True:
        item = await queue.get()
        print(f"Consumer {consumer_id} consumed: {item}")
        await asyncio.sleep(0.2)
        queue.task_done()


async def queue_example():
    """队列示例"""
    queue = asyncio.Queue(maxsize=10)

    # 创建生产者和消费者
    producers = [asyncio.create_task(producer(queue, i)) for i in range(2)]
    consumers = [asyncio.create_task(consumer(queue, i)) for i in range(3)]

    # 等待生产者完成
    await asyncio.gather(*producers)

    # 等待队列清空
    await queue.join()

    # 取消消费者
    for c in consumers:
        c.cancel()


# asyncio.run(queue_example())


# ==================== 8. 使用Semaphore ====================

async def limited_worker(semaphore: asyncio.Semaphore, worker_id: int):
    """限流的工作者"""
    async with semaphore:
        print(f"Worker {worker_id} is working...")
        await asyncio.sleep(1)
        print(f"Worker {worker_id} finished")


async def semaphore_example():
    """信号量示例"""
    # 限制同时运行的协程数量
    semaphore = asyncio.Semaphore(2)

    workers = [
        limited_worker(semaphore, i)
        for i in range(5)
    ]

    await asyncio.gather(*workers)


# asyncio.run(semaphore_example())


# ==================== 9. 异步生成器 ====================

async def async_range(n: int):
    """异步生成器"""
    for i in range(n):
        await asyncio.sleep(0.1)
        yield i


async def async_generator_example():
    """异步生成器示例"""
    start = time.time()
    result = [x async for x in async_range(5)]
    end = time.time()
    print(f"Results: {result}, Time: {end - start:.2f}s")


# asyncio.run(async_generator_example())


# ==================== 10. 异步上下文管理器 ====================

class AsyncContextManager:
    """异步上下文管理器"""

    async def __aenter__(self):
        print("Entering context")
        await asyncio.sleep(0.1)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("Exiting context")
        await asyncio.sleep(0.1)
        return False


async def async_context_example():
    """异步上下文管理器示例"""
    async with AsyncContextManager() as manager:
        print("Inside context")


# asyncio.run(async_context_example())


# ==================== 11. 异步迭代器 ====================

class AsyncIterator:
    """异步迭代器"""

    def __init__(self, n: int):
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


async def async_iterator_example():
    """异步迭代器示例"""
    async for value in AsyncIterator(3):
        print(f"Got: {value}")


# asyncio.run(async_iterator_example())


# ==================== 12. 批量请求 ====================

async def fetch_url(url: str) -> str:
    """模拟URL获取"""
    await asyncio.sleep(0.5)
    return f"Data from {url}"


async def batch_requests(urls: List[str]) -> List[str]:
    """批量请求"""
    print(f"Fetching {len(urls)} URLs...")
    start = time.time()

    tasks = [fetch_url(url) for url in urls]
    results = await asyncio.gather(*tasks)

    end = time.time()
    print(f"Fetched {len(results)} URLs in {end - start:.2f}s")
    return results


async def batch_example():
    """批量请求示例"""
    urls = [f"http://example.com/{i}" for i in range(10)]
    results = await batch_requests(urls)
    print(f"Got {len(results)} results")


# asyncio.run(batch_example())


# ==================== 13. 异常处理 ====================

async def failing_coro(task_id: int):
    """会失败的协程"""
    await asyncio.sleep(0.1)
    if task_id == 2:
        raise ValueError(f"Task {task_id} failed")
    return f"Success {task_id}"


async def exception_example():
    """异常处理示例"""
    try:
        results = await asyncio.gather(
            failing_coro(1),
            failing_coro(2),
            failing_coro(3),
            return_exceptions=True  # 不抛出异常
        )

        for result in results:
            if isinstance(result, Exception):
                print(f"Error: {result}")
            else:
                print(f"Result: {result}")

    except Exception as e:
        print(f"Caught exception: {e}")


# asyncio.run(exception_example())


# ==================== 14. 任务取消 ====================

async def cancellable_task(task_id: int):
    """可取消的任务"""
    try:
        print(f"Task {task_id} started")
        await asyncio.sleep(5)
        print(f"Task {task_id} completed")
    except asyncio.CancelledError:
        print(f"Task {task_id} was cancelled!")
        raise


async def cancellation_example():
    """取消示例"""
    task = asyncio.create_task(cancellable_task(1))

    # 等待1秒后取消
    await asyncio.sleep(1)
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        print("Task was cancelled")


# asyncio.run(cancellation_example())


# ==================== 15. 协程链接 ====================

async def step1():
    """步骤1"""
    print("Step 1")
    await asyncio.sleep(0.1)
    return "Result 1"


async def step2(previous_result):
    """步骤2"""
    print(f"Step 2 with {previous_result}")
    await asyncio.sleep(0.1)
    return "Result 2"


async def step3(previous_result):
    """步骤3"""
    print(f"Step 3 with {previous_result}")
    await asyncio.sleep(0.1)
    return "Final result"


async def chaining_example():
    """协程链接示例"""
    result1 = await step1()
    result2 = await step2(result1)
    result3 = await step3(result2)
    print(result3)


# asyncio.run(chaining_example())


# ==================== 16. 动态任务管理 ====================

async def dynamic_task(task_id: int):
    """动态任务"""
    print(f"Task {task_id} started")
    await asyncio.sleep(1)
    return task_id


async def dynamic_tasks_example():
    """动态任务管理示例"""
    tasks = set()

    # 动态添加任务
    for i in range(5):
        task = asyncio.create_task(dynamic_task(i))
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    # 等待所有任务完成
    await asyncio.gather(*tasks)
    print("All tasks completed")


# asyncio.run(dynamic_tasks_example())


# ==================== 17. 超时装饰器 ====================

def with_timeout(timeout: float):
    """超时装饰器"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                print(f"{func.__name__} timed out after {timeout}s")
                return None
        return wrapper
    return decorator


@with_timeout(1.0)
async def slow_function():
    """慢函数"""
    await asyncio.sleep(2)
    return "Done"


async def timeout_decorator_example():
    """超时装饰器示例"""
    result = await slow_function()
    print(f"Result: {result}")


# asyncio.run(timeout_decorator_example())


# ==================== 18. 进度报告 ====================

async def task_with_progress(task_id: int, progress_queue: asyncio.Queue):
    """带进度的任务"""
    for i in range(1, 11):
        await asyncio.sleep(0.1)
        progress = i * 10
        await progress_queue.put((task_id, progress))
    return task_id


async def progress_reporter(progress_queue: asyncio.Queue):
    """进度报告器"""
    while True:
        task_id, progress = await progress_queue.get()
        print(f"Task {task_id}: {progress}% complete")
        progress_queue.task_done()


async def progress_example():
    """进度报告示例"""
    progress_queue = asyncio.Queue()

    # 启动进度报告器
    reporter = asyncio.create_task(progress_reporter(progress_queue))

    # 创建任务
    tasks = [
        task_with_progress(i, progress_queue)
        for i in range(3)
    ]

    # 等待所有任务完成
    await asyncio.gather(*tasks)

    # 等待队列清空
    await progress_queue.join()

    # 取消报告器
    reporter.cancel()


# asyncio.run(progress_example())


# ==================== 19. 资源池 ====================

class ResourcePool:
    """资源池"""

    def __init__(self, size: int):
        self.pool = asyncio.Queue(maxsize=size)
        self.size = size

    async def acquire(self):
        """获取资源"""
        return await self.pool.get()

    def release(self, resource):
        """释放资源"""
        self.pool.put_nowait(resource)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


async def use_pool(pool: ResourcePool, user_id: int):
    """使用资源池"""
    async with pool:
        resource = await pool.acquire()
        print(f"User {user_id} acquired resource {resource}")
        await asyncio.sleep(1)
        print(f"User {user_id} releasing resource {resource}")
        pool.release(resource)


async def pool_example():
    """资源池示例"""
    pool = ResourcePool(2)

    # 初始化资源
    for i in range(pool.size):
        await pool.pool.put(f"Resource-{i}")

    # 创建多个用户
    users = [use_pool(pool, i) for i in range(5)]
    await asyncio.gather(*users)


# asyncio.run(pool_example())


# ==================== 20. 异步HTTP请求（模拟） ====================

async def mock_http_request(url: str, method: str = "GET") -> dict:
    """模拟HTTP请求"""
    await asyncio.sleep(0.5)
    return {
        "url": url,
        "method": method,
        "status": 200,
        "data": f"Response from {url}"
    }


async def fetch_multiple_urls(urls: List[str]) -> List[dict]:
    """获取多个URL"""
    print(f"Fetching {len(urls)} URLs...")
    start = time.time()

    tasks = [mock_http_request(url) for url in urls]
    results = await asyncio.gather(*tasks)

    end = time.time()
    print(f"Completed in {end - start:.2f}s")
    return results


async def http_example():
    """HTTP请求示例"""
    urls = [
        "https://api.github.com",
        "https://www.python.org",
        "https://stackoverflow.com"
    ]

    results = await fetch_multiple_urls(urls)

    for result in results:
        print(f"{result['url']}: {result['status']}")


# asyncio.run(http_example())


if __name__ == "__main__":
    print("异步编程示例")
    print("注意：这些示例需要Python 3.7+")

    # 运行示例
    asyncio.run(concurrent_fetch())
