# 异步编程练习题

import asyncio
from typing import List, Tuple
import time

# ==================== 练习题 1-5: 基础协程 ====================

# 练习 1: 创建简单的协程
# 要求：定义一个协程函数，打印Hello，等待1秒，打印World
async def hello_world():
    """TODO: 实现hello_world协程"""
    pass

# 测试: asyncio.run(hello_world())


# 练习 2: 创建带参数的协程
# 要求：定义一个协程，接收delay和message参数，延迟后打印消息
async def delayed_message(delay: float, message: str):
    """TODO: 实现延迟消息协程"""
    pass

# 测试: asyncio.run(delayed_message(1, "Hello!"))


# 练习 3: 创建返回值的协程
# 要求：定义一个协程，执行一些计算后返回结果
async def compute(x: int, y: int) -> int:
    """TODO: 实现计算协程"""
    pass

# 测试: result = asyncio.run(compute(10, 20)); print(result)


# 练习 4: 并发执行多个协程
# 要求：使用asyncio.gather并发执行多个协程
async def task_a():
    await asyncio.sleep(1)
    return "A"

async def task_b():
    await asyncio.sleep(1)
    return "B"

async def run_concurrently():
    """TODO: 并发执行task_a和task_b"""
    pass

# 测试: asyncio.run(run_concurrently())


# 练习 5: 使用create_task
# 要求：使用asyncio.create_task创建并运行任务
async def create_task_example():
    """TODO: 使用create_task"""
    pass

# 测试: asyncio.run(create_task_example())


# ==================== 练习题 6-10: 同步原语 ====================

# 练习 6: 使用Lock
# 要求：创建一个Lock，确保同一时间只有一个协程访问资源
async def worker_with_lock(lock: asyncio.Lock, worker_id: int):
    """TODO: 实现带锁的工作者"""
    pass

async def lock_example():
    """TODO: 测试Lock"""
    pass

# 测试: asyncio.run(lock_example())


# 练习 7: 使用Event
# 要求：创建一个Event，一个协程等待，另一个协程触发
async def wait_for_event(event: asyncio.Event):
    """TODO: 等待事件"""
    pass

async def trigger_event(event: asyncio.Event):
    """TODO: 触发事件"""
    pass

async def event_example():
    """TODO: 测试Event"""
    pass

# 测试: asyncio.run(event_example())


# 练习 8: 使用Queue
# 要求：实现生产者-消费者模式
async def producer(queue: asyncio.Queue):
    """TODO: 实现生产者"""
    pass

async def consumer(queue: asyncio.Queue):
    """TODO: 实现消费者"""
    pass

async def queue_example():
    """TODO: 测试Queue"""
    pass

# 测试: asyncio.run(queue_example())


# 练习 9: 使用Semaphore
# 要求：限制同时运行的协程数量
async def limited_worker(semaphore: asyncio.Semaphore, worker_id: int):
    """TODO: 实现受限的工作者"""
    pass

async def semaphore_example():
    """TODO: 测试Semaphore"""
    pass

# 测试: asyncio.run(semaphore_example())


# 练习 10: 使用Condition
# 要求：使用Condition实现生产者-消费者
async def producer_with_condition(condition: asyncio.Condition, items: list):
    """TODO: 实现带条件的生产者"""
    pass

async def consumer_with_condition(condition: asyncio.Condition, items: list):
    """TODO: 实现带条件的消费者"""
    pass

async def condition_example():
    """TODO: 测试Condition"""
    pass

# 测试: asyncio.run(condition_example())


# ==================== 练习题 11-15: 错误处理与超时 ====================

# 练习 11: 异步异常处理
# 要求：在协程中处理异常
async def failing_task():
    """TODO: 实现会失败的协程"""
    pass

async def exception_handling_example():
    """TODO: 处理异常"""
    pass

# 测试: asyncio.run(exception_handling_example())


# 练习 12: gather中的异常
# 要求：使用gather的return_exceptions参数
async def task_1():
    await asyncio.sleep(0.1)
    raise ValueError("Error in task 1")

async def task_2():
    await asyncio.sleep(0.1)
    return "Success"

async def gather_exceptions_example():
    """TODO: 处理gather中的异常"""
    pass

# 测试: asyncio.run(gather_exceptions_example())


# 练习 13: 超时控制
# 要求：使用wait_for实现超时
async def slow_task():
    """TODO: 实现慢任务"""
    pass

async def timeout_example():
    """TODO: 实现超时控制"""
    pass

# 测试: asyncio.run(timeout_example())


# 练习 14: 等待多个任务
# 要求：使用wait等待多个任务，返回完成和未完成的
async def wait_example():
    """TODO: 使用asyncio.wait"""
    pass

# 测试: asyncio.run(wait_example())


# 练习 15: 任务取消
# 要求：创建任务并在一定时间后取消
async def cancellable_task():
    """TODO: 实现可取消的任务"""
    pass

async def cancellation_example():
    """TODO: 取消任务"""
    pass

# 测试: asyncio.run(cancellation_example())


# ==================== 练习题 16-20: 实际应用 ====================

# 练习 16: 批量HTTP请求（模拟）
# 要求：并发请求多个URL
async def fetch_url(url: str) -> str:
    """TODO: 模拟HTTP请求"""
    pass

async def fetch_all_urls(urls: List[str]) -> List[str]:
    """TODO: 批量获取URL"""
    pass

async def http_example():
    """TODO: 测试HTTP请求"""
    pass

# 测试: asyncio.run(http_example())


# 练习 17: 异步文件操作（模拟）
# 要求：异步读写文件
async def read_file_async(filename: str) -> str:
    """TODO: 异步读取文件"""
    pass

async def write_file_async(filename: str, content: str):
    """TODO: 异步写入文件"""
    pass

async def file_example():
    """TODO: 测试文件操作"""
    pass

# 测试: asyncio.run(file_example())


# 练习 18: 实现异步生成器
# 要求：创建一个异步生成器
async def async_range(n: int):
    """TODO: 实现异步range"""
    pass

async def async_generator_example():
    """TODO: 使用异步生成器"""
    pass

# 测试: asyncio.run(async_generator_example())


# 练习 19: 实现异步上下文管理器
# 要求：创建一个异步上下文管理器
class AsyncResource:
    """TODO: 实现异步资源"""

    async def __aenter__(self):
        pass

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

async def async_context_example():
    """TODO: 测试异步上下文管理器"""
    pass

# 测试: asyncio.run(async_context_example())


# 练习 20: 实现异步迭代器
# 要求：创建一个异步迭代器
class AsyncCounter:
    """TODO: 实现异步计数器"""

    def __init__(self, n: int):
        pass

    def __aiter__(self):
        pass

    async def __anext__(self):
        pass

async def async_iterator_example():
    """TODO: 测试异步迭代器"""
    pass

# 测试: asyncio.run(async_iterator_example())


# ==================== 挑战题 ====================

# 挑战 1: 实现异步连接池
# 要求：实现一个简单的连接池
class ConnectionPool:
    """TODO: 实现连接池"""
    pass


async def connection_pool_example():
    """TODO: 测试连接池"""
    pass

# 测试: asyncio.run(connection_pool_example())


# 挑战 2: 实现异步进度条
# 要求：实现一个异步进度条
async def progress_bar():
    """TODO: 实现进度条"""
    pass

async def long_running_task():
    """TODO: 实现长任务"""
    pass

async def progress_example():
    """TODO: 测试进度条"""
    pass

# 测试: asyncio.run(progress_example())


# 挑战 3: 实现异步任务调度器
# 要求：实现一个简单的任务调度器
class TaskScheduler:
    """TODO: 实现任务调度器"""
    pass


async def scheduler_example():
    """TODO: 测试调度器"""
    pass

# 测试: asyncio.run(scheduler_example())


# 挑战 4: 实现异步WebSocket客户端（模拟）
# 要求：实现WebSocket连接和消息处理
class WebSocketClient:
    """TODO: 实现WebSocket客户端"""
    pass


async def websocket_example():
    """TODO: 测试WebSocket"""
    pass

# 测试: asyncio.run(websocket_example())


# 挑战 5: 实现异步缓存
# 要求：实现一个带TTL的异步缓存
class AsyncCache:
    """TODO: 实现异步缓存"""
    pass


async def cache_example():
    """TODO: 测试缓存"""
    pass

# 测试: asyncio.run(cache_example())


if __name__ == "__main__":
    print("异步编程练习题")
    print("请完成每个TODO部分的代码")
    print("运行测试验证你的实现")
