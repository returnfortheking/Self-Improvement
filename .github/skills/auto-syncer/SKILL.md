---
name: auto-syncer
description: 自动同步器。自动监控数据变化，执行 git commit 和 push/pull，完全自动。
metadata:
  category: utility
  triggers: "同步, 同步数据, push, pull"
  autonomous: true
---

# Auto-Syncer - 自动同步器

你是**自动同步器**，完全自主地管理 Git 同步。

> **目标**：自动 commit + push/pull，用户无需干预
> **触发**: 文件变化或定时任务

---

## 工作流程

### Step 1: 监听文件变化

```python
def watch_files():
    # 监听 data/ 和 progress/ 目录
    observer = Observer()
    observer.schedule(FileChangeHandler(), "data/", recursive=True)
    observer.schedule(FileChangeHandler(), "progress/", recursive=True)
    observer.start()
```

### Step 2: 自动 Commit

```python
def auto_commit(changed_files):
    # 自动添加文件
    repo.index.add(changed_files)
    
    # 生成智能 commit message
    commit_msg = generate_commit_message(changed_files)
    
    # 自动提交
    repo.index.commit(commit_msg)
    
    return commit_msg
```

### Step 3: 自动 Push

```python
def auto_push():
    # 自动推送到远程
    origin = repo.remote(name='origin')
    origin.push()
    
    # 记录推送结果
    log_push_result()
```

### Step 4: 自动 Pull

```python
def auto_pull():
    # 定时拉取（每 5 分钟）
    while True:
        try:
            repo.remotes.origin.pull()
            await asyncio.sleep(300)  # 5 分钟
        except Exception:
            await asyncio.sleep(60)  # 失败后 1 分钟重试
```

---

## 智能特性

### 特性 1: 智能合并

```python
def smart_merge(conflict_files):
    # 根据文件类型自动合并
    for file in conflict_files:
        if is_progress_file(file):
            # 进度文件：合并数据
            merge_progress_data(file)
        elif is_question_file(file):
            # 题库文件：保留最新时间戳
            keep_newer_version(file)
        else:
            # 其他文件：标记冲突
            mark_conflict(file)
```

### 特性 2: 防抖处理

```python
def debounce_changes():
    # 1 秒内的多次变化合并为一次 commit
    if file_changed_recently(file, 1 second):
        queue_change(file)
    else:
        process_queued_changes()
```

### 特性 3: 自动冲突解决

```python
def auto_resolve_conflict(file):
    # 检查冲突类型
    conflict_type = detect_conflict_type(file)
    
    if conflict_type == "timestamp_conflict":
        return resolve_by_timestamp(file)
    elif conflict_type == "data_conflict":
        return resolve_by_data_merge(file)
    else:
        # 无法自动解决，标记并继续
        return mark_for_manual_resolution(file)
```

---

## 配置

```yaml
sync:
  auto_commit: true              # 自动 commit
  auto_push: true               # 自动 push
  auto_pull: true               # 自动 pull
  pull_interval_seconds: 300     # pull 间隔（5 分钟）
  debounce_seconds: 1             # 防抖间隔（1 秒）
  max_retries: 3                # 最大重试次数
  
conflict:
  auto_resolve_timestamp: true     # 自动解决时间戳冲突
  auto_resolve_data: true         # 自动解决数据冲突
  backup_before_resolve: true    # 解决前备份
```

---

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-02-07 | 初始版本 |

---

**维护者**: Auto-Syncer Team
**最后更新**: 2026-02-07
