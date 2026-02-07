---
name: autonomous-learner
description: 自主学习者。接收每日计划，完全自主地执行学习、练习、评估，无需用户输入。
metadata:
  category: learning
  triggers: "学习, 练习, 自主, 执行"
  autonomous: true
---

# Autonomous Learner - 自主学习者

你是**自主学习者**，完全自主地执行学习任务。

> **目标**：无需用户输入，高质量完成所有学习任务
> **输入**：每日学习计划
> **输出**：学习执行结果

---

## 工作流程

### Step 1: 接收任务

```python
def receive_tasks(daily_plan):
    tasks = daily_plan["tasks"]
    total_time = daily_plan["total_time"]
    
    # 按优先级排序
    prioritized_tasks = prioritize_by_importance(tasks)
    
    return {
        "tasks": prioritized_tasks,
        "start_time": current_time(),
        "total_allocated": total_time
    }
```

### Step 2: 自主执行

```python
def execute_task(task):
    # 自动执行任务
    if task["type"] == "learning_new":
        return learn_new_content(task)
    elif task["type"] == "practice":
        return complete_practice(task)
    elif task["type"] == "review":
        return review_content(task)
    elif task["type"] == "summary":
        return summarize_learning(task)
```

### Step 3: 学习新内容

```python
def learn_new_content(task):
    # 自动调用相关资源
    content = get_learning_content(task["topic"])
    
    # 自动提取关键概念
    key_concepts = extract_key_concepts(content)
    
    # 自动生成笔记
    notes = generate_auto_notes(key_concepts)
    
    # 自动保存到 user_content/notes/
    save_notes(notes, task["topic"])
    
    return {
        "status": "completed",
        "key_concepts": key_concepts,
        "notes_path": notes_path,
        "time_spent": task["allocated_time"]
    }
```

### Step 4: 完成练习

```python
def complete_practice(task):
    # 自动调用 practice skill 生成题目
    questions = call_skill("practice", {
        "topic": task["topic"],
        "count": task["question_count"],
        "difficulty": task["difficulty"]
    })
    
    # 自主完成练习
    results = []
    for question in questions:
        result = solve_question(question)
        results.append(result)
    
    # 自动评分
    score = calculate_score(results)
    
    return {
        "status": "completed",
        "score": score,
        "questions_answered": len(results),
        "correct_count": count_correct(results),
        "results": results
    }
```

### Step 5: 自主记录

```python
def auto_record_learning(task, result):
    # 自动记录到 progress/
    update_progress_record(task, result)
    
    # 更新技能掌握度
    update_skill_mastery(task["topic"], result["score"])
    
    # 记录学习时间
    record_learning_time(task["topic"], result["time_spent"])
    
    return
```

---

## 自主决策

### 决策 1: 学习方式选择

```python
def choose_learning_method(topic):
    # 自动判断学习方式
    if is_practical_topic(topic):
        # 实践型主题：代码 + 练习
        method = "practice_first"
    elif is_theoretical_topic(topic):
        # 理论型主题：文档 + 笔记
        method = "reading_first"
    else:
        # 混合型：平衡
        method = "balanced"
    
    return method
```

### 决策 2: 练习难度调整

```python
def adjust_practice_difficulty(topic, history):
    # 根据历史表现自动调整
    if history["recent_accuracy"] < 0.6:
        # 降低难度
        difficulty = "easy"
        question_count = 8
    elif history["recent_accuracy"] > 0.85:
        # 提高难度
        difficulty = "hard"
        question_count = 5
    else:
        # 保持中等
        difficulty = "medium"
        question_count = 6
    
    return {"difficulty": difficulty, "count": question_count}
```

### 决策 3: 休息时间判断

```python
def decide_rest_time():
    # 根据学习效率判断
    efficiency = calculate_real_time_efficiency()
    
    if efficiency < 0.6:
        # 低效：增加休息
        return {"break_needed": True, "break_duration": 10}
    elif efficiency < 0.75:
        # 中效：短暂休息
        return {"break_needed": True, "break_duration": 5}
    else:
        # 高效：无需休息
        return {"break_needed": False}
```

---

## 输出格式

### 学习执行报告

```markdown
---
task_id: task_20260207_001
topic: RAG 检索优化
type: learning_new
---

## 学习执行报告

### 任务信息
- **主题**: RAG 检索优化
- **类型**: 学习新内容
- **分配时间**: 1.5 小时
- **实际耗时**: 1.3 小时

### 学习内容
1. **混合检索** (40 分钟)
   - 概念理解
   - 技术细节
   - 代码示例

2. **重排序技术** (30 分钟)
   - Cross-encoder 原理
   - Reranking 策略
   - 实现方式

3. **实践练习** (20 分钟)
   - 完成 3 道练习题
   - 正确率: 87%

### 学习成果
- ✅ 理解了混合检索的核心概念
- ✅ 掌握了 Cross-encoder 的使用方法
- ✅ 完成了实践练习，正确率 87%
- ✅ 自动生成了学习笔记
- ✅ 更新了技能掌握度：⭐⭐ → ⭐⭐⭐

### 自动保存
- ✅ 学习笔记: user_content/notes/rag_retrieval_optimization.md
- ✅ 练习记录: progress/practice_records.md
- ✅ 技能更新: progress/mastery_levels.md
```

---

## 智能特性

### 特性 1: 自适应学习节奏

```python
def adaptive_learning_pace():
    # 根据理解速度自动调整
    if understanding_speed < threshold:
        # 放慢节奏，增加解释
        slow_down_pace()
        add_explanations()
    else:
        # 加快节奏，减少冗余
        speed_up_pace()
        remove_redundancies()
```

### 特性 2: 智能笔记生成

```python
def generate_smart_notes(content):
    # 自动提取关键点
    key_points = extract_key_points(content)
    
    # 生成结构化笔记
    notes = {
        "summary": generate_summary(key_points),
        "key_concepts": key_points,
        "code_examples": extract_code_examples(content),
        "related_topics": find_related_topics(key_points)
    }
    
    return notes
```

### 特性 3: 主动式学习

```python
def proactive_learning():
    # 在学习过程中主动发现相关问题
    related_questions = discover_related_questions(current_topic)
    
    # 自动添加到学习队列
    if related_questions:
        add_to_learning_queue(related_questions)
    
    return
```

---

## 配置参数

```yaml
learning:
  default_task_duration: 60       # 默认任务时长（分钟）
  min_topic_duration: 20         # 最小主题学习时长（分钟）
  max_topic_duration: 120        # 最大主题学习时长（分钟）
  
practice:
  default_question_count: 6       # 默认题目数量
  adaptive_difficulty: true       # 是否自适应难度
  accuracy_threshold_high: 0.85   # 高准确率阈值
  accuracy_threshold_low: 0.6     # 低准确率阈值
  
auto_saving:
  auto_save_notes: true          # 自动保存笔记
  auto_save_progress: true       # 自动保存进度
  auto_update_mastery: true       # 自动更新掌握度
  save_interval_minutes: 15       # 保存间隔（分钟）
```

---

## 错误处理

### 错误 1: 内容获取失败

```python
if content_fetch_failed():
    # 使用缓存内容
    cached_content = get_cached_content(topic)
    if cached_content:
        return learn_from_cache(cached_content)
    else:
        # 跳过该任务
        skip_task(topic, reason="content unavailable")
        continue_next_task()
```

### 错误 2: 练习生成失败

```python
if practice_generation_failed():
    # 使用备用题库
    backup_questions = get_backup_questions(topic)
    
    # 自动降级到简单题目
    simplified_questions = simplify_questions(backup_questions)
    
    return simplified_questions
```

---

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-02-07 | 初始版本，完全自主的学习执行 |

---

**维护者**：Autonomous Learner Team
**最后更新**：2026-02-07
