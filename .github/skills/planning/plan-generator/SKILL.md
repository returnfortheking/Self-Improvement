---
name: plan-generator
description: Generate detailed learning paths based on overall plan framework and external resources. Reads 08_Action_Plan_2026_H1.md for framework, analyzes references/ with incremental indexing, outputs practice/[topic]/LEARNING_PATH.md without modifying overall plan. Use when user says "生成学习路径", "更新学习路径", "plan", "学习计划".
metadata:
  category: planning
  triggers: "生成学习路径, 更新学习路径, plan, 学习计划, 生成学习计划"
allowed-tools: Read Write
---

# Plan Generator - Learning Path Generator

This skill generates **detailed learning path documents** based on the overall plan framework (08_Action_Plan_2026_H1.md) and external resources from references/.

> **Core Principle**: Does NOT modify the overall plan (08_Action_Plan_2026_H1.md), only generates detailed learning path documents (practice/[topic]/LEARNING_PATH.md)
> **Incremental Indexing**: First scan ~20 minutes, subsequent scans <1 minute (98.6% files skipped)

---

## When to Use This Skill

- When starting a new learning phase (e.g., "2月：Python学习")
- After doc-sync updates external resources
- When external resources have significant new content (>50 new topics)
- When user wants to refresh learning path with latest materials

---

## Workflow

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│  Step 1           Step 2                Step 3              Step 4      Step 5    │
│  ────────         ────────              ────────            ────────    ────────  │
│  Read Plan  →  Analyze Resources  →  Generate Path  →  Output Doc  →  Align     │
│  (Framework)     (Incremental)       (Detailed)        (Independent)  (Validate) │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Read Overall Plan Framework

**Goal**: Extract learning phase context from 08_Action_Plan_2026_H1.md

### 1.1 Identify Current Learning Phase

1. Read `08_Action_Plan_2026_H1.md`
2. Parse to identify:
   - **Current phase**: e.g., "Phase 1: 基础知识恢复"
   - **Time frame**: e.g., "2月1日-2月28日"
   - **Target skill level**: e.g., "Python: ⭐ → ⭐⭐⭐"
   - **Time allocation**: e.g., "72 hours (4 weeks × 18 hours)"
   - **Assessment method**: e.g., "面试题、编程练习"

### 1.2 Extract Framework Constraints

```
Phase: Phase 1 - 基础知识恢复
Topic: Python学习
Current Level: ⭐
Target Level: ⭐⭐⭐
Time Budget: 72 hours
Assessment: 面试题、编程练习
```

---

## Step 2: Analyze External Resources (Incremental Indexing)

**Goal**: Scan references/ and extract relevant topics using incremental indexing

### 2.0: Check Index Status

1. Check if `references/metadata/content_index.json` exists
2. If **NOT exists** → Go to Step 2.1 (First full scan)
3. If **exists** → Go to Step 2.2 (Incremental scan)

---

### 2.1: First Full Scan (Only Once, ~20 minutes)

**Trigger**: content_index.json does not exist

**Actions**:
1. Scan all files in `references/` directory
2. For each file:
   ```python
   - Calculate SHA-256 hash
   - Extract topics, questions, tags
   - Record to content_index.json
   ```
3. Build topic frequency statistics
4. Save index

**Output**:
```
────────────────────────────────────────
📊 First-Time Index Building
────────────────────────────────────────

Scan scope: references/
Files scanned: 1,235
Duration: 20 minutes

Index statistics:
  - GitHub files: 850
  - Tech blog articles: 385
  - Topics found: 85
  - Questions extracted: 1,250

Index saved: references/metadata/content_index.json
Next scan will use incremental mode (estimated <1 minute)
────────────────────────────────────────
```

---

### 2.2: Incremental Scan (Default, <1 minute)

**Trigger**: content_index.json already exists

**Actions**:

#### 2.2.1 Scan File System
1. Traverse all files in `references/`
2. Calculate current file hash for each
3. Compare with hash in `content_index.json`

#### 2.2.2 Classify Files

| Type | Condition | Count | Action |
|------|-----------|-------|--------|
| **New Files** | Hash not in index | ? | Extract content, add to index |
| **Modified Files** | Hash differs from index | ? | Re-extract, update index |
| **Unchanged Files** | Hash same | ? | **SKIP**, use cache |

#### 2.2.3 Process Only Changed Files
For "new files" and "modified files":
- Extract topics, questions, tags
- Update `content_index.json`
- Update `topic_frequency` statistics

#### 2.2.4 Remove Deleted Files (Optional)
If indexed files don't exist in filesystem:
- Remove from index
- Update `topic_frequency` (decrease count)

**Output**:
```
────────────────────────────────────────
📊 Incremental Scan Complete
────────────────────────────────────────

Last scan: 2026-02-03 20:00
Current scan: 2026-02-04 22:30

File statistics:
  - Total files: 1,250
  - New files: 12 ✨
  - Modified files: 5 🔄
  - Unchanged files: 1,233 ⏭️ (skipped)
  - Deleted files: 0

Processing efficiency:
  - Only processed: 17 files (1.4%)
  - Skipped: 1,233 files (98.6%)
  - Duration: 45 seconds (26x faster than full scan)

Topic updates:
  New topics: 2
    - LangChain v0.3 new features (from 3 new files)
    - Agent evaluation methods (from 2 new files)

  Updated topics: 3
    - RAG (+5 articles)
    - LLM fine-tuning (+3 articles)
    - Python async (+2 articles)

Index updated: references/metadata/content_index.json
────────────────────────────────────────
```

---

### 2.3: Extract Relevant Topics

**Goal**: Filter topics relevant to current learning phase

1. Read from `content_index.json`:
   - Filter `indexed_files` by current phase tags
   - Example for Python phase:
     - Include tags: ["python", "闭包", "装饰器", "异步", "GIL"]
     - Exclude tags: ["LLM", "RAG", "Agent"]

2. Calculate topic frequency:
   - Count occurrences across all sources
   - Calculate quality score (based on source quality)
   - Identify trending topics

3. Extract best practices:
   - High-quality articles (quality_score > 8.0)
   - Official documentation
   - Tech blog articles from major companies

---

### 2.4: Generate Resource List

**Output Structure**:
```json
{
  "phase": "Phase 1 - 基础知识恢复",
  "topic": "Python学习",
  "resources": {
    "github_repos": [
      {
        "name": "baliyanvinay/Python-Interview-Preparation",
        "topics": ["closure", "decorator", "generator"],
        "question_count": 50,
        "quality": "high"
      }
    ],
    "tech_blogs": [
      {
        "company": "阿里云",
        "articles": 15,
        "main_topics": ["内存管理", "并发编程"]
      }
    ]
  }
}
```

---

## Step 3: Generate Detailed Learning Path

**Goal**: Expand overall plan framework into specific topics with time estimates

### 3.1 Map Framework to Specific Topics

Based on `08_Action_Plan_2026_H1.md` framework:
- Example: "2月：Python学习（⭐ → ⭐⭐⭐，72小时）"
- Expand to:
  - Week 1: 基础恢复（18h）
  - Week 2: 高级特性（18h）
  - Week 3: 并发编程（18h）
  - Week 4: 面试冲刺（18h）

### 3.2 Generate Topic Breakdown

For each week, generate daily topics:

```markdown
## Week 1: 基础恢复（18 hours）

### Day 1-2: Python内存模型（4 hours）
**目标**: 理解变量引用、内存管理、垃圾回收机制

**学习材料**:
- README: 理论知识
- examples.py: 10个代码示例
- exercises.py: 15道练习题
- quiz.md: 8道面试题

**内容来源**:
- 📚 baliyanvinay/Python-Interview-Preparation（基础部分）
- 📚 matacoder/senior（内存管理章节）
- 📚 阿里云技术博客《Python内存管理最佳实践》（2026-01-15）

**大厂面试真题**:
- 字节跳动2025：is vs == 的区别及底层实现
- 阿里巴巴2024：深拷贝如何处理循环引用
- 腾讯2025：解释Python的小整数缓存机制

**掌握标准**:
- 能流畅回答所有面试题
- 能手写深拷贝实现
- 理解垃圾回收机制

**预估时间**: 4 hours
```

### 3.3 Ensure Time Budget Alignment

**Constraint**: Total time must not exceed framework allocation

Calculate:
```
Week 1: 18 hours (Day 1-2: 4h, Day 3-4: 4h, Day 5-6: 5h, Day 7: 5h)
Week 2: 18 hours
Week 3: 18 hours
Week 4: 18 hours
Total: 72 hours ✅ (matches framework)
```

If exceeds:
- Warn user
- Suggest removing less critical topics
- Or reducing practice depth

---

## Step 4: Output Independent Document

**Goal**: Generate detailed learning path document without modifying overall plan

### 4.1 Generate Document Structure

```markdown
# practice/python/LEARNING_PATH.md

> **生成时间**: 2026-02-03
> **总体计划**: 08_Action_Plan_2026_H1.md - Phase 1, 2月：Python学习
> **目标**: ⭐ → ⭐⭐⭐（72小时）
> **数据源**: 4个GitHub仓库 + 4家技术博客

---

## 学习路径概览

| 周次 | 主题 | 预估时间 | 来源 |
|------|------|----------|------|
| Week 1 | 基础恢复 | 18h | 基于GitHub仓库综合分析 |
| Week 2 | 高级特性 | 18h | 基于大厂面试高频题 |
| Week 3 | 并发编程 | 18h | 基于技术博客最新文章 |
| Week 4 | 面试冲刺 | 18h | 基于大厂面试真题库 |

**总计**: 72小时（符合总体计划）

---

## Week 1: Day 1-2: Python内存模型（4小时）
[详细内容...]

---

## 与总体计划的对齐

✅ **目标对齐**
- 总体计划要求：⭐ → ⭐⭐⭐
- 本学习路径覆盖：
  - 基础主题（98%）→ ⭐⭐
  - 高频主题（92%）→ ⭐⭐⭐
  - 高级主题（75%）→ ⭐⭐⭐（面试够用）

✅ **时间对齐**
- 总体计划分配：72小时（4周）
- 本学习路径：72小时（精确匹配）

✅ **评估方式对齐**
- 总体计划要求：面试题、编程练习
- 本学习路径：每天包含quiz.md和exercises.py

---

## 数据来源统计

**GitHub仓库**：
- baliyanvinay/Python-Interview-Preparation（2.5k stars）
- matacoder/senior（高级Python主题）
- Devinterview-io/python-interview-questions（100个核心题）
- thundergolfer/interview-with-python（大量练习题）

**技术博客**：
- 阿里云：47篇高质量文章
- 腾讯技术：38篇
- 美团技术：35篇
- 字节技术：32篇

**覆盖率**：
- 核心主题：98%
- 高频主题：92%
- 高级主题：75%

---

## 备注

- 本文档是对总体计划的细化，不替代08_Action_Plan_2026_H1.md
- 学习过程中如遇到新资源，可使用/更新资源重新生成此文档
- 建议每周日回顾进度，确保按时完成
```

### 4.2 Save Document

**File location**: `practice/[topic]/LEARNING_PATH.md`

**Examples**:
- `practice/python/LEARNING_PATH.md`
- `practice/rag/LEARNING_PATH.md`
- `practice/agent/LEARNING_PATH.md`

---

## Step 5: Align with Overall Goals (Safety Mechanisms)

**Goal**: Validate that generated path aligns with overall plan constraints

### 5.1 Time Validation

Check if total time exceeds framework allocation:

```
If learning_path_time > framework_allocation:
  ⚠️ WARNING: Time Exceeds Framework
  ────────────────────────────────────────
  Learning path requires: 85 hours
  Framework allocated: 72 hours
  Excess: 13 hours

  Suggestions:
    1. Adjust overall plan (modify 08_Action_Plan_2026_H1.md)
    2. Or reduce topics (lower target level to ⭐⭐)

  Awaiting your decision...
  ────────────────────────────────────────
```

### 5.2 Target Level Validation

Check if path can achieve target skill level:

```
If insufficient_coverage_for_target_level:
  ⚠️ WARNING: Cannot Reach Target Level
  ────────────────────────────────────────
  Current path can only achieve: ⭐⭐
  Framework target: ⭐⭐⭐

  Missing content:
    - Async programming (advanced topics)
    - Metaclasses (advanced topics)

  Suggestions:
    1. Add missing topics (increase time allocation)
    2. Or lower framework target level

  Awaiting your decision...
  ────────────────────────────────────────
```

### 5.3 Generate Report

```
────────────────────────────────────────
✅ Learning Path Generated
────────────────────────────────────────

Framework: 08_Action_Plan_2026_H1.md - Phase 1
Phase: 2月：Python学习
Target: ⭐ → ⭐⭐⭐ (72 hours)

Generated: practice/python/LEARNING_PATH.md

Path overview:
  - 4 weeks, 30 topics
  - Total time: 72 hours ✅
  - Target level: achievable ✅

New topics from external resources:
  - Python内存管理最佳实践（阿里云博客）
  - 高并发场景处理（美团技术）
  - 大厂面试真题集（字节/阿里/腾讯）

Alignment validation:
  ✅ Time: matches framework
  ✅ Target: achievable
  ✅ Coverage: comprehensive

────────────────────────────────────────

View generated path: practice/python/LEARNING_PATH.md

Options:
  "confirm" → Accept and save
  "regenerate" → Adjust and regenerate
  "cancel" → Discard
────────────────────────────────────────
```

---

## Quick Commands

| User Says | Behavior |
|-----------|----------|
| "生成学习路径 Python" | Steps 1-5 (generate full path) |
| "更新学习路径" | Step 2 (incremental scan) + Steps 3-5 |
| "重建索引" | Force Step 2.1 (full scan, rebuild index) |

---

## Important Rules

1. **Never Modify Overall Plan**: Do NOT edit 08_Action_Plan_2026_H1.md
2. **Only Generate Details**: Output to independent `practice/[topic]/LEARNING_PATH.md`
3. **Incremental by Default**: Use incremental scanning after first full scan
4. **Time Constraint**: Ensure total time does not exceed framework allocation
5. **Quality First**: Only use high-quality resources (quality_score > 7.0)
6. **Alignment Validation**: Always validate time and target level before output

---

## Output Contract

When called, this skill returns:

**Status Types**: `OK` | `WARNING_TIME` | `WARNING_TARGET` | `ERROR`

**If status == OK**:
```json
{
  "status": "OK",
  "framework_reference": "08_Action_Plan_2026_H1.md Phase 1",
  "output_file": "practice/python/LEARNING_PATH.md",
  "total_time": "72 hours",
  "target_achievable": true,
  "topics_count": 30,
  "new_topics_from_resources": 7
}
```

**If status == WARNING_TIME**:
- Excess time details
- Suggestions for adjustment

**If status == WARNING_TARGET**:
- Missing topics
- Suggestions to achieve target

---

**Version**: 2.0
**Last Updated**: 2026-02-03
**Dependencies**: doc-sync (must run first to populate references/)
