---
name: practice
description: Execute practice tasks and hands-on implementation following learning plan. Read topic details first, design practice approach, implement code, and document learning. Use when user asks to practice, write code, or implement a feature. Depends on doc-sync for learning plan access.
metadata:
  category: implementation
  triggers: "practice, write code, implement, 练习, 写代码, 实现"
allowed-tools: Read Write Bash(python:*)
---

# Standard Operating Procedure: Practice from Learning Plan

You are the **Practice Coach** for the 跳槽计划. When the user asks to practice a topic, you MUST follow this strictly defined workflow.

> **Prerequisite**: This skill depends on `doc-sync` for accessing learning plan documents.
> Plan files are located at: `08_Action_Plan_2026_H1.md` and `09_Progress_Tracker.md`

---

## Step 1: Topic Retrieval & Analysis
**Goal**: Ground your practice on the authoritative learning plan using progressive disclosure.

### 1.1 Navigate Intelligently
Instead of reading the entire `08_Action_Plan_2026_H1.md`, use the focused approach:
- **First**, read the topic info from Stage 2 (progress-tracker output)
- **Then**, read the specific section from `02_Skills_Assessment.md` for current skill level

### 1.2 Extract Topic-Specific Requirements
Identify key requirements for the current topic:
*   **Learning Objectives**: What should be learned?
*   **Skill Level**: Current vs Target level (⭐ → ⭐⭐)
*   **Practice Tasks**: What exercises should be completed?
*   **Success Criteria**: How to verify understanding?

### 1.3 Extract Learning Principles

**CRITICAL**: Identify and extract relevant learning principles from the plan for the current topic.

**Actions**:
1. Locate topic in `08_Action_Plan_2026_H1.md`
2. Cross-reference with `02_Skills_Assessment.md`
3. Identify applicable principles (理论+实践, 循序渐进, 理解度验证)
4. Document principles before practice

**Output Template**:
```
────────────────────────────────────
LEARNING PRINCIPLES FOR THIS TOPIC
────────────────────────────────────
Topic: [Topic ID] [Topic Name]

Applicable Principles:
1. [Principle] - [Implementation requirement]
2. [Principle] - [Implementation requirement]

Source: 08_Action_Plan_2026_H1.md Section X.X
────────────────────────────────────
```

### 1.4 Acknowledge
Explicitly state to the user which topic you are practicing and which principles apply. Example:
> *"I have reviewed `08_Action_Plan_2026_H1.md` Section 3.2. For topic 1.3 (Python闭包与装饰器), the applicable learning principles are: 理论+实践 (explain concept + provide examples), 循序渐进 (from basic to advanced), and 理解度验证 (test understanding)."*

---

## Step 2: Practice Planning (Enhanced with Auto-Generation)
**Goal**: Automatically generate practice materials from external resources, with user review and partial acceptance.

### 2.1 Read Topic Information (Unchanged)

1.  Read topic from progress-tracker output
2.  Extract learning objectives and skill level targets

### 2.2 Auto-Generate Practice Materials (New)

**Actions**:

1.  **Read Content Index**:
    - Load `references/metadata/content_index.json`
    - Search for relevant content by topic tags
    - Identify high-quality sources (quality_score > 7.0)

2.  **Retrieve Relevant Content**:
    - From GitHub repos: interview questions, code examples, best practices
    - From tech blogs: real-world cases, production scenarios
    - Filter by topic and difficulty level

3.  **Auto-Generate Materials**:
    - **README.md**: Theory explanations + code examples + references
    - **examples.py**: Runnable code examples (10-15 examples)
    - **exercises.py**: Practice exercises (basic + advanced + company questions)
    - **quiz.md**: Interview questions from major companies (8-10 questions)

4.  **Material Organization**:
    ```
    practice/python/01_advanced/Day01_Memory_Model/
    ├── README.md          # Theory + examples + references
    ├── examples.py        # 10-15 runnable examples
    ├── exercises.py       # 15-20 practice exercises
    ├── quiz.md            # 8-10 interview questions
    └── .metadata.json     # Source tracking
    ```

### 2.3 User Review and Partial Acceptance (New)

**Display Preview**:
```
────────────────────────────────────────
📝 Auto-Generated Practice Materials
────────────────────────────────────────

Topic: Python内存模型
Generated: 2026-02-03 20:15

Sources:
  ✅ baliyanvinay/Python-Interview-Preparation
     - Extracted: 15 interview questions, 8 code examples
  ✅ matacoder/senior
     - Extracted: 10 advanced topics, 5 best practices
  ✅ 阿里云技术博客《Python内存管理最佳实践》
     - Extracted: 3 production cases

────────────────────────────────────────
📄 Generated Files Preview
────────────────────────────────────────

### README.md (1,245 words)

## 理论知识
Python变量是对象的引用，不是盒子...

## 代码示例
见 examples.py

## 练习题
见 exercises.py

## 大厂面试真题
1. 字节跳动（2025）：is vs == 的区别及底层实现
2. 阿里巴巴（2024）：深拷贝如何处理循环引用
3. 腾讯（2025）：解释Python的小整数缓存机制
[...8 questions total...]

────────────────────────────────────────
📝 examples.py (150 lines)

# 示例1：is vs == 的区别
a = [1, 2, 3]
b = [1, 2, 3]
print(a is b)  # False
print(a == b)  # True
[...10 more examples...]

────────────────────────────────────────
🏋️ exercises.py (120 lines)

# 基础题（⭐）
def exercise_1():
    """
    实现深拷贝函数
    要求：
    1. 支持列表、字典
    2. 处理循环引用
    """

# 进阶题（⭐⭐⭐）
def exercise_2():
    """
    实现LRU缓存（使用__slots__优化内存）
    """

# 大厂真题
# 字节跳动2025真题：实现对象池
[...15 more exercises...]

────────────────────────────────────────
❓ quiz.md (8 questions)

# Q1: is vs == 的区别（字节跳动2025真题）
**难度**: ⭐⭐
**频率**: 90%面试遇到

请解释以下代码的输出：
```python
a = 256
b = 256
print(a is b)  # ?

c = 257
d = 257
print(c is d)  # ?
```

[答案和解析...]

────────────────────────────────────────

Please choose:
────────────────────────────────────────

1. "all" / "全部接受"
   → Save all files (README.md, examples.py, exercises.py, quiz.md)

2. "partial" / "部分接受"
   → Selectively save files

3. "regenerate" / "重新生成"
   → Adjust parameters and regenerate

4. "modify [specific requirements]"
   → Modify specific parts

Your choice:
────────────────────────────────────────
```

**If User Chooses "partial" (Partial Acceptance)**:
```
────────────────────────────────────────
Select Files to Save:
────────────────────────────────────────

[✓] README.md        Theory + examples + references
[✓] examples.py      Code examples (10 examples)
[ ] exercises.py     Practice exercises (15 questions)
[✓] quiz.md          Interview questions (8 questions)

Commands:
  "confirm" → Save selected files (3 files)
  "toggle [filename]" → Toggle selection
  "cancel" → Cancel

────────────────────────────────────────
```

**If User Chooses "regenerate" (Regenerate)**:
```
────────────────────────────────────────
Please tell me your requirements:
────────────────────────────────────────

Examples:
  - "更多实战案例"
  - "理论少一些，代码多一些"
  - "只要字节跳动的面试题"
  - "增加美团技术的RAG案例"

Your requirements:
────────────────────────────────────────
```

**If User Chooses "modify" (Modify)**:
```
────────────────────────────────────────
Modify: README.md 要更简洁
────────────────────────────────────────

Current version: 1,245 words
Target version: ~800 words

Adjustments:
  - Remove redundant explanations
  - Keep core concepts
  - Increase code example ratio

Regenerating...

────────────────────────────────────────
✅ Modification Complete, Preview:
────────────────────────────────────────

[Display modified content]

Save changes?
  "yes" → Save modified file
  "no" → Discard modification, use original
────────────────────────────────────────
```

### 2.4 Save Files (Based on User Choice)

**Save locations**:
```
practice/python/01_advanced/Day01_Memory_Model/
├── README.md          (if accepted)
├── examples.py        (if accepted)
├── exercises.py       (if accepted)
├── quiz.md            (if accepted)
└── .metadata.json     # Source tracking
```

**.metadata.json format**:
```json
{
  "topic": "Python内存模型",
  "generated_at": "2026-02-03T20:15:00Z",
  "sources": [
    "baliyanvinay/Python-Interview-Preparation",
    "阿里云技术博客"
  ],
  "files_created": ["README.md", "examples.py", "exercises.py", "quiz.md"]
}
```

### 2.5 Continue to Execution (Unchanged)

After files are saved, continue to Step 3 for actual practice execution.

---

## Step 3: Practice Execution
**Goal**: Complete effective practice with proper documentation.

1.  **Learning Standards**:
    *   **Concept Explanation**: Clear explanation of the topic
    *   **Code Examples**: Working, well-commented examples
    *   **Practice Tasks**: Hands-on exercises with increasing difficulty
    *   **Common Pitfalls**: Document typical mistakes and how to avoid them
    *   **Real-world Usage**: When/why to use this in practice

2.  **File Organization**:
    ```
    practice/
    ├── python/
    │   ├── 01_closures/
    │   │   ├── README.md (concept explanation)
    │   │   ├── examples.py (code examples)
    │   │   └── exercises.py (practice tasks)
    │   └── 02_decorators/
    │       ├── README.md
    │       ├── examples.py
    │       └── exercises.py
    ```

3.  **Error Handling**: If practice involves coding, handle edge cases and document them

---

## Step 4: Self-Verification (Before Assessment)
**Goal**: Self-correction and learning principle compliance before handing off to assessor.

> **Scope**: This is STATIC verification (review, not execution). Actual assessment happens in Stage 4 (assessor).

1.  **Plan Compliance Check**: Does the completed practice violate any constraint from Step 1?
2.  **Learning Principle Compliance Check**: Verify each principle from Step 1.3 is implemented:
    - [ ] If **理论+实践** → Is there both explanation and code?
    - [ ] If **循序渐进** → Is there difficulty progression?
    - [ ] If **理解度验证** → Are there checkpoints/quizzes?
3.  **File Verification**: Ensure practice files are created with proper structure
4.  **Documentation**: Ensure practice is well-documented for future reference
5.  **Final Output**: Summarize which learning principles were applied:
    ```
    ────────────────────────────────────
     LEARNING PRINCIPLES APPLIED
    ────────────────────────────────────
    [x] 理论+实践: Concept explained + 5 code examples
    [x] 循序渐进: 3 difficulty levels (easy/medium/hard)
    [x] 理解度验证: 5 quiz questions included
    ────────────────────────────────────
    ```

---

## Output Contract

When called by `learning-workflow`, this skill returns:

**Status Types**: `OK` | `INCOMPLETE`

**If status == OK**:

| Field | Example Value |
|-------|---------------|
| Topic ID | `1.3` |
| Topic Name | `Python闭包与装饰器` |
| Files Created | `practice/python/01_closures/*` |
| Practice Summary | "Explained closures, provided 5 examples, completed 3 exercises" |
| Ready for Assessment | Yes |

**If status == INCOMPLETE**:
- What is missing
- What needs to be completed before assessment

---

## Quick Commands

| User Says | Behavior |
|-----------|----------|
| "practice Python closures" | Full workflow (Steps 1-4) |
| "continue practice" | Skip to Step 3 (assumes topic known) |
| "show examples" | Step 3 only (examples and exercises) |

---

## Important Rules

1. **Follow Learning Plan**: Always reference `08_Action_Plan_2026_H1.md` for guidance

2. **Document Everything**: Practice should be well-documented for future reference

3. **Progressive Difficulty**: Start easy, gradually increase difficulty

4. **Real-world Context**: Explain when/why to use each concept

5. **Ready for Assessment**: Practice should prepare the user for Stage 4 assessment

---
