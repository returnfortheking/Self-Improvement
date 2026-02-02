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

## Step 2: Practice Planning
**Goal**: Ensure effective practice design before writing any code.

1.  **File Strategy**: List the files to create for practice (code files, notes, etc.)
2.  **Practice Design**: Based on the learning principles extracted in Step 1.3:
    - If **理论+实践** principle applies → Start with concept explanation, then code examples
    - If **循序渐进** principle applies → Plan difficulty progression (easy → medium → hard)
    - If **理解度验证** principle applies → Prepare quiz questions or checkpoints
3.  **Environment Check**: Verify the development environment is ready (Python installed, etc.)
4.  **Learning Checklist**: Before proceeding, verify your plan addresses each principle from Step 1.3.

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
