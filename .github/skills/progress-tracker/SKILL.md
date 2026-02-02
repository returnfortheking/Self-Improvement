---
name: progress-tracker
description: Identify next learning topic from action plan and validate claimed progress against actual skill state. Serves as GPS for learning - tells you where you are and where to go next. Stage 2 of learning-workflow pipeline. Use when user says "检查进度", "status", "下一个学什么", "what's next", "查找任务".
metadata:
  category: progress-tracking
  triggers: "status, what's next, find topic, 检查进度, 下一个学什么, 查找任务"
allowed-tools: Read
---

# Progress Tracker & Topic Discovery

This skill identifies the **next learning topic** from the action plan and **validates** that claimed progress matches actual skill state. It serves as the "GPS" for learning - telling you where you are and where to go next.

> **Single Responsibility**: Locate → Validate → Confirm

---

## When to Use This Skill

- When you need to **find the next topic** to learn
- When you want to **check current learning progress**
- When you suspect **progress tracking is out of sync** with actual skills
- As **Stage 2** of the `learning-workflow` pipeline
- After a break to **resume learning** from the correct point

---

## Workflow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Step 1              Step 2                Step 3              Step 4        │
│  ────────            ────────              ────────            ────────      │
│  Data Collection  →  Progress Validation → Topic Identification → Confirm    │
│  (Data Prep)         (Validation)          (Topic Confirm)       (User OK)    │
│                          │                                                   │
│                          ▼                                                   │
│                     ️ Mismatch? → Escalate to User → Fix 09_Progress      │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Data Collection

**Goal**: Gather information about claimed progress and actual skill state.

### 1.1 Read Action Plan and Progress

1. Read `08_Action_Plan_2026_H1.md` (Learning Plan)
2. Read `09_Progress_Tracker.md` (Progress Tracking)
3. Parse to identify:
   - All learning topics and their status markers
   - Current phase (Python恢复, RAG学习, etc.)
   - Topics marked as completed vs in-progress vs not-started

### 1.2 Status Marker Recognition

| Marker | Meaning | Status |
|--------|---------|--------|
| `[ ]` | Not started | `NOT_STARTED` |
| `` | Not started | `NOT_STARTED` |
| `[~]` | In progress | `IN_PROGRESS` |
| `` | In progress | `IN_PROGRESS` |
| `[x]` | Completed | `COMPLETED` |
| `` | Completed | `COMPLETED` |

### 1.3 Build Topic List

**Output Structure**:
```
Phase 1: 基础知识恢复
  [x] 1.1: Python环境搭建
  [x] 1.2: Python基础语法
  [~] 1.3: Python闭包与装饰器  ← CURRENT (in progress)
  [ ] 1.4: Python异步编程

Phase 2: RAG核心知识
  [ ] 2.1: Vector Database基础
  [ ] 2.2: Embedding原理
  ...
```

---

## Step 2: Progress Validation

**Goal**: Verify that claimed progress matches actual skill state.

### 2.1 Identify Verification Targets

For each topic marked as `COMPLETED` or `IN_PROGRESS`, identify expected evidence:

| Topic | Expected Evidence |
|------|-------------------|
| 1.1: Python环境搭建 | `python --version` works, virtualenv created |
| 1.2: Python基础语法 | Can answer basic questions, practice code exists |
| 1.3: Python闭包与装饰器 | Can explain concepts, completed practice tasks |
| 2.1: Vector DB基础 | `02_Skills_Assessment.md` shows ⭐⭐ level |

### 2.2 Verify Evidence Exists

For each expected evidence:
1. Check if `02_Skills_Assessment.md` reflects the skill level
2. Check if practice code exists in project
3. Check if conversations/teacher/ has relevant learning records

**Verification Commands**:
```bash
# Check skill assessment
grep "Python" 02_Skills_Assessment.md

# Check practice code
find . -name "*python*practice*" -type f

# Check learning records
ls conversations/teacher/ | grep python
```

### 2.3 Detect and Handle Mismatches

**Mismatch Types**:

| Type | Description | Severity |
|------|-------------|----------|
| `SKILL_LEVEL_MISMATCH` | Topic marked complete but skill level doesn't reflect | High |
| `NO_PRACTICE_CODE` | Topic marked complete but no practice code found | Medium |
| `STALE_PROGRESS` | Topic marked "in progress" for multiple sessions | Medium |
| `MISSING_RECORDS` | No learning records found for completed topic | Low |

**If any mismatch detected**, escalate to user:

```
────────────────────────────────────────────────────
️ PROGRESS INCONSISTENCY DETECTED
────────────────────────────────────────────────────

Plan Claims: Phase 1.3 - Python闭包与装饰器 (in progress)
Actual State: Phase 1.2 - Python基础语法 (incomplete)

Missing Items:
   02_Skills_Assessment.md shows Python: ⭐ (expected ⭐⭐)
   No practice code for decorators found
   No learning records for closures

────────────────────────────────────────────────────
OPTIONS:
────────────────────────────────────────────────────

1. Fix progress tracking in 09_Progress_Tracker.md
    → Update markers to reflect actual state
    → Re-run doc-sync
    → Restart topic discovery

2. Confirm previous topics as completed
    → Learning may be in different location/format
    → Provide explanation and proceed

3. Continue from actual progress
    → Skip incomplete topics
    → Start from where skills actually are

Please choose an option (1/2/3):
────────────────────────────────────────────────────
```

---

## Step 3: Topic Identification

**Goal**: Clearly identify the single next topic to learn.

### 3.1 Determine Next Topic

**Priority Logic**:
1. If any topic is `IN_PROGRESS` → That is the current topic
2. Otherwise, find the first `NOT_STARTED` topic → That is the next topic
3. If all topics complete → Report "All topics complete"

### 3.2 Gather Topic Context

For the identified topic, collect:
- **Topic ID**: e.g., `1.3`, `2.1`
- **Topic Name**: e.g., "Python闭包与装饰器"
- **Phase**: e.g., "Phase 1: 基础知识恢复"
- **Current Skill Level**: From `02_Skills_Assessment.md`
- **Dependencies**: Previous topics that should be complete
- **Learning Resources**: References to documentation, tutorials, etc.

### 3.3 Output Topic Information

```
────────────────────────────────────────────────────
 CURRENT TOPIC IDENTIFIED
────────────────────────────────────────────────────

Phase:    1 - 基础知识恢复
Topic ID: 1.3
Name:     Python闭包与装饰器
Status:   IN_PROGRESS ()

Current Skill Level: ⭐ (out of ⭐⭐⭐⭐)
Target Level: ⭐⭐

Plan Reference:
  Schedule: 08_Action_Plan_2026_H1.md line XX
  Details:  02_Skills_Assessment.md Section X.X

Dependencies:
   1.1: Python环境搭建
   1.2: Python基础语法

Verification: Progress validated
────────────────────────────────────────────────────
```

---

## Step 4: User Confirmation

**Goal**: Get explicit user confirmation before proceeding.

### 4.1 Request Confirmation

```
────────────────────────────────────────────────────
 CONFIRM TOPIC
────────────────────────────────────────────────────

Ready to learn:
  [1.3] Python闭包与装饰器

Options:
   Confirm / 确认 - Proceed with this topic
   Override / 指定其他 - Specify a different topic
   Cancel / 取消 - Stop and review

Your choice:
────────────────────────────────────────────────────
```

### 4.2 Handle User Response

| Response | Action |
|----------|--------|
| Confirm / 确认 / Yes | Return topic info to caller (learning-workflow Stage 3) |
| Override / 指定其他 | Ask for topic ID, validate it exists, return that topic |
| Cancel / 取消 | Stop the workflow, return to idle state |

---

## Quick Commands

| User Says | Behavior |
|-----------|----------|
| "status" / "检查进度" | Steps 1-3 (report current state, no confirmation needed) |
| "what's next" / "下一个学什么" | Steps 1-3 (identify next topic) |
| "find topic" / "查找任务" | Full workflow (Steps 1-4) |
| "validate" / "验证进度" | Steps 1-2 only (validation report) |
| "fix progress" / "修正进度" | Step 2.4 workflow (mismatch handling) |

---

## Output Contract

When called by `learning-workflow`, this skill returns:

**Status Types**: `OK` | `MISMATCH` | `ALL_COMPLETE` | `CANCELLED`

**If status == OK**:

| Field | Example Value |
|-------|---------------|
| Topic ID | `1.3` |
| Topic Name | `Python闭包与装饰器` |
| Phase | `1 - 基础知识恢复` |
| Current Skill Level | `⭐` |
| Target Skill Level | `⭐⭐` |
| Plan Reference | `08_Action_Plan_2026_H1.md` line 142 |
| Dependencies Met | Yes/No |

**If status == MISMATCH**:
- Claimed Topic vs Actual Topic
- List of missing evidence
- User choice needed: Fix 09 / Confirm / Continue from actual

---

## Important Rules

1. **Always Validate Before Proceeding**: Never assume the plan is accurate. Always check actual skill state.

2. **User Confirmation Required**: Don't auto-proceed to practice. Wait for explicit user confirmation.

3. **Single Topic Focus**: Identify ONE topic at a time. Don't batch-identify multiple topics.

4. **Dependency Awareness**: Warn if previous topics appear incomplete, but let user decide how to proceed.

5. **Non-Destructive**: This skill only READS and REPORTS. It doesn't modify core documents (except when user explicitly chooses Option 1 in mismatch handling).

6. **Graceful Degradation**: If doc-sync cache is missing, fall back to reading `08_Action_Plan_2026_H1.md` and `09_Progress_Tracker.md` directly.

---
