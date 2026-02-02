---
name: learning-workflow
description: Master orchestrator for learning workflow. Use when user says "开始学习", "继续学习", "next topic", "practice", or asks to start learning. Coordinates doc-sync, progress-tracker, practice, assessor, and checkpoint skills in a pipeline to complete one learning topic per iteration.
metadata:
  category: orchestration
  triggers: "开始学习, 继续学习, next topic, practice, 学习"
allowed-tools: Read
---

# Learning Workflow Orchestrator

You are the **Learning Manager AI** for the 跳槽计划. When the user asks to start learning, you MUST execute the following pipeline **in order**.

> **This is a Meta-Skill**: It orchestrates other skills. Each stage's **specific implementation details** are defined in the respective skill's SKILL.md file. This file only defines **pipeline flow** and **inter-stage coordination**.

---

## Pipeline Stages

| Stage | Skill | Description | Skill File |
|-------|-------|-------------|------------|
| 1 | `doc-sync` | Sync core documents | `.github/skills/doc-sync/SKILL.md` |
| 2 | `progress-tracker` | Find next learning topic | `.github/skills/progress-tracker/SKILL.md` |
| 3 | `practice` | Execute practice/implementation | `.github/skills/practice/SKILL.md` |
| 4 | `assessor` | Assess understanding | `.github/skills/assessor/SKILL.md` |
| 5 | `checkpoint` | Save progress | `.github/skills/checkpoint/SKILL.md` |

> **For detailed execution steps, completion criteria, and output formats for each stage, refer to the corresponding SKILL.md file.**

---

## Pipeline Flow

```
                    ┌──────────────────┐
                    │  User: "开始学习"  │
                    └────────┬─────────┘
                             ▼
                  ┌──────────────────────┐
                  │  Stage 1: doc-sync   │
                  └────────┬─────────────┘
                           ▼
                  ┌──────────────────────┐
                  │ Stage 2: progress-   │
                  │         tracker      │
                  └────────┬─────────────┘
                           │
                     ️ Exception? ──→ User Confirm → Update 09 → Back to Stage1
                           │
                           ▼
                  ┌──────────────────────┐
          ┌──────▶│ Stage 3: practice    │
          │       └────────┬─────────────┘
          │                ▼
          │       ┌──────────────────────┐
          │       │ Stage 4: assessor    │
          │       └────────┬─────────────┘
          │                ▼
          │           ┌─────────┐
          │           │ Pass?   │
          │           └────┬────┘
          │     No         │         Yes
          │     ┌──────────┴──────────┐
          │     ▼                     ▼
          │ Iteration < 3?     ┌──────────────────────┐
          │     │              │  Stage 5: checkpoint │
          │ Yes │              └──────────────────────┘
          └─────┘
                │ No (iteration >= 3)
                ▼
          ┌──────────────────┐
          │ Escalate to User │
          └──────────────────┘
```

---

## Inter-Stage Data Flow

The orchestrator is responsible for passing context between stages:

| From | To | Data Passed |
|------|----|-------------|
| Stage 2 | Stage 3 | Topic ID, Topic Name, Current Skill Level |
| Stage 3 | Stage 4 | Practice Results, Code Submitted |
| Stage 4 | Stage 3 | Assessment Failures (on failure, for iteration) |
| Stage 4 | Stage 5 | Assessment Results, Iteration Count |
| Stage 2,3,4 | Stage 5 | Topic ID, New Skill Level, Learning Summary |

---

## Quick Commands

> **Important Note**: Each execution of "开始学习" completes **one learning topic** (e.g., Python闭包 → Python装饰器), not an entire phase (e.g., Python基础 → RAG).

| User Says | Pipeline Behavior |
|-----------|-------------------|
| "开始学习" / "开始学习Python" | Full pipeline (Stage 1-5), completes **next topic** |
| "继续学习" / "继续练习" | Skip to Stage 3 (assumes topic known) |
| "检查进度" / "status" | Stage 2 only |
| "测试我" / "test me" | Stage 4 only |
| "保存进度" / "save progress" | Stage 5 only |

---

## Orchestrator Rules

1. **Delegation**: Each stage's specific logic is defined by its skill; the orchestrator only handles invocation and flow
2. **Action Plan is Source of Truth**: Learning progress is based on `08_Action_Plan_2026_H1.md` and `09_Progress_Tracker.md`
3. **Stage Order**: Execute in 1→2→3→4→5 order unless explicitly specified otherwise
4. **Single Topic**: Each pipeline run completes **one learning topic**
5. **User Confirmation**: Wait for user confirmation after Stage 2 before continuing
6. **Assess Before Checkpoint**: Stage 4 must pass before entering Stage 5
7. **Iteration Discipline**: Enforce 3-iteration limit
8. **Two-Step Checkpoint**: Stage 5 requires two user confirmations:
   - First: Verify learning summary (then auto-update 09_Progress_Tracker.md)
   - Second: Decide whether to execute git commit
