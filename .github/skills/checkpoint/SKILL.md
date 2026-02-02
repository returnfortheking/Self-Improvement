---
name: checkpoint
description: Summarize completed learning, update progress tracking in 09_Progress_Tracker.md and 02_Skills_Assessment.md, and prepare for next iteration. Final stage of learning-workflow pipeline. Use when learning and assessment is completed, or when user says "完成检查点", "checkpoint", "保存进度", "save progress", "学习完成".
metadata:
  category: progress-tracking
  triggers: "checkpoint, save progress, 完成检查点, 保存进度, 学习完成"
allowed-tools: Read Write
---

# Checkpoint & Progress Persistence

This skill handles **learning completion summarization** and **progress tracking synchronization**. It ensures that completed learning is properly documented and the project progress in `09_Progress_Tracker.md` and `02_Skills_Assessment.md` stays up-to-date.

> **Single Responsibility**: Summarize → Persist → Prepare Next

---

## When to Use This Skill

- When a learning topic and assessment is **completed**
- When you need to **manually update progress** in core documents
- When you want to **generate a commit message** for completed learning
- As the **final stage** of the `learning-workflow` pipeline

---

## Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 1           Step 1.5                 Step 2              Step 3      │
│  ────────         ────────                 ────────            ────────     │
│  Summarize   →   User Confirm (WHAT)  →   Persist Progress →  Commit Prep │
│  (Summarize)      (Verify learning done)    (Update docs)      (WHETHER)   │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────┐
                    │ Assessment Passed │
                    └────────┬─────────┘
                             ▼
                  ┌──────────────────────┐
                  │  Step 1: Summarize   │
                  │  Generate summary    │
                  └────────┬─────────────┘
                           ▼
                  ┌──────────────────────┐
                  │ Step 1.5: User       │
                  │ Confirmation         │
                  │ Wait for user OK     │
                  └────────┬─────────────┘
                           │
                     User OK? ──No──→ Revise summary → Back to Step1
                           │
                       Yes ▼
                  ┌──────────────────────┐
                  │ Step 2: Persist      │
                  │ Progress             │
                  │ Update core docs     │
                  └────────┬─────────────┘
                           ▼
                  ┌──────────────────────┐
                  │ Step 3: Commit Prep  │
                  │ Generate commit msg  │
                  │ Wait for user OK     │
                  └────────┬─────────────┘
                           │
                     User OK? ──No──→ Skip commit → Flow end
                           │
                       Yes ▼
                  ┌──────────────────────┐
                  │  Execute git commit  │
                  └────────┬─────────────┘
                           ▼
                  ┌──────────────────────┐
                  │   Checkpoint Done  │
                  └──────────────────────┘
```

---

## Step 1: Learning Summary

**Goal**: Generate a clear, structured summary of completed learning.

### 1.1 Collect Information

Gather the following from the current session:
- **Topic ID**: e.g., `1.3`, `2.1`
- **Topic Name**: e.g., "Python闭包与装饰器"
- **Practice Results**: What was practiced (files, exercises)
- **Assessment Results**: Quiz score, understanding level
- **Skill Level Change**: Before and after

### 1.2 Generate Summary Report

**Output Format**:
```
────────────────────────────────────────────────────
 TOPIC COMPLETED: [Topic ID] [Topic Name]
────────────────────────────────────────────────────

 Practice Files:
    - practice/python/01_closures/README.md
    - practice/python/01_closures/examples.py
    - practice/python/01_closures/exercises.py

 Assessment Results:
    - Quiz: 4/5 correct (80%)
    - Coding: Completed successfully
    - Understanding: ⭐⭐⭐⭐ (good)

 Skill Level: ⭐ → ⭐⭐ (upgraded)

 Plan Reference: 08_Action_Plan_2026_H1.md Section [X.Y]
────────────────────────────────────────────────────
```

---

## Step 1.5: User Confirmation (Verify WHAT Was Done)

**Goal**: Present summary to user for verification before persisting progress.

**This confirms WHAT learning was completed** - validating the summary accuracy, not whether to save it.

### 1.5.1 Confirmation Prompt

**Output Format**:
```
════════════════════════════════════════════════════
 Please Verify Learning Summary / 请验证学习总结
════════════════════════════════════════════════════

 Topic: [Topic ID] [Topic Name]
 Plan Reference: 08_Action_Plan_2026_H1.md Section [X.Y]

 Practice Files:
    - practice/python/01_closures/README.md
    - practice/python/01_closures/examples.py
    - practice/python/01_closures/exercises.py

 Assessment Results:
    - Quiz: 4/5 correct (80%)
    - Understanding: ⭐⭐⭐⭐

 Skill Level: ⭐ → ⭐⭐

════════════════════════════════════════════════════
 Is this summary accurate?
 以上总结是否准确？

   Please reply: "confirm" / "确认" to save progress
                "revise" / "修改" to regenerate summary

 Note: This only verifies the summary. Core documents will be
 updated after confirmation. Git commit decision comes later.
════════════════════════════════════════════════════
```

### 1.5.2 Handle User Response

| User Response | Action |
|---------------|--------|
| "confirm" / "yes" / "确认" / "是" | Proceed to Step 2 |
| "revise" / "no" / "修改" / "否" | Ask user what needs to be corrected, then regenerate summary |

**Important**: Do NOT proceed to Step 2 until user explicitly confirms.

---

## Step 2: Persist Progress

**Goal**: Update `09_Progress_Tracker.md` and `02_Skills_Assessment.md` to mark the topic as completed.

> **Auto-Execute**: This step runs automatically after Step 1.5 user confirmation. No additional user input required.

### 2.1 Locate Topic in Progress Documents

1. Read `09_Progress_Tracker.md`
2. Find the topic by its identifier pattern
3. Also read `02_Skills_Assessment.md` for skill level updates

### 2.2 Update Progress Markers

**Supported Marker Styles**:

| Before | After | Style |
|--------|-------|-------|
| `[ ]` | `[x]` | Checkbox |
| `` | `` | Emoji |
| `### 1.3：任务名` | `### 1.3：任务名 ` | Title suffix |

### 2.3 Update Skill Levels in 02_Skills_Assessment.md

**CRITICAL**: After updating the topic status, you MUST also update the skill level in `02_Skills_Assessment.md`.

**What to Update**:
1. **Current Skill Level**: Update the star rating (⭐ → ⭐⭐)
2. **Last Learned Topic**: Update to current topic
3. **Next Topic**: Suggest next learning topic

**Example**:
```markdown
Before:
| Python | ⭐ | 基础知识恢复 | 2026-01-28 |

After (when 1.3 completed):
| Python | ⭐⭐ | 闭包与装饰器 | 2026-01-31 |
```

### 2.4 Step 2 Output Format

**Output after updating documents**:
```
────────────────────────────────────
✅ Progress Documents Updated
────────────────────────────────────
Topic: [Topic ID] [Topic Name]
Status: [ ] -> [x]
Skill Level: ⭐ -> ⭐⭐

Files Updated:
  - 09_Progress_Tracker.md
  - 02_Skills_Assessment.md
────────────────────────────────────
```

---

## Step 3: Commit Preparation

**Goal**: Generate structured commit message and ask user whether to commit.

### 3.1 Commit Message Template

**Subject Format**:
```
[Learning] [Topic X.Y] <brief description>
```

**Template Definition**:
| Field | Description | Example |
|-------|-------------|---------|
| `[Learning]` | Commit type indicator | `Learning` |
| `[Topic X.Y]` | Topic number | `[Topic 1.3]` |
| `<brief description>` | What was learned (< 50 chars) | `learn Python closures` |

### 3.2 Generate Commit Message

**Output Format**:
```
════════════════════════════════════════════════════
 COMMIT MESSAGE / 提交信息
════════════════════════════════════════════════════

【Subject】
[Learning] [Topic 1.3] learn Python closures and decorators

【Description】
Completed 08_Action_Plan_2026_H1.md Topic 1.3: Python闭包与装饰器

Learning:
- Explained closure concepts and __closure__ attribute
- Provided 5 code examples
- Completed 3 practice exercises
- Passed assessment (4/5 correct, 80%)

Practice:
- practice/python/01_closures/README.md
- practice/python/01_closures/examples.py
- practice/python/01_closures/exercises.py

Skill Level: ⭐ → ⭐⭐

Refs: 08_Action_Plan_2026_H1.md Section 3.2
════════════════════════════════════════════════════
```

### 3.3 User Commit Confirmation (Decide WHETHER to Commit)

**This confirms WHETHER to commit** - deciding if changes should be committed to git now or manually later.

**Prompt User**:
```
────────────────────────────────────
 Do you want me to commit these changes?
 是否需要帮您执行 git commit？
────────────────────────────────────

Please reply / 请回复:
  "yes" / "commit" / "是" → Execute git add + git commit
  "no" / "skip" / "否"   → End flow, you can commit manually later
────────────────────────────────────
```

### 3.4 Execute Commit (If Confirmed)

**If user confirms**:
```bash
# Stage all changed files
git add 09_Progress_Tracker.md 02_Skills_Assessment.md practice/

# Commit with generated message
git commit -m "<subject>" -m "<description>"
```

**Success Output**:
```
────────────────────────────────────
 COMMIT SUCCESSFUL
────────────────────────────────────
Commit: <short hash>
Branch: <current branch>

Progress saved, topic [Topic ID] completed!
进度已保存，主题 [Topic ID] 已完成！
────────────────────────────────────
```

### 3.5 Skip Commit (If Declined)

**If user declines**:
```
────────────────────────────────────
 WORKFLOW COMPLETED (No Commit)
────────────────────────────────────
 Core documents updated
 Git commit skipped

You can manually commit later with:
  git add .
  git commit -m "<subject>" -m "<description>"

Topic [Topic ID] checkpoint completed!
主题 [Topic ID] 检查点完成！
────────────────────────────────────
```

---

## Quick Commands

| User Says | Behavior |
|-----------|----------|
| "checkpoint" / "完成检查点" | Full workflow (Step 1-3) with confirmations |
| "save progress" / "保存进度" | Step 1.5-2 only (confirm + persist) |
| "commit message" / "生成提交信息" | Step 3 only (generate commit message) |
| "commit for me" / "帮我提交" | Step 3 + execute git commit |

---

## Important Rules

1. **Always Update Core Documents**: Update both `09_Progress_Tracker.md` and `02_Skills_Assessment.md`.

2. **Preserve Existing Format**: Match the marker style already used in the document.

3. **Atomic Updates**: Update ONE topic at a time. Don't batch-update multiple topics.

4. **Two User Confirmations Required**:
   - Step 1.5: User must confirm learning summary before persisting
   - Step 3.3: User must confirm before git commit
   - **NEVER skip these confirmations!**

5. **Update Both Progress and Skill Level**: When marking a topic complete, update both the topic status in 09 and the skill level in 02.

6. **Traceability**: Every checkpoint must reference the specific plan section that defined the topic.

---
