---
name: doc-sync
description: Synchronize core documents status and generate cache. Read core documents (01-09) and generate JSON cache for other skills to consume. Foundation for all learning operations. Use when user says "同步文档", "sync docs", or before any learning-dependent task.
metadata:
  category: documentation
  triggers: "同步文档, sync docs, update docs"
allowed-tools: Read Write
---

# Doc Sync

This skill synchronizes the core documents (01-09) and generates JSON cache files for other skills to consume.

> **This is a prerequisite for all learning-based operations.** Other skills depend on the cache files to perform their tasks.

---

## How to Use

### Used in learning-workflow (Automatic)

When you trigger learning-workflow (e.g., "开始学习" or "继续学习"), **doc-sync runs automatically as Stage 1**. No manual action needed.

### Manual Sync (Edge Cases Only)

Only manually run if:
- You edited core documents (01-09) outside of workflow
- Cache files are corrupted or missing
- Testing a single skill in isolation

---

## Core Documents

This skill reads and synchronizes the following documents:

| Document | Content | Priority |
|----------|---------|----------|
| `01_Personal_Profile.md` | 个人信息与求职意向 | High |
| `02_Skills_Assessment.md` | 技术栈评估与规划 | High |
| `08_Action_Plan_2026_H1.md` | 2026年上半年行动计划 | Critical |
| `09_Progress_Tracker.md` | 进度跟踪 | Critical |

---

## Directory Structure

```
.github/skills/doc-sync/
├── SKILL.md              ← This file
├── sync_docs.py          ← Sync script (TODO: implement)
├── .docs_hash            ← Hash file for change detection
└── cache/                ← Generated cache files
    ├── 01_personal_profile.json
    ├── 02_skills_assessment.json
    ├── 08_action_plan.json
    └── 09_progress_tracker.json
```

---

## What the Sync Script Does

The script performs these operations:
1. Read core documents from project root
2. Calculate hash to detect changes
3. Parse documents and generate JSON cache
4. Save cache files to `cache/` directory

---

## Cache File Format

### 08_action_plan.json

```json
{
  "last_updated": "2026-01-31",
  "current_phase": "Phase 1: 基础知识恢复",
  "topics": [
    {
      "id": "1.1",
      "name": "Python基础恢复",
      "status": "in_progress",
      "priority": "critical",
      "estimated_hours": 40
    },
    {
      "id": "1.2",
      "name": "RAG基础知识",
      "status": "not_started",
      "priority": "high",
      "estimated_hours": 30
    }
  ]
}
```

### 09_progress_tracker.json

```json
{
  "last_updated": "2026-01-31",
  "overall_progress": "15%",
  "skills": {
    "python": {
      "current_level": "⭐",
      "target_level": "⭐⭐⭐⭐",
      "last_topic": "Python闭包",
      "next_topic": "Python装饰器"
    },
    "rag": {
      "current_level": "⭐⭐",
      "target_level": "⭐⭐⭐⭐⭐",
      "last_topic": "Vector DB基础",
      "next_topic": "RAG架构设计"
    }
  }
}
```

---

## Output Contract

When called by `learning-workflow`, this skill returns:

**Status Types**: `OK` | `CHANGED` | `ERROR`

**If status == OK**:

```json
{
  "status": "OK",
  "cache_path": ".github/skills/doc-sync/cache/",
  "documents_synced": ["01", "02", "08", "09"],
  "last_sync": "2026-01-31T15:30:00Z"
}
```

---

## Important Notes

- **Never edit cache files directly** — they are auto-generated
- **Always edit core documents (01-09.md)** and re-run the sync script
- Cache files are used by other skills for fast access to structured data

---

## Implementation TODO

This skill needs a Python script `sync_docs.py` to:
1. Parse markdown files
2. Extract structured data
3. Generate JSON cache
4. Detect changes via hash

**For now, this skill is a placeholder. Other skills should read core documents directly until this is implemented.**
