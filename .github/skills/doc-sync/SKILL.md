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

---

## Step 1.5: JD Data Auto-Sync (Automatic)

**Goal**: Automatically detect and parse new JD data, update core documents.

> **This runs automatically every time** learning-workflow Stage 1 executes.
> User never needs to manually trigger this.
> Uses Claude's native multimodal capabilities (no Python scripts needed).

### 1.5.1 Detect New JD Images

**Actions**:
1. Read `jd_data/metadata.json`
2. Scan `jd_data/images/` directory
3. Compare with metadata.json to identify new images
4. List new image files

**Detection Logic**:
```python
# Pseudo-code
known_images = metadata.json.get('processed_images', [])
current_images = os.listdir('jd_data/images/')
new_images = [img for img in current_images if img not in known_images]
```

### 1.5.2 Parse New JDs (Using Claude's Vision)

**For each new image**:

1. **Read the image** using Read tool
2. **Extract text** using `extract_text_from_screenshot` tool
3. **Parse JD information**:
   ```
   Company: [从文本中提取]
   Position: [从文本中提取]
   Salary: [解析薪资范围]
   Location: [从文本中提取]
   Requirements: [从文本中提取]
   ```

**Example Extraction Process**:
```
Input: jd_data/images/2026-01-28_001_字节_大模型应用.jpg
→ Step 1: Read image file
→ Step 2: Use extract_text_from_screenshot tool
→ Output:
  "公司：字节跳动
   岗位：大模型应用算法工程师
   薪资：80-110K·15薪
   地点：上海
   要求：
   - 熟悉PyTorch
   - 有大模型应用经验
   - ..."

→ Step 3: Parse to structured data:
  {
    "company": "字节跳动",
    "position": "大模型应用算法工程师",
    "salary_min": 80,
    "salary_max": 110,
    "salary_months": 15,
    "location": "上海",
    "requirements": ["PyTorch", "大模型应用"]
  }
```

### 1.5.3 Update Core Documents

**Target 1: 03_Market_Research_JD_Analysis.md**

Update sections:
- Update total position count (87 → 92)
- Add new positions to relevant category
- Update salary statistics
- Update source information

**Update Location in Document**:
Find section `## 📊 岗位数据统计` and update:
```markdown
| 数据集 | 岗位数 | 采集时间 | 来源 |
|--------|--------|----------|------|
| 初始数据集 | 87 | 2026-01-28 | 招聘网站 |
| 新增数据 | 5 | 2026-02-02 | jd_data/images/ |
| **总计** | **92** | - | - |
```

**Target 2: 04_Target_Positions_Analysis.md**

Update sections:
- Add new position details to relevant category
- Update skill requirements summary
- Update company list if new companies found

### 1.5.4 Update Metadata

**Update jd_data/metadata.json**:
```json
{
  "last_updated": "2026-02-02",
  "total_positions": 92,
  "processed_images": [
    "2026-01-28_001_字节_大模型应用.jpg",
    "2026-01-28_002_阿里_RAG开发.jpg",
    "2026-02-02_003_腾讯_AI架构.jpg"
  ],
  "last_sync": "2026-02-02T16:30:00Z",
  "collections": [
    {
      "date": "2026-01-28",
      "source": "招聘网站截图",
      "count": 87
    },
    {
      "date": "2026-02-02",
      "source": "jd_data/images/",
      "count": 5,
      "note": "Auto-synced by doc-sync"
    }
  ]
}
```

### 1.5.5 Generate Update Report

```
────────────────────────────────────────────────────
✅ JD DATA AUTO-SYNCED
────────────────────────────────────────────────────
New JDs Found: 5
Images Processed:
  ✅ 2026-01-28_001_字节_大模型应用.jpg
  ✅ 2026-01-28_002_阿里_RAG开发.jpg
  ✅ 2026-01-30_003_腾讯_AI架构.jpg
  ✅ 2026-02-02_004_百度_大模型.jpg
  ✅ 2026-02-02_005_美团_AI应用.jpg

Parsed Information:
  - Companies: 字节跳动, 阿里, 腾讯, 百度, 美团
  - Positions: 5
  - Salary Range: 30-110K

Documents Updated:
  ✅ 03_Market_Research_JD_Analysis.md
     - Positions: 87 → 92
     - Added: 5 new positions to category

  ✅ 04_Target_Positions_Analysis.md
     - Updated: 5 position details
     - Updated: skill requirements summary

Metadata Updated:
  ✅ jd_data/metadata.json
  - last_sync: 2026-02-02T16:30:00Z
────────────────────────────────────────────────────
```

### 1.5.6 No New JDs Case

If no new images detected:

```
────────────────────────────────────────────────────
ℹ️  NO NEW JD DATA
────────────────────────────────────────────────────
Current positions: 92
Last scan: 2026-02-02T16:30:00Z
Scanned directory: jd_data/images/
Status: No new images to process
────────────────────────────────────────────────────
```

### 1.5.7 Error Handling

**If image extraction fails**:
```
⚠️ WARNING: Failed to extract text from image
  Image: 2026-02-02_XXX.jpg
  Error: [Error details]
  Action: Skip this image, continue with others
```

**If document update fails**:
```
❌ ERROR: Failed to update document
  Document: 03_Market_Research_JD_Analysis.md
  Error: [Error details]
  Action: Rollback metadata changes, report to user
```

---

## Important Notes

- **Never edit cache files directly** — they are auto-generated
- **Always edit core documents (01-09.md)** and re-run the sync script
- Cache files are used by other skills for fast access to structured data
- **JD parsing is fully automatic** — triggered on every learning-workflow run
- **No Python scripts needed** — uses Claude's native multimodal capabilities

---

## Implementation Status

### ✅ Implemented
- JD data auto-detection
- JD text extraction using `extract_text_from_screenshot`
- Automatic document updates (03, 04)
- Metadata tracking

### 📋 TODO (Optional Enhancements)
- Cache file generation for faster access
- Hash-based change detection
- Advanced JD categorization

**Current Implementation**: JD parsing uses Claude's native vision capabilities directly, no Python scripts required.
