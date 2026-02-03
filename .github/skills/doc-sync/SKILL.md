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

## Step 2: External Resources Sync (New)

**Trigger**:
- Manual: `/更新资源`
- Automatic: Called by plan-generator when needed
- Before generating learning paths

**Goal**: Synchronize external resources (GitHub repos + tech blogs) for learning material generation

### 2.1 GitHub Repository Synchronization

**Actions**:

1. **Read Configuration**:
   - Load `references/metadata/repos_to_sync.json`
   - Get list of repositories to sync

2. **For Each Repository**:
   ```bash
   if repository does not exist locally:
       git clone --depth 1 <repo_url> <local_path>
       record: first clone successful
   else:
       git pull
       record: updated to latest commit
   ```

3. **Statistics Collection**:
   - Count new files added
   - Count modified files
   - Record commit hashes

4. **Space Limit Check**:
   - Check available space before cloning
   - Ensure total < 10GB
   - Skip and warn if limit exceeded

5. **Error Handling**:
   - Each repository independent try-catch
   - Failure does not interrupt others
   - Log errors for manual retry

**Repository Sources** (from repos_to_sync.json):
- baliyanvinay/Python-Interview-Preparation (Python interview questions)
- matacoder/senior (Advanced Python topics)
- Devinterview-io/python-interview-questions (100 core questions)
- thundergolfer/interview-with-python (Practice exercises)
- coderion/awesome-llm-and-aigc (LLM interview questions)

---

### 2.2 Tech Blog Synchronization (High-Quality Filtering)

**Actions**:

1. **Read Configuration**:
   - Load `references/metadata/blogs_to_sync.json`
   - Get blog sources and quality filters

2. **For Each Blog Source**:
   - Use `WebReader` to fetch article list
   - **Intelligent Filtering** (only keep high-quality articles):
     * Min views/likes (varies by source, 500-1500)
     * Must include tags: [Python, LLM, RAG, Agent, 算法, 系统设计]
     * Length > 1000 characters
     * Must have code examples or diagrams
     * ❌ Exclude: ads, promotions, activity notifications

3. **Download New Articles**:
   - Only download articles published since last sync
   - Save as Markdown files
   - Organize by company and date

4. **Concurrent Control**:
   - No concurrency limit (as requested)
   - Each source independent try-catch

**Blog Sources** (from blogs_to_sync.json):
- 阿里云开发者社区 (developer.aliyun.com)
- 腾讯技术 (cloud.tencent.com/developer)
- 美团技术团队 (tech.meituan.com)
- 字节技术团队 (techblog.toutiao.com)

---

### 2.3 Update Content Index

**Trigger**: After repository/blog sync completes

**Actions**:

1. **Check Index Status**:
   - Check if `references/metadata/content_index.json` exists
   - Determine: first scan or incremental scan

2. **If First Scan** (content_index.json does not exist):
   - Full scan of all files in `references/`
   - Calculate SHA-256 hash for each file
   - Extract topics, questions, tags
   - Build content_index.json
   - Estimated time: ~20 minutes

3. **If Incremental Scan** (content_index.json exists):
   - Compare file hashes
   - Only process new/modified files (skip 98.6% unchanged files)
   - Update index
   - Estimated time: <1 minute

4. **Update Statistics**:
   - Update `topic_frequency` counts
   - Calculate quality scores
   - Identify trending topics

---

### 2.4 Generate Sync Report

```
────────────────────────────────────────
✅ External Resources Sync Complete
────────────────────────────────────────

Time: 2026-02-03 23:00 - 23:45
Duration: 45 minutes

────────────────────────────────────────
📦 GitHub Repositories (5)
────────────────────────────────────────

✅ baliyanvinay/Python-Interview-Preparation
   - Status: Updated (12 new files)
   - Latest commit: abc123 (2026-02-02)

✅ matacoder/senior
   - Status: Updated (5 new files)
   - Latest commit: def456 (2026-02-01)

⚠️ awesome-llm-and-aigc
   - Status: Sync failed (connection timeout)
   - Action: Use /重试同步 awesome-llm-and-aigc

────────────────────────────────────────
📰 Tech Blogs (152 new articles)
────────────────────────────────────────

✅ 阿里云 (47 articles)
  - High-quality: 42 (filtered 5 low-quality)
  - Main topics: RAG (15), LLM (18), Agent (9)

✅ 腾讯技术 (38 articles)
  - High-quality: 35
  - Main topics: 系统设计 (12), 算法 (15)

✅ 美团技术 (35 articles)
  - High-quality: 32
  - Main topics: 分布式系统 (18)

✅ 字节技术 (32 articles)
  - High-quality: 28
  - Main topics: 推荐系统 (10), 算法 (12)

────────────────────────────────────────
💾 Storage Space
────────────────────────────────────────

Used: 2.3GB / 10GB
Available: 7.7GB

────────────────────────────────────────
📊 Content Index Updated
────────────────────────────────────────

Indexed files: 1,250
Topics found: 85
Questions extracted: 1,250

Index saved: references/metadata/content_index.json

────────────────────────────────────────

⚠️ Note: 1 repository sync failed, use /重试同步 to retry
────────────────────────────────────────
```

---

### 2.5 Automatic Follow-up Actions

**If significant new content detected** (>50 new topics):
- Automatically trigger plan-generator
- Generate updated learning path
- Present update recommendations to user

---

## Quick Commands (Updated)

| User Says | Behavior |
|-----------|----------|
| "同步文档" / "sync docs" | Step 1 only (core documents) |
| "更新资源" / "更新外部资源" | Step 2 only (external resources) |
| "更新资源 [repo-name]" | Step 2.1 only (specific repo) |
| "重试同步 [repo-name]" | Retry failed repository sync |
| "重建索引" | Force Step 2.3 full scan (rebuild index) |

---

## Implementation Status

### ✅ Implemented (v1.0)
- JD data auto-detection
- JD text extraction using `extract_text_from_screenshot`
- Automatic document updates (03, 04)
- Metadata tracking

### ✅ Implemented (v2.0 - New)
- GitHub repository synchronization
- Tech blog crawling with quality filtering
- Content index with incremental scanning
- External resource sync reports

### 📋 TODO (Optional Enhancements)
- Cache file generation for core documents
- Hash-based change detection for core documents
- Advanced JD categorization

**Current Implementation**: Both JD parsing and external resources sync use Claude's native capabilities directly, no Python scripts required.

---

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
