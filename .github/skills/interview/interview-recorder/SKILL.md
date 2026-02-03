---
name: interview-recorder
description: Record interview experience, process audio files, and generate interview summary documents. Use when user says "记录面试", "面试总结", "interview".
metadata:
  category: interview
  triggers: "记录面试, 面试总结, interview, 面试记录, 面经"
allowed-tools: Read Write Bash
---

# Interview Recorder

Record interview experience, process audio files, and generate structured interview documents.

> **核心价值**：沉淀面试经验，识别知识盲区，为后续面试做准备
> **数据管理**：使用interview_data/统一管理所有面试相关文件
> **更新时机**：每次面试后立即记录

---

## File Structure (类似jd_data)

```
interview_data/
├── audio/              # 音频文件（原始）
│   ├── 20260215_字节_AI前端.mp3
│   └── 20260220_小红书_Agent研发.m4a
├── transcripts/        # 转录文本
│   ├── 20260215_字节_AI前端_transcript.md
│   └── 20260220_小红书_Agent研发_transcript.md
├── summaries/          # 面经文档
│   ├── 20260215_字节_AI前端.md
│   └── 20260220_小红书_Agent研发.md
├── metadata.json       # 元数据索引
└── README.md           # 说明文档
```

---

## When to Use This Skill

### 触发时机
- 面试结束后立即使用
- 用户说 "记录面试"、"面试总结"、"interview"
- 用户提供面试音频文件

---

## Workflow

```
Collect Basic Info
       ↓
Process Audio (if provided) → Move + Transcribe
       ↓
Extract Interview Content
       ↓
Generate Summary Document
       ↓
Update Metadata
       ↓
Update Progress Documents (08/09)
       ↓
Git Commit (with confirmation)
```

---

## Step 1: Collect Basic Information

询问并收集以下信息：

### 1.1 必填信息

- **公司名称**：e.g., "字节跳动"
- **岗位名称**：e.g., "AI前端工程师"
- **面试轮次**：1st/2nd/3rd/HR/Boss
- **面试日期**：YYYY-MM-DD
- **面试方式**：onsite/remote/phone

### 1.2 可选信息

- **薪资范围**：e.g., "40-70K"
- **面试结果**：Waiting/Rejected/Offer
- **面试官**：[optional]

---

## Step 2: Process Audio (if provided)

### 2.1 移动音频文件

如果用户提供了音频文件路径：

```bash
# 示例：用户提供 /path/to/recording.mp3
# 移动到 interview_data/audio/
mv "/path/to/recording.mp3" "interview_data/audio/20260215_字节_AI前端.mp3"
```

### 2.2 转录音频

**说明**：本skill期望音频转录由外部工具或用户完成。

如果提供转录文本，保存到：
```
interview_data/transcripts/YYYYMMDD_公司_岗位_transcript.md
```

**转录文本格式**：

```markdown
# 转录文本

**面试日期**：YYYY-MM-DD
**公司**：[公司名称]
**岗位**：[岗位名称]

[Full transcript text]

---
转录时间：YYYY-MM-DD HH:MM
```

**如果没有转录**：跳过此步骤，直接进入Step 3。

---

## Step 3: Extract Interview Content

### 3.1 技术问题

提取所有技术问题：

```markdown
### Q1: [Question]

**我的回答**：
[Your answer]

**面试官反馈**：
[Optional feedback]

**改进方向**：
- [ ] Review: [Topic 1]
- [ ] Practice: [Topic 2]
```

### 3.2 行为问题

```markdown
### Q: [Behavioral Question]

**我的回答**：
[Your answer]

**评估**：
[Self-assessment]
```

### 3.3 系统设计（如果有）

```markdown
### 系统设计：[Design Title]

**题目要求**：
[Requirements]

**我的方案**：
[Approach]

**反馈**：
[Feedback]
```

---

## Step 4: Generate Summary Document

创建 `interview_data/summaries/YYYYMMDD_公司_岗位.md`：

```markdown
# 面经：[公司名称] - [岗位名称]

**面试日期**：YYYY-MM-DD
**面试轮次**：1st/2nd/3rd
**面试方式**：onsite/remote
**面试官**：[optional]

---

## 基本信息

- **公司**：[公司名称]
- **岗位**：[岗位名称]
- **薪资范围**：[optional]
- **结果**：Waiting/Rejected/Offer

---

## 技术问题

### Q1: [Question]

**我的回答**：
[Your answer]

**面试官反馈**：
[Optional feedback]

**改进方向**：
- [ ] [Topic 1]
- [ ] [Topic 2]

### Q2: [Question]
...

---

## 行为问题

### Q: [Tell me about a time you...]

**我的回答**：
[Your answer]

**评估**：
[Self-assessment]

---

## 系统设计（如果有）

**题目**：[Design X]

**我的方案**：[Approach]

**反馈**：[Feedback]

---

## 总结与反思

### 做得好的地方 ✅

- [ ] Good point 1
- [ ] Good point 2

### 需要改进 🔴

- [ ] Weakness 1
- [ ] Weakness 2

### 后续行动 📋

- [ ] Review: [Topic 1]
- [ ] Practice: [Topic 2]
- [ ] Learn: [Topic 3]

---

## 文件链接

- **音频记录**：`audio/20260215_字节_AI前端.mp3`
- **转录文本**：`transcripts/20260215_字节_AI前端_transcript.md`
- **关联JD**：`jd_data/images/xxx.png` (optional)

---

**文档创建时间**：YYYY-MM-DD HH:MM
**最后更新**：YYYY-MM-DD HH:MM
```

---

## Step 5: Update Metadata

更新 `interview_data/metadata.json`：

```json
{
  "last_updated": "2026-02-02",
  "total_interviews": 1,
  "interviews": [
    {
      "date": "2026-02-15",
      "company": "字节跳动",
      "position": "AI前端工程师",
      "round": "1st",
      "status": "Waiting",
      "audio": "audio/20260215_字节_AI前端.mp3",
      "transcript": "transcripts/20260215_字节_AI前端_transcript.md",
      "summary": "summaries/20260215_字节_AI前端.md"
    }
  ]
}
```

---

## Step 6: Update Progress Documents

### 6.1 更新 09_Progress_Tracker.md

添加到 "三、每周进度更新" section：

```markdown
### 第X周（2026.MM.DD - 2026.MM.DD）

**面试记录**：
- [ ] 2026-02-15 字节 AI前端 (1st) - Waiting
- [ ] 2026-02-20 小红书 Agent研发 (2nd) - Rejected
```

### 6.2 更新 08_Action_Plan_2026_H1.md

如果面试发现技能差距，更新学习计划：

```markdown
### 面试反馈调整

根据面试反馈，需要补充：
- [ ] Python异步编程（字节面试问题）
- [ ] RAG生产级实践（小红书问题）
```

---

## Step 7: Git Commit (with confirmation)

### 7.1 生成Commit Message

**Subject**：
```
[Interview] [Company] [Position] interview record
```

**Description**：
```
Interview Date: YYYY-MM-DD
Position: [Company] [Position]
Round: 1st/2nd/3rd

Questions Recorded:
- X technical questions
- Y behavioral questions
- Z system design

Areas to Improve:
- [ ] Topic 1
- [ ] Topic 2

Interview Doc: interview_data/summaries/YYYYMMDD_公司_岗位.md
```

### 7.2 询问用户

```
────────────────────────────────────
是否提交面试记录到Git？
────────────────────────────────────

回复：
  "yes" / "commit" / "是" → 执行 git add + git commit
  "no" / "skip" / "否"   → 跳过提交
────────────────────────────────────
```

---

## Quick Commands

| 用户命令 | 行为 |
|---------|------|
| `/记录面试` | 完整流程（收集信息 → 生成文档 → 更新元数据） |
| `/查看面试` | 显示所有面试记录列表 |
| `/面试总结 [公司] [岗位]` | 生成指定公司的面试总结 |
| `/转录音频 [文件路径]` | 转录音频文件（需要外部工具支持） |

---

## Important Rules

1. **文件命名规范**：严格遵循 `YYYYMMDD_公司_岗位` 格式
2. **音频处理**：音频文件先移动，再更新metadata
3. **结构化存储**：使用interview_data/统一管理
4. **进度同步**：每次记录后更新08/09文档
5. **改进导向**：总是识别需要改进的地方
6. **隐私保护**：谨慎处理敏感信息

---

## Creating Directory Structure

如果目录不存在，自动创建：

```bash
mkdir -p interview_data/{audio,transcripts,summaries}
touch interview_data/metadata.json
```

---

## 示例

### 输入

```
用户: /记录面试

AI引导过程：
1. 询问基本信息
   - 公司：字节跳动
   - 岗位：AI前端工程师
   - 轮次：1st
   - 日期：2026-02-15

2. 询问音频文件
   - 用户：提供了 /path/to/recording.mp3

3. 询问面试内容
   - 用户口述或文字输入

4. 生成面经文档
   - 保存到 interview_data/summaries/20260215_字节_AI前端.md

5. 更新元数据
   - 更新 metadata.json

6. 询问是否提交
   - 用户：是

7. Git提交
```

---

## Error Handling

### 错误1：interview_data目录不存在

**处理**：自动创建目录结构

### 错误2：metadata.json格式错误

**处理**：备份现有文件，重新创建

### 错误3：音频文件不存在

**处理**：跳过音频处理，继续生成文档

---

**更新时间**：2026-02-02
**维护者**：learning-workflow orchestrator
