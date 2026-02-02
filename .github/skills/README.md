# Skills System Documentation

> **Version**: 1.0
> **Last Updated**: 2026-01-31
> **Inspired By**: [MODULAR-RAG-MCP-SERVER](https://github.com/jerry-ai-dev/MODULAR-RAG-MCP-SERVER)

---

## 📚 What is this Skills System?

This is a **standardized workflow system** based on Claude Code Skills, designed to automate and structure your learning process for the 2026 job search plan.

### Core Concept

Instead of manually managing learning, testing, and progress tracking, the **Skills system** provides:

- **Standard Operating Procedures (SOPs)** for each learning stage
- **Automated pipelines** that coordinate multiple skills
- **Progress validation** to ensure claimed progress matches reality
- **Consistent documentation** with automatic updates

---

## 🏗️ Architecture

```
.github/skills/
├── META-SKILLS
│   └── learning-workflow/      # Main orchestrator
│
├── FOUNDATION-SKILLS
│   ├── doc-sync/               # Document synchronization
│   └── progress-tracker/       # Find next topic + validate
│
├── LEARNING-SKILLS
│   ├── practice/               # Hands-on practice
│   └── assessor/               # Test understanding
│
└── UTILITY-SKILLS
    └── checkpoint/             # Save progress + git commit
```

---

## 🔄 How It Works

### Example: Learning Python Closures

**Without Skills** (Manual):
```
1. Read 08_Action_Plan_2026_H1.md to find what to learn
2. Search for tutorials online
3. Practice coding
4. Test yourself
5. Update 09_Progress_Tracker.md manually
6. Git commit manually
```

**With Skills** (Automated):
```
User says: "开始学习Python闭包"
  ↓
learning-workflow automatically:
  Stage 1: doc-sync - Sync documents
  Stage 2: progress-tracker - Identify topic 1.3
  Stage 3: practice - Practice with examples
  Stage 4: assessor - Test understanding (get 80%)
  Stage 5: checkpoint - Update progress + git commit
  ↓
Result: Python skill ⭐ → ⭐⭐, fully documented
```

---

## 📖 Skills Reference

### 1. learning-workflow (Meta-Skill)

**Purpose**: Orchestrates the complete learning pipeline

**Usage**:
```
"开始学习Python"
"继续学习"
"下一个知识点"
```

**Pipeline**: doc-sync → progress-tracker → practice → assessor → checkpoint

**File**: [learning-workflow/SKILL.md](learning-workflow/SKILL.md)

---

### 2. doc-sync

**Purpose**: Synchronize core documents and generate cache

**Usage**:
```
"同步文档"
```

**What it does**:
- Reads 01-09 core documents
- Generates JSON cache for other skills
- Detects changes via hash

**File**: [doc-sync/SKILL.md](doc-sync/SKILL.md)

---

### 3. progress-tracker

**Purpose**: Find next learning topic + validate progress

**Usage**:
```
"检查进度"
"下一个学什么"
"status"
```

**What it does**:
- Reads 08_Action_Plan_2026_H1.md
- Identifies next topic to learn
- Validates claimed progress vs actual skill state
- Handles mismatches

**File**: [progress-tracker/SKILL.md](progress-tracker/SKILL.md)

---

### 4. practice

**Purpose**: Execute hands-on practice

**Usage**:
```
"练习Python闭包"
"写代码练习"
"practice decorators"
```

**What it does**:
- Reads learning plan for topic
- Creates practice files (examples, exercises)
- Documents learning with README
- Organizes by topic in `practice/` directory

**File**: [practice/SKILL.md](practice/SKILL.md)

---

### 5. assessor

**Purpose**: Test understanding and assess skill level

**Usage**:
```
"测试我"
"assess"
"验证理解"
```

**What it does**:
- Determines assessment type (quiz, coding, interview)
- Conducts interactive assessment
- Evaluates score (need 80%+ to pass)
- Recommends skill level upgrade

**File**: [assessor/SKILL.md](assessor/SKILL.md)

---

### 6. checkpoint

**Purpose**: Save progress and update documents

**Usage**:
```
"保存进度"
"checkpoint"
"学习完成"
```

**What it does**:
- Generates learning summary
- Updates 09_Progress_Tracker.md
- Updates 02_Skills_Assessment.md
- Generates git commit message
- Asks for confirmation before committing

**File**: [checkpoint/SKILL.md](checkpoint/SKILL.md)

---

## 🎯 Quick Start

### First Time Setup

1. **Read the Meta-Skill**:
   ```
   Read: .github/skills/learning-workflow/SKILL.md
   ```

2. **Check your current status**:
   ```
   User says: "检查进度"
   ```

3. **Start learning**:
   ```
   User says: "开始学习Python"
   ```

### Daily Usage

```
Morning:
  "开始学习" → Full learning pipeline (1 topic)

Afternoon:
  "继续学习" → Continue from where you left off

Evening:
  "保存进度" → Save today's progress

Weekly:
  "status" → Check overall progress
```

---

## 📊 Design Philosophy

### Inspired by MODULAR-RAG-MCP-SERVER

This Skills system is directly inspired by [jerry-ai-dev/MODULAR-RAG-MCP-SERVER](https://github.com/jerry-ai-dev/MODULAR-RAG-MCP-SERVER), which demonstrates:

- **Meta-Skill orchestration**: Coordinating multiple skills in pipelines
- **Standard operating procedures**: Detailed SOPs for each skill
- **Output contracts**: Clear input/output specifications
- **User confirmation**: Critical decision points require user approval
- **Iteration discipline**: Limit loops to prevent infinite cycles

### Adaptations for Learning

While MODULAR focuses on **software development**, this system adapts the same patterns for **learning and skill development**:

| MODULAR | This System |
|---------|-------------|
| dev-workflow | learning-workflow |
| spec-sync | doc-sync |
| implement | practice |
| testing-stage | assessor |
| DEV_SPEC.md | 08_Action_Plan_2026_H1.md + 09_Progress_Tracker.md |

---

## ⚙️ Configuration

### File Structure

```
.github/skills/
├── README.md                   # This file
├── learning-workflow/
│   └── SKILL.md
├── doc-sync/
│   ├── SKILL.md
│   ├── sync_docs.py (TODO)
│   └── cache/ (auto-generated)
├── progress-tracker/
│   └── SKILL.md
├── practice/
│   └── SKILL.md
├── assessor/
│   └── SKILL.md
└── checkpoint/
    └── SKILL.md
```

### Document Dependencies

Skills depend on these core documents:

| Document | Purpose | Used By |
|----------|---------|---------|
| 01_Personal_Profile.md | Personal info | doc-sync |
| 02_Skills_Assessment.md | Skill levels | all skills |
| 08_Action_Plan_2026_H1.md | Learning plan | all skills |
| 09_Progress_Tracker.md | Progress tracking | all skills |

---

## 🔧 Troubleshooting

### Issue: Skills not found

**Solution**: Make sure `.github/skills/` is in your project root.

### Issue: Progress mismatch

**Solution**: Use progress-tracker's "fix progress" option to realign.

### Issue: Assessment fails repeatedly

**Solution**: After 3 iterations, the system will escalate to you for manual intervention.

---

## 🚀 Roadmap

### Phase 1: Foundation (Current)
- ✅ Create 6 core skills
- ✅ Document skill system
- ⏳ Implement sync_docs.py
- ⏳ Test full pipeline

### Phase 2: Enhancement
- ⏳ Add weekly-routine Meta-Skill
- ⏳ Add interview-prep Meta-Skill
- ⏳ Create more practice templates

### Phase 3: Automation
- ⏳ Auto-skill triggering
- ⏳ Progress analytics
- ⏳ Learning recommendations

---

## 📝 Contributing

When adding new skills:

1. Follow the MODULAR pattern (YAML frontmatter + SOP)
2. Include Output Contract section
3. Add Quick Commands table
4. Document Important Rules
5. Update this README

---

## 🙏 Acknowledgments

- **Inspired by**: [jerry-ai-dev/MODULAR-RAG-MCP-SERVER](https://github.com/jerry-ai-dev/MODULAR-RAG-MCP-SERVER)
- **Built with**: Claude Code Skills
- **Purpose**: 2026年跳槽计划

---

**Last Updated**: 2026-01-31
**Status**: 🚧 Work in Progress
**Version**: 1.0
