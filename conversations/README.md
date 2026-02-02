# 对话历史目录

> **最后更新**：2026-01-31
> **说明**：5个Claude的对话历史按角色隔离存储

---

## 目录结构

```
conversations/
├── README.md                   # 本文件
├── summaries/                  # 通用助手生成的汇总
│   ├── 00_weekly_summary.md   # 每周汇总
│   ├── 01_learning_progress.md # 学习进度（给教学用）
│   ├── 02_market_updates.md   # 市场动态（给研究用）
│   └── 03_assessment_summary.md # 测试评估摘要（给测试用）
├── general/                   # 通用助手历史
├── assessor/                  # 测试评估历史
├── teacher/                   # 教学历史
├── researcher/                # 研究历史
├── planner/                   # 计划协调历史
└── current/                   # 当前临时对话（Git忽略）
```

---

## 对话历史可见性

### 可读所有历史
- **通用助手**（General）- 协调者
- **计划协调**（Planner）- 全局视角

### 只读自己的历史 + 汇总
- **测试评估**（Assessor）
- **教学**（Teacher）
- **研究**（Researcher）

**理由**：
- 专用Claude专注自己的职责
- 避免信息过载
- 需要全局信息时读summaries/即可

---

## 文件命名规范

### 历史对话
```
YYYYMMDD_主题.md

示例：
- 20260201_python_closure.md
- 20260204_jd_collection.md
- 20260205_weekly_review.md
```

### 汇总文档
```
00_weekly_summary.md        # 每周汇总（通用助手）
01_learning_progress.md      # 学习进度（通用助手）
02_market_updates.md         # 市场动态（通用助手）
03_assessment_summary.md     # 测试评估摘要（通用助手）
```

---

## Git跟踪规则

### ✅ Git跟踪（提交到仓库）
- `summaries/*` - 汇总文档
- `general/*` - 通用助手历史
- `assessor/*` - 测试评估历史
- `teacher/*` - 教学历史
- `researcher/*` - 研究历史
- `planner/*` - 计划协调历史

### ❌ Git忽略（不提交）
- `current/*` - 临时对话

---

## 使用说明

### 1. 查看某个Claude的历史

```bash
# 查看教学历史
ls conversations/teacher/

# 查看研究历史
ls conversations/researcher/
```

### 2. 查看汇总

```bash
# 查看每周汇总
cat conversations/summaries/00_weekly_summary.md

# 查看学习进度
cat conversations/summaries/01_learning_progress.md
```

### 3. 搜索历史

```bash
# 搜索Python相关的教学记录
grep -r "Python" conversations/teacher/

# 搜索本周的所有对话
find conversations/ -name "202602*.md"
```

---

## 协作流程

```
专用Claude（测试/教学/研究）
  ↓ 完成工作，保存到各自目录
  ↓
通用助手（每周汇总）
  ↓ 读取所有历史，生成summaries/
  ↓
计划协调（每周协调）
  ↓ 读取所有历史和summaries/，更新08/09
```

---

**相关文档**：
- [COMMIT_CONVENTIONS.md](../COMMIT_CONVENTIONS.md) - Commit规范
- [../system_prompts/](../system_prompts/) - 系统提示词
