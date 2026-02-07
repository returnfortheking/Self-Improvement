# 对话历史目录

> **最后更新**：2026-02-07
> **说明**：Skills v3.0系统的对话历史和汇总文档
> **⚠️ 注意**：5个Claude协作系统已废弃，对话历史结构已简化

---

## 目录结构

```
conversations/
├── README.md                   # 本文件
├── summaries/                  # 自动生成的汇总文档
│   ├── 00_weekly_summary.md   # 每周汇总（待创建）
│   ├── 01_learning_progress.md # 学习进度（待创建）
│   ├── 02_market_updates.md   # 市场动态（待创建）
│   └── 03_assessment_summary.md # 测试评估摘要（待创建）
├── general/                   # 通用助手历史（已废弃）
├── assessor/                  # 测试评估历史（已废弃）
├── teacher/                   # 教学历史（已废弃）
├── researcher/                # 研究历史（已废弃）
├── planner/                   # 计划协调历史（已废弃）
└── current/                   # 当前临时对话（Git忽略）
```

---

## ⚠️ 系统变更说明

### 旧系统（已废弃）
- 5个Claude协作系统
- 需要切换VSCode工作区
- 对话历史按角色隔离
- **状态**：⚠️ 已废弃，文件已移动到`archive/old_systems/`

### 新系统（Skills v3.0）
- 单一Skills系统（14个Skills）
- 无需切换工作区
- 对话历史由Claude Code自动管理
- 自动生成汇总文档到`summaries/`
- **状态**：✅ 当前使用

---

## 对话历史管理

### Skills v3.0的对话历史

**自动管理**：
- Claude Code自动保存所有对话历史
- 无需手动管理
- 无需按角色隔离

**自动汇总**：
- `auto-syncer`自动生成汇总文档到`summaries/`
- 汇总内容：
  - 每周学习进度
  - JD分析和市场更新
  - 技能评估总结
  - 面试准备情况

### 废弃角色目录

以下目录已废弃，保留为历史参考：
- `general/` - 通用助手历史
- `assessor/` - 测试评估历史
- `teacher/` - 教学历史
- `researcher/` - 研究历史
- `planner/` - 计划协调历史

**建议**：
- 这些目录已不再使用
- 可以删除或保留为历史记录
- Skills系统不依赖这些目录

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
