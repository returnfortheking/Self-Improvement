# 5个Claude协作系统 - 设计文档

> **创建日期**：2026-01-31
> **目的**：记录5个Claude协作系统的设计决策和演进过程
> **重要性**：⭐⭐⭐⭐⭐ 所有Claude的全局视野基础文档

---

## 🎯 设计目标

### 用户需求
- **背景**：华为16级工程师，目标在2026年6月30日前完成跳槽
- **目标薪资**：70-100万/年
- **工作地点**：上海，必须保证双休
- **核心问题**：需要一个高效的AI协作系统来管理学习、测试、研究、计划

### 解决方案
设计5个专业化的Claude角色，通过VSCode工作区切换实现分工协作。

---

## 📐 架构设计

### 5个角色定义

| 角色 | 主要职责 | 权限范围 |
|------|----------|----------|
| **通用助手** (General) | 协调者、信息枢纽 | 可读所有对话历史，可写summaries/ |
| **测试评估** (Assessor) | 技能测试、评估 | 可写02/05/00文档 |
| **教学** (Teacher) | 技术教学、问答 | 可写conversations/teacher/ |
| **研究** (Researcher) | JD收集、市场分析 | 可写03/04文档 |
| **计划协调** (Planner) | 进度跟踪、计划调整 | 可读所有对话，可写08/09文档 |

### 对话历史可见性设计

**关键决策**：分层可见性，避免信息过载

```
┌─────────────────────────────────────┐
│ 通用助手 (General)                   │
│ - 可读：所有对话历史                  │
│ - 理由：需要全局视野，生成每周汇总    │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ 计划协调 (Planner)                   │
│ - 可读：所有对话历史                  │
│ - 理由：需要协调全局，识别风险        │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ 测试评估 (Assessor)                  │
│ - 可读：自己的历史 + summaries/      │
│ - 理由：专注测试，不需要所有细节      │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ 教学 (Teacher)                       │
│ - 可读：自己的历史 + summaries/      │
│ - 理由：专注教学，避免干扰            │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ 研究 (Researcher)                   │
│ - 可读：自己的历史 + summaries/      │
│ - 理由：专注研究，提高效率            │
└─────────────────────────────────────┘
```

---

## 🔄 设计演进历程

### 迭代1：手动保存对话历史（已废弃）

**初始想法**：
- 在每个Claude的系统提示词中添加手动保存指令
- 用户需要明确说"保存对话到xxx"

**用户反馈**：
> "我还是有疑问，claude不是会自动保存对话历史吗，为什么你要在prompt里要求他保存？"

**问题**：
- Claude Code插件已经自动保存对话历史
- 手动保存是重复功能
- 混淆了"对话历史"和"工作成果"的概念

### 迭代2：工作成果自动更新（当前方案）

**核心理念**：
```
对话历史 → Claude Code自动管理（我们不关心）
工作成果 → 我们自动更新到核心文档（重点）
```

**实现机制**：
1. **对话结束触发**：用户说"结束"、"完成"、"下次再聊"
2. **自动更新核心文档**：
   - 教学Claude → 更新 `conversations/summaries/01_learning_progress.md`
   - 测试评估Claude → 更新 `conversations/summaries/03_assessment_summary.md`
   - 研究Claude → 更新 `conversations/summaries/02_market_updates.md`
3. **保存工作记录**：到 `conversations/{role}/YYYYMMDD_主题.md`

---

## 📂 文件结构设计

### 核心文档（01-09）
- 00-09编号，表示文档优先级和阅读顺序
- 所有Claude都可以读取（除了教学/研究/测试评估只能读特定的）
- Git版本控制

### 对话历史结构
```
conversations/
├── summaries/              # 汇总文档（所有Claude可读）⭐
│   ├── 00_weekly_summary.md
│   ├── 01_learning_progress.md
│   ├── 02_market_updates.md
│   └── 03_assessment_summary.md
├── general/                # 通用助手历史
├── assessor/               # 测试评估历史
├── teacher/                # 教学历史
├── researcher/             # 研究历史
└── planner/                # 计划协调历史
```

### 系统提示词
```
system_prompts/
├── 00_general_prompt.md
├── 01_assessor_prompt.md
├── 02_teacher_prompt.md
├── 03_researcher_prompt.md
└── 04_planner_prompt.md
```

### VSCode工作区
```
.vscode/
├── workspace_general.code-workspace
├── workspace_assessor.code-workspace
├── workspace_teacher.code-workspace
├── workspace_researcher.code-workspace
└── workspace_planner.code-workspace
```

---

## 🎯 关键设计决策

### 决策1：为什么需要5个Claude？

**问题**：单个Claude对话太长，容易超出上下文窗口

**解决方案**：
- 按任务类型分角色
- 每个角色专注自己的领域
- 通过summaries/实现信息共享

### 决策2：为什么使用VSCode工作区切换？

**问题**：如何让Claude知道自己的角色？

**解决方案**：
- 每个.code-workspace文件指定对应的系统提示词
- 切换工作区 = 切换Claude角色
- 避免每次都要说"你是教学专家"

### 决策3：为什么分层可见性？

**问题**：所有Claude都读所有对话会信息过载

**解决方案**：
- General和Planner：读所有（需要全局视野）
- 其他Claude：只读自己的 + summaries/（专注领域）

### 决策4：如何解决上下文丢失问题？

**问题**：切换Claude对话后，可能丢失之前的设计决策

**解决方案**：
1. ✅ README.md - 记录Git commit历史
2. ✅ CHANGELOG.md - 记录项目变更
3. ✅ **本文档** - 记录设计决策和演进
4. ✅ conversations/summaries/ - 自动更新的工作成果

---

## 🚀 使用流程

### 每周协作流程

```
周一（研究Claude）
├─ 收集JD → 更新03、04文档
└─ 说"完成" → 自动更新summaries/02_market_updates.md

周二（计划协调Claude）
├─ 读取所有对话历史
├─ 更新08、09文档
└─ 识别风险，提出调整建议

周三-周五（教学Claude）
├─ 教学Python/React
└─ 说"结束" → 自动更新summaries/01_learning_progress.md

周日（通用助手Claude）
├─ 读取所有对话历史
├─ 生成每周汇总 → summaries/00_weekly_summary.md
└─ 更新CHANGELOG.md

月末（测试评估Claude）
├─ 测试技能
└─ 说"测试完成" → 自动更新summaries/03_assessment_summary.md
```

---

## 🔧 Git管理策略

### Commit格式
```bash
[Role] Brief description (max 50 chars)

- 详细说明
- 背景/原因
- 已更新：文件列表
```

### 角色标识符
- `[General]` - 通用助手
- `[Assessor]` - 测试评估
- `[Teacher]` - 教学
- `[Researcher]` - 研究
- `[Planner]` - 计划协调
- `[System]` - 系统级变更

### Tag规范
```bash
vYYYY-MM-description

例如：v2026-01-5-claude-system
```

---

## ⚠️ 常见误区

### 误区1："对话历史需要手动保存"

**纠正**：
- Claude Code自动保存对话历史
- 我们只需要自动更新工作成果（核心文档）

### 误区2："每个Claude都需要读所有对话"

**纠正**：
- 只有General和Planner读所有
- 其他Claude只读自己的 + summaries/

### 误区3："切换工作区很麻烦"

**纠正**：
- 一次性配置，一键切换
- 比每次说"你是教学专家"更高效

---

## 📊 成功指标

### 系统效果评估

| 指标 | 目标 | 测量方法 |
|------|------|----------|
| 对话效率 | 每个Claude专注领域 | 对话长度不超过5轮/任务 |
| 信息共享 | summaries/每周更新 | Git commit频率 |
| 进度跟踪 | 每周识别风险 | Planner每周输出 |
| 技能提升 | 每月测试一次 | Assessor每月测试 |

---

## 🎓 如何使用本文档

### 对于新启动的Claude

1. **首次启动时**：
   - 读取本文档（design_documents/00_5_claude_system_design.md）
   - 理解系统设计理念和架构
   - 了解自己的角色定位

2. **执行任务时**：
   - 参考"使用流程"章节
   - 遵循"关键设计决策"
   - 执行"自动更新机制"

3. **遇到问题时**：
   - 查看"常见误区"章节
   - 理解"设计演进历程"

### 对于用户

1. **验收系统**：
   - 检查5个VSCode工作区是否正确配置
   - 验证对话历史可见性设置
   - 测试自动更新机制

2. **监控系统**：
   - 查看 `conversations/summaries/` 了解进度
   - 查看 Git commit 历史了解变更
   - 查看本文档了解系统设计

---

## 🚀 下一步

### 待优化项
- [ ] 完成其他4个Claude的自动更新机制（目前只有Teacher完成）
- [ ] 添加更多自动化触发词
- [ ] 优化summaries/的模板

### 待验证项
- [ ] VSCode工作区切换是否顺畅
- [ ] 自动更新机制是否有效
- [ ] 分层可见性是否合理

---

## 📝 版本历史

| 日期 | 版本 | 更新内容 |
|------|------|----------|
| 2026-01-31 | v1.0 | 初始版本，记录完整设计决策 |

---

**文档所有者**：用户（returnfortheking）
**维护者**：通用助手Claude
**重要级别**：⭐⭐⭐⭐⭐ 全局视野核心文档
