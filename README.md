# 2026年跳槽计划 - 自主学习与面试准备系统

> **目标**：在2026年6月30日前完成跳槽，目标薪资70-100万/年
> **当前状态**：📋 阶段三 - 技能提升与面试准备（进行中）
> **核心系统**：Skills v3.0 - 完全自主的AI驱动系统
> **最后更新**：2026-02-07

---

## 🎯 项目概述

这是一个**AI驱动的自主学习和面试准备系统**，帮助你：
- ✅ 自动规划和执行学习计划
- ✅ 智能分析JD并准备面试
- ✅ 自动跟踪进度和评估效果
- ✅ 完全自动同步多设备数据

### 核心特性

| 特性 | 说明 | 自动化程度 |
|------|------|-----------|
| **自主决策** | 系统自动决定学什么、学多久、何时复习 | 90% |
| **智能规划** | 基于进度、计划、效率动态调整 | 85% |
| **自动评估** | 学习后自动评估效果，识别薄弱点 | 90% |
| **自动同步** | Git自动commit+push，多设备无缝协作 | 100% |
| **面试准备** | 分析JD、匹配技能、生成学习路径 | 85% |

**用户只需**：每天说一句"今天学什么"或"准备面试"，系统自动完成所有工作。

---

## 📑 文档结构

### 核心数据文档（01-09）

| 文件 | 内容 | 状态 | 最后更新 |
|------|------|------|---------|
| [01_Personal_Profile.md](01_Personal_Profile.md) | 个人信息与求职意向 | ✅ 已完成 | 2026-01-28 |
| [02_Skills_Assessment.md](02_Skills_Assessment.md) | 技术栈评估与规划 | ✅ 已完成 | 2026-01-28 |
| [03_Market_Research_JD_Analysis.md](03_Market_Research_JD_Analysis.md) | 市场调研：JD分析 | ✅ 已完成 | 2026-01-28 |
| [04_Target_Positions_Analysis.md](04_Target_Positions_Analysis.md) | 目标岗位详细分析 | ✅ 已完成 | 2026-01-28 |
| [05_Skills_Gap_Analysis.md](05_Skills_Gap_Analysis.md) | 技能差距详细分析 | ✅ 已完成 | 2026-01-28 |
| [06_Salary_Feasibility_Analysis.md](06_Salary_Feasibility_Analysis.md) | 薪资可达性评估 | ✅ 已完成 | 2026-01-28 |
| [07_Strengths_Risks_Analysis.md](07_Strengths_Risks_Analysis.md) | 核心优势与风险分析 | ✅ 已完成 | 2026-01-28 |
| [08_Action_Plan_2026_H1.md](08_Action_Plan_2026_H1.md) | 2026年上半年行动计划 | ✅ 已完成 | 2026-01-28 |
| [09_Progress_Tracker.md](09_Progress_Tracker.md) | 进度跟踪（自动更新） | 🔄 进行中 | 自动更新 |

### Skills系统（.github/skills/）

**核心设计**：14个模块化Skills，2个用户入口，完全自主的AI驱动

| 类型 | Skills | 描述 |
|------|--------|------|
| **入口** | autonomous-orchestrator | 学习流程主编排器 |
| **入口** | interview-oracle | 面试准备入口 |
| **规划** | daily-planner | 每日智能规划器 |
| **学习** | autonomous-learner | 自主学习者 |
| **评估** | auto-assessor | 自动评估器 |
| **分析** | job-analyzer | 职位分析器 |
| **推荐** | smart-recommender | 智能推荐器 |
| **同步** | auto-syncer | 自动同步器 |
| **工具** | doc-sync, practice, assessor, checkpoint, progress-tracker, interview-recorder | 支持工具 |

**详细文档**：
- [.github/skills/README_v3_Autonomous.md](.github/skills/README_v3_Autonomous.md) - Skills系统完整说明
- [.github/skills/registry.yaml](.github/skills/registry.yaml) - Skills注册表

### JD数据管理（全自动）

| 目录 | 内容 | 自动化 |
|------|------|--------|
| [jd_data/images/](jd_data/images/) | JD截图 | 添加新截图后，autonomous-orchestrator自动检测、解析、更新03/04文档 |
| [jd_data/raw/](jd_data/raw/) | 原始JD文本 | 保留原始数据 |
| [jd_data/metadata.json](jd_data/metadata.json) | 元数据 | 自动更新 |

**使用方式**：
1. 将新JD截图放入`jd_data/images/`（命名：`YYYY-MM-DD_序号_公司_岗位.jpg`）
2. 说"开始学习"或"今天学什么"
3. 系统自动检测、解析、更新文档
4. 无需手动触发

### 学习材料（practice/）

| 目录 | 内容 |
|------|------|
| [practice/python/](practice/python/) | Python学习路径（Week1-4，包含examples和exercises） |
| [practice/python/LEARNING_PATH.md](practice/python/LEARNING_PATH.md) | Python学习路径规划 |

### 参考资料

| 目录 | 内容 |
|------|------|
| [references/](references/) | 外部参考资料（GitHub仓库、博客等） |
| [references/MODULAR-RAG-MCP-SERVER/](references/MODULAR-RAG-MCP-SERVER/) | Skills设计参考项目 |

---

## 🚀 快速开始

### 第一次使用

1. **阅读快速入门**（3分钟）
   - [00_Quick_Start.md](00_Quick_Start.md) - 快速入门指南

2. **了解Skills系统**（5分钟）
   - [.github/skills/README_v3_Autonomous.md](.github/skills/README_v3_Autonomous.md) - Skills系统说明

3. **验证数据准确性**（5分钟）
   - 检查[01-09核心文档](#核心数据文档01-09)是否准确
   - 确认[08_Action_Plan_2026_H1.md](08_Action_Plan_2026_H1.md)的学习计划

4. **开始使用**（现在！）
   ```
   在Claude Code中说："今天学什么"
   系统自动完成：规划→学习→评估→保存→同步
   ```

### 日常使用

**早上/晚上：**
```
说："今天学什么" 或 "开始学习"
系统自动执行当日学习流程，生成学习报告
```

**查看进度：**
```
说："查看进度"
系统展示当前进度、技能等级、薄弱点
```

**准备面试：**
```
说："准备面试 Trae" 或 "准备面试 字节跳动"
系统自动：分析JD→匹配技能→生成学习路径→准备面试包
```

---

## 📊 核心数据摘要

### 个人背景
- **学历**：华中科技大学（985）本科 + 哈工大（C9）硕士
- **工作年限**：4年（华为）
- **当前薪资**：约60万/年
- **目标薪资**：70-100万/年

### 目标岗位
- **主要方向**：AI IDE、RAG、Agent开发、AI应用
- **工作地点**：上海
- **工作强度**：保证双休（965/995）

### 关键差距（需要弥补）
1. 🔴 完整LLM应用（必须弥补）
2. 🔴 PyTorch精通（必须弥补）
3. 🔴 RAG生产级别（必须弥补）
4. 🔴 模型微调（必须弥补）
5. 🔴 Agent架构（必须弥补）

### 时间安排
- **总准备时间**：5个月（2026.01-2026.06）
- **每日学习时间**：3-4小时（工作日），6-8小时（周末）
- **关键里程碑**：6月前完成完整LLM应用

---

## 🎯 系统架构

### Skills v3.0工作流

```
用户触发（对话）
    ↓
┌─────────────────────────────────────────┐
│  autonomous-orchestrator (学习入口）   │
│  interview-oracle (面试入口）          │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  自主学习流程                            │
│  daily-planner → autonomous-learner →    │
│  auto-assessor → progress-tracker →     │
│  auto-syncer                            │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  面试准备流程                            │
│  job-analyzer → smart-recommender →     │
│  auto-assessor → auto-syncer            │
└─────────────────────────────────────────┘
    ↓
自动报告（学习报告/面试准备包）
```

### 数据流

```
Markdown文档（01-09） ← Skills系统读取状态
    ↓
智能决策（规划/评估/推荐）
    ↓
执行学习/分析
    ↓
更新文档 + Git commit
    ↓
自动同步到GitHub
```

---

## 📝 使用指南

### 查看核心文档

| 需求 | 文档 |
|------|------|
| 查看个人信息 | [01_Personal_Profile.md](01_Personal_Profile.md) |
| 查看技能评估 | [02_Skills_Assessment.md](02_Skills_Assessment.md) |
| 查看市场调研 | [03_Market_Research_JD_Analysis.md](03_Market_Research_JD_Analysis.md) |
| 查看目标岗位 | [04_Target_Positions_Analysis.md](04_Target_Positions_Analysis.md) |
| 查看技能差距 | [05_Skills_Gap_Analysis.md](05_Skills_Gap_Analysis.md) |
| 查看薪资分析 | [06_Salary_Feasibility_Analysis.md](06_Salary_Feasibility_Analysis.md) |
| 查看优势风险 | [07_Strengths_Risks_Analysis.md](07_Strengths_Risks_Analysis.md) |
| 查看行动计划 | [08_Action_Plan_2026_H1.md](08_Action_Plan_2026_H1.md) |
| 查看学习进度 | [09_Progress_Tracker.md](09_Progress_Tracker.md) |

### 更新数据

1. **个人信息变更**：编辑`01_Personal_Profile.md`
2. **技能提升后**：自动更新到`02_Skills_Assessment.md`和`09_Progress_Tracker.md`
3. **发现新的JD**：截图放入`jd_data/images/`，说"开始学习"，系统自动解析并更新03、04文档
4. **每周进度**：自动更新到`09_Progress_Tracker.md`

### 文档命名规范

- 核心文档：`01_`, `02_`等前缀表示文档顺序
- Python文件：`snake_case.py`（如`examples.py`, `exercises.py`）
- Markdown文件：`snake_case.md`或`CamelCase.md`
- 使用英文文件名，避免中文路径问题

---

## 📜 Git历史（简化版）

### 关键里程碑

| 日期 | Commit | 描述 |
|------|--------|------|
| 2026-01-31 | bc264ef | Skills系统创建（基于MODULAR模式） |
| 2026-02-02 | 814cc46 | 9个核心文档适配Skills系统 |
| 2026-02-02 | 897dee4 | 清理旧JD目录和临时文件 |
| 2026-02-07 | 0cfa142 | 添加CS336 Assignments 1-5学习材料 |
| 2026-02-07 | c839b6b | 添加Assignment 5 Alignment和RLHF学习材料 |
| 2026-02-07 | 23973f2 | Doc sync - CS336 focused update |
| 2026-02-07 | 36dffbe | 更新学习路径，整合CS336内容 |
| 2026-02-07 | 18a4f99 | 添加CS336 assignment repositories |

### 查看完整历史

```bash
# 查看所有commits
git log --oneline

# 查看最近10个commits
git log --oneline -10

# 查看某个commit的详细信息
git show <commit-id>
```

---

## 🔧 系统配置

### 自动化配置（.github/skills/registry.yaml）

```yaml
automation:
  auto_commit: true      # 自动git commit
  auto_push: true       # 自动git push
  auto_report: true      # 自动生成报告
  auto_adjust: true      # 自动调整计划
  max_auto_adjust: 0.2   # 最大自动调整幅度（20%）
```

### 学习配置

```yaml
learning:
  daily_hours: 4        # 每日学习目标时间（小时）
  efficiency_threshold: 0.6  # 效率阈值（低于此值建议休息）
  review_interval: 7     # 复习间隔（天）
  mastery_threshold: 0.8    # 掌握度阈值
```

---

## 📞 常见问题

### Q1: 我需要知道哪个skill做什么吗？

**A: 不需要！**

- 只需要记住2种触发方式：
  - 学习场景：说"今天学什么"、"开始学习"、"继续学习"
  - 面试场景：说"准备面试 [公司名]"、"分析JD"
- 系统自动判断用哪个skill
- 内部调用对你完全透明

### Q2: 系统如何知道今天该学什么？

**A: 系统自动分析：**

1. 读取今日日期和时间
2. 读取当前进度（09_Progress_Tracker.md）
3. 读取学习计划（08_Action_Plan_2026_H1.md）
4. 查看昨日完成情况
5. 检查是否有薄弱点
6. 智能决策今天该做什么

### Q3: 我可以干预学习内容吗？

**A: 可以！**

- 明确说出主题（如"今天学 LangGraph"）
- 系统会自动调整计划
- 也可以在过程中随时调整

### Q4: 系统会自动同步吗？

**A: 是的！**

- 所有数据变更自动 commit + push
- 每5分钟自动 pull
- 你无需手动操作

### Q5: 如何查看系统在做什么？

**A: 系统会实时告知：**

- 每个步骤开始时告知你
- 重要决策时说明理由
- 完成后生成详细报告

---

## 📊 项目状态

### 已完成

- ✅ 9个核心文档（01-09）
- ✅ Skills v3.0系统（14个Skills）
- ✅ JD数据收集（15+个JD）
- ✅ Python学习路径（Week1-4）
- ✅ 自动化工作流（学习+面试准备）
- ✅ Git自动同步

### 进行中

- 🔄 技能提升（Python、PyTorch、RAG、Agent）
- 🔄 面试准备（持续进行）
- 🔄 进度跟踪（自动更新）

### 待完成

- 📋 完整LLM应用项目
- 📋 前端应用（Electron + React）- 未来计划
- 📋 移动应用（React Native）- 未来计划
- 📋 后端API（FastAPI）- 未来计划

---

## 📧 反馈与支持

**遇到问题？**

1. **查看文档**：
   - [00_Quick_Start.md](00_Quick_Start.md) - 快速入门
   - [.github/skills/README_v3_Autonomous.md](.github/skills/README_v3_Autonomous.md) - Skills系统说明

2. **检查数据**：
   - 核心文档（01-09）是否准确
   - 学习进度是否正常

3. **调整配置**：
   - 修改`.github/skills/registry.yaml`
   - 调整自动化程度

**如有任何错误或需要改进的地方，请告诉我！**

---

## 🎉 开始你的求职准备之旅！

**项目状态**：🔄 阶段三 - 技能提升与面试准备（进行中）
**核心系统**：✅ Skills v3.0 - 完全自主
**下一步**：说"今天学什么"，开始自主学习！

---

**最后更新**：2026-02-07
**版本**：v3.0
**维护者**：returnfortheking
