# AGENTS.md

This document provides guidelines for agentic coding assistants working in the LearningSystem repository.

## 项目概述

**LearningSystem** 是一个 **AI驱动的自主学习和面试准备系统**，基于 Skills v3.0 架构，帮助用户在2026年6月30日前完成跳槽（目标：AI IDE/RAG/Agent岗位）。

### 核心系统

- **Skills v3.0 系统** (`.github/skills/`) - 14个模块化Skills，完全自主的AI驱动
- **核心数据文档** (01-09.md) - 个人信息、技能评估、市场调研、行动计划
- **JD数据管理** (jd_data/) - 自动化JD收集和分析
- **学习材料** (practice/) - Python学习路径和实践材料

### 项目目标

- **目标薪资**: 70-100万/年
- **目标岗位**: AI IDE、RAG、Agent开发
- **准备时间**: 5个月（2026.01-2026.06）
- **自动化程度**: 88%（AI主导，最小人工输入）

---

## 快速入门

### 第一次使用

1. **阅读项目文档**
   - [README.md](README.md) - 项目主页和快速开始
   - [.github/skills/README_v3_Autonomous.md](.github/skills/README_v3_Autonomous.md) - Skills系统说明（5分钟）

2. **了解Skills系统**
   - 14个模块化Skills
   - 2个用户入口（学习 + 面试准备）
   - 完全自主的AI驱动流程

3. **开始使用**
   ```
   在Claude Code中说："今天学什么" 或 "开始学习"
   系统自动完成：规划→学习→评估→保存→同步
   ```

### 核心文档

| 文件 | 用途 |
|------|------|
| [README.md](README.md) | 项目主页，Skills v3.0系统说明 ⭐ |
| [00_Quick_Start.md](00_Quick_Start.md) | 快速入门指南 |
| [.github/skills/README_v3_Autonomous.md](.github/skills/README_v3_Autonomous.md) | Skills系统完整说明 ⭐ |
| [.github/skills/registry.yaml](.github/skills/registry.yaml) | Skills注册表 |

### 数据文档

| 文件 | 用途 |
|------|------|
| [01_Personal_Profile.md](01_Personal_Profile.md) | 个人信息与求职意向 |
| [02_Skills_Assessment.md](02_Skills_Assessment.md) | 技术栈评估与规划 |
| [03_Market_Research_JD_Analysis.md](03_Market_Research_JD_Analysis.md) | 市场调研：JD分析 |
| [04_Target_Positions_Analysis.md](04_Target_Positions_Analysis.md) | 目标岗位详细分析 |
| [05_Skills_Gap_Analysis.md](05_Skills_Gap_Analysis.md) | 技能差距详细分析 |
| [06_Salary_Feasibility_Analysis.md](06_Salary_Feasibility_Analysis.md) | 薪资可达性评估 |
| [07_Strengths_Risks_Analysis.md](07_Strengths_Risks_Analysis.md) | 核心优势与风险分析 |
| [08_Action_Plan_2026_H1.md](08_Action_Plan_2026_H1.md) | 2026年上半年行动计划 |
| [09_Progress_Tracker.md](09_Progress_Tracker.md) | 进度跟踪（自动更新） |

---

## Build/Lint/Test Commands

### Python学习材料

```bash
# 运行Python示例代码
python practice/python/Week1-2/Day01-02_Basics/examples.py

# 运行特定练习
python practice/python/Week1-2/Day01-02_Basics/exercises.py
```

### 测试

- **无自动化测试套套**：Python学习材料通过手动运行验证
- **验证方法**：运行examples.py，检查输出是否符合预期

### Git操作

```bash
# 查看所有commits
git log --oneline

# 查看最近10个commits
git log --oneline -10

# 查看某个commit的详细信息
git show <commit-id>
```

---

## Code Style Guidelines

### 文件命名

- **Python文件**: `snake_case.py` (如 `examples.py`, `exercises.py`)
- **Markdown文件**: `snake_case.md` 或 `CamelCase.md`
- **目录名**: `snake_case` (如 `Day01-02_Basics`)
- **编号前缀**: 使用数字前缀表示顺序 (如 `01_Personal_Profile.md`)

### Python代码风格

- **类型提示**: `from typing import ...` 用于所有函数签名
- **文档字符串**: 中文docstrings，使用三引号 `"""`
- **命名约定**:
  - 函数/变量: `snake_case`
  - 类: `PascalCase`
  - 常量: `UPPER_SNAKE_CASE`
- **字符串格式化**: 使用f-strings: `f"Value: {value}"`
- **主程序保护**: `if __name__ == "__main__":` 用于可执行脚本
- **导入顺序**:
  1. 标准库导入
  2. 第三方导入
  3. 本地导入
  - 每组之间用空行分隔

### 格式化

- 使用4空格缩进（不使用制表符）
- 最大行长度: 100字符（软限制），120（硬限制）
- 顶级定义之间用空行分隔
- 逗号后加空格，切片中冒号前加空格
- 无尾随空格

### 异常处理

- 使用特定异常类型: `except ValueError`, `except TypeError`
- 包含错误上下文信息
- 使用 `raise ExceptionType("descriptive message")` 抛出自定义异常
- 在异步代码中处理 `asyncio.CancelledError`
- 使用 `return_exceptions=True` 配合 `asyncio.gather()` 收集错误

### 异步代码

- 异步函数前缀 `async def`
- 始终 `await` 协程
- 使用 `asyncio.run(main())` 执行异步代码
- 优先使用 `asyncio.gather()` 进行并发执行
- 使用上下文管理器 (`async with`) 管理资源

### 文档风格

- 章节标题: `# === 章节名称 ===`
- 示例注释: `# ==================== N. 示例名称 ====================`
- 学习材料中使用中文注释和docstrings
- 保持示例自包含且可运行

### 内容指南

- 所有教育内容使用中文
- 每个Python模块应有模块docstring说明其用途
- 提供示例 (`examples.py`) 和练习 (`exercises.py`)
- 按顺序编号示例以便引用
- 包含打印语句显示中间步骤

### Git提交信息

- 使用约定式提交: `type: description`
- 类型: `[System]`, `[Docs]`, `[Fix]`, `[Feat]`
- 如需要，在消息正文中包含上下文
- 引用相关的issue编号

### 在本项目中工作

1. **添加示例**: 创建 `examples.py` 和 `exercises.py` 配对文件
2. **更新文档**: 保持中文语言一致性
3. **代码结构**: 遵循 `practice/python/` 目录中的现有模式
4. **测试**: 手动运行示例验证其工作
5. **注释**: 使用中文以便教育清晰

---

## Skills系统规范

### Skill文件结构

每个Skill目录应包含：
```
skill-name/
├── SKILL.md              # Skill定义和说明
```

### SKILL.md格式

```yaml
---
name: skill-name
description: 简短描述
metadata:
  category: category-name
  triggers: "trigger1, trigger2"
  autonomous: true/false
  auto_start: true/false
---

# Skill Name

详细说明...
```

### Skills分类

- **编排** (orchestration): 用户入口，工作流路由
- **学习** (learning): 学习相关活动
- **规划** (planning): 学习计划生成
- **分析** (analysis): 技能评估，进度分析
- **推荐** (recommendation): 智能推荐学习内容
- **面试** (interview): 面试记录，经验沉淀
- **工具** (utility): 数据同步，版本控制

---

## 项目结构

```
LearningSystem/
├── 00_*.md                      # 核心文档（快速入门、架构设计等）
├── 01-09_*.md                   # 核心数据文档
├── .github/skills/                # Skills v3.0系统 ⭐
│   ├── autonomous-orchestrator/   # 学习流程主编排器
│   ├── interview-oracle/          # 面试准备入口
│   ├── daily-planner/             # 每日规划器
│   ├── autonomous-learner/        # 自主学习者
│   ├── auto-assessor/             # 自动评估器
│   ├── job-analyzer/              # 职位分析器
│   ├── smart-recommender/         # 智能推荐器
│   ├── auto-syncer/               # 自动同步器
│   ├── doc-sync/                 # 文档同步
│   ├── practice/                  # 练习生成
│   ├── assessor/                  # 技能评估
│   ├── checkpoint/                # 检查点保存
│   ├── interview/interview-recorder/ # 面试记录
│   ├── progress-tracker/          # 进度跟踪
│   ├── registry.yaml             # Skills注册表
│   ├── README_v3_Autonomous.md    # Skills系统说明 ⭐
│   └── SKILLS_V3_ARCHITECTURE.md # Skills架构文档
├── jd_data/                      # JD数据（自动管理）
│   ├── images/                    # JD截图（git跟踪）
│   ├── raw/                       # 原始JD文本
│   └── metadata.json              # 元数据（自动更新）
├── interview_data/                # 面试数据
├── practice/                      # 练习材料
│   └── python/                   # Python学习路径
│       ├── Week1-2/
│       ├── Week3-4/
│       └── LEARNING_PATH.md
├── conversations/                 # 对话历史
│   ├── summaries/                # 自动生成的汇总文档
│   ├── general/                  # 通用助手历史（已废弃）
│   ├── assessor/                 # 测试评估历史（已废弃）
│   ├── teacher/                  # 教学历史（已废弃）
│   ├── researcher/               # 研究历史（已废弃）
│   └── planner/                  # 计划协调历史（已废弃）
├── references/                    # 参考资料
│   └── MODULAR-RAG-MCP-SERVER/  # Skills设计参考
├── archive/                      # 归档文件
│   ├── JD_Details/               # 旧JD详细文档
│   ├── Old_Assessments/           # 旧评估文档
│   └── old_systems/              # 废弃系统（5个Claude） ⚠️
├── diagrams/                      # 架构图
│   └── architecture_diagrams.html # 在线渲染的架构图
└── README.md                     # 项目主页 ⭐
```

---

## 重要说明

### 特殊性质

- **这是一个学习系统，不是生产代码**
- **清晰度和教育价值优先于性能优化**
- **示例应逐步演示概念**
- **所有代码必须可以运行，无需标准Python库之外的依赖**

### 避免事项

- **不要**引入复杂的外部依赖
- **不要**使用晦涩的编程技巧
- **不要**创建需要用户额外配置的代码
- **不要**在示例中省略错误处理

---

## 常见问题

### Q: 5个Claude协作系统还存在吗？

**A: 不存在！**
- 5个Claude协作系统已被Skills v3.0完全取代
- 废弃文件已移动到 `archive/old_systems/`
- 当前系统是Skills v3.0（`.github/skills/`）

### Q: 00_Architecture_Design.md描述的是当前系统吗？

**A: 不是！**
- `00_Architecture_Design.md` 描述的是**未来计划**的InterviewPrep系统（前端+后端）
- 文档开头有明确的警告说明
- 当前实际系统是Skills v3.0（`.github/skills/`）

### Q: 如何开始使用Skills系统？

**A: 非常简单！**
```
在Claude Code中说："今天学什么"
系统自动完成：规划→学习→评估→保存→同步
```

### Q: 如何查看当前进度？

**A:**
```
说："查看进度"
系统自动生成进度报告
```

### Q: 如何准备面试？

**A:**
```
说："准备面试 [公司名]" 或 "分析JD"
系统自动：分析JD→匹配技能→生成学习路径→准备面试包
```

---

## 需要帮助？

### 文档资源

- [README.md](README.md) - 项目主页和快速开始 ⭐
- [.github/skills/README_v3_Autonomous.md](.github/skills/README_v3_Autonomous.md) - Skills系统完整说明 ⭐
- [00_Quick_Start.md](00_Quick_Start.md) - 快速入门指南
- [archive/old_systems/README.md](archive/old_systems/README.md) - 废弃系统说明

### 快速参考

| 需求 | 文档 |
|------|------|
| 项目概览 | [README.md](README.md) |
| Skills系统 | [.github/skills/README_v3_Autonomous.md](.github/skills/README_v3_Autonomous.md) |
| 快速入门 | [00_Quick_Start.md](00_Quick_Start.md) |
| 个人信息 | [01_Personal_Profile.md](01_Personal_Profile.md) |
| 技能评估 | [02_Skills_Assessment.md](02_Skills_Assessment.md) |
| 学习计划 | [08_Action_Plan_2026_H1.md](08_Action_Plan_2026_H1.md) |
| 当前进度 | [09_Progress_Tracker.md](09_Progress_Tracker.md) |

---

**最后更新**: 2026-02-07
**维护者**: returnfortheking
