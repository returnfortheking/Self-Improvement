# 2026年跳槽计划 - 主索引

> **目标**：在2026年6月30日前完成跳槽，目标薪资70-100万/年
> **当前状态**：📋 阶段一 - 信息收集与分析（已完成）
> **最后更新**：2026-01-28

---

## 📑 文档结构

本文档已按模块拆分，方便独立更新和维护。

### 核心文档

| 文件 | 内容 | 状态 | 最后更新 |
|------|------|------|---------|
| [01_Personal_Profile.md](01_Personal_Profile.md) | 个人信息与求职意向 | ✅ 已完成 | 2026-01-28 |
| [02_Skills_Assessment.md](02_Skills_Assessment.md) | 技术栈评估与规划 | ✅ 已完成 | 2026-01-28 |
| [03_Market_Research_JD_Analysis.md](03_Market_Research_JD_Analysis.md) | 市场调研：18个JD分析 | ✅ 已完成 | 2026-01-28 |
| [04_Target_Positions_Analysis.md](04_Target_Positions_Analysis.md) | 目标岗位详细分析 | ✅ 已完成 | 2026-01-28 |
| [05_Skills_Gap_Analysis.md](05_Skills_Gap_Analysis.md) | 技能差距详细分析 | ✅ 已完成 | 2026-01-28 |
| [06_Salary_Feasibility_Analysis.md](06_Salary_Feasibility_Analysis.md) | 薪资可达性评估 | ✅ 已完成 | 2026-01-28 |
| [07_Strengths_Risks_Analysis.md](07_Strengths_Risks_Analysis.md) | 核心优势与风险分析 | ✅ 已完成 | 2026-01-28 |
| [08_Action_Plan_2026_H1.md](08_Action_Plan_2026_H1.md) | 2026年上半年行动计划 | ✅ 已完成 | 2026-01-28 |
| [09_Progress_Tracker.md](09_Progress_Tracker.md) | 进度跟踪（每周更新） | 📋 待开始 | - |

### 设计文档

> **重要**：这些文档记录了系统的设计决策和演进历程，是所有Claude的全局视野基础

| 文件 | 内容 | 重要性 | 最后更新 |
|------|------|--------|----------|
| [design_documents/00_5_claude_system_design.md](design_documents/00_5_claude_system_design.md) | 5个Claude协作系统设计文档 | ⭐⭐⭐⭐⭐ | 2026-01-31 |

### JD数据管理（全自动）

> **重要**：新增JD后，learning-workflow自动解析并更新03、04文档

| 目录 | 内容 | 自动化 |
|------|------|--------|
| [jd_data/images/](jd_data/images/) | JD截图 | 添加新截图后，learning-workflow自动检测、解析并更新03、04文档 |
| [jd_data/raw/](jd_data/raw/) | 原始JD文本 | 保留原始数据 |
| [jd_data/metadata.json](jd_data/metadata.json) | 元数据 | 自动更新（采集记录、统计信息） |

**使用方式**：
1. 将新JD截图放入 `jd_data/images/`（按命名规范：`YYYY-MM-DD_序号_公司_岗位.jpg`）
2. 运行任何学习命令（如"开始学习"）
3. 系统自动检测、解析、更新文档
4. 无需手动触发

### 归档文件

| 目录 | 内容 | 说明 |
|------|------|------|
| [archive/JD_Details/](archive/JD_Details/) | 详细JD数据 | 已被03、04文档整合 |
| [archive/Old_Assessments/](archive/Old_Assessments/) | 旧评估文档 | 已被01、02文档整合 |

### 参考资料

| 目录 | 内容 | 来源 |
|------|------|------|
| [references/MODULAR-RAG-MCP-SERVER/](references/MODULAR-RAG-MCP-SERVER/) | Skills设计参考项目 | [GitHub](https://github.com/jerry-ai-dev/MODULAR-RAG-MCP-SERVER) |

---

## 🎯 快速导航

### 如果你需要：

- **查看/修改个人信息** → [01_Personal_Profile.md](01_Personal_Profile.md)
- **查看/修改技术栈评估** → [02_Skills_Assessment.md](02_Skills_Assessment.md)
- **查看市场调研数据** → [03_Market_Research_JD_Analysis.md](03_Market_Research_JD_Analysis.md)
- **了解目标岗位详情** → [04_Target_Positions_Analysis.md](04_Target_Positions_Analysis.md)
- **查看技能差距** → [05_Skills_Gap_Analysis.md](05_Skills_Gap_Analysis.md)
- **了解薪资可行性** → [06_Salary_Feasibility_Analysis.md](06_Salary_Feasibility_Analysis.md)
- **查看优势与风险** → [07_Strengths_Risks_Analysis.md](07_Strengths_Risks_Analysis.md)
- **查看行动计划** → [08_Action_Plan_2026_H1.md](08_Action_Plan_2026_H1.md)
- **更新学习进度** → [09_Progress_Tracker.md](09_Progress_Tracker.md)

### 系统相关

- **了解5个Claude协作系统** → [design_documents/00_5_claude_system_design.md](design_documents/00_5_claude_system_design.md)
- **查看使用指南** → [HOW_TO_USE.md](HOW_TO_USE.md)
- **查看变更日志** → [CHANGELOG.md](CHANGELOG.md)

---

## 📊 核心数据摘要

### 个人背景
- **学历**：华中科技大学（985）本科 + 哈工大（C9）硕士
- **工作年限**：4年（华为）
- **当前薪资**：约60万/年
- **目标薪资**：70-100万/年

### 目标岗位
- **主要方向**：AI应用开发、AI Agent开发
- **工作地点**：上海
- **工作强度**：保证双休（965/995）

### 关键差距
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

## ✅ 待办事项

### 当前阶段（2026年1月28日）

- [x] 信息收集与分析
- [x] 18个JD详细分析
- [x] 技能差距评估
- [x] 制定初步计划
- [ ] **等待用户确认文档准确性**
- [ ] 制定详细学习计划
- [ ] 制定LLM应用项目方案

### 下一步

1. **用户确认**：检查所有文档的准确性
2. **制定详细计划**：5个月的月度/周计划
3. **开始执行**：按照计划开始学习和项目开发

---

## 📝 文档更新日志

| 日期 | 文件 | 更新内容 | 更新人 |
|------|------|---------|--------|
| 2026-01-28 | 全部文档 | 初始版本创建 | Claude |
| 2026-01-31 | 全部文档 | 基于87个岗位数据更新 | Claude |
| 2026-01-31 | 19个新文件 | 创建5个Claude协作系统 | Claude |
| 2026-01-31 | HOW_TO_USE.md | 添加使用指南 | Claude |

---

## 📜 Git提交历史记录（重要！）

### 提交记录（所有commit的详细信息）

#### Commit 1: 项目初始化
- **日期**：2026-01-31
- **Commit ID**: 393ea11
- **Message**: "System: Project initialization"
- **变更**：
  - 创建87个岗位市场调研
  - 完成12个技术栈测试
  - 创建核心评估文档（01-09）
  - 测试发现：Python严重遗忘、4项技能超预期
- **文件数**：78个文件
- **背景**：信息收集与分析阶段完成

#### Commit 2: 5个Claude协作系统创建
- **日期**：2026-01-31
- **Commit ID**: 1579b9b
- **Message**: "System: Create 5 Claude collaboration architecture"
- **变更**：
  - 创建5个Claude角色系统提示词
  - 创建对话历史目录结构
  - 创建VSCode工作区配置
  - 创建Git配置文件
  - 创建变更日志和规范文档
- **文件数**：19个新文件
- **背景**：实现5个Claude协作分工机制
- **核心设计**：
  - 通用助手：协调者，可读所有历史
  - 测试评估：技能测试，可写02/05/00文档
  - 教学：技术教学，可写conversations/teacher/
  - 研究：JD收集，可写03/04文档
  - 计划协调：进度跟踪，可写08/09文档

#### Commit 3: 工作成果自动更新机制
- **日期**：2026-01-31
- **Commit ID**: 3187145
- **Message**: "System: Add auto-update mechanism for work results"
- **变更**：
  - 在教学Claude系统提示词中添加自动更新机制
  - 教学Claude会在对话结束时自动更新学习进度
- **文件数**：1个文件修改
- **核心功能**：
  - 对话结束时（用户说"结束"、"完成"）
  - 自动更新conversations/summaries/01_learning_progress.md
  - 自动保存教学记录到conversations/teacher/
- **触发方式**：用户说"结束"、"完成"、"下次再聊"

#### Commit 4: 使用指南文档
- **日期**：2026-01-31
- **Commit ID**: cfdd13b
- **Message**: "Docs: Add comprehensive usage guide"
- **变更**：
  - 创建HOW_TO_USE.md使用指南
  - 详细说明5个Claude的使用方法
  - 说明自动更新机制
- **文件数**：1个新文件
- **内容**：
  - 工作区切换方法
  - 对话历史管理策略
  - 自动更新机制说明
  - 协作流程示例

#### Commit 5: 设计文档与全局视野
- **日期**：2026-01-31
- **Commit ID**: 9d359dc
- **Message**: "[System] Add design documents for global context"
- **变更**：
  - 创建design_documents/00_5_claude_system_design.md（5个Claude系统设计文档）
  - 创建design_documents/README.md（设计文档说明）
  - 更新所有5个系统提示词，引用设计文档
  - 更新README.md，添加设计文档章节
- **文件数**：8个文件（2个新文件，6个修改）
- **背景**：
  - 用户问题2："我们当前的这个对话，是不是没法继承给新增的这5个claude？这是否会导致每个claude缺乏全局视野？"
  - 用户担心切换Claude对话后丢失设计决策和上下文
- **核心功能**：
  - 记录完整的设计决策和演进历程
  - 所有5个Claude都可以访问全局视野
  - General和Planner：首次启动必读
  - Assessor、Teacher、Researcher：建议阅读
- **设计文档内容**：
  - 设计目标和架构
  - 5个角色定义和职责分工
  - 对话历史可见性设计理念
  - 设计迭代历程（手动保存 → 自动更新）
  - 关键设计决策和理由
  - 使用流程和协作模式
  - Git管理策略
  - 常见误区说明

#### Commit 6: 更新README记录Commit 5
- **日期**：2026-01-31
- **Commit ID**: 39595b7
- **Message**: "[System] Update README with Commit 5 record"
- **变更**：
  - 在README.md中添加Commit 5详细记录
  - 记录设计文档创建过程
  - 记录解决用户问题2的方案
- **文件数**：1个文件修改
- **背景**：保持Git commit历史的完整性

#### Commit 7: Skills系统创建（基于MODULAR模式）
- **日期**：2026-02-02
- **Commit ID**: bc264ef
- **Message**: "[System] Create Skills system inspired by MODULAR-RAG-MCP-SERVER"
- **变更**：
  - 创建6个核心Skills：learning-workflow, doc-sync, progress-tracker, practice, assessor, checkpoint
  - 创建完整Skills系统文档
  - 严格遵循MODULAR-RAG-MCP-SERVER项目的设计模式
- **文件数**：8个文件（1804行）
- **背景**：
  - 用户选择方案B：完全迁移到Skills
  - 用户要求："尽量和github项目保持一致，不要增加自己的小巧思"
  - 深入学习MODULAR-RAG-MCP-SERVER的Skills设计
- **核心功能**：
  - **learning-workflow**（Meta-Skill）：编排5个stage的完整学习流程
  - **doc-sync**：同步核心文档，生成缓存
  - **progress-tracker**：查找下一个学习主题，验证进度真实性
  - **practice**：执行实战练习，生成练习文件
  - **assessor**：测试理解程度，评定技能等级
  - **checkpoint**：保存进度，更新核心文档，生成git commit
- **设计模式**：
  - 完全遵循MODULAR项目的YAML frontmatter结构
  - 每个Skill包含详细的SOP（Standard Operating Procedure）
  - 明确的Output Contract定义
  - 关键决策点需要用户确认
  - 限制迭代次数（最多3次）
  - Quick Commands表格
  - Important Rules列表
- **关键决策**：
  - 保留现有01-09文档结构（不拆分）
  - doc-sync生成JSON缓存（待实现sync_docs.py）
  - 进度验证基于08_Action_Plan_2026_H1.md和09_Progress_Tracker.md
  - 技能等级更新到02_Skills_Assessment.md
- **与5个Claude系统的关系**：
  - Skills系统是**工作流程自动化**层
  - 5个Claude系统是**角色分工**层
  - 两者可以并存，Skills提供更结构化的流程

#### Commit 8: 更新README记录Commit 6和7
- **日期**：2026-02-02
- **Commit ID**: a906bf1
- **Message**: "[System] Update README with Commit 6 and 7 records"
- **变更**：
  - 添加Commit 6：更新README记录Commit 5
  - 添加Commit 7：Skills系统创建记录
- **文件数**：1个文件修改
- **背景**：保持Git commit历史的完整性

#### Commit 9: 目录结构整理与JD自动解析
- **日期**：2026-02-02
- **Commit ID**: d9a9360
- **Message**: "System: Reorganize directory and add auto-JD-parsing"
- **变更**：
  - **目录结构整理**：
    - 创建 `jd_data/` 统一管理JD数据
      - `images/`：61个JD截图（git跟踪）
      - `raw/`：原始JD文本数据
      - `metadata.json`：元数据（自动更新）
    - 创建 `archive/` 归档旧文档
      - `JD_Details/`：4个JD详细文档（43, 44, AI_Infra, All_Positions_70Plus）
      - `Old_Assessments/`：2个旧评估文档（Comprehensive_Skills, Mission_Overview）
    - 创建 `references/` 外部参考资料
      - `MODULAR-RAG-MCP-SERVER/`：Skills设计参考项目
  - **JD自动解析功能**：
    - 更新 `doc-sync/SKILL.md` 添加完整JD解析SOP（Step 1.5）
    - 更新 `learning-workflow/SKILL.md` Stage 1包含JD解析
    - 使用Claude原生多模态能力（`extract_text_from_screenshot`）
    - 自动检测 `jd_data/images/` 新JD
    - 自动更新 `03_Market_Research_JD_Analysis.md`
    - 自动更新 `04_Target_Positions_Analysis.md`
    - 永不要求用户手动触发
  - **文档更新**：
    - 更新 `.gitignore`（忽略references下的.git/）
    - 更新 `README.md`（添加jd_data, archive, references章节）
- **文件数**：357个文件
- **背景**：
  - 用户需求1：jd和JD截图统一到一个目录管理
  - 用户需求2：jd数据由git跟踪
  - 用户需求3：非核心文档需要归档
  - 用户需求4：实现JD自动解析，永不手动触发
- **核心创新**：
  - 完全自动化的JD数据管理
  - 每次learning-workflow运行时自动检测并解析新JD
  - 使用Claude原生能力，无需Python脚本
  - 保证03、04文档永远最新

---

## 🔍 如何查看Commit历史

### 查看所有commit
```bash
git log --oneline
```

### 查看某个commit的详细信息
```bash
git show <commit-id>
```

### 按Claude过滤commit
```bash
git log --grep="Teacher" --oneline
git log --grep="System" --oneline
```

---

## 💡 关于Commit信息记录

### 为什么需要记录？

**问题**：更换Claude对话后，commit上下文会丢失
- 不知道为什么做这个修改
- 不知道这个修改解决了什么问题
- 无法追溯决策过程

**解决方案**：
1. ✅ **README.md** - 记录所有commit的详细信息
2. ✅ **CHANGELOG.md** - 记录项目变更历史
3. ✅ **详细的commit message** - 每个commit说明原因和内容

### 换Claude后如何找回上下文？

**方法1：查看这个README.md**
- 所有commit信息都记录在这里
- 包含日期、变更内容、背景

**方法2：查看CHANGELOG.md**
- 记录项目变更历史
- 按时间顺序排列

**方法3：查看Git历史**
```bash
git log --oneline
git show <commit-id>
```

---

## 💡 使用说明

### 如何更新文档

1. **个人信息变更**：编辑 `01_Personal_Profile.md`
2. **技能提升后**：更新 `02_Skills_Assessment.md` 和 `09_Progress_Tracker.md`
3. **发现新的JD**：追加到 `03_Market_Research_JD_Analysis.md`
4. **目标岗位调整**：更新 `04_Target_Positions_Analysis.md`
5. **每周进度**：更新 `09_Progress_Tracker.md`

### 文档命名规范

- `01_`, `02_` 等前缀表示文档顺序
- 使用英文文件名，避免中文路径问题
- 使用下划线分隔单词
- 使用 `.md` 扩展名

---

## 📧 反馈与确认

**请仔细检查所有模块的文档，重点确认：**

1. ✅ 个人信息是否准确？
2. ✅ 技术栈评估是否客观？
3. ✅ JD分析是否完整？
4. ✅ 技能差距是否准确？
5. ✅ 薪资目标是否合理？
6. ✅ 时间规划是否可行？

**如有任何错误或需要修改的地方，请告诉我！**

---

**项目状态**：📋 等待确认
**下一步**：制定详细学习计划
