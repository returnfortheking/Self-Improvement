# 5个Claude协作系统 - 使用指南

> **最后更新**：2026-01-31
> **目的**：说明如何使用5个Claude协作系统

---

## 🎯 核心设计理念

### 对话历史 vs 工作成果

```
┌─────────────────────────────────────────┐
│ Claude Code 插件                          │
│ ↓ 自动保存对话历史                         │
│ 位置：Claude Code内部存储（统一管理）      │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│ 我们关注的是：工作成果自动更新           │
└─────────────────────────────────────────┘
```

### 不关心对话历史，关心工作成果

**对话历史**：
- 由Claude Code自动管理
- 用户无需手动保存
- 按需查看即可

**工作成果**（重点）：
- 核心文档自动更新（02、03、04、08、09等）
- 汇总文档自动同步（summaries/）
- Git版本跟踪

---

## 🚀 如何使用5个Claude

### 方式1：VSCode工作区切换（推荐）

#### 步骤1：打开工作区

```
在VSCode中：
1. File → Open Workspace...
2. 选择 .vscode/workspace_teacher.code-workspace
3. VSCode会打开新窗口，标题栏显示"教学助手"
```

#### 步骤2：启动Claude Code

在VSCode中启动Claude Code插件

#### 步骤3：开始对话

```
直接说："教我Python闭包"
```

**系统提示词自动加载**，Claude知道自己是教学专家。

---

### 方式2：手动指定角色（备选）

如果不想切换工作区：

```
"你是教学专家（参考 system_prompts/02_teacher_prompt.md），
请教我Python闭包"
```

---

## 📝 自动更新机制（已实现）

### 教学Claude自动更新

**触发条件**：对话结束时（你说"结束"、"完成"、"下次再聊"）

**自动执行**：
```
1. ✅ 更新学习进度 → conversations/summaries/01_learning_progress.md
2. ✅ 保存教学记录 → conversations/teacher/YYYYMMDD_主题.md
3. ✅ 告知更新内容
```

**示例**：
```
你："教我Python闭包...（对话）...好了，今天就到这里"
Claude："✅ 本次教学已记录
     学习进度：Python闭包（理解程度：⭐⭐⭐）
     下次重点：装饰器
     已更新：conversations/summaries/01_learning_progress.md"
```

### 测试评估Claude自动更新

**触发条件**：测试完成时（你说"测试完成"）

**自动执行**：
```
1. ✅ 更新测试摘要 → conversations/summaries/03_assessment_summary.md
2. ✅ 保存测试记录 → conversations/assessor/YYYYMMDD_技术栈.md
```

### 研究Claude自动更新

**触发条件**：研究完成时（你说"完成"）

**自动执行**：
```
1. ✅ 更新市场动态 → conversations/summaries/02_market_updates.md
2. ✅ 保存研究记录 → conversations/researcher/YYYYMMDD_主题.md
```

---

## 🔄 协作流程

### 每周循环

```
周一（研究Claude）
├─ 收集JD → 更新03、04文档 → 说"完成"
└─ 自动更新：conversations/summaries/02_market_updates.md

周二（计划协调Claude）
├─ 分析新JD → 调整08、09文档 → 说"完成"
└─ 自动更新：读取所有对话，识别问题

周三-周五（教学Claude）
├─ 教学Python/React → 说"结束"
└─ 自动更新：conversations/summaries/01_learning_progress.md

周日（通用助手Claude）
├─ 读取所有对话历史
├─ 生成每周汇总 → conversations/summaries/00_weekly_summary.md
└─ 更新CHANGELOG.md

月末（测试评估Claude）
├─ 测试技能 → 说"测试完成"
└─ 自动更新：conversations/summaries/03_assessment_summary.md
```

---

## 💾 Git管理策略

### 何时commit

**频繁commit，随时备份**：
- 完成一个小任务就commit
- 对话结束后commit
- 更新核心文档后commit

### Commit格式

```bash
[Teacher] 完成Python闭包教学

- 理解程度：⭐⭐⭐
- 下次重点：装饰器
- 已更新：conversations/summaries/01_learning_progress.md
```

### 查看commit历史

```bash
# 按Claude过滤
git log --grep="Teacher" --oneline

# 查看本周commit
git log --since="1 week ago" --oneline
```

---

## 🎓 使用示例

### 示例1：学习Python（使用教学Claude）

```
1. 打开 workspace_teacher.code-workspace
2. 启动Claude Code
3. 说："教我Python闭包"
4. 学习...
5. 说："结束"
6. Claude自动：
   - 更新学习进度汇总
   - 保存教学记录
   - 告知下次学习重点
```

### 示例2：测试技能（使用测试评估Claude）

```
1. 打开 workspace_assessor.code-workspace
2. 启动Claude Code
3. 说："测试我的Python水平"
4. 测试...
5. 说："测试完成"
6. Claude自动：
   - 更新测试评估摘要
   - 保存测试记录
   - 更新技能等级
```

### 示例3：每周协调（使用计划协调Claude）

```
1. 打开 workspace_planner.code-workspace
2. 启动Claude Code
3. 说："更新本周进度"
4. Claude自动：
   - 读取所有对话历史
   - 更新08、09文档
   - 识别风险
   - 提出调整建议
```

---

## 📂 文档结构总览

```
项目根目录/
├── 00-09_核心文档.md        # 工作成果（Git跟踪）
├── system_prompts/           # 5个Claude系统提示词
├── conversations/
│   ├── summaries/          # 汇总文档（自动更新）⭐
│   ├── general/            # 通用助手对话
│   ├── assessor/           # 测试评估对话
│   ├── teacher/            # 教学对话
│   ├── researcher/         # 研究对话
│   └── planner/            # 计划协调对话
└── .vscode/
    └── workspace_*.code-workspace  # 5个Claude工作区
```

**重点关注的文件**（自动更新的汇总）：
1. `conversations/summaries/00_weekly_summary.md` - 每周汇总
2. `conversations/summaries/01_learning_progress.md` - 学习进度
3. `conversations/summaries/02_market_updates.md` - 市场动态
4. `conversations/summaries/03_assessment_summary.md` - 测试评估摘要

---

## ⚠️ 注意事项

### 1. 对话历史管理

**Claude Code自动保存**：
- 对话历史由Claude Code插件管理
- 保存在Claude Code内部存储
- 用户无需手动保存

**我们管理的**：
- 工作成果（核心文档）
- 汇总文档（summaries/）
- 教学记录、测试记录等

### 2. 自动更新触发

**方式1：明确表示结束**
- 说："结束"、"完成"、"下次再聊"、"好了"

**方式2：明确指令**
- 说："更新进度"、"记录学习"、"保存成果"

### 3. 工作区切换

**建议**：
- 不同任务用不同的Claude
- 切换到对应的工作区
- 让Claude自动识别角色

---

## 🎯 快速上手

### 第一次使用

1. **选择工作区**
   ```
   学习Python → 打开 workspace_teacher.code-workspace
   测试技能 → 打开 workspace_assessor.code-workspace
   ```

2. **开始对话**
   ```
   直接说你要做什么，不需要说角色
   ```

3. **结束对话**
   ```
   说"结束"或"完成"
   Claude自动更新工作成果
   ```

### Git备份

```bash
# 随时备份
git add .
git commit -m "[Teacher] 完成Python闭包教学"
git push origin main
```

---

## 📞 需要帮助？

### 常见问题

**Q1：对话历史保存在哪里？**
A：Claude Code自动管理，保存在扩展内部存储，你无需关心。

**Q2：如何查看之前的对话？**
A：
- 方法1：在Claude Code中查看历史记录
- 方法2：查看 `conversations/` 目录下的记录

**Q3：如何知道工作成果已更新？**
A：查看 `conversations/summaries/` 下的4个汇总文档

**Q4：如何在5个Claude之间切换？**
A：
- 方法1：切换VSCode工作区
- 方法2：直接说角色："你是教学专家"

---

## 🚀 下一步

1. **验收**：检查上述功能是否符合需求
2. **使用**：开始使用5个Claude协作系统
3. **反馈**：有问题随时告诉我
4. **打Tag**：验收通过后打tag

**准备好了吗？验收通过请告诉我！**
