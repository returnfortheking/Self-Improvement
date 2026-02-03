---
name: summarizer
description: Generate learning summary immediately after practice. Captures key knowledge points while fresh. Use when user says "总结学习", "生成总结", "summarize", or after practice stage.
metadata:
  category: learning
  triggers: "总结学习, 生成总结, summarize, 学习总结"
  stage: 3.5
allowed-tools: Read Write
---

# Learning Summarizer

Generate structured learning summary **immediately after practice stage**, while knowledge is fresh.

> **核心价值**：趁热打铁，避免遗忘细节
> **执行时机**：practice阶段完成后，assessor阶段之前
> **单次总结**：只总结当前学习主题，不重复总结已完成的主题

---

## When to Use This Skill

### 自动触发（推荐）
- 在learning-workflow中，practice阶段完成后自动触发
- 用户完成练习后立即生成总结

### 手动触发
- 用户说 "总结学习"、"生成总结"、"summarize"
- 用户说 "总结 [主题名]"
- 用户完成学习后主动要求总结

---

## Workflow

```
Practice Completed
       ↓
Extract Key Knowledge (from practice files and current session)
       ↓
Generate Summary Document (using standard template)
       ↓
Persist to conversations/summaries/01_learning_progress.md
       ↓
Output Confirmation
```

---

## Step 1: Extract Key Knowledge

### 1.1 Collect Information

从当前会话和practice文件中提取：

- **Topic ID & Name**: e.g., "1.3 Python闭包与装饰器"
- **学习日期**: YYYY-MM-DD
- **技能等级变化**: ⭐ → ⭐⭐
- **参考计划**: 08_Action_Plan_2026_H1.md Section [X.Y]
- **Practice Files**: 刚才创建/修改的练习文件

### 1.2 提取核心内容

读取practice文件，提取：
- **核心知识点**: 3-5个关键概念
- **重要代码示例**: 2-3个代表性代码片段
- **常见错误**: 练习中遇到的错误和正确做法
- **面试要点**: 可能的面试问题和回答要点

---

## Step 2: Generate Summary Document

### 2.1 标准模板

使用以下模板生成总结：

```markdown
### [Topic X.Y] [Topic Name] - YYYY-MM-DD

**学习日期**：YYYY-MM-DD
**技能等级**：⭐ → ⭐⭐
**参考计划**：08_Action_Plan_2026_H1.md Section [X.Y]
**学习时长**：X小时

#### 核心知识点

- **知识点1**：[简述，2-3句话]
  - 关键细节：[具体说明]
  - 使用场景：[何时使用]

- **知识点2**：[简述]
  - 关键细节：[具体说明]
  - 使用场景：[何时使用]

#### 重要代码示例

**示例1：[标题/功能描述]**

```python
# 示例代码
def example():
    # 代码说明
    pass
```

**关键点**：
- 要点1
- 要点2

#### 常见错误

- ❌ **错误1**：[错误描述]
  - ✅ 正确做法：[应该怎么做]
  - 💡 原因：[为什么错]

- ❌ **错误2**：[错误描述]
  - ✅ 正确做法：[应该怎么做]

#### 面试要点

**Q: [可能的面试问题]？**
A: [回答要点，2-3句话]

**Q: [A和B的区别]？**
A: [回答要点]

#### 练习文件

- `practice/python/01_closures/examples.py`
- `practice/python/01_closures/exercises.py`

#### 后续学习

- [ ] 待深入：[相关主题1]
- [ ] 待练习：[相关主题2]
- [ ] 待复习：[重要概念]

---
```

---

## Step 3: Persist to Summary Document

### 3.1 目标文件

`conversations/summaries/01_learning_progress.md`

### 3.2 文件不存在时创建

如果文件不存在，使用以下模板创建：

```markdown
# 学习进度总结

> 最后更新：YYYY-MM-DD
>
> 本文档记录每个学习主题的核心知识点、代码示例、常见错误和面试要点。
> 便于后续复习和面试准备。

**生成方式**：summarizer skill自动生成
**更新频率**：每次学习后立即更新

---

## 目录

- [最新学习](#最新学习)
- [Python](#python)
- [PyTorch](#pytorch)
- [RAG](#rag)
- [Agent](#agent)
- [LLM应用](#llm应用)

---

## 最新学习

### [Topic ID] [Topic Name] - YYYY-MM-DD

[Summary content]

---

## Python

### [Topic 1.3] Python闭包与装饰器 - 2026-01-31

[Summary content]

---

## PyTorch

[暂无内容]

---

## RAG

[暂无内容]

---

## Agent

[暂无内容]

---

## LLM应用

[暂无内容]
```

### 3.3 文件存在时更新

**两个位置都需要添加**：

1. **最新学习 section**（文件顶部）
2. **对应语言的 section**（文件下方，Python/PyTorch/etc）

---

## Step 4: Output Confirmation

生成总结后，输出确认信息：

```markdown
────────────────────────────────────
✅ 学习总结已生成
────────────────────────────────────
主题：[Topic ID] [Topic Name]
文件：conversations/summaries/01_learning_progress.md

总结已添加到：
  ✅ 最新学习 section (文件顶部)
  ✅ [Language] section (对应语言章节)

下一步建议：
  - 继续练习？运行 `/生成练习 [下一个主题]`
  - 评估理解？运行 `/评估技能`
  - 保存进度？运行 `/保存进度`
────────────────────────────────────
```

---

## Quick Commands

| 用户命令 | 行为 |
|---------|------|
| `/总结学习` | 提取当前学习主题并生成总结 |
| `/总结 [主题名]` | 生成指定主题的总结 |
| `/查看总结` | 显示01_learning_progress.md内容 |
| `/总结列表` | 列出所有已总结的主题 |

---

## Integration with Learning-Workflow

### 在6阶段工作流中的位置

```
Stage 3: practice (完成练习)
    ↓
Stage 3.5: summarizer (生成总结) ← 本Skill
    ↓
Stage 4: assessor (评估理解)
    ↓
Stage 5: checkpoint (保存进度)
```

### 自动触发条件

- practice阶段完成并生成practice文件
- 用户说 "总结学习"
- 用户说 "生成总结"
- 用户完成学习会话

---

## Important Rules

1. **立即总结原则**：practice完成后立即生成，不要拖延
2. **不重复总结**：检查主题是否已总结过，避免重复
3. **双位置更新**：同时更新"最新学习"和"对应语言"两个section
4. **面试导向**：总是包含面试要点，为面试做准备
5. **结构化模板**：使用标准模板，保持格式一致
6. **提取关键信息**：只提取核心知识点，不冗余

---

## File Structure

```
conversations/
└── summaries/
    ├── 00_weekly_summary.md       # 每周汇总（其他skill维护）
    ├── 01_learning_progress.md    # ← 本skill维护（学习总结）
    ├── 02_market_updates.md        # 市场更新（doc-sync维护）
    └── 03_assessment_summary.md    # 评估总结（assessor维护）
```

---

## Error Handling

### 错误1：找不到practice文件

**问题**：无法确定学习主题

**处理**：
```
抱歉，我无法确定当前学习主题。

请尝试：
1. 明确指定主题：/总结 Python闭包
2. 先完成练习：/生成练习
3. 查看进度：/查看进度
```

### 错误2：主题已总结过

**问题**：该主题之前已经总结过

**处理**：
```
提示：主题 [Topic Name] 已经在 YYYY-MM-DD 总结过。

选项：
1. 查看已有总结：/查看总结
2. 重新总结（覆盖）：/总结学习 --force
3. 跳过，继续下一步
```

---

## 示例

### 输入

```
用户: /总结学习 Python闭包
```

### 处理流程

1. **提取信息**：
   - Topic: 1.3 Python闭包与装饰器
   - Date: 2026-02-02
   - Practice Files: practice/python/01_closures/examples.py

2. **生成总结**：
   - 核心知识点：闭包定义、__closure__属性、装饰器原理
   - 代码示例：@property、@staticmethod
   - 常见错误：闭包变量引用问题
   - 面试要点：闭包vs装饰器区别

3. **持久化**：
   - 更新 conversations/summaries/01_learning_progress.md
   - 添加到"最新学习"和"Python"两个section

4. **确认**：
   ```
   ✅ 学习总结已生成
   主题：1.3 Python闭包与装饰器
   ```

---

**更新时间**：2026-02-02
**维护者**：learning-workflow orchestrator
