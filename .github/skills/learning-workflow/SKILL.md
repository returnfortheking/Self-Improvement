---
name: learning-workflow
description: Intelligent router for learning workflow. Analyzes current state and executes required stages. Use when user says "开始学习", "继续学习", or any learning command.
metadata:
  category: orchestration
  triggers: "开始学习, 继续学习, 学习"
allowed-tools: Read
---

# Learning Workflow Router

You are the **Intelligent Router** for the 跳槽计划 learning system.

> **核心职责**：根据用户命令和当前状态，智能路由到相应的Skill，**不总是执行所有阶段**
> **数据源**：读取 `.github/skills/registry.yaml` 获取所有Skills配置
> **执行原则**：按需执行，避免资源浪费

---

## 核心设计原则

### 1. 模块化执行
- 每个Stage都是独立的Skill，可单独调用
- 不强制执行完整流程
- 用户可以精确控制

### 2. 智能路由
- 读取registry.yaml获取Skills配置
- 根据用户命令匹配对应Skill
- 根据当前状态判断需要的阶段

### 3. 配置驱动
- 所有Skill关系在registry.yaml中定义
- 添加新Skill无需修改learning-workflow代码
- 支持未来扩展

---

## 启动时加载配置

每次执行时，首先读取注册表：

```
1. 读取 .github/skills/registry.yaml
2. 解析所有Skills配置
3. 解析所有Workflows配置
4. 读取当前状态（09_Progress_Tracker.md）
```

---

## 路由决策逻辑

### Step 1: 解析用户命令

```python
# 伪代码
def parse_user_command(user_input):
    # 检查是否明确指定Skill
    for skill in registry.skills:
        for trigger in skill.triggers:
            if trigger in user_input:
                return {
                    "type": "direct_skill",
                    "skill_id": skill.id,
                    "user_input": user_input
                }

    # 检查是否是workflow命令
    if "开始学习" in user_input or "继续学习" in user_input:
        return {
            "type": "workflow",
            "workflow": "learning",
            "user_input": user_input
        }

    # 无法判断
    return {
        "type": "unknown",
        "user_input": user_input
    }
```

### Step 2: 执行路由

```
┌─────────────────────────────────────────┐
│  解析用户命令                              │
└──────────────┬──────────────────────────┘
               │
       ┌────────┴────────┐
       ▼                 ▼
  明确Skill         Workflow命令
       │                 │
       ▼                 ▼
  直接调用        分析当前状态
  指定Skill           │
                     ▼
              ┌───────────────┴──────────────┐
              │ 检查 09_Progress_Tracker │
              └───────────────┬──────────────┘
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
              有明确状态         无明确状态
                    │                 │
                    ▼                 ▼
              执行对应阶段        显示状态让用户选择
```

---

## Quick Commands Matrix

基于registry.yaml，支持以下命令：

### 直接Skill命令（执行单个Skill）

| 命令 | 执行Skill | 使用场景 |
|------|---------|----------|
| `/同步文档` | doc-sync | 添加新JD后 |
| `/查看进度` | progress-tracker | 检查当前状态 |
| `/生成练习 [主题]` | practice | 需要练习题 |
| `/总结学习` | summarizer | 完成学习后 |
| `/评估技能` | assessor | 测试理解程度 |
| `/保存进度` | checkpoint | 保存并提交 |
| `/记录面试` | interview-recorder | 面试结束后 |
| `/help` | - | 显示所有命令 |

### Workflow命令（智能路由）

| 命令 | 行为 | 说明 |
|------|------|------|
| `/开始学习 [主题]` | practice → summarizer → (ask) → assessor → checkpoint | 开始学习指定主题 |
| `/继续学习` | 分析当前状态 → 找到下一步 → 执行 | 从上次中断处继续 |
| `/开始学习` (无主题) | 分析当前状态 → 找到下一步 → 执行 | 智能判断下一步 |

---

## 智能状态判断

### 读取当前状态

从 `09_Progress_Tracker.md` 和 `conversations/summaries/01_learning_progress.md` 读取：

| 当前状态特征 | 下一步行动 | 执行的Skills |
|-------------|-----------|-------------|
| `jd_data/images/` 有新文件 | 需要同步JD | doc-sync |
| 用户指定主题（如"Python闭包"） | 开始学习这个主题 | practice → summarizer |
| 刚完成practice（有practice文件） | 需要总结 | summarizer |
| 刚总结完（01_learning_progress.md已更新） | 询问是否评估 | (ask) → assessor or checkpoint |
| 无明确状态 | 显示当前进度 | progress-tracker |

### 示例判断流程

```
用户: /开始学习

AI执行:
1. 读取 09_Progress_Tracker.md
2. 检查 jd_data/images/ 有新JD?
   → Yes: 执行 doc-sync
3. 检查用户是否指定主题?
   → Yes (如"Python闭包"): 执行 practice → summarizer
   → No: 检查当前状态
4. 根据状态决定下一步...
```

---

## 执行单个Skill

当用户明确指定Skill时：

### 示例1：用户说 "/生成练习 Python装饰器"

```yaml
执行: practice skill
参数:
  topic: "Python装饰器"
  source: "用户指定"
输出: 生成练习题
```

### 示例2：用户说 "/查看进度"

```yaml
执行: progress-tracker skill
参数: {}
输出: 显示当前技能水平和进度
```

### 示例3：用户说 "/记录面试"

```yaml
执行: interview-recorder skill
参数: {}
输出: 引导用户记录面试
```

---

## 执行Workflow

当用户说 "/开始学习" 或 "/继续学习" 时：

### 场景1：用户指定主题

```
用户: /开始学习 Python闭包

AI执行:
1. practice (topic="Python闭包")
2. summarizer (topic="Python闭包")
3. 询问: "是否评估?"
   → Yes: assessor → checkpoint
   → No: 结束
```

### 场景2：用户未指定主题（智能判断）

```
用户: /开始学习

AI执行:
1. 检查当前状态
   - 如果有新JD → doc-sync
   - 如果有未完成主题 → 继续
   - 如果无明确状态 → progress-tracker
2. 根据状态执行相应Skills
```

---

## Registry-based Configuration

### 读取Skills配置

从 `.github/skills/registry.yaml` 读取：

```yaml
skills:
  - id: practice
    name: "练习生成"
    category: "learning"
    stage: 3
    triggers: ["生成练习", "practice", "练习"]
    path: "practice/SKILL.md"
```

### 动态调用Skill

```python
# 伪代码
def invoke_skill(skill_id, context):
    skill_config = registry.get_skill(skill_id)
    skill_path = skill_config.path

    # 读取Skill的SKILL.md
    skill_definition = read_skill_md(skill_path)

    # 执行Skill
    return execute_skill(skill_definition, context)
```

---

## 可扩展性设计

### 添加新Skill无需修改learning-workflow

**示例**：添加 `project-planner` skill

1. 创建Skill目录和SKILL.md
2. 在registry.yaml中添加配置：
   ```yaml
   - id: project-planner
     name: "项目规划"
     category: "project"
     triggers: ["规划项目", "project plan"]
   ```
3. 完成！learning-workflow自动识别新Skill

### 添加新Workflow

**示例**：添加 `project` workflow

1. 在registry.yaml中添加：
   ```yaml
   workflows:
     project:
       name: "项目工作流"
       stages:
         - project-planner
         - task-tracker
   ```
2. 完成！learning-workflow自动支持新workflow

---

## Orchestrator Rules

1. **Registry作为单一数据源**：所有Skills配置从registry.yaml读取
2. **不硬编码Skill名称**：使用registry中的id和triggers
3. **支持动态扩展**：添加新Skill无需修改本文件
4. **智能路由优先**：优先执行用户明确指定的Skill
5. **状态驱动**：根据当前状态智能判断下一步
6. **资源优化**：按需执行，不浪费资源

---

## Error Handling

### 错误1：registry.yaml不存在

**处理**：
```
错误：无法读取Skills注册表 (.github/skills/registry.yaml)

请确保：
1. registry.yaml 文件存在
2. YAML格式正确
3. Skills目录结构完整
```

### 错误2：找不到指定Skill

**处理**：
```
错误：找不到Skill "xxx"

可用的Skills：
- doc-sync
- progress-tracker
- practice
- summarizer
- assessor
- checkpoint
- interview-recorder

使用 /help 查看所有可用命令
```

### 错误3：当前状态无法判断

**处理**：
```
抱歉，我无法确定下一步操作。

当前状态：[状态描述]

建议：
1. 查看进度：/查看进度
2. 选择具体命令：
   - /生成练习 [主题]
   - /总结学习
   - /评估技能
```

---

## 示例执行流程

### 示例1：学了10分钟Python

```bash
用户: /开始学习 Python闭包

AI分析:
- 用户指定主题：Python闭包
- 无需跑完整流程

AI执行:
1. practice skill (生成Python闭包练习)
2. summarizer skill (生成学习总结)

询问: "是否评估?"
用户: 不需要

结束 ✅ (只执行2个阶段)
```

### 示例2：添加新JD

```bash
用户: (把3个新JD截图放到 jd_data/images/)
用户: /同步文档

AI执行:
1. doc-sync skill (解析新JD + 更新03/04)

结束 ✅ (只执行1个阶段)
```

### 示例3：查看进度

```bash
用户: /查看进度

AI执行:
1. progress-tracker skill (显示当前状态)

结束 ✅ (只执行1个阶段)
```

---

## 更新日志

| 版本 | 日期 | 变更 |
|------|------|------|
| 2.0 | 2026-02-02 | 重构为智能路由器，支持模块化执行 |
| 1.0 | 2026-01-31 | 初始版本，固定5阶段流程 |

---

**维护者**：Learning System Architect
**最后更新**：2026-02-02
