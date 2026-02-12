# Nanobot 开源贡献完整指南：避免重复提交

> **目标**：教你如何参与开源项目，避免重复提交
> **案例**：Nanobot 项目实战
> **策略**：从新手到核心贡献者的完整路径

---

## 第一部分：Nanobot 项目现状分析

### 📊 当前 Issues 状态（基于 GitHub API 数据）

我分析了 Nanobot 的最新 50 个 Issues（按评论数排序），以下是关键发现：

#### 热门 Issues（评论数最多）

| Issue # | 标题 | 状态 | 类型 |
|---------|------|------|------|
| **#108** | feat: add ability to chat with different agents | ✅ 已合并 | PR |
| **#143** | Feature Request: Support custom apiBase for OpenAI-compatible endpoints | 🔵 开放 | Feature Request |
| **#144** | feat: add tool output truncation to prevent context overflow | 🔵 开放 | PR |

**关键发现**：
1. ✅ 项目**活跃**（最新的 Issue 是 2026-02-08，即几天前）
2. ✅ PR **会被合并**（#108 已合并）
3. ✅ 社区**接受新贡献者**（多位不同作者）
4. ⚠️ **Feature Request** 存在（#143）- 机会

---

### 🗺️ Roadmap 在哪里？

**好消息**：Nanobot 的 Roadmap 就在 README 里！

**README 中的 Roadmap**（来自我之前读取的内容）：

```markdown
## Features & Roadmap

Nanobot aims to be a **fully compliant MCP Host** and support all MCP + MCP-UI features.

| Feature Category    | Feature | Status        |
|---------------------|---------|---------------|
| **MCP Core**         | TODO    | ✅ Implemented |
|                     | TODO    | 🚧 Partial     |
|                     | TODO    | ❌ Not yet     |
| **MCP-UI**           | TODO    | 🚧 Partial     |
|                     | TODO    | ✅ Implemented |
|                     | TODO    | ❌ Not yet     |

### Roadmap

- Full MCP + MCP-UI compliance
- More robust multi-agent support
- Production-ready UI
- Expanded model provider support
- Expanded authentication and security features
- Frontend integrations (Slack, SMS, email, embedded web agents)
- Easy embedding into existing apps and websites
```

**关键 Roadmap 项**：
1. **More robust multi-agent support** → 这是你想做的！✅
2. **Production-ready UI** → UI 贡献机会
3. **Expanded authentication and security** → 安全护栏系统
4. **Frontend integrations** → Slack、Email 集成

---

### 📌 Open Questions（设计文档中）

**DESIGN.md** 明确列出的未解决问题：

```markdown
## Open Questions
- Do we support multi-agent (manager/worker) out of the gate?
- Do we allow skills to embed sub-steps (nested structured execution)?
- What is the minimal built-in toolset? (fs, shell, web, git)
- Where does secrets management live (env, vault integration, per-run ephemeral tokens)?
```

**这意味着**：
- ✅ **Multi-Agent 还在讨论阶段** = 机会！
- ✅ 你可以参与设计讨论
- ✅ 你可以提出你的方案

---

## 第二部分：如何避免重复提交？（开源新人必读）

### 🔍 为什么重复提交是新人最常犯的错误？

**典型场景**：
```
新人A：花了2周实现功能X
新人B：同时花了2周实现同样的功能X
结果：
  - Maintainer：只能合并一个
  - 另一个人的时间浪费了
  - 两个人的PR都被拒绝（因为冲突）
```

**原因**：
1. ❌ 没有查看 Issues
2. ❌ 没有与社区讨论
3. ❌ 直接开始写代码
4. ❌ 提交 PR 时才发现有人已经在做了

---

### ✅ 避免重复提交的 5 步流程

#### 步骤 1：搜索 Issues（必做！）

**方法 A：使用 GitHub 网页搜索**

1. 打开 https://github.com/nanobot-ai/nanobot/issues
2. 在搜索框输入关键词：
   ```
   multi-agent
   OR
   MCP protocol
   OR
   tool truncation
   ```

**方法 B：使用 GitHub 高级搜索**

```
is:open is:issue multi-agent repo:nanobot-ai/nanobot
is:open is:pull-request multi-agent repo:nanobot-ai/nanobot
```

**方法 C：使用标签过滤**

```
label:"enhancement"
label:"bug"
label:"documentation"
```

---

#### 步骤 2：查看 PR（Pull Requests）

**为什么重要**：有人可能已经在做你想做的功能！

**查看方法**：
```bash
# 打开所有 PR
https://github.com/nanobot-ai/nanobot/pulls

# 搜索 PR
multi-agent
```

**Nanobot 当前 PR**（基于数据）：
- **#144**：tool output truncation（正在进行）
- **#108**：chat with different agents（已合并）

**如果找到相关的 PR**：
- ✅ 先不要做
- 💬 在 PR 下评论："我可以帮忙测试/改进XX"
- 🤝 询问作者是否需要帮助

---

#### 步骤 3：在 Discussions 中讨论（关键！）

**为什么重要**：
- 提前沟通设计思路
- 获得 Maintainer 反馈
- 避免方向错误

**如何使用 Discussions**：

1. 打开 https://github.com/nanobot-ai/nanobot/discussions
2. 点击 "New discussion"
3. 选择类别（例如 "Ideas"）
4. **标题示例**：
   ```
   Proposal: Multi-Agent Architecture Design
   Idea: Implement Multi-Agent using Manager-Worker Pattern
   Question: What's the roadmap for multi-agent support?
   ```

**Discussion 模板**：

```markdown
## Proposal: Multi-Agent Architecture for Nanobot

### 背景
我注意到 DESIGN.md 中提到 "Do we support multi-agent (manager/worker)?" 作为 Open Question。我想贡献这个功能。

### 设计思路
我设计了一个 Manager-Worker 架构：

1. **Manager Agent**：负责任务分解和分发
2. **Worker Agents**：执行具体任务
3. **通信协议**：使用 MCP 协议进行 Agent 间通信

### 询问社区
1. 这个设计是否符合 Nanobot 的架构方向？
2. 是否有人已经在做类似的工作？
3. 有什么建议或改进？

### 下一步计划
如果社区认可，我计划：
- Week 1-2: 完善设计文档
- Week 3-4: 实现基础功能
- Week 5-6: 测试和优化

欢迎任何反馈！
```

**等待反馈**（通常 2-7 天）：
- 如果 Maintainer 说 "Good idea!" → 开始做
- 如果有人说 "I'm already working on this" → 协作或放弃
- 如果有人说 "This doesn't fit" → 调整方向

---

#### 步骤 4：创建 Issue（可选但推荐）

**为什么创建 Issue**：
- 宣布你的计划
- 让其他人知道你在做
- 获得反馈

**Issue 模板**：

```markdown
## Feature: Multi-Agent Support

### 概述
实现 Multi-Agent 架构，支持 Manager-Worker 模式。

### Motivation
[参考 DESIGN.md 的 Open Question]
> Do we support multi-agent (manager/worker) out of the gate?

### 设计方案
详见 Discussion: [链接到你的 Discussion]

### 实现计划
- [ ] Phase 1: 扩展 Agent 配置支持 `agents` 字段
- [ ] Phase 2: 实现 `invokeAgent` 方法
- [ ] Phase 3: 支持嵌套调用
- [ ] Phase 4: 结果聚合

### 时间线
预计 6-8 周（每周 4-6 小时）

### 协作
欢迎任何反馈和建议！如果有其他人想做类似的工作，欢迎联系我一起合作。
```

---

#### 步骤 5：再开始写代码

**只有完成了前面 4 步，才开始写代码！**

**流程总结**：
```
1. 搜索 Issues → 确认没人做
   ↓
2. 查看 PR → 确认没有进行中的 PR
   ↓
3. Discussions 讨论 → 获得社区认可
   ↓
4. 创建 Issue → 宣布计划
   ↓
5. 开始写代码
```

---

## 第三部分：Nanobot 具体行动指南

### 🎯 2 月行动（避免重复提交）

#### Week 1（2.10-2.16）：调研与讨论

**Day 1-2**：
- [ ] 阅读 README.md 中的 Roadmap
- [ ] 阅读 DESIGN.md 的 Open Questions
- [ ] 搜索 Issues：`multi-agent`、`MCP protocol`
- [ ] 查看 PR：是否有人在做 Multi-Agent

**Day 3-4**：
- [ ] 在 GitHub Discussions 发起讨论：
  ```markdown
  Title: Proposal: Multi-Agent Architecture Design

  内容：使用上面的模板
  ```

**Day 5-7**：
- [ ] 等待反馈（通常 2-7 天）
- [ ] 如果有人回复，积极回应
- [ ] 如果有人说已经在做，询问是否需要帮助

---

#### Week 2（2.17-2.23）：设计与规划

**如果社区认可**：
- [ ] 编写详细设计文档
- [ ] 创建 Issue 宣布计划
- [ ] 制定时间表

**如果社区不认可**：
- [ ] 调整方向
- [ ] 或选择其他贡献方向（如 MCP 协议完善）

---

#### Week 3-4（2.24-3.2）：实现与测试

- [ ] 让 AI 生成代码（Vibe Coding）
- [ ] 本地测试
- [ ] 提交 PR

---

### 📋 检查清单（提交前必看）

**提交 PR 前，回答这些问题**：

- [ ] 我搜索过 Issues 吗？
- [ ] 我查看过 PR 吗？
- [ ] 我在 Discussions 中讨论过吗？
- [ ] 我创建 Issue 宣布计划了吗？
- [ ] 有人已经在做这个功能吗？
- [ ] Maintainer 认可这个方向吗？

**如果任何一个是 NO**：
→ ❌ **不要提交 PR**
→ 返回前面的步骤

---

## 第四部分：实际案例演示

### 案例 1：避免重复提交（正确做法）

**场景**：你想做 Multi-Agent 功能

**错误做法** ❌：
```
1. 直接开始写代码（2周）
2. 提交 PR
3. Maintainer：有人已经在做了
4. 你的 PR 被关闭
5. 浪费 2 周时间
```

**正确做法 ✅**：
```
1. 搜索 Issues → 发现 DESIGN.md 的 Open Question
2. 查看 PR → 确认没有人做 Multi-Agent
3. 发起 Discussions → 社区认可设计
4. 创建 Issue → 宣布计划
5. 开始写代码 → 提交 PR
6. PR 被合并 ✅
```

---

### 案例 2：发现有人已经在做（协作做法）

**场景**：搜索后发现有人已经在做 Multi-Agent

**正确做法 ✅**：
```
1. 在那个 Issue 下评论：
   "Hi! 我也感兴趣这个功能，我可以帮忙测试/文档/部分实现吗？"

2. 作者回复：欢迎帮助

3. 协作方式：
   - 你负责文档
   - 他负责核心代码
   - 一起 Review
   - 一起提交 PR（Co-author）

4. 结果：双赢
   - 你获得了贡献经验
   - 他减轻了工作量
   - 社区获得了功能
```

---

### 案例 3：Feature Request（快速贡献）

**场景**：你看到 Issue #143（支持自定义 API Base）

**快速贡献**：
```
1. 确认没有人做
2. 在 Issue 下评论：
   "我可以实现这个功能！计划如下：..."
3. 等待 Maintainer 回复
4. 开始实现
```

**时间成本**：2-3 周（比 Multi-Agent 简单）

---

## 第五部分：工具与技巧

### 🔍 搜索技巧

**GitHub 高级搜索语法**：

```
# 搜索开放 Issues
is:issue is:open multi-agent repo:nanobot-ai/nanobot

# 搜索 PR
is:pr is:open multi-agent repo:nanobot-ai/nanobot

# 搜索标签
label:"enhancement" is:open repo:nanobot-ai/nanobot

# 搜索评论数多的
comments:>5 is:open repo:nanobot-ai/nanobot

# 搜索更新时间
updated:>2025-02-01 is:open repo:nanobot-ai/nanobot
```

---

### 📧 如何联系 Maintainer？

**方法 1：在 Issue 下评论**
```
@maintainer_id 你好，我想实现这个功能，你觉得这个方向如何？
```

**方法 2：在 Discussions 中提**
```
@maintainer_id 请问你对 Multi-Agent 的设计有什么建议？
```

**方法 3：查看 CONTRIBUTING.md**
```
很多项目有贡献指南，告诉新人如何开始。
```

---

### ⏰ 何时跟进？

**时间线建议**：

| 动作 | 等待时间 |
|------|---------|
| Discussions 反馈 | 2-7 天 |
| Issue 回复 | 1-3 天 |
| PR Review | 3-14 天（视复杂度）|
| PR 合并 | 1-7 天 |

**如果太久没回复**：
- 1 周后：在 Issue 下友好提醒
- 2 周后：在 Discussions 中询问
- 1 个月后：可能 Maintainer 忙碌，考虑其他项目

---

## 第六部分：Nanobot 特定建议

### 🎯 推荐的贡献方向（基于 Issues 分析）

#### 方向 1：Multi-Agent 架构（首选）⭐⭐⭐⭐⭐

**为什么推荐**：
- ✅ DESIGN.md 明确列为 Open Question
- ✅ Roadmap 提到 "More robust multi-agent support"
- ✅ 没有人在做（根据 Issues 分析）
- ✅ 展示系统设计能力

**行动步骤**：
1. **Week 1**：在 Discussions 提案设计
2. **Week 2**：等待反馈，完善设计
3. **Week 3-6**：实现基础功能
4. **Week 7-8**：测试和优化

**预期成果**：
- 成为这个功能的核心贡献者
- 简历上的重磅亮点

---

#### 方向 2：MCP 协议完善（次选）⭐⭐⭐⭐⭐

**为什么推荐**：
- ✅ README 显示 "Partial" 状态
- ✅ 你的 Agent 理论可以应用
- ✅ 时间较短（4-6 周）

**行动步骤**：
1. 查看 README 的 "Features & Roadmap"
2. 找到 "Partial" 的功能
3. 在 Issues 中搜索相关讨论
4. 实现

---

#### 方向 3：Feature Request 快速贡献（备选）⭐⭐⭐⭐

**Issue #143**：支持自定义 API Base

**行动步骤**：
1. 在 Issue 下评论："我可以实现这个功能"
2. 等待 Maintainer 确认
3. 开始实现

**时间成本**：2-3 周

---

## 总结：避免重复提交的黄金法则

### ✅ 5 步流程（必遵守）

```
1. 搜索 Issues
   ↓
2. 查看 PR
   ↓
3. Discussions 讨论
   ↓
4. 创建 Issue
   ↓
5. 开始写代码
```

### ❌ 永远不要做的事

- ❌ 直接开始写代码（不搜索）
- ❌ 直接提交 PR（不讨论）
- ❌ 假设没人做（要验证）
- ❌ 忽略 Maintainer 的反馈

### ✅ 新人心态

- 🤝 **协作优先**：愿意与他人合作
- 📚 **学习优先**：先理解，再实现
- 💬 **沟通优先**：多讨论，少假设
- 🧪 **实验优先**：从小功能开始

---

## 下一步

**本周行动（2.10-2.16）**：
1. 阅读 README.md 中的 Roadmap
2. 阅读 DESIGN.md 中的 Open Questions
3. 搜索 Issues：`multi-agent`、`MCP protocol`
4. 在 Discussions 发起你的第一个讨论

**你想深入哪个方向**？
- Multi-Agent 架构设计细节
- 如何撰写 Discussions 提案
- 如何与 Maintainer 有效沟通
- 或者其他
