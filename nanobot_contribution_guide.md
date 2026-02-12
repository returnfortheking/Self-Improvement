# Nanobot 贡献方向选择指南与技术深度解析

> **目标**：教你如何评估和选择开源项目的贡献方向
> **案例**：Nanobot 项目实战
> **策略**：Vibe Coding（AI 辅助编程，不学 Go）

---

## 第一部分：如何选择贡献方向

### 📊 贡献方向评估框架

选择贡献方向时，需要从 5 个维度评估：

#### 维度 1：面试价值（权重 40%）

**问题**：这个方向能让面试官记住你吗？

| 贡献方向 | 面试价值 | 理由 |
|---------|---------|------|
| **Multi-Agent 架构** | ⭐⭐⭐⭐⭐ | 2025 最热话题，展示系统设计能力 |
| **MCP 协议完善** | ⭐⭐⭐⭐⭐ | 标准化协议经验，展示前瞻性 |
| **安全护栏系统** | ⭐⭐⭐⭐ | 展示安全意识和工程化能力 |
| **性能优化** | ⭐⭐⭐ | 基础能力，但不突出 |
| **文档/测试** | ⭐⭐ | 展示责任心，但技术深度不够 |

**面试话术对比**：
- ❌ 文档贡献："我写了 Nanobot 的 API 文档"
- ✅ Multi-Agent："我设计了 Nanobot 的 Multi-Agent 架构，实现了 Manager-Worker 模式和任务分发机制"

---

#### 维度 2：技术匹配度（权重 25%）

**问题**：这个方向需要的技术栈你掌握多少？

| 贡献方向 | 所需技术 | 你的水平 | 匹配度 |
|---------|---------|---------|--------|
| **Multi-Agent 架构** | Agent 理论、系统设计 | ⭐⭐⭐⭐ 理论扎实 | ✅ 完美匹配 |
| **MCP 协议完善** | 协议理解、网络编程 | ⭐⭐⭐ 有基础 | ✅ 良好匹配 |
| **安全护栏系统** | 安全策略、沙箱机制 | ⭐⭐ 理论有 | ⚠️ 需要学习 |
| **性能优化** | Go 性能调优、并发 | ⭐ | ❌ 不匹配 |
| **UI 开发** | Svelte、前端 | ⭐⭐⭐ 熟练 | ✅ 可行（但不是首选）|

**Vibe Coding 补偿**：
- 即使不熟悉 Go，AI 也能帮你写代码
- 但你需要理解**设计**和**架构**
- 你的 Agent 理论 ⭐⭐⭐⭐ 是最大优势

---

#### 维度 3：项目需求度（权重 20%）

**问题**：项目真的需要这个功能吗？是不是核心功能？

**如何判断**：

1. **查看 README 的 Roadmap**
   ```markdown
   | Feature Category | Feature | Status |
   |------------------|---------|--------|
   | MCP Core | TODO | 🚧 Partial |
   | MCP-UI | TODO | ❌ Not yet |
   ```
   → **Partial/Not yet** = 高需求

2. **查看 Issues**
   - 如果有相关 Issue = 社区需要
   - 如果没人提 = 可能不重要

3. **查看 DESIGN.md 的 Open Questions**
   ```markdown
   ## Open Questions
   - Do we support multi-agent (manager/worker)?
   ```
   → **Open Question** = 设计讨论中 = 机会

**Nanobot 的需求度排序**：
1. **Multi-Agent** - DESIGN.md 明确问 "Do we support multi-agent?"
2. **MCP 协议完善** - README 显示 "Partial"
3. **安全护栏** - DESIGN.md 定义了但可能未实现

---

#### 维度 4：竞争激烈度（权重 10%）

**问题**：有多少人在做这个？容易抢占先机吗？

**判断方法**：
- 查看 GitHub Contributors 数量
- 查看 PR 中是否有相关方向
- 查看 Issues 的活跃度

**Nanobot 当前状态**（2025-02-10）：
- 总 Contributors：较少（刚推出）
- 相关 PR：几乎没有 Multi-Agent 的 PR
- **结论**：竞争小，机会大

---

#### 维度 5：可行性与时间成本（权重 5%）

**问题**：你能在 4-8 周内完成吗？

| 贡献方向 | 时间估算 | 可行性 |
|---------|---------|--------|
| **Multi-Agent 架构** | 6-8周 | ⚠️ 需要规划 |
| **MCP 协议完善** | 4-6周 | ✅ 可行 |
| **安全护栏系统** | 4-6周 | ✅ 可行 |
| **示例 Agent Pack** | 2-3周 | ✅ 非常可行 |

---

### 🎯 综合评分矩阵

基于以上 5 个维度，给你做一个综合评分：

| 贡献方向 | 面试价值 | 技术匹配 | 项目需求 | 竞争激烈度 | 可行性 | **总分** |
|---------|---------|---------|---------|-----------|--------|---------|
| **Multi-Agent 架构** | 5/5 | 5/5 | 5/5 | 5/5 | 4/5 | **24/25** ⭐⭐⭐⭐⭐ |
| **MCP 协议完善** | 5/5 | 4/5 | 4/5 | 5/5 | 5/5 | **23/25** ⭐⭐⭐⭐⭐ |
| **安全护栏系统** | 4/5 | 2/5 | 4/5 | 5/5 | 5/5 | **20/25** ⭐⭐⭐⭐ |
| **示例 Agent Pack** | 3/5 | 4/5 | 3/5 | 4/5 | 5/5 | **19/25** ⭐⭐⭐ |
| **性能优化** | 3/5 | 1/5 | 3/5 | 5/5 | 3/5 | **15/25** ⭐⭐ |

**推荐顺序**：
1. **Multi-Agent 架构**（24分） - 面试最强，但需要设计能力
2. **MCP 协议完善**（23分） - 协议理解，时间较短
3. **安全护栏系统**（20分） - 安全能力展示

---

## 第二部分：Nanobot 技术深度解析

基于我刚才读取的核心代码，让我给你讲解 Nanobot 的核心技术。

---

### 🏗️ 核心架构：Config-Driven Agent System

**设计理念**（来自 DESIGN.md）：
> "Configuration is the product. Code is a stable execution engine; behavior lives in versioned Markdown."

**三层架构**：

```
┌─────────────────────────────────────────┐
│   Agent (角色/Persona)                   │
│   ├── model: claude-3-7-sonnet-latest   │
│   ├── tools: [Read, Write, Edit]        │
│   ├── mcpServers: [store, github]       │
│   └── instructions: "You are a..."      │
└─────────────────────────────────────────┘
         │ 调用
         ▼
┌─────────────────────────────────────────┐
│   Skill (能力/技能)                      │
│   ├── name: write_code                  │
│   ├── description: "Write clean code"   │
│   ├── tools: [Write, Edit]              │
│   └── inputs: {topic, language}         │
└─────────────────────────────────────────┘
         │ 组合成
         ▼
┌─────────────────────────────────────────┐
│   Task (具体任务)                        │
│   ├── name: refactor_database           │
│   ├── agent: senior_dev                 │
│   ├── skills: [analyze, refactor]       │
│   └── steps: [analyze.md, plan.md, ...] │
└─────────────────────────────────────────┘
```

---

### 🔄 Agent 运行流程（代码级解析）

基于我刚才读取的 `pkg/agents/run.go`，这是真实的运行流程：

```go
// 1. 用户请求
req := types.CompletionRequest{
    Agent: "shopping-assistant",
    Input: []types.Message{
        {Role: "user", Content: "帮我找一件红色的衬衫"},
    },
}

// 2. 加载 Agent 配置
agent := config.Agents["shopping-assistant"]
// agent.Model = "claude-3-7-sonnet-latest"
// agent.MCPServers = ["store"]

// 3. 添加 Tools（来自 run.go:addTools）
toolMappings := registry.BuildToolMappings(ctx, agent.MCPServers)
// toolMappings = {
//   "store/search": {MCPServer: "store", TargetName: "search_products"}
// }

// 4. 调用 LLM（来自 run.go:run）
resp := completer.Complete(ctx, req)
// LLM 返回：
// {
//   "Output": {
//     "Items": [{
//       "ToolCall": {
//         "Name": "store/search",
//         "Arguments": {"color": "red", "type": "shirt"}
//       }
//     }]
//   }
// }

// 5. 执行 Tool Call（来自 toolcall.go:toolCalls）
for _, toolCall := range resp.Output.Items {
    targetServer := toolMappings[toolCall.Name]
    result := registry.Call(ctx, targetServer.MCPServer, toolCall.Arguments)
    // 调用 MCP Server 的工具
}

// 6. 将 Tool Result 返回给 LLM（多轮对话）
run.ToolOutputs[callID] = result
// 再次调用 LLM，继续对话
```

---

### 🔌 MCP 协议实现细节

基于 `pkg/mcp/types.go`，这是 MCP 协议的核心类型定义：

#### 1️⃣ 初始化握手

```go
// 客户端发送
type InitializeRequest struct {
    ProtocolVersion string             // "2024-11-05"
    Capabilities    ClientCapabilities // 客户端能力
    ClientInfo      ClientInfo         // 客户端信息
}

// 服务端返回
type InitializeResult struct {
    ProtocolVersion string            // 协议版本
    Capabilities    ServerCapabilities // 服务端能力
    ServerInfo      ServerInfo        // 服务端信息
    Instructions    string            // 使用说明
}
```

**面试话术**：
> "我参与了 MCP 协议的初始化握手实现，包括能力协商（Capabilities Negotiation）和版本兼容性处理。"

---

#### 2️⃣ Tool Call 机制

```go
// 工具定义（来自 MCP Server）
type Tool struct {
    Name         string          // 工具名称
    Description  string          // 工具描述
    InputSchema  json.RawMessage // 输入 Schema（JSON Schema）
    OutputSchema json.RawMessage // 输出 Schema
}

// 工具调用请求
type CallToolRequest struct {
    Name      string         // 工具名称
    Arguments map[string]any  // 参数
    Meta      map[string]any  // 元数据
}

// 工具调用结果
type CallToolResult struct {
    Content           []Content // 内容
    IsError           bool      // 是否错误
}
```

**完整的 Tool Call 流程**：

```
1. LLM 决定调用工具
   LLM → "我需要调用 store/search 工具"

2. Nanobot 构建请求
   CallToolRequest{
       Name: "store/search",
       Arguments: {"color": "red", "type": "shirt"}
   }

3. 通过 MCP 协议发送到 MCP Server
   HTTP POST http://mcpstorefront.com/tools/call
   Body: {"name": "store/search", "arguments": {...}}

4. MCP Server 执行并返回
   CallToolResult{
       Content: [{
           Type: "text",
           Text: "找到 3 件红色衬衫：..."
       }]
   }

5. Nanobot 将结果返回给 LLM
   继续对话
```

**面试话术**：
> "我深入理解了 MCP 协议的 Tool Call 机制，包括 Schema 验证、参数序列化、错误处理等核心环节。"

---

#### 3️⃣ Session 管理

```go
// Session 抽象（来自 pkg/mcp/session.go）
type Session interface {
    ID() string
    Get(key string, value any) bool
    Set(key string, value any)
    Parent() Session
    Root() Session
}

// 聊天上下文存储
type Execution struct {
    Request          types.CompletionRequest
    PopulatedRequest *types.CompletionRequest
    Response         *types.CompletionResponse
    ToolOutputs      map[string]types.ToolOutput
    ToolToMCPServer  types.ToolMappings
    Done             bool
}
```

**Session 的作用**：
- 存储聊天历史（多轮对话）
- 存储 Tool Call 结果
- 支持 Thread（对话线程）

**面试话术**：
> "我实现了 MCP Session 管理，支持多轮对话、工具调用历史、Thread 隔离等核心功能。"

---

### 🎨 实际代码示例

#### 示例 1：添加一个新的 Tool

假设你想添加一个 `git_commit` 工具：

**vibe coding 流程**：

```yaml
# prompt.yaml
role: Go 语言专家
task: |
  在 Nanobot 中添加一个新的内置工具：git_commit

  要求：
  1. 工具名称：git_commit
  2. 输入参数：message（string，required），files（array of string，optional）
  3. 功能：执行 git commit 命令
  4. 参考现有的工具实现：pkg/tools/service.go

context: |
  - 项目使用 Go 语言
  - 工具注册在 pkg/tools/service.go
  - 需要实现 Tool interface

output_format: |
  1. 修改代码的 diff
  2. 测试方法
  3. 可能的问题
```

**让 AI 帮你生成代码**，然后你：
1. 理解代码逻辑
2. 测试是否工作
3. 提交 PR

---

#### 示例 2：实现 Multi-Agent（Manager-Worker）

**设计思路**（不需要写 Go 代码）：

```markdown
## Multi-Agent 架构设计

### 1. Manager Agent 配置

```yaml
# .nanobot/agents/manager/AGENT.md
---
name: Task Manager
model: claude-3-7-sonnet-latest
agents:
  - worker_code
  - worker_test
  - worker_docs
---

You are a task manager agent.

你的职责：
1. 接收用户任务
2. 将任务分解为子任务
3. 分配给合适的 Worker Agent
4. 聚合 Worker 的结果

工作流程：
- 分析任务类型
- 选择 Worker：code → worker_code, test → worker_test
- 发送任务给 Worker
- 收集结果并整合
```

### 2. Worker Agent 配置

```yaml
# .nanobot/agents/worker_code/AGENT.md
---
name: Code Worker
model: claude-3-5-sonnet-latest
tools:
  - Read
  - Write
  - Edit
---

You are a code generation worker agent.

你的职责：
- 接收 Manager 分配的编码任务
- 生成高质量代码
- 返回结果给 Manager
```

### 3. 实现步骤（让 AI 帮你写 Go 代码）

1. **扩展 Agent 配置**
   - 添加 `agents` 字段到 Agent schema
   - 支持引用其他 Agent

2. **实现 Agent 调用**
   - 在 `pkg/agents/run.go` 添加 `invokeAgent` 方法
   - 复用现有的 `Complete` 方法

3. **实现结果聚合**
   - 收集所有 Worker 的结果
   - 合并后返回给用户
```

**面试话术**：
> "我设计了 Nanobot 的 Multi-Agent 架构，实现了 Manager-Worker 模式。设计了 Agent 间通信协议、任务分发机制、结果聚合策略。"

---

## 第三部分：Vibe Coding 实战指南

### 🤖 什么是 Vibe Coding？

**定义**：用 AI 辅助编程，自己专注于**设计**和**理解**，让 AI 写代码。

**核心思想**：
- 你懂 Agent 理论、系统设计 → 你的优势
- AI 懂 Go 语法、API 调用 → AI 的优势
- 你做架构师，AI 做实现者

---

### 📝 Vibe Coding 工作流

#### 步骤 1：设计方案（你来做）

**你做的事情**：
1. 理解需求
2. 设计架构
3. 定义接口
4. 编写设计文档

**示例**：Multi-Agent 架构设计

```markdown
## Multi-Agent 通信协议设计

### 消息格式

```json
{
  "type": "agent_call",
  "from_agent": "manager",
  "to_agent": "worker_code",
  "task_id": "task_123",
  "task": {
    "description": "实现一个用户登录功能",
    "context": "使用 React + TypeScript"
  }
}
```

### 调用流程

1. Manager 调用 Worker
2. Worker 执行任务
3. Worker 返回结果给 Manager
4. Manager 聚合所有结果
```

**你不需要写 Go 代码，只需要**：
- 理解 Agent 理论 ✅（你有）
- 设计系统架构 ✅（可以学）
- 编写设计文档 ✅（AI 帮你润色）

---

#### 步骤 2：让 AI 生成代码

**Prompt 模板**：

```yaml
role: Go 语言后端开发专家

task: |
  基于 Multi-Agent 架构设计文档，实现 Nanobot 的 Agent 调用功能

context: |
  项目地址：https://github.com/nanobot-ai/nanobot
  核心文件：
  - pkg/agents/run.go（Agent 运行逻辑）
  - pkg/mcp/types.go（MCP 类型定义）

  当前 Agent 配置结构（pkg/types/config.go）：
  type Agent struct {
      Name      string
      Model     string
      Tools     []string
      Agents    []string  // 新增：支持引用其他 Agent
      // ...
  }

design: |
  ## 架构设计

  1. 扩展 Agent 配置，添加 `agents` 字段
  2. 实现 `invokeAgent` 方法，复用 `Complete` 方法
  3. 支持嵌套调用（Manager → Worker → SubWorker）

requirements:
  - 使用 Go 1.21+
  - 遵循项目现有代码风格
  - 添加单元测试
  - 更新文档

output:
  1. 代码 diff（显示修改的文件和行）
  2. 测试方法
  3. 可能的边界情况和错误处理
```

**AI 会给你**：
- 完整的 Go 代码
- 代码解释
- 测试方法

---

#### 步骤 3：理解代码（重要！）

**AI 生成代码后，你必须**：

1. **阅读代码，理解逻辑**
   ```go
   func (a *Agents) invokeAgent(ctx context.Context, targetAgent string, task Task) (*Response, error) {
       // 1. 加载目标 Agent 配置
       agent := config.Agents[targetAgent]

       // 2. 构建请求
       req := CompletionRequest{
           Agent: targetAgent,
           Input: []Message{
               {Role: "user", Content: task.Description},
           },
       }

       // 3. 调用 Complete（复用现有逻辑）
       return a.Complete(ctx, req)
   }
   ```
   **你理解的**：这就是调用另一个 Agent，和调用用户一样简单！

2. **手动测试**
   ```bash
   cd nanobot
   make
   nanobot run ./examples/multi-agent.yaml
   ```

3. **调试问题**
   - 如果出错，问 AI："这个错误是什么原因？"
   - 如果测试失败，问 AI："如何调试这个功能？"

---

#### 步骤 4：提交 PR

**PR 描述模板**：

```markdown
## Multi-Agent Support: Manager-Worker Pattern

### 概述
实现了 Nanobot 的 Multi-Agent 架构，支持 Manager-Worker 模式。

### 设计文档
[链接到你的设计文档]

### 主要变更
1. 扩展 Agent 配置，添加 `agents` 字段
2. 实现 `invokeAgent` 方法
3. 支持嵌套 Agent 调用

### 测试
- [x] 单元测试
- [x] 集成测试（examples/multi-agent.yaml）
- [x] 手动测试

### 设计考虑
- 复用现有的 `Complete` 方法，避免重复代码
- 支持 Agent 间的任意嵌套
- 保持向后兼容（`agents` 字段可选）

### 已知限制
- 暂不支持并行调用（可以后续优化）
- 暂不支持 Agent 间直接通信（只能通过 Manager）

### 示例
参见 `examples/multi-agent.yaml`
```

---

### 💡 Vibe Coding 最佳实践

#### 1. 从小做起

**❌ 错误做法**：
- 直接实现完整的 Multi-Agent 系统
- 一次性改 10 个文件

**✅ 正确做法**：
- 先实现一个简单的 Agent 调用
- 测试通过后，再扩展功能

**第一个 PR**：
- 只添加 1 个功能
- 只改 1-2 个文件
- 容易 Code Review

---

#### 2. 理解大于代码

**你不需要**：
- 背诵 Go 语法
- 记住标准库 API
- 手写每一行代码

**但你必须**：
- **理解架构设计**
- **理解数据流**
- **理解错误处理**

**示例**：
```
AI 写的代码：
func (a *Agents) invokeAgent(...) {
    agent := config.Agents[targetAgent]
    ...
}

你理解的：
这个函数做了什么？
1. 从配置中加载目标 Agent
2. 构建请求
3. 调用 Complete 方法

如果面试官问：
"你如何实现 Agent 调用的？"
→ 你可以说出上面 3 步，而不是说代码细节
```

---

#### 3. 利用 AI 学习

**学习 Go 的方法**（针对 Nanobot）：

**Week 1**：基础语法
```yaml
task: |
  我是 Nanobot 项目的贡献者，需要理解 Go 代码。

  请用对比的方式教我 Go 语法（参考 Python/JavaScript）：

  1. 变量声明（var vs let）
  2. 函数定义（func vs function）
  3. 结构体（struct vs class）
  4. 接口（interface vs interface）
  5. 错误处理（error vs try-catch）

  每个概念给我：
  - 语法对比
  - 实际例子（来自 Nanobot 代码）
  - 常见陷阱
```

**Week 2**：并发模型
```yaml
task: |
  教我 Go 的并发模型（Goroutines + Channels）

  参考 Nanobot 中的并发代码：
  - pkg/mcp/callback.go
  - pkg/mcp/runner.go

  我需要理解：
  1. Goroutine vs Thread
  2. Channel 的使用
  3. Select 语句
  4. 并发安全
```

**Week 3**：HTTP/WebSocket
```yaml
task: |
  教我 Go 的 HTTP/WebSocket 编程

  参考 Nanobot 的网络代码：
  - pkg/mcp/httpclient.go
  - pkg/mcp/httpserver.go

  我需要理解：
  1. HTTP Client 使用
  2. HTTP Server 实现
  3. WebSocket 协议
  4. 错误处理
```

---

#### 4. Code Review 是学习机会

**当 Maintainer Review 你的代码时**：

**Maintainer**：
```go
// 这个函数有问题，应该使用 context.Context
func (a *Agents) invokeAgent(targetAgent string) (*Response, error) {
    // ...
}
```

**你的回答**（先问 AI）：
```yaml
task: |
  Maintainer 说我的代码应该使用 context.Context

  请解释：
  1. context.Context 是什么？
  2. 为什么需要它？
  3. 如何修改我的代码？
  4. 有什么最佳实践？

  原代码：
  [你的代码]

  期望：修改后的代码 + 解释
```

**然后你回复 Maintainer**：
> "感谢反馈！我理解了 context.Context 的作用，它是用于控制请求生命周期和超时的。我已经更新了代码，请再次 Review。"

**这样你就在学习 + 贡献！**

---

## 第四部分：具体行动计划

### 🎯 2 月行动（熟悉阶段）

**Week 1（2.10-2.16）**：
- [ ] 本地运行 Nanobot
  ```bash
  cd d:/AI/2026/LearningSystem/nanobot
  make
  nanobot run ./examples/shopping.yaml
  ```
- [ ] 阅读核心代码（用 AI 帮忙理解）
  - `pkg/mcp/types.go`（MCP 协议类型）
  - `pkg/agents/run.go`（Agent 运行逻辑）
  - `docs/agents/DESIGN.md`（设计文档）
- [ ] 理解 MCP 协议（看文档 + AI 教学）
- [ ] 学习 Go 基础（对比 Python/JavaScript）

**Week 2（2.17-2.23）**：
- [ ] 选择贡献方向（Multi-Agent 或 MCP 协议）
- [ ] 编写设计文档
- [ ] 在 GitHub Discussions 讨论设计
- [ ] 等待 Maintainer 反馈

**Week 3-4（2.24-3.2）**：
- [ ] 根据反馈完善设计
- [ ] 让 AI 生成代码
- [ ] 本地测试
- [ ] 提交第一个 PR（简单功能）

---

### 🎯 3-4 月行动（贡献阶段）

**3 月**：
- [ ] 第 1 个 PR：简单的功能或文档
- [ ] 建立 Trust（建立信任）
- [ ] 参与社区讨论

**4 月**：
- [ ] 第 2-3 个 PR：核心功能实现
- [ ] 成为 Active Contributor
- [ ] 参与 Multi-Agent 或 MCP 协议设计

---

## 总结

### 如何选择贡献方向？

**5 个维度评估**：
1. 面试价值（40%）
2. 技术匹配度（25%）
3. 项目需求度（20%）
4. 竞争激烈度（10%）
5. 可行性（5%）

**Nanobot 最佳选择**：
1. **Multi-Agent 架构**（24/25 分） - 面试最强
2. **MCP 协议完善**（23/25 分） - 协议理解

### Vibe Coding 能行吗？

**答案**：完全可以！

**你只需要**：
- 理解 Agent 理论 ✅（你有）
- 设计系统架构 ✅（可以学）
- 理解 AI 生成的代码 ✅（AI 帮忙）

**AI 负责**：
- 写 Go 代码 ✅
- 调试错误 ✅
- 生成测试 ✅

**面试时**：
- 你讲架构设计
- 你讲 Agent 理论
- 你讲 Multi-Agent 通信协议
- 不用担心 Go 语法细节

---

**下一步**：你想深入哪个方向？
- Multi-Agent 架构设计细节
- MCP 协议深入解析
- Go 语言快速学习路径
- 或者直接开始第一个 PR
