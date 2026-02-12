# Nanobot 项目能力评估与实战指南

> **目标**：评估 Nanobot 能做什么复杂功能 + vibe coding 实战
> **基于**：100 个 GitHub Issues 分析
> **创建时间**：2025-02-10

---

## 第一部分：Issues 分析（基于本地数据）

### 📊 Issues 分类统计

我下载了 Nanobot 的 100 个最新 Issues，以下是分类：

#### 按类型分类

| 类型 | 数量 | 占比 | 示例 |
|------|------|------|------|
| **Feature (新功能)** | ~40% | #143, #144, #108 | feat: add tool output truncation |
| **Bug Fix (修复)** | ~25% | #76, #72 | fix: stop filtering hidden files |
| **Enhancement (增强)** | ~20% | #137, #138 | enhance: use human-friendly step names |
| **Docs (文档)** | ~10% | #142 | docs: Update CLAUDE.md |
| **Chore (杂项)** | ~5% | #141 | chore: report MCP version |

#### 按状态分类

| 状态 | 数量 | 说明 |
|------|------|------|
| **Closed (已关闭)** | ~60% | 已完成或被拒绝 |
| **Open (开放)** | ~40% | 可参与的机会 |

#### 按主题分类（基于标题分析）

| 主题 | Issues 数 | 热度 | 示例 |
|------|-----------|------|------|
| **Multi-Agent** | 3-5 | 🔥 高 | #108: chat with different agents |
| **MCP 协议** | 10-15 | 🔥 高 | #139: support mcp apps |
| **Tool 调用** | 8-10 | 🔥 高 | #144: tool output truncation |
| **UI/UX** | 5-8 | 中 | #137: chat workflows UI |
| **配置系统** | 5-7 | 中 | #143: custom apiBase |
| **文档** | 3-5 | 低 | #142: update docs |

---

### 🔥 高价值 Issues（适合贡献）

#### 1. Multi-Agent 相关

**#108**: feat: add ability to chat with different agents
- **状态**：✅ 已合并
- **作者**：thedadams
- **意义**：已经有人做了基础的 Multi-Agent 支持
- **你的机会**：**增强 Multi-Agent 功能**

**可做的延伸**：
- [ ] 实现 Manager-Worker 模式
- [ ] 实现 Agent 间任务分发
- [ ] 实现 Multi-Agent 协作工作流
- [ ] 添加 Multi-Agent 监控和调试

---

#### 2. MCP 协议相关

**#139**: feat: support mcp apps
- **状态**：✅ 已合并
- **意义**：支持 MCP Apps 扩展

**可做的延伸**：
- [ ] 完善未实现的 MCP Features（README 中标记 Partial）
- [ ] 优化 MCP Client 性能
- [ ] 添加 MCP Server 自动发现
- [ ] 实现 MCP 协议的高级特性

---

#### 3. Tool 调用相关

**#144**: feat: add tool output truncation to prevent context overflow
- **状态**：🔵 开放（最新 PR）
- **作者**：cjellick (CONTRIBUTOR)
- **功能**：工具输出截断，防止超出上下文长度

**可做的延伸**：
- [ ] 添加 Tool 输出缓存
- [ ] 实现 Tool 调用优化（批量、并行）
- [ ] 添加 Tool 执行超时控制
- [ ] 实现 Tool 结果聚合

---

### 📋 Open Issues（可以直接参与的）

基于分析，以下是**适合新人参与**的开放 Issues：

| Issue # | 标题 | 难度 | 适合你吗？ |
|---------|------|------|-----------|
| **#143** | Support custom apiBase | ⭐⭐ | ✅ 容易，API 配置 |
| **#76** | Fix: space in folder path | ⭐ | ✅ 简单，Bug 修复 |
| **#72** | Layout shift | ⭐⭐ | ⚠️ UI 相关 |
| **#139** | Support mcp apps | ⭐⭐⭐ | ✅ 已合并，可延伸 |
| **#144** | Tool output truncation | ⭐⭐⭐ | ✅ 进行中，可帮忙测试 |

**推荐**：
- **首选**：#143（custom apiBase）- 简单，API 经验
- **次选**：#76（space in folder path）- Bug 修复，容易上手
- **长期**：Multi-Agent 增强 - 展示系统设计能力

---

## 第二部分：Nanobot 能力评估

### 🤔 什么是"能力边界"？

**问题**：Nanobot 能做什么样的复杂功能？

**类比**：
- Nanobot = **操作系统内核**（提供基础能力）
- Agent = **应用程序**（使用内核能力）
- MCP Servers = **驱动程序**（提供具体工具）

**能力边界**：
```
┌─────────────────────────────────────────┐
│         Nanobot (MCP Host)               │
│                                         │
│   提供：                                │
│   ✅ Agent 配置与运行                    │
│   ✅ MCP Server 管理                    │
│   ✅ LLM 调用（OpenAI/Anthropic）        │
│   ✅ Tool 调用路由                      │
│   ✅ Session 管理                        │
│   ✅ 审批网关                           │
│   ✅ UI 框架                             │
│                                         │
│   不提供：                              │
│   ❌ 具体的业务逻辑                      │
│   ❌ 领域知识（代码、医学、法律）        │
│   ❌ 外部 API（除了 MCP）               │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│      你设计的 Agent (应用层)             │
│                                         │
│   提供：                                │
│   ✅ 业务逻辑                            │
│   ✅ 领域知识                            │
│   ✅ 任务编排                            │
│   ✅ 错误处理                            │
└─────────────────────────────────────────┘
```

---

### 🎯 Nanobot 能做什么复杂功能？

#### ✅ 能做的复杂功能（推荐）

##### 1️⃣ 代码审查 Agent（Multi-Agent）

**复杂度**：⭐⭐⭐⭐⭐

**架构**：
```
Manager Agent (代码审查经理)
  ├─ Worker Agent 1 (代码风格检查)
  │   └─ Tools: ESLint, Prettier
  ├─ Worker Agent 2 (安全扫描)
  │   └─ Tools: Semgrep, CodeQL
  ├─ Worker Agent 3 (性能分析)
  │   └─ Tools: Lighthouse, Benchmark
  └─ Worker Agent 4 (测试生成)
      └─ Tools: Jest, Playwright
```

**实现**：
```yaml
# .nanobot/agents/code-review-manager/AGENT.md
---
name: Code Review Manager
model: claude-3-7-sonnet-latest
agents:
  - style_checker
  - security_scanner
  - performance_analyzer
  - test_generator
temperature: 0.3

You are a code review manager.

你的职责：
1. 接收代码审查请求
2. 将任务分解为 4 个子任务
3. 分配给对应的 Worker Agent
4. 聚合所有 Worker 的结果
5. 生成综合审查报告

工作流程：
- style_checker: 检查代码风格
- security_scanner: 扫描安全问题
- performance_analyzer: 分析性能
- test_generator: 生成测试用例
```

**Nanobot 提供**：
- ✅ Agent 配置（YAML + Markdown）
- ✅ Worker Agent 调用
- ✅ Tool 路由（ESLint, Semgrep 等通过 MCP）
- ✅ 结果聚合

**你提供**：
- ✅ 业务逻辑（如何分解任务）
- ✅ Prompt Engineering（如何让 Worker 更准确）
- ✅ 结果整合（如何合并 4 个 Worker 的输出）

**面试价值**：⭐⭐⭐⭐⭐
> "我设计了一个 Multi-Agent 代码审查系统，使用 Manager-Worker 模式，支持 4 个专业 Worker 并行分析代码，包括风格检查、安全扫描、性能分析、测试生成。使用 Nanobot 实现 Agent 间通信和任务分发。"

---

##### 2️⃣ 自动化研究 Agent（复杂工具编排）

**复杂度**：⭐⭐⭐⭐

**场景**：自动完成学术研究流程

**架构**：
```
Research Agent
  ├─ Skill: Literature Search
  │   └─ Tools: arxiv_api, google_scholar_api
  ├─ Skill: Data Collection
  │   └─ Tools: kaggle_download, web_scraper
  ├─ Skill: Data Analysis
  │   └─ Tools: pandas, jupyter
  └─ Skill: Report Generation
      └─ Tools: latex_compiler, pandoc
```

**实现**：
```yaml
# .nanobot/agents/researcher/AGENT.md
---
name: Research Assistant
model: claude-3-7-sonnet-latest
skills:
  - literature_search
  - data_collection
  - data_analysis
  - report_generation

tools:
  - arxiv_api
  - google_scholar
  - kaggle_download
  - jupyter
---

You are a research assistant.

你可以：
1. 搜索相关论文
2. 下载实验数据
3. 进行数据分析
4. 生成研究报告
```

**Nanobot 提供**：
- ✅ Skill 调用链
- ✅ Tool 编排
- ✅ 执行追踪

**你提供**：
- ✅ 研究 Prompt
- ✅ 数据处理逻辑
- ✅ 报告模板

---

##### 3️⃣ 智能客服系统（真实场景）

**复杂度**：⭐⭐⭐⭐⭐

**场景**：企业级智能客服，支持多个渠道

**架构**：
```
Customer Service Agent
  ├─ MCP Server 1: Shopify API (商品查询)
  ├─ MCP Server 2: Stripe API (退款处理)
  ├─ MCP Server 3: Zendesk API (工单管理)
  └─ MCP Server 4: Email API (邮件通知)
```

**实现**：
```yaml
# .nanobot/agents/customer-service/AGENT.md
---
name: Customer Service Bot
model: claude-3-7-sonnet-latest
mcpServers:
  - shopify_store
  - stripe_payment
  - zendesk_tickets
  - email_notification

tools:
  - shopify_store/search_products
  - shopify_store/get_order_status
  - stripe_payment/refund
  - zendesk_tickets/create_ticket
  - email_notification/send

---

You are a customer service agent.

你可以：
- 查询商品信息
- 处理退款请求
- 创建工单
- 发送邮件通知
```

**Nanobot 提供**：
- ✅ 多 MCP Server 集成
- ✅ Tool 调用编排
- ✅ 审批机制（敏感操作需要审批）

**你提供**：
- ✅ 客服对话逻辑
- ✅ 业务规则（退款政策）
- ✅ 多语言支持

---

### ❌ 不能做的复杂功能（超出边界）

#### 1️⃣ 实时系统（如游戏、高频交易）

**原因**：
- Nanobot 基于 HTTP/WebSocket（请求-响应）
- 不支持实时性要求 < 100ms 的场景

**替代方案**：
- Nanobot + 独立的实时服务
- 或选择其他框架

---

#### 2️⃣ 大规模分布式计算

**原因**：
- Nanobot 是**单机** MCP Host
- 没有分布式计算能力

**替代方案**：
- 使用 Nanobot + Kubernetes（每个 Pod 运行一个 Nanobot）
- 或使用 Ray、Dask 等分布式框架

---

#### 3️⃣ 深度学习训练

**原因**：
- Nanobot 没有 GPU 支持
- 没有 Model Training 能力

**替代方案**：
- Nanobot + 外部训练服务（调用训练 API）
- 或使用 PyTorch、TensorFlow 直接训练

---

### 🎯 Nanobot 最佳适用场景

**最适合**：
1. ✅ **Multi-Agent 协作系统**（Manager-Worker）
2. ✅ **工具编排复杂任务**（10+ 步骤）
3. ✅ **人机协作流程**（审批、修正、迭代）
4. ✅ **知识密集型任务**（需要 LLM + 工具）
5. ✅ **可解释性要求高**（需要追踪执行过程）

**不适合**：
1. ❌ 实时系统（游戏、高频交易）
2. ❌ 大规模分布式计算
3. ❌ 深度学习训练
4. ❌ 极低延迟要求（< 100ms）

---

## 第三部分：Vibe Coding 实战示例

### 🤖 使用 AI 辅助 Go 开发（完整流程）

#### 示例：添加一个 `git_commit` 工具

**目标**：在 Nanobot 中添加 Git Commit 工具

---

### Step 1：让 AI 生成代码（你只需要写 Prompt）

**Prompt 模板**：

```yaml
role: Go 语言后端开发专家，精通 Nanobot 项目架构

task: |
  在 Nanobot 中添加一个新的内置工具：git_commit

context: |
  项目信息：
  - 仓库：https://github.com/nanobot-ai/nanobot
  - 分支：main
  - 当前工具注册位置：pkg/tools/service.go

  参考现有工具实现：
  - Bash 工具：pkg/tools/service.go 中的 BashTool
  - Read 工具：pkg/tools/service.go 中的 ReadTool

requirements:
  1. 添加 GitCommitTool 结构体
  2. 实现 Execute 方法
  3. 注册到工具注册表
  4. 添加单元测试
  5. 遵循项目代码风格

tool_specification:
  name: git_commit
  description: "执行 git commit 命令"
  input_schema:
    type: object
    properties:
      message:
        type: string
        description: "Commit message"
      files:
        type: array
        items:
          type: string
        description: "Files to commit (optional, commit all if not specified)"
    required:
      - message

output_format: |
  请提供：
  1. 修改的代码（显示完整的函数）
  2. 需要修改的文件列表
  3. 测试方法
  4. 可能的问题和解决方案

reference_code: |
  参考现有 Bash 工具的实现：

  type BashTool struct {
      Name         string
      Description string
  }

  func (t *BashTool) Definition() schema.Tool {
      return schema.Tool{
          Name:         t.Name,
          Description:  t.Description,
          InputSchema:  map[string]any{
              "type": "object",
              "properties": map[string]any{
                  "command": map[string]any{
                      "type":        "string",
                      "description": "Command to execute",
                  },
              },
          },
      }
  }

  func (t *BashTool) Execute(ctx context.Context, input map[string]any) (any, error) {
      // 执行逻辑
  }
```

**让 AI 生成代码** → AI 会给你完整的 Go 代码

---

### Step 2：理解代码（你需要做的）

**AI 生成的代码**（示例）：

```go
// GitCommitTool represents the git_commit tool
type GitCommitTool struct {
    Name         string
    Description string
}

// Definition returns the tool schema
func (t *GitCommitTool) Definition() schema.Tool {
    return schema.Tool{
        Name:         t.Name,
        Description:  t.Description,
        InputSchema: map[string]any{
            "type": "object",
            "properties": map[string]any{
                "message": map[string]any{
                    "type":        "string",
                    "description": "Commit message",
                },
                "files": map[string]any{
                    "type": "array",
                    "items": map[string]any{
                        "type": "string",
                    },
                    "description": "Files to commit",
                },
            },
            "required": []string{"message"},
        },
    }
}

// Execute executes git commit
func (t *GitCommitTool) Execute(ctx context.Context, input map[string]any) (any, error) {
    message := input["message"].(string)

    // 构建 git commit 命令
    args := []string{"commit", "-m", message}

    if files, ok := input["files"].([]string); ok && len(files) > 0 {
        args = append(args, files...)
    } else {
        args = append(args, "-a")
    }

    // 执行命令
    cmd := exec.Command("git", args...)
    output, err := cmd.CombinedOutput()

    if err != nil {
        return nil, fmt.Errorf("git commit failed: %w: %s", err, string(output))
    }

    return map[string]any{
        "success": true,
        "output":  string(output),
    }, nil
}
```

**你理解的**：
1. **Definition()**：定义工具的 Schema（JSON Schema）
2. **Execute()**：执行 Git Commit 命令
3. **错误处理**：如果失败，返回错误

**你不需要**：
- ❌ 背诵 Go 语法
- ❌ 记住 exec.Command 的参数
- ❌ 记住错误处理模式

**但你必须**：
- ✅ 理解这个工具做了什么
- ✅ 理解输入输出格式
- ✅ 能解释给面试官

---

### Step 3：测试代码（让 AI 帮忙）

**Prompt**：

```yaml
task: |
  为上面的 GitCommitTool 编写单元测试

requirements:
  - 使用 Go 的 testing 包
  - 测试成功场景
  - 测试失败场景（没有 Git 仓库）
  - 测试指定文件 vs 全部文件

output:
  1. 测试文件路径（如 pkg/tools/git_commit_test.go）
  2. 完整的测试代码
  3. 运行测试的方法
```

---

### Step 4：提交 PR

**PR 描述**：

```markdown
## feat: add git_commit tool

### 概述
添加了 `git_commit` 内置工具，允许 Agent 执行 Git Commit 命令。

### 主要变更
- 添加 `GitCommitTool` 结构体
- 实现 `Execute` 方法
- 添加单元测试
- 注册到工具注册表

### 测试
- [x] 单元测试通过
- [x] 手动测试：成功提交代码
- [x] 边界情况测试：无 Git 仓库时的错误处理

### 使用示例

```yaml
# .nanobot/agents/developer/AGENT.md
---
name: Developer Agent
model: claude-3-5-sonnet-20241022
tools:
  - git_commit
---

你可以使用 git_commit 工具提交代码。
```

### 设计考虑
- 支持 commit message 参数
- 支持指定文件或全部文件
- 错误处理：无 Git 仓库时返回友好错误

---

## 第四部分：立即行动（本周）

### 📅 Week 1 行动计划（2.10-2.16）

#### Day 1-2：搜索与分析

**Day 1**：
- [ ] 搜索 Issues：`multi-agent`
- [ ] 搜索 PR：`multi-agent`
- [ ] 阅读 README 中的 Roadmap
- [ ] 阅读 DESIGN.md 中的 Open Questions

**Day 2**：
- [ ] 总结搜索结果
- [ ] 识别空缺（没有人做的功能）
- [ ] 选择贡献方向

---

#### Day 3-4：发起讨论

**Day 3**：
- [ ] 在 GitHub Discussions 创建提案
  ```markdown
  Title: Proposal: Multi-Agent Architecture for Nanobot

  内容：使用模板（见上一份文档）
  ```

**Day 4**：
- [ ] 等待反馈（通常 2-7 天）
- [ ] 如果有人回复，积极回应

---

#### Day 5-7：本地熟悉

**Day 5**：
- [ ] 本地运行 Nanobot
  ```bash
  cd d:/AI/2026/LearningSystem/nanobot
  make
  nanobot run ./examples/shopping.yaml
  ```

**Day 6-7**：
- [ ] 尝试修改示例 Agent
- [ ] 理解配置格式
- [ ] 理解 MCP 工作原理

---

### 🎯 关键检查点

**每天问自己**：
- [ ] 我是否搜索过 Issues？
- [ ] 我是否查看过 PR？
- [ ] 我是否和社区讨论过？

**如果都是 YES**：
→ ✅ 继续下一步

**如果有任何 NO**：
→ ❌ 停下来，回到搜索步骤

---

## 总结

### 1. Nanobot 能力边界

**能做的复杂功能**：
- ✅ Multi-Agent 协作系统（Manager-Worker）
- ✅ 工具编排复杂任务（10+ 步骤）
- ✅ 人机协作流程
- ✅ 知识密集型任务

**不能做的**：
- ❌ 实时系统（< 100ms 延迟）
- ❌ 大规模分布式计算
- ❌ 深度学习训练

**最佳场景**：**Multi-Agent 系统** ⭐⭐⭐⭐⭐

---

### 2. Vibe Coding 可行性

**完全可行**！

**你提供**：
- ✅ Agent 理论（你有）
- ✅ 系统设计（可以学）
- ✅ 理解代码（AI 帮忙）

**AI 提供**：
- ✅ Go 代码
- ✅ 测试代码
- ✅ 调试建议

---

### 3. 下一步

**本周开始**：
1. 搜索 Issues
2. 发起 Discussions
3. 本地运行 Nanobot

**你想先做哪一个**？
- A. 我帮你搜索并分析 Issues
- B. 我帮你撰写 Discussions 提案
- C. 我帮你搭建本地环境
- D. 我教你 vibe coding 第一个示例

选择一个，我们开始！
