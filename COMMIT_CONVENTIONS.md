# Git Commit 规范

> **最后更新**：2026-01-31
> **目的**：规范Git提交记录，便于追溯和协作

---

## Commit Message 格式

```
[角色] 简短描述（50字以内）

详细说明（可选）

- 变更点1
- 变更点2
```

## 角色标识

| 标识 | 对应Claude | 说明 |
|------|-----------|------|
| [General] | 通用助手 | 生成汇总、处理杂事 |
| [Assessor] | 测试评估 | 技术测试、技能评估 |
| [Teacher] | 教学 | 教学记录、答疑 |
| [Researcher] | 研究 | JD收集、市场分析 |
| [Planner] | 计划协调 | 进度跟踪、计划调整 |
| [System] | 系统配置 | 文件结构调整、配置更新 |

## Commit 示例

### 示例1：测试评估Claude
```bash
[Assessor] 完成Python技能测试

- 测试Python基础：闭包、装饰器、生成器
- 发现严重遗忘（⭐⭐ → ⭐）
- 更新02_Skills_Assessment.md
- 更新05_Skills_Gap_Analysis.md
- 更新00_Technical_Skills_Test_Record.md
```

### 示例2：教学Claude
```bash
[Teacher] 添加Python闭包教学记录

- 教学内容：闭包原理和应用
- 学员理解程度：基本掌握
- 下次教学重点：装饰器
- 保存教学记录：conversations/teacher/20260201_python_closure.md
```

### 示例3：研究Claude
```bash
[Researcher] 更新本周JD分析

- 收集5个新岗位
- 更新03_Market_Research_JD_Analysis.md
- 更新04_Target_Positions_Analysis.md
- 分析匹配度变化
```

### 示例4：计划协调Claude
```bash
[Planner] 根据测试结果调整学习计划

- Python最高优先级系统重学
- 调整08_Action_Plan_2026_H1.md
- 更新09_Progress_Tracker.md
- 识别风险：Python学习时间可能不足
```

### 示例5：通用助手
```bash
[General] 生成第4周汇总

- 汇总本周工作成果
- 更新conversations/summaries/00_weekly_summary.md
- 更新学习进度汇总
- 更新市场动态汇总
```

### 示例6：系统配置
```bash
[System] 创建5个Claude角色配置

- 创建system_prompts/目录
- 创建5个系统提示词文件
- 创建VSCode工作区配置
- 创建conversations/目录结构
```

---

## Commit 频率建议

| Claude | Commit频率 | 时机 |
|--------|-----------|------|
| 通用助手 | 每周1次 | 生成汇总后 |
| 测试评估 | 每次测试后 | 完成技能测试后 |
| 教学 | 每次教学后 | 完成一次教学后 |
| 研究 | 每周1次 | 完成JD收集后 |
| 计划协调 | 每周1次 | 更新进度后 |
| 系统配置 | 按需 | 配置变更后 |

---

## 重要节点：打Tag

### Tag格式

```
格式：vYYYY-MM-描述
示例：
- v2026-01-project-start（项目启动）
- v2026-01-monthly-test（1月测试完成）
- v2026-02-python-recovered（Python恢复）
- v2026-03-rag-production（RAG生产级）
- v2026-04-llm-app-release（LLM应用发布）
```

### 建议打Tag的时机

1. **月度测试完成后**
   ```bash
   git tag -a v2026-01-monthly-test -m "1月测试完成

   - Python：⭐⭐⭐
   - Agent：⭐⭐⭐
   - PyTorch：⭐⭐⭐
   - 关键发现：Python严重遗忘，需要系统重学"
   ```

2. **重要里程碑**
   ```bash
   git tag -a v2026-02-python-recovered -m "Python恢复到⭐⭐⭐"
   git tag -a v2026-03-rag-production -m "RAG达到生产级"
   git tag -a v2026-04-llm-app-release -m "LLM应用发布"
   ```

3. **半年总结**
   ```bash
   git tag -a v2026-06-half-year-review -m "半年总结：完成跳槽目标"
   ```

### 查看Tag

```bash
# 列出所有tag
git tag

# 查看tag详情
git show v2026-01-monthly-test

# 查看某个时间点的代码
git checkout v2026-01-monthly-test
```

---

## 查看Commit历史

```bash
# 查看所有commit
git log --oneline

# 按角色过滤
git log --grep="Teacher" --oneline
git log --grep="Assessor" --oneline
git log --grep="Researcher" --oneline

# 查看某个时间范围的commit
git log --since="2026-01-01" --until="2026-01-31" --oneline

# 查看某个Claude的所有commit
git log --grep="Assessor" --pretty=format:"%h %s" --name-only

# 查看最近一周的commit
git log --since="1 week ago" --oneline
```

---

## Commit最佳实践

### ✅ 应该做的

1. **频繁提交**：每次完成一个小任务就commit
2. **清晰的描述**：commit message要说明白做了什么
3. **使用角色标识**：方便追溯哪个Claude做的修改
4. **重要节点打Tag**：标记里程碑

### ❌ 不应该做的

1. **不要一次提交太多**：拆分成多个逻辑相关的commit
2. **不要使用模糊的描述**：如"update"、"fix"
3. **不要提交敏感信息**：如密码、密钥等
4. **不要忘记写角色标识**：无法追溯是谁做的修改

---

## 常见问题

### Q1：如果忘记打角色标识怎么办？

```bash
# 修改最近的commit message
git commit --amend -m "[Assessor] 正确的commit message"

# 注意：不要修改已经push的commit
```

### Q2：如何撤销一次commit？

```bash
# 撤销commit，保留修改
git reset --soft HEAD~1

# 撤销commit，丢弃修改
git reset --hard HEAD~1

# 注意：不要撤销已经push的commit
```

### Q3：如何查看某个文件的修改历史？

```bash
# 查看某个文件的commit历史
git log --follow --oneline -- 02_Skills_Assessment.md

# 查看某个文件在某次commit中的修改
git show <commit-hash>:02_Skills_Assessment.md
```

---

**参考文档**：
- [CHANGELOG.md](CHANGELOG.md) - 查看项目变更历史
- [00_Technical_Skills_Test_Record.md](00_Technical_Skills_Test_Record.md) - 查看测试记录
