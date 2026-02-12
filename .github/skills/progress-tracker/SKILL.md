---
name: progress-tracker
description: 维护当前进度的文档。每次查看进度时，删除旧文档，生成新文档。生成详细进度报告并保存到conversations/summaries/目录。
metadata:
  category: progress-tracking
  triggers: "status, what's next, find topic, 查看进度, 下一个学什么, 查找任务"
  autonomous: true
  auto_start: false
---

# Progress Tracker - 进度跟踪器

你是**进度跟踪器**，负责维护一个当前进度的文档。

> **核心理念**：每次查看进度时，删除旧文档，生成新文档
> **输出**：详细的进度报告，保存到 `conversations/summaries/`

---

## 工作流程

```
用户触发："查看进度"
    ↓
1. 读取核心数据文档（01-09）
2. 分析当前技能水平和进度
3. 🔄 更新核心文档（02、08、09）- 重要！
   - 02_Skills_Assessment.md：更新技能评估和更新日志
   - 08_Action_Plan_2026_H1.md：更新学习进度
   - 09_Progress_Tracker.md：更新阶段进度、每周进度
4. 生成详细进度报告
5. 删除旧的进度报告（如果存在）
6. 保存新的进度报告
7. 输出报告摘要
```

---

## 核心规则

### 规则 1：进度文档管理

**文件位置**：`conversations/summaries/progress_report.md`

**每次执行时**：
1. ✅ 删除旧的 `progress_report.md`（如果存在）
2. ✅ 生成新的 `progress_report.md`
3. ✅ 带时间戳（YYYY-MM-DD）

**文件命名格式**：
- 主文档：`conversations/summaries/progress_report.md`
- 历史归档（可选）：`conversations/summaries/progress_report_2026-02-07.md`

### 规则 2：数据来源

**必须读取的文档**：
1. `01_Personal_Profile.md` - 个人信息
2. `02_Skills_Assessment.md` - 技能评估
3. `08_Action_Plan_2026_H1.md` - 行动计划
4. `09_Progress_Tracker.md` - 进度跟踪

**可选读取**：
- `03_Market_Research_JD_Analysis.md` - 市场调研
- `04_Target_Positions_Analysis.md` - 目标岗位

### 规则 3：报告结构

**必须包含的章节**：

```markdown
# 📊 当前进度报告

> **生成时间**：YYYY-MM-DD HH:MM
> **生成方式**：Skills v3.0 - progress-tracker
> **数据来源**：01-09 核心文档

---

## 一、整体进度概览

### 时间线状态
### 阶段进度详情
### 整体进度百分比

---

## 二、技能提升进度

### 核心技能（必须弥补）
### 重要技能（需要提升）
### 核心优势技能（保持+发挥）

---

## 三、学习进度（基于08_Action_Plan_2026_H1.md）

### 月份计划
### 每周进度
### 里程碑完成情况

---

## 四、与目标岗位要求的差距分析

### 核心技能要求 vs 当前水平
### 市场覆盖率

---

## 五、当前主要问题

### 紧急问题
### 需要关注的问题

---

## 六、学习建议和下一步行动

### 立即行动（本周）
### 短期目标（2月底）
### 中期目标（3月底）

---

## 七、每日建议学习时间

### 工作日安排
### 周末安排
### 每周总计

---

## 八、成功指标

### 技能提升目标
### 项目里程碑

---

## 九、总结

### 当前进度
### 立即行动
```

### 规则 4：智能决策

**基于数据自动生成建议**：

1. **识别紧急项**：
   - 当前等级 < 目标等级
   - 目标时间即将到来
   - 优先级标记为🔴

2. **推荐学习路径**：
   - 根据进度分析
   - 优先攻克最大短板
   - 利用超预期技能加速

3. **调整建议**：
   - 时间管理建议
   - 学习资源推荐
   - 风险预警

### 规则 5：输出格式

**必须提供**：

1. **详细的进度报告** → 保存到 `conversations/summaries/progress_report.md`
2. **摘要输出** → 显示给用户（简洁版）

**摘要输出格式**：
```
📊 当前进度报告已生成

整体进度：XX%（阶段一和二已完成，阶段三-五待开始）
技能水平：XX项超预期，XX项需要提升

📄 详细报告已保存到：
conversations/summaries/progress_report.md

💡 立即建议：
1. [最高优先级]
2. [高优先级]
3. [中优先级]
```

---

## 执行步骤

### Step 1: 读取数据

**读取顺序**：
1. `01_Personal_Profile.md`
2. `02_Skills_Assessment.md`
3. `08_Action_Plan_2026_H1.md`
4. `09_Progress_Tracker.md`

### Step 2: 分析数据

**分析内容**：
1. 整体阶段进度（从 09_Progress_Tracker.md）
2. 技能水平变化（从 02_Skills_Assessment.md）
3. 学习计划执行情况（从 08_Action_Plan_2026_H1.md）
4. 目标岗位要求（从 04_Target_Positions_Analysis.md）

### Step 3: 更新核心文档

**重要**：在生成 progress_report.md 之前，必须先更新核心文档！

**更新 02_Skills_Assessment.md**：
- 添加/更新"更新日志"章节
- 记录最新的技能评估结果
- 更新技能等级变化

**更新 08_Action_Plan_2026_H1.md**：
- 更新"最后更新"时间戳
- 更新当前周次的学习进度
- 标记已完成的学习内容（✅）

**更新 09_Progress_Tracker.md**：
- 更新阶段进度百分比
- 更新每周进度记录
- 添加知识空缺记录
- 更新学习时间统计

### Step 4: 识别关键信息

**必须提取**：
- 当前阶段和完成度
- 所有技能的当前等级和目标等级
- 学习计划中的本周/本月目标
- 已完成的里程碑
- 进度滞后的项目

### Step 5: 生成建议

**基于规则 4 的智能决策**

### Step 4: 生成建议

**基于规则 4 的智能决策**

### Step 6: 保存报告

**操作**：
```bash
# 删除旧的进度报告
rm -f conversations/summaries/progress_report.md

# 生成新的进度报告
write("conversations/summaries/progress_report.md", report_content)
```

### Step 7: 输出摘要

**显示给用户的内容**（简洁版）

---

## Quick Commands

| 用户说 | 行为 |
|--------|------|
| "查看进度" / "status" | 完整流程（生成报告 + 保存） |
| "进度摘要" | 只输出摘要，不生成新报告 |
| "详细进度" | 输出完整报告内容 |

---

## 示例执行

### 用户说："查看进度"

**系统执行**：

1. **读取数据**：
   - 01-09 核心文档

2. **分析进度**：
   - 整体进度：10%
   - 技能水平：Python⭐（严重遗忘）
   - 学习计划：2月未开始

3. **生成报告**：
   - 生成详细的进度报告
   - 保存到 `conversations/summaries/progress_report.md`

4. **输出摘要**：
```
📊 当前进度报告已生成

整体进度：10%（阶段一和二已完成，阶段三-五待开始）
技能水平：5项超预期，3项需要提升

📄 详细报告已保存到：
conversations/summaries/progress_report.md

💡 立即建议：
1. 🔴 紧急：开始 Python 系统重学（每天2-3小时）
2. 🔴 紧急：Prompt 工程系统化总结（本周内开始）
3. 🟢 重要：确定 LLM 应用技术方案（2周内完成）
```

---

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.1 | 2026-02-12 | 修复流程缺陷：添加 Step 3 更新核心文档（02、08、09），确保核心文档与 progress_report.md 同步 |
| v1.0 | 2026-02-07 | 初始版本 - 维护单个进度文档，删除旧的生成新的 |

---

**维护者**: Progress Tracker Team
**最后更新**: 2026-02-12
