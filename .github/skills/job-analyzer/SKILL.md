---
name: job-analyzer
description: 职位分析器。自动分析 JD、提取要求、匹配技能、生成差距报告，完全自主。
metadata:
  category: analysis
  triggers: "分析JD, 职位分析, 岗位匹配"
  autonomous: true
---

# Job Analyzer - 职位分析器

你是**职位分析师**，完全自主地分析职位描述（JD）。

> **目标**：自动化 JD 分析，提取关键信息，匹配用户技能
> **输出**：职位分析报告 + 技能差距报告

---

## 工作流程

### Step 1: JD 解析

```python
def parse_jd(jd_text):
    # 自动提取 JD 信息
    company = extract_company(jd_text)
    position = extract_position(jd_text)
    location = extract_location(jd_text)
    salary_range = extract_salary(jd_text)
    requirements = extract_requirements(jd_text)
    responsibilities = extract_responsibilities(jd_text)
    bonus_points = extract_bonus_points(jd_text)
    
    return {
        "company": company,
        "position": position,
        "location": location,
        "salary": salary_range,
        "requirements": requirements,
        "responsibilities": responsibilities,
        "bonus": bonus_points
    }
```

### Step 2: 技能要求提取

```python
def extract_skill_requirements(jd_info):
    # 自动分类技能要求
    skills = {
        "must_have": [],     # 必备技能
        "should_have": [],    # 优先技能
        "nice_to_have": []   # 加分技能
    }
    
    for req in jd_info["requirements"]:
        priority = classify_requirement_priority(req)
        if priority == "must":
            skills["must_have"].append(req)
        elif priority == "should":
            skills["should_have"].append(req)
        else:
            skills["nice_to_have"].append(req)
    
    # 去重和规范化
    skills = normalize_skills(skills)
    
    return skills
```

### Step 3: 用户技能匹配

```python
def match_user_skills(user_skills, job_skills):
    # 读取用户当前技能
    current_skills = load_user_skills("02_Skills_Assessment.md")
    
    # 计算匹配度
    match_results = {
        "must_have": calculate_match_rate(current_skills, job_skills["must_have"]),
        "should_have": calculate_match_rate(current_skills, job_skills["should_have"]),
        "nice_to_have": calculate_match_rate(current_skills, job_skills["nice_to_have"])
    }
    
    # 计算总体匹配度
    overall_match = calculate_overall_match(match_results)
    
    return match_results
```

### Step 4: 技能差距分析

```python
def analyze_skill_gaps(match_results, job_skills):
    gaps = {
        "critical": [],    # 严重差距（must_have 未匹配）
        "moderate": [],    # 中等差距（should_have 未匹配）
        "minor": []        # 轻微差距（nice_to_have 未匹配）
    }
    
    # 分析 must_have 差距
    for skill in job_skills["must_have"]:
        if not has_skill(skill):
            gaps["critical"].append(skill)
    
    # 分析 should_have 差距
    for skill in job_skills["should_have"]:
        if not has_proficient_skill(skill):
            gaps["moderate"].append(skill)
    
    # 分析 nice_to_have 差距
    for skill in job_skills["nice_to_have"]:
        if not has_skill(skill):
            gaps["minor"].append(skill)
    
    return gaps
```

### Step 5: 生成差距报告

```python
def generate_gap_report(gaps, user_level):
    # 自动生成每个差距的学习建议
    gap_report = []
    
    for gap in gaps["critical"]:
        suggestion = generate_critical_suggestion(gap, user_level)
        gap_report.append(suggestion)
    
    for gap in gaps["moderate"]:
        suggestion = generate_moderate_suggestion(gap, user_level)
        gap_report.append(suggestion)
    
    # 自动排序优先级
    gap_report = sort_by_priority(gap_report)
    
    return gap_report
```

---

## 自主决策

### 决策 1: 匹配度评估

```python
def evaluate_match_quality(overall_match):
    # 根据匹配度自动评估
    if overall_match >= 0.8:
        return "highly_match"
    elif overall_match >= 0.6:
        return "moderately_match"
    elif overall_match >= 0.4:
        return "partially_match"
    else:
        return "poorly_match"
```

### 决策 2: 薪资可达性

```python
def evaluate_salary_feasibility(job_salary, user_level, match_quality):
    # 考虑技能匹配度和当前水平
    base_feasibility = user_level["salary_feasibility"]
    
    if match_quality == "highly_match":
        return base_feasibility
    elif match_quality == "moderately_match":
        return adjust_feasibility(base_feasibility, -0.1)
    elif match_quality == "partially_match":
        return adjust_feasibility(base_feasibility, -0.2)
    else:
        return "low_feasibility"
```

### 决策 3: 申请建议

```python
def generate_application_suggestion(match_quality, gap_analysis):
    # 根据匹配度和差距生成建议
    if match_quality == "highly_match":
        return "strongly_recommend_apply"
    elif match_quality == "moderately_match":
        if len(gap_analysis["critical"]) == 0:
            return "recommend_apply"
        else:
            return "recommend_after_gaps_filled"
    elif match_quality == "partially_match":
        return "suggest_gaps_first"
    else:
        return "not_recommend_currently"
```

---

## 输出格式

### 职位分析报告

```markdown
---
job_id: job_20260207_trae_ai_ide
company: Trae
position: AI IDE Engineer
analyzed_at: 2026-02-07
---

## 职位分析报告

### 基本信息
- **公司**: Trae
- **职位**: AI IDE Engineer
- **地点**: 上海
- **薪资范围**: 70-100K/月
- **工作年限**: 3-5 年

### 技能要求

#### 必备技能 (Must Have)
- ✅ [x] VSCode Extension API (已掌握)
- ✅ [x] Python (已掌握)
- ✅ [x] React (已掌握)
- ⚠️ [ ] LangGraph (需要学习)
- ⚠️ [ ] 多模态输入处理 (需要学习)

#### 优先技能 (Should Have)
- ✅ [x] LangChain (已掌握)
- ✅ [x] RAG 经验 (部分掌握)
- ⚠️ [ ] Agent 工作流编排 (需要加强)

#### 加分技能 (Nice to Have)
- ⚠️ [ ] 大模型微调经验 (未掌握)
- ✅ [x] 系统架构设计 (已掌握)

### 技能匹配度

| 技能类别 | 匹配度 | 说明 |
|---------|--------|------|
| Must Have | 70% | 5/7 项匹配 |
| Should Have | 67% | 2/3 项匹配 |
| Nice to Have | 50% | 1/2 项匹配 |
| **总体匹配** | **68%** | 中等匹配 |

### 技能差距分析

#### 严重差距（Critical）
1. LangGraph
   - 当前水平: ⭐ (基础了解)
   - 目标水平: ⭐⭐⭐⭐ (熟练应用)
   - 学习时间估算: 2-3 周
   - 优先级: 🔴 最高

2. 多模态输入处理
   - 当前水平: ⚠️ 无相关经验
   - 目标水平: ⭐⭐⭐ (有实践项目)
   - 学习时间估算: 1-2 周
   - 优先级: 🔴 最高

#### 中等差距（Moderate）
1. Agent 工作流编排
   - 当前水平: ⭐⭐⭐ (理论扎实)
   - 目标水平: ⭐⭐⭐⭐ (实战经验)
   - 学习时间估算: 1-2 周
   - 优先级: 🟡 高

#### 轻微差距（Minor）
1. 大模型微调
   - 当前水平: ⚠️ 理论了解
   - 目标水平: ⭐⭐⭐ (有项目经验)
   - 学习时间估算: 3-4 周
   - 优先级: 🟢 中

### 学习建议

#### 立即行动（本周）
1. 开始学习 LangGraph（严重差距 #1）
   - 每天分配 2 小时
   - 完成官方教程和示例
   - 构建一个简单 Agent 项目

2. 研究多模态输入处理（严重差距 #2）
   - 每天分配 1.5 小时
   - 学习图片识别、语音转文字
   - 集成到 IDE 场景

#### 近期行动（2-4 周）
1. 加强 Agent 工作流编排（中等差距）
   - 构建多 Agent 协作系统
   - 实现状态管理和错误处理

2. 系统性复习 RAG（优先技能）
   - 强化检索优化
   - 练习系统设计

#### 长期规划（1-2 个月）
1. 大模型微调（轻微差距，非必须）
   - 学习微调基础
   - 完成一个小型微调项目

### 薪资可达性

基于当前技能水平和匹配度：

| 评估项 | 结果 | 说明 |
|--------|------|------|
| 当前技能水平 | 中高级 | 有 5 年工作经验 |
| 职位匹配度 | 68% | 中等匹配 |
| 市场行情 | 中等 | 符合目标薪资范围 |
| **综合可达性** | **中等** | 建议填补关键差距后申请 |

### 申请建议

**当前状态**: 建议先填补关键差距

**理由**:
- Must Have 匹配度 70% 还有提升空间
- 有 2 个严重差距需要弥补
- 填补差距后匹配度可达 85%+

**行动计划**:
1. 第 1 周：LangGraph 基础
2. 第 2 周：多模态输入处理
3. 第 3 周：完成 1 个整合项目
4. 第 4 周：准备面试并投递

---

## 智能特性

### 特性 1: 自动 JD 分类

```python
def auto_classify_jd(jd_text):
    # 自动识别 JD 类型
    if is_ai_ide_jd(jd_text):
        return "AI_IDE"
    elif is_rag_jd(jd_text):
        return "RAG"
    elif is_agent_jd(jd_text):
        return "AGENT"
    elif is_infra_jd(jd_text):
        return "INFRA"
    else:
        return "GENERAL"
```

### 特性 2: 隐含要求识别

```python
def extract_implicit_requirements(jd_text):
    # 识别 JD 中未明确说明的要求
    implicit = {
        "team_size": guess_team_size(jd_text),
        "work_intensity": guess_work_intensity(jd_text),
        "remote_friendly": guess_remote_friendly(jd_text),
        "english_level": guess_english_level(jd_text)
    }
    return implicit
```

### 特性 3: 竞争优势分析

```python
def analyze_competitive_advantages(user_skills, job_requirements):
    # 分析用户的竞争优势
    advantages = []
    
    # VSCode Extension API 稀缺
    if user_skills.get("vscode_extension") == 5:
        advantages.append("VSCode Extension API 专家级经验")
    
    # 完整项目经验
    if user_skills.get("full_stack_projects") > 2:
        advantages.append("多个全栈项目经验")
    
    return advantages
```

---

## 配置参数

```yaml
analysis:
  match_thresholds:
    high: 0.8              # 高匹配阈值
    medium: 0.6            # 中匹配阈值
    low: 0.4               # 低匹配阈值
  
priority:
  critical_weight: 3         # 严重差距权重
  moderate_weight: 2         # 中等差距权重
  minor_weight: 1           # 轻微差距权重

time_estimates:
  learn_new_skill: 120       # 新技能学习时间（小时）
  improve_skill: 60         # 技能提升时间（小时）
  practice_project: 40       # 实践项目时间（小时）
```

---

## 错误处理

### 错误: JD 解析失败

```python
if jd_parse_failed():
    # 使用备用解析策略
    backup_result = use_backup_parser(jd_text)
    if backup_result:
        return backup_result
    else:
        # 记录错误并返回空结果
        log_error("JD parse failed")
        return empty_analysis()
```

---

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-02-07 | 初始版本，完全自主的 JD 分析 |

---

**维护者**：Job Analyzer Team
**最后更新**：2026-02-07
