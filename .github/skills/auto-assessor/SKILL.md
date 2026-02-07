---
name: auto-assessor
description: 自动评估器。自动评估学习效果、技能掌握度，生成评估报告。
metadata:
  category: assessment
  triggers: "评估, 测试, 检查, 评估"
  autonomous: true
---

# Auto-Assessor - 自动评估器

你是**自动评估器**，完全自主地评估学习效果和技能掌握度。

> **目标**：自动评估，生成准确报告
> **输入**：学习记录、练习结果
> **输出**：评估报告

---

## 工作流程

### Step 1: 收集评估数据

```python
def collect_assessment_data():
    # 读取学习记录
    learning_records = load_learning_records("progress/daily_progress.md")
    
    # 读取练习结果
    practice_results = load_practice_results("progress/practice_records.md")
    
    # 读取技能掌握度
    mastery_levels = load_mastery_levels("progress/mastery_levels.md")
    
    return {
        "learning": learning_records,
        "practice": practice_results,
        "mastery": mastery_levels
    }
```

### Step 2: 自动评估

```python
def auto_assess(data):
    assessments = []
    
    # 评估学习效率
    learning_efficiency = assess_learning_efficiency(data["learning"])
    assessments.append(learning_efficiency)
    
    # 评估技能掌握度
    skill_mastery = assess_skill_mastery(data["practice"], data["mastery"])
    assessments.append(skill_mastery)
    
    # 评估知识巩固度
    knowledge_retention = assess_knowledge_retention(data)
    assessments.append(knowledge_retention)
    
    return assessments
```

### Step 3: 生成报告

```python
def generate_assessment_report(assessments):
    report = {
        "date": current_date(),
        "overall_score": calculate_overall_score(assessments),
        "strengths": identify_strengths(assessments),
        "weaknesses": identify_weaknesses(assessments),
        "recommendations": generate_recommendations(assessments)
    }
    
    return report
```

---

## 自动评估规则

### 规则 1: 学习效率

```python
def assess_learning_efficiency(learning_records):
    # 计算实际用时 / 计划用时
    actual_time = sum(learning_records["time_spent"])
    planned_time = sum(learning_records["allocated_time"])
    
    efficiency = actual_time / planned_time
    
    if efficiency > 1.1:
        return {"metric": "learning_efficiency", "score": 0.9, "status": "overtime"}
    elif efficiency > 0.9:
        return {"metric": "learning_efficiency", "score": 0.95, "status": "efficient"}
    elif efficiency > 0.7:
        return {"metric": "learning_efficiency", "score": 0.8, "status": "normal"}
    else:
        return {"metric": "learning_efficiency", "score": 0.6, "status": "low_efficiency"}
```

### 规则 2: 技能掌握度

```python
def assess_skill_mastery(practice_results, mastery_levels):
    for skill in mastery_levels:
        # 计算近期准确率
        recent_accuracy = calculate_recent_accuracy(skill, practice_results, days=7)
        
        # 计算掌握度趋势
        mastery_trend = calculate_trend(skill["history"])
        
        # 自动更新掌握度
        new_mastery = update_mastery_level(skill["current_level"], recent_accuracy, mastery_trend)
        
        skill["new_level"] = new_mastery
        skill["accuracy"] = recent_accuracy
        skill["trend"] = mastery_trend
    
    return mastery_levels
```

### 规则 3: 薄弱点识别

```python
def identify_weaknesses(assessments):
    weaknesses = []
    
    # 识别准确率低的技能
    for assessment in assessments:
        if assessment["metric"] == "skill_mastery" and assessment["score"] < 0.6:
            weaknesses.append({
                "skill": assessment["skill"],
                "type": "low_accuracy",
                "priority": "high" if assessment["score"] < 0.5 else "medium"
            })
    
    # 识别学习效率低的内容
    if assessments[0]["status"] == "low_efficiency":
        weaknesses.append({
            "type": "learning_efficiency",
            "priority": "medium"
        })
    
    return weaknesses
```

---

## 配置参数

```yaml
assessment:
  efficiency:
    high_threshold: 0.9        # 高效率阈值
    low_threshold: 0.7         # 低效率阈值
  
  mastery:
    min_samples: 5              # 最小样本数
    trend_window: 7             # 趋势窗口（天）
    improvement_threshold: 0.1  # 提升阈值（10%）
  
  auto_update:
    update_mastery: true        # 自动更新掌握度
    generate_report: true         # 自动生成报告
    save_to_progress: true       # 保存到进度文件
```

---

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-02-07 | 初始版本，完全自主的自动评估 |

---

**维护者**: Auto-Assessor Team
**最后更新**: 2026-02-07
