# Day 26-27: Assignment 4 - Data Processing - 面试题

> **题目难度**: ⭐⭐ ~ ⭐⭐⭐⭐
> **考察重点**: 文本处理、去重算法、质量评分、数据管道
> **建议时间**: 35分钟

---

## Part 1: 文本处理

### Q1: 数据清洗的主要步骤？⭐⭐

**参考答案**:

**主要步骤**:
1. **去除HTML标签**: 提取纯文本
2. **Unicode标准化**: 统一字符编码
3. **去除多余空白**: 连续空格、换行
4. **特殊字符处理**: 保留中英文和标点
5. **去除过短行**: 过滤噪声

**工具**:
- **trafilatura**: 生产环境推荐
- **BeautifulSoup**: 简单场景
- **unstructured**: 复杂布局

---

### Q2: 如何处理中文文本分词？⭐⭐⭐

**参考答案**:

**jieba分词**:
```python
import jieba

text = "机器学习是人工智能的重要分支"

# 精确模式（默认）
words = jieba.cut(text)

# 全模式（粒度更细）
words = jieba.cut(text, cut_all=True)

# 搜索引擎模式（适合搜索）
words = jieba.cut_for_search(text)
```

**选择建议**:
- NLP任务: 精确模式
- 搜索引擎: 搜索引擎模式
- 需要快速: 全模式

---

## Part 2: 去重算法

### Q3: MinHash LSH的原理？⭐⭐⭐⭐

**参考答案**:

**核心思想**: 用固定长度的签名表示文档，相似的文档有相似的签名

**步骤**:
1. **Shingling**: 将文档转换为连续k个词的集合
2. **MinHash**: 对每个词计算哈希，取最小值作为签名
3. **LSH**: 使用局部敏感哈希快速查找相似文档

**优势**:
- 速度: 比对对比快1000倍
- 可扩展: 适合大规模数据集
- 可调: 通过阈值调整相似度

**公式**:
```
Jaccard相似度 = |A∩B| / |A∪B|
```

---

### Q4: 精确去重vs模糊去重？⭐⭐⭐⭐

**参考答案**:

| 维度 | 精确去重 | 模糊去重（MinHash） |
|------|---------|------------------|
| **匹配** | 文本完全相同 | 语义相似 |
| **速度** | 快（O(n)） | 较慢（O(n log n)） |
| **准确性** | 100% | 90-95% |
| **适用** | 明显重复 | 改写、转载 |

**选择建议**:
- 同源数据: 精确去重
- 多源数据: MinHash LSH
- 先精确后模糊: 提高效率

---

### Q5: 如何调整MinHash阈值？⭐⭐⭐⭐

**参考答案**:

**阈值范围**: 0.7-0.85

**调整建议**:
```python
threshold = 0.7  # 宽松（更多重复）
threshold = 0.8  # 平衡（推荐）
threshold = 0.9  # 严格（更少重复）
```

**影响**:
- 阈值太高: 重复未去除
- 阈值太低: 误删（把不同文档判为重复）

**最佳实践**:
1. 从0.8开始
2. 在验证集上测试
3. 根据结果调整

---

## Part 3: 质量评分

### Q6: 质量评分的关键指标？⭐⭐⭐

**参考答案**:

**主要指标**:
1. **文本长度**: 50-5000字符（过短质量低）
2. **句子数量**: 2-50个句子（完整性）
3. **标点符号比例**: 1%-10%（合理范围）
4. **特殊词**: 广告词、有害词

**评分函数**:
```python
def quality_score(text):
    score = 0.0
    if 50 <= len(text) <= 5000:
        score += 0.3
    if 2 <= count_sentences(text) <= 50:
        score += 0.3
    if 0.01 <= punctuation_ratio(text) <= 0.1:
        score += 0.2
    if not contains_spam(text):
        score += 0.2
    return score
```

---

### Q7: 如何过滤有害内容？⭐⭐⭐

**参考答案**:

**方法1: 关键词过滤**
```python
harmful_keywords = ['暴力', '色情', '赌博']
if any(kw in text for kw in harmful_keywords):
    return False
```

**方法2: 正则表达式**
```python
# 匹配敏感模式
pattern = r'(暴力|色情).*内容'
if re.search(pattern, text):
    return False
```

**方法3: ML分类器**（生产环境）
```python
from transformers import pipeline

classifier = pipeline("text-classification", model="harmful-content-detector")
result = classifier(text)
if result['label'] == 'HARMFUL':
    return False
```

---

## Part 4: 数据管道

### Q8: 如何构建生产级数据管道？⭐⭐⭐⭐

**参考答案**:

**关键设计**:
1. **模块化**: 每个处理步骤独立
2. **可配置**: 通过参数调整处理逻辑
3. **可监控**: 记录每个步骤的统计信息
4. **容错**: 单个文档失败不影响整体
5. **增量处理**: 支持添加新数据

**示例**:
```python
class ProductionPipeline:
    def __init__(self, config):
        self.config = config
        self.stats = {}

    def process(self, raw_data):
        for step in ['clean', 'deduplicate', 'filter']:
            raw_data = self._run_step(raw_data, step)
            self._update_stats(step, raw_data)
        return raw_data
```

---

### Q9: 如何处理大规模数据集？⭐⭐⭐⭐⭐

**参考答案**:

**策略1: 批量处理**
```python
batch_size = 1000
for i in range(0, len(files), batch_size):
    batch = files[i:i+batch_size]
    process_batch(batch)
```

**策略2: 并行处理**
```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(process, f) for f in files]
    results = [f.result() for f in futures]
```

**策略3: 流式处理**
```python
def stream_process(file_iterator):
    for batch in file_iterator:
        yield process_batch(batch)
```

**策略4: 增量索引**
```python
lsh = IncrementalLSH()
for new_docs in stream_documents():
    lsh.add(new_docs)
```

---

### Q10: 如何评估数据处理质量？⭐⭐⭐⭐

**参考答案**:

**评估指标**:

1. **去重效果**:
   - 去重率 = (原始数量 - 去重后数量) / 原始数量
   - 典型值: 5-20%

2. **质量过滤**:
   - 保留率 = 过滤后数量 / 过滤前数量
   - 典型值: 60-80%

3. **最终质量**:
   - 人工抽检100个文档
   - 计算合格率

4. **性能**:
   - 处理速度: 文档/秒
   - 内存占用: 峰值内存

**验证方法**:
```python
# 1. 统计分析
print(f"原始: {n_original}个")
print(f"去重: {n_dedup}个 (去重率:{(1-n_dedup/n_original)*100:.1f}%)")
print(f"过滤: {n_filtered}个 (保留率:{n_filtered/n_dedup*100:.1f}%)")

# 2. 人工验证
import random
sample = random.choice(final_docs, 100)
quality = manual_evaluate(sample)
print(f"抽检质量: {quality['pass_rate']:.1%}")
```

---

## 总结

**必会题目** (面试高频):
- Q1: 数据清洗步骤
- Q3: MinHash LSH原理
- Q6: 质量评分指标
- Q8: 生产级数据管道设计

**加分题目** (深入理解):
- Q4: 去重策略选择
- Q7: 有害内容过滤
- Q9: 大规模数据处理
- Q10: 数据质量评估

**建议**:
1. 掌握完整数据处理流程
2. 理解各种去重算法的trade-off
3. 能够设计生产级数据管道
4. 关注数据质量而非数量

---

**下一步**: [Day 28: RAG进阶](../Day28_RAG_Advanced/README.md)
