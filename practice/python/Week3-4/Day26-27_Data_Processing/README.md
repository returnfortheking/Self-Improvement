# Day 26-27: Assignment 4 - Data Processing

> **学习目标**: 掌握数据处理管道，理解去重算法，实现生产级数据清洗流程
> **时间分配**: 6小时（理论2h + 实践4h）
> **难度**: ⭐⭐⭐⭐
> **重要性**: ⭐⭐⭐⭐⭐ (LLM应用的数据质量关键)
> **来源**: CS336 Assignment 4 - Data

---

## 📚 核心概念

### 1. 数据处理管道概述

**完整流程**:
```
原始数据 → 清洗 → 去重 → 分块 → 索引
   ↓         ↓      ↓      ↓      ↓
 HTML/文本  纯文本  唯一文档  chunks  向量DB
```

**为什么重要**:
- **数据质量**直接影响模型性能（Garbage In, Garbage Out）
- RAG系统的检索质量80%取决于数据处理
- 训练数据去重可以提升训练效率

---

### 2. 文本处理

#### 2.1 HTML转文本

**问题**: 网页数据包含大量噪声（HTML标签、CSS、JavaScript）

**工具对比**:
| 工具 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **BeautifulSoup** | 简单易用 | 需要手动处理 | 简单HTML |
| **trafilatura** | 自动提取正文 | 依赖重 | 生产环境 |
| **unstructured** | 支持复杂布局 | 慢 | 多模态文档 |

**最佳实践**:
```python
import trafilatura

def html_to_text(html_content):
    """使用trafilatura提取正文"""
    return trafilatura.extract(
        html_content,
        include_comments=False,
        include_tables=True,
        no_fallback=False
    )
```

#### 2.2 文本清洗

**清洗步骤**:
1. **去除多余空白**: 连续空格、换行符
2. **特殊字符处理**: Unicode标准化
3. **去除广告/导航栏**: 基于规则或ML
4. **语言检测**: 过滤非目标语言

```python
import re
import unicodedata

def clean_text(text):
    """文本清洗"""
    # 1. Unicode标准化
    text = unicodedata.normalize('NFKC', text)

    # 2. 去除多余空白
    text = re.sub(r'\s+', ' ', text)

    # 3. 去除特殊字符（保留中文、英文、标点）
    text = re.sub(r'[^\u4e00-\u9fff\u0041-\u007a\u0020-\u007e\uff0a\uff1f\uff08\uff09]', '', text)

    # 4. 去除过短行
    lines = [l for l in text.split('\n') if len(l.strip()) > 10]

    return '\n'.join(lines)
```

---

### 3. 去重算法

#### 3.1 精确去重

**方法**: 文本完全相同或MD5哈希相同

```python
import hashlib

def exact_deduplicate(documents):
    """精确去重"""
    seen = set()
    unique_docs = []

    for doc in documents:
        # 计算MD5哈希
        doc_hash = hashlib.md5(doc['text'].encode()).hexdigest()

        if doc_hash not in seen:
            seen.add(doc_hash)
            unique_docs.append(doc)

    return unique_docs
```

**适用场景**:
- ✅ 相同来源的数据
- ✅ 明显的重复内容
- ❌ 语义相似但文本不同

#### 3.2 MinHash LSH（局部敏感哈希）

**原理**:
1. 将文档转换为Shingle集合（连续k个词）
2. 计算MinHash签名（固定长度）
3. 使用LSH（局部敏感哈希）快速查找相似文档

**优势**:
- 比对比快1000倍
- 可调节相似度阈值
- 适合大规模数据集

**实现**:
```python
from datasketch import MinHashLSH, MinHash

def minhash_deduplicate(documents, threshold=0.8):
    """使用MinHash LSH去重"""

    # 1. 创建LSH索引
    lsh = MinHashLSH(threshold=threshold, num_perm=128)

    # 2. 为每个文档计算MinHash
    minhashes = {}
    for idx, doc in enumerate(documents):
        # 分词
        words = doc['text'].split()

        # 计算MinHash签名
        m = MinHash(num_perm=128)
        for word in words:
            m.update(word.encode())

        # 添加到索引
        lsh.insert(idx, m)
        minhashes[idx] = m

    # 3. 查找重复
    duplicates = set()
    for idx in range(len(documents)):
        # 查找相似文档
        result = lsh.query(minhashes[idx])
        for similar_idx in result:
            if similar_idx != idx and similar_idx not in duplicates:
                print(f"文档{idx}与文档{similar_idx}相似")
                duplicates.add(similar_idx)

    # 4. 返回去重后的文档
    return [doc for idx, doc in enumerate(documents) if idx not in duplicates]
```

#### 3.3 SimHash（相似度哈希）

**原理**:
1. 将文档转换为词向量
2. 计算simhash指纹（固定长度）
3. 比较汉明距离

**适用场景**:
- 需要快速去重
- 相似度阈值固定
- 中文文本

---

### 4. 内容过滤

#### 4.1 质量评分

**评分指标**:
1. **文本长度**: 过短的文档质量低
2. **句子数量**: 句子太少可能不完整
3. **特殊词比例**: 过多"点击这里"等广告词
4. **标点符号比例**: 过少标点可能是噪声

```python
def quality_score(text):
    """计算文档质量分数（0-1）"""
    score = 0.0

    # 1. 长度评分
    length = len(text)
    if 100 <= length <= 10000:
        score += 0.3
    elif length > 10000:
        score += 0.2

    # 2. 句子数量
    sentences = text.split('。')
    if 3 <= len(sentences) <= 100:
        score += 0.3

    # 3. 标点符号比例
    punctuation_ratio = sum(1 for c in text if c in '。，！？；：') / len(text)
    if 0.02 <= punctuation_ratio <= 0.15:
        score += 0.2

    # 4. 广告词检测
    spam_keywords = ['点击', '广告', '推广']
    if not any(kw in text for kw in spam_keywords):
        score += 0.2

    return score
```

#### 4.2 有害内容过滤

**方法**:
1. **关键词过滤**: 黑名单词汇
2. **正则表达式**: 匹配有害模式
3. **ML模型**: 使用分类器识别

```python
def filter_harmful_content(texts):
    """过滤有害内容"""
    harmful_keywords = [
        '暴力', '色情', '赌博',
        # ... 更多关键词
    ]

    filtered = []
    for text in texts:
        # 检查是否包含有害关键词
        is_harmful = any(kw in text for kw in harmful_keywords)

        if not is_harmful:
            filtered.append(text)

    return filtered
```

---

## 🔧 实战案例

### 案例1: 完整数据处理管道

```python
class DataPipeline:
    """完整的数据处理管道"""

    def __init__(self):
        self.documents = []

    def load_html(self, html_files):
        """加载HTML文件"""
        from trafilatura import extract

        for html_file in html_files:
            with open(html_file, 'r', encoding='utf-8') as f:
                html = f.read()

            # 提取正文
            text = extract(html)

            self.documents.append({
                'source': html_file,
                'text': text
            })

    def clean(self):
        """清洗文本"""
        for doc in self.documents:
            doc['text'] = clean_text(doc['text'])

    def deduplicate(self, method='minhash'):
        """去重"""
        if method == 'exact':
            self.documents = exact_deduplicate(self.documents)
        elif method == 'minhash':
            self.documents = minhash_deduplicate(self.documents, threshold=0.8)

    def filter_quality(self, min_score=0.6):
        """质量过滤"""
        filtered = []
        for doc in self.documents:
            score = quality_score(doc['text'])
            if score >= min_score:
                doc['quality_score'] = score
                filtered.append(doc)

        self.documents = filtered

    def process(self, html_files):
        """完整处理流程"""
        print(f"加载文档: {len(html_files)}个文件")
        self.load_html(html_files)

        print(f"清洗后: {len(self.documents)}个文档")
        self.clean()

        print(f"去重前: {len(self.documents)}个文档")
        self.deduplicate(method='minhash')
        print(f"去重后: {len(self.documents)}个文档")

        print(f"质量过滤前: {len(self.documents)}个文档")
        self.filter_quality(min_score=0.6)
        print(f"质量过滤后: {len(self.documents)}个文档")

        return self.documents

# 使用
pipeline = DataPipeline()
html_files = ['data/file1.html', 'data/file2.html']
documents = pipeline.process(html_files)
```

---

## 💡 实现技巧

### 1. 批量处理优化

```python
def batch_process(html_files, batch_size=100):
    """批量处理HTML文件"""
    results = []

    for i in range(0, len(html_files), batch_size):
        batch = html_files[i:i+batch_size]

        # 并行处理
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_html, f) for f in batch]
            results.extend([f.result() for f in futures])

    return results
```

### 2. 增量去重

```python
class IncrementalDeduplicator:
    """增量去重器"""

    def __init__(self, threshold=0.8):
        self.lsh = MinHashLSH(threshold=threshold, num_perm=128)
        self.processed_count = 0

    def add_documents(self, new_docs):
        """添加新文档"""
        for idx, doc in enumerate(new_docs):
            # 计算MinHash
            words = doc['text'].split()
            m = MinHash(num_perm=128)
            for word in words:
                m.update(word.encode())

            # 检查是否重复
            duplicates = self.lsh.query(m)

            if not duplicates:
                # 不重复，添加到索引
                global_idx = self.processed_count + idx
                self.lsh.insert(global_idx, m)
                self.processed_count += 1
                yield doc
```

---

## 🎯 学习检验

### 关键问题

1. **数据处理**:
   - 为什么数据清洗重要？
   - 如何选择合适的去重方法？
   - 质量评分的指标有哪些？

2. **去重算法**:
   - MinHash LSH的原理？
   - 精确去重vs模糊去重？
   - 如何调整相似度阈值？

3. **实践应用**:
   - 如何构建生产级数据管道？
   - 如何处理大规模数据集？
   - 如何监控数据质量？

### 代码练习

完成 [examples.py](examples.py) 中的练习题。

---

## 📖 延伸阅读

**论文**:
- "Text 덴전자링을 위한 한국어 BPE 모델" (BPE论文)
- "MapReduce: Simplified Data Processing on Large Clusters"

**代码参考**:
- [CS336 Assignment 4](references/github/assignment4-data/)
- [trafilatura](https://github.com/adbartra/trafilatura)
- [datasketch](https://github.com/boundedregression/datasketch)

---

## ⚠️ 常见陷阱

1. **过度清洗**:
   - ❌ 删除了有用信息
   - ✅ 保留原始文档，生成清洗后的副本

2. **去重阈值设置**:
   - ❌ 阈值太高（0.95）导致重复未去除
   - ❌ 阈值太低（0.5）导致误删
   - ✅ 典型值：0.7-0.85

3. **内存管理**:
   - ❌ 一次性加载所有文档
   - ✅ 使用生成器和批量处理

4. **中文文本处理**:
   - ❌ 使用英文分词器
   - ✅ 使用jieba等中文分词工具

---

**下一步**: [Day 28: RAG进阶技巧](../Day28_RAG_Advanced/README.md)
