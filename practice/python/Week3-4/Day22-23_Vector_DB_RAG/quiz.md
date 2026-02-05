# Day 22-23: 向量数据库与RAG基础 - 面试题

> **题目难度**: ⭐⭐ ~ ⭐⭐⭐⭐⭐
> **考察重点**: 向量数据库原理、RAG架构、Embedding、检索优化
> **建议时间**: 40分钟

---

## Part 1: 向量数据库基础

### Q1: 向量数据库与传统数据库的主要区别？⭐⭐⭐

**参考答案**:

| 维度 | 传统数据库 | 向量数据库 |
|------|-----------|-----------|
| **数据模型** | 结构化（表、文档） | 向量（Embedding） |
| **查询方式** | 精确匹配、范围查询 | 相似度搜索 |
| **索引结构** | B-Tree、Hash | HNSW、IVF、PQ |
| **应用场景** | 事务处理、业务数据 | 语义搜索、推荐系统 |
| **典型产品** | MySQL、MongoDB | Chroma、Pinecone、Milvus |

**为什么需要向量数据库**:
- 传统数据库无法高效处理高维向量相似度搜索
- 向量数据库专门优化了近似最近邻（ANN）搜索
- 支持高维向量（512-4096维）的快速检索

---

### Q2: 什么是Embedding？为什么它能表示语义？⭐⭐⭐⭐

**参考答案**:

**定义**: 将高维稀疏数据（文本、图像）映射到低维密集向量

**为什么能表示语义**:
1. **训练目标**: 神经网络学习使相似对象的向量接近
2. **空间结构**: 语义相似的文本在向量空间中距离近
3. **连续表示**: 向量的每个维度都编码某种语义特征

**示例**:
```
文本: "机器学习是AI的分支"
      "深度学习属于机器学习"
      "今天天气很好"

Embedding后:
vec1 = [0.12, -0.34, 0.56, ...]  # 机器学习相关
vec2 = [0.13, -0.32, 0.54, ...]  # 与vec1距离近（语义相似）
vec3 = [-0.45, 0.78, -0.23, ...]  # 与vec1距离远（语义不相似）
```

**关键模型**:
- OpenAI: text-embedding-3-small/large (1536/3072维)
- BGE: bge-m3 (1024维，中文优化)
- Cohere: embed-english-v3.0 (1024维)

---

### Q3: 余弦相似度与欧氏距离的区别？⭐⭐⭐⭐

**参考答案**:

**余弦相似度**:
```
cos_sim(A, B) = (A · B) / (||A|| × ||B||)
范围: [-1, 1]，越大越相似
```
- 只关心方向，不关心大小
- 适合文本检索（不受文档长度影响）

**欧氏距离**:
```
euclidean(A, B) = sqrt(Σ(Ai - Bi)²)
范围: [0, +∞)，越小越相似
```
- 考虑向量的实际大小
- 适合图像检索

**示例**:
```python
vec1 = [1, 1]      # 长度√2
vec2 = [2, 2]      # 长度√8，方向相同
vec3 = [1, -1]     # 长度√2，方向相反

cos_sim(vec1, vec2) = 1.0  # 完全相似（方向相同）
cos_sim(vec1, vec3) = 0.0  # 不相似（方向垂直）

euclidean(vec1, vec2) = 1.414  # 有一定距离
euclidean(vec1, vec3) = 2.828  # 距离更大
```

**选择建议**:
- 文本搜索: **余弦相似度**
- 图像检索: **欧氏距离**
- 推荐系统: **点积**（最快的近似）

---

## Part 2: RAG架构

### Q4: RAG相比Fine-tuning的优势？⭐⭐⭐⭐

**参考答案**:

| 维度 | RAG | Fine-tuning |
|------|-----|-------------|
| **知识更新** | 实时更新索引 | 需要重新训练 |
| **数据要求** | 只需文档 | 需要大量训练数据 |
| **成本** | 低 | 高（GPU、时间） |
| **幻觉问题** | 减少幻觉（基于文档） | 可能产生幻觉 |
| **可解释性** | 高（可查看检索到的文档） | 低（黑盒） |
| **知识边界** | 清晰（检索范围） | 模型内部知识 |

**适用场景**:
- **RAG**: 知识密集、实时更新、私有数据
- **Fine-tuning**: 特定任务、特定风格、领域适应

**最佳实践**: RAG + Fine-tuning结合

---

### Q5: 如何优化RAG的检索质量？⭐⭐⭐⭐⭐

**参考答案**:

**策略1: 优化文档分块**
```python
# 根据内容类型调整chunk_size
- 代码: 512 tokens（函数级别）
- 论文: 1024-2048 tokens（段落级别）
- QA对: 256-512 tokens（问题级别）
```

**策略2: 混合检索**
```python
# 向量检索 + BM25关键词检索
results = (
    alpha * vector_search(query) +
    (1-alpha) * bm25_search(query)
)
```

**策略3: 查询重写**
```python
# 扩展查询
"机器学习" → "机器学习 AI 深度学习"
```

**策略4: Rerank**
```python
# 两阶段检索
candidates = vector_search(query, top_k=100)  # 召回
reranked = reranker.rank(candidates, query)     # 精排
```

**策略5: 元数据过滤**
```python
# 先过滤再检索
where={"category": "技术文档", "date": {"$gte": "2024-01-01"}}
```

---

### Q6: 什么是混合检索（Hybrid Search）？⭐⭐⭐⭐

**参考答案**:

**定义**: 结合向量检索和关键词检索

**为什么需要**:
- **向量检索**: 擅长语义相似，可能忽略精确匹配
- **关键词检索**: 擅长精确匹配，但无法理解语义
- **混合检索**: 结合两者优势

**实现**:
```python
def hybrid_search(query, alpha=0.5):
    # 1. 向量检索
    vector_results = vector_store.search(query, top_k=100)
    vector_scores = {doc.id: score for doc, score in vector_results}

    # 2. BM25检索
    bm25_results = bm25_index.search(query, top_k=100)
    bm25_scores = {doc.id: score for doc, score in bm25_results}

    # 3. 融合打分
    final_scores = {}
    for doc_id in set(vector_scores) | set(bm25_scores):
        vec_score = vector_scores.get(doc_id, 0)
        bm25_score = bm25_scores.get(doc_id, 0)
        final_scores[doc_id] = alpha * vec_score + (1-alpha) * bm25_score

    # 4. 排序
    return sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
```

**调优**:
- **alpha=1.0**: 纯向量检索（语义查询）
- **alpha=0.0**: 纯BM25检索（关键词查询）
- **alpha=0.5**: 平衡混合

---

## Part 3: 文档处理

### Q7: 如何选择合适的chunk_size？⭐⭐⭐⭐

**参考答案**:

**原则**: 根据文档类型和检索任务调整

**chunk_size选择**:

| 文档类型 | 推荐chunk_size | overlap | 原因 |
|---------|---------------|---------|------|
| 技术文档 | 1024-2048 | 128-256 | 保留完整概念 |
| 代码 | 512 | 50-100 | 函数/类完整性 |
| 新闻 | 256-512 | 25-50 | 单个事件 |
| QA对 | 256-512 | 0 | 问题独立性 |

**权衡**:
- **太大**:
  - ✅ 保留更多上下文
  - ❌ 检索不精确（噪声多）
  - ❌ 单次检索token数多

- **太小**:
  - ✅ 检索精确
  - ❌ 上下文不足
  - ❌ 需要更多chunks

**最佳实践**:
```python
# 根据文档类型动态调整
if doc_type == "code":
    chunk_size = 512
elif doc_type == "paper":
    chunk_size = 1536
else:
    chunk_size = 1024
```

---

### Q8: overlap的作用是什么？如何设置？⭐⭐⭐

**参考答案**:

**作用**: 保持上下文连续性

**问题**: 不使用overlap导致边界信息丢失
```
chunk1: "...机器学习是AI的"
chunk2: "分支。深度学习使用..."
```
→ "AI的"和"分支"被拆开

**使用overlap**:
```
chunk1: "...机器学习是AI的"
chunk2: "AI的分支。深度学习..."  # overlap了"AI的"
```

**overlap设置**:
```python
# 一般为chunk_size的10-25%
chunk_size = 1024
overlap = 128  # ~12.5%
```

**注意事项**:
- overlap越大，检索结果冗余越多
- 代码类文档: overlap=50（保持语法完整）
- 叙述类文档: overlap=100-200（保持语义完整）

---

## Part 4: 实现细节

### Q9: ChromaDB的高级用法？⭐⭐⭐⭐

**参考答案**:

**1. 元数据过滤**:
```python
# 精确匹配
results = collection.query(
    query_texts=["query"],
    where={"category": "技术"}
)

# 范围查询
results = collection.query(
    query_texts=["query"],
    where={"date": {"$gte": "2024-01-01", "$lte": "2024-12-31"}}
)

# 逻辑运算
results = collection.query(
    query_texts=["query"],
    where={"$and": [
        {"category": "技术"},
        {"date": {"$gte": "2024-01-01"}}
    ]}
)
```

**2. 自动Embedding**:
```python
from chromadb.utils import embedding_functions

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="key",
    model_name="text-embedding-3-small"
)

collection = client.get_or_create_collection(
    "docs",
    embedding_function=openai_ef
)

# 无需手动提供embeddings
collection.add(
    documents=["文本1", "文本2"],
    ids=["doc1", "doc2"]
)
```

**3. 持久化**:
```python
# 持久化到磁盘
client = chromadb.PersistentClient(path="./data/chroma")

# 临时存储（内存）
client = chromadb.Client()
```

---

### Q10: 如何评估RAG系统？⭐⭐⭐⭐⭐

**参考答案**:

**检索指标**:

1. **召回率 (Recall)**:
   ```
   Recall = |检索到的相关文档| / |所有相关文档|
   ```
   - 衡量检索的全面性

2. **精确率 (Precision)**:
   ```
   Precision = |检索到的相关文档| / |检索到的总文档|
   ```
   - 衡量检索的准确性

3. **MRR (Mean Reciprocal Rank)**:
   ```
   MRR = 1 / rank_of_first_relevant_doc
   ```
   - 衡量第一个相关文档的排名

**生成指标**:

4. **忠实度 (Faithfulness)**:
   - 答案是否基于检索到的文档
   - 使用RAGAS框架评估

5. **答案相关性 (Answer Relevance)**:
   - 答案是否回答了问题

**评估工具**:
```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall
)

results = evaluate(
    dataset=test_dataset,
    metrics=[faithfulness, answer_relevancy, context_recall]
)
```

---

### Q11: 如何处理多模态RAG（文本+图像）？⭐⭐⭐⭐⭐

**参考答案**:

**挑战**: 文档包含图像，如何检索？

**方案1: 多模态Embedding**
```python
# 使用CLIP模型（文本+图像）
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('clip-ViT-B-32')

# 文本embedding
text_emb = model.encode(["搜索查询"])

# 图像embedding
image_emb = model.encode(["image1.png", "image2.png"])

# 跨模态检索
similarity = text_emb @ image_emb.T
```

**方案2: 图像描述**
```python
# 生成图像描述，转为文本检索
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

image = Image.open("figure.png")
inputs = processor(image, return_tensors="pt")
caption = model.generate(**inputs)
description = processor.decode(caption[0])

# 现在可以和文本一起索引
```

**方案3: 向量库分离**
```python
# 文本用文本embedding
text_collection.add(
    documents=[...],
    embeddings=text_embeddings
)

# 图像用图像embedding
image_collection.add(
    documents=[...],
    embeddings=image_embeddings
)

# 分别检索后合并
text_results = text_collection.query(...)
image_results = image_collection.query(...)
```

---

### Q12: RAG系统的常见问题及解决方案？⭐⭐⭐⭐

**参考答案**:

**问题1: 检索召回率低**

**原因**: 分块不合理、embedding模型不匹配

**解决**:
```python
# 1. 优化分块
chunker = SemanticChunker(embedding_model)

# 2. 混合检索
hybrid_results = vector_search + bm25_search

# 3. 查询扩展
expanded_query = expand_query(query)
```

**问题2: 生成答案不基于文档**

**原因**: Prompt设计不当、LLM未正确使用上下文

**解决**:
```python
prompt = """
你是一个助手。请基于以下文档回答问题。
如果文档中没有相关信息，请说"我不知道"。

文档:
{context}

问题: {question}

请只基于文档回答，不要编造信息。
"""
```

**问题3: 检索速度慢**

**原因**: 数据量大、未优化索引

**解决**:
```python
# 1. 使用近似索引（HNSW）
collection = client.create_collection(
    "docs",
    metadata={"hnsw:space": "cosine", "hnsw:M": 16}
)

# 2. 减少检索数量
results = collection.query(..., n_results=5)  # 而不是100

# 3. 使用元数据过滤缩小范围
results = collection.query(..., where={"category": "tech"})
```

---

## Part 5: 进阶话题

### Q13: 如何实现增量更新向量索引？⭐⭐⭐⭐

**参考答案**:

**问题**: 文档更新后，如何更新索引？

**方案**:
```python
def update_document(doc_id, new_text, collection):
    # 1. 删除旧文档
    collection.delete(ids=[doc_id])

    # 2. 添加新文档
    collection.add(
        documents=[new_text],
        ids=[doc_id]
    )

# 批量更新
def batch_update(updates, collection):
    for doc_id, new_text in updates:
        collection.delete(ids=[doc_id])

    new_docs = [new_text for _, new_text in updates]
    new_ids = [doc_id for doc_id, _ in updates]

    collection.add(
        documents=new_docs,
        ids=new_ids
    )
```

**最佳实践**:
- 使用upsert（如果支持）
- 定期重建索引（维护索引质量）
- 版本控制（追踪文档变更）

---

### Q14: 如何处理中英文混合检索？⭐⭐⭐⭐

**参考答案**:

**挑战**: 中英文embedding空间不同

**方案1: 多语言模型**
```python
# 使用支持中英文的模型
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# 或
model = SentenceTransformer('BAAI/bge-m3')  # 多语言支持
```

**方案2: 翻译后检索**
```python
def translate_query(query):
    # 检测语言
    if is_chinese(query):
        # 翻译成英文
        translated = translate_to_english(query)
        return [query, translated]  # 返回双语
    else:
        return [query]

# 使用双语查询检索
queries = translate_query(user_query)
results = [collection.query(q) for q in queries]
```

**方案3: 独立索引**
```python
# 中文和英文分别索引
zh_collection = client.get_collection("zh_docs")
en_collection = client.get_collection("en_docs")

# 根据查询语言选择索引
if is_chinese(query):
    results = zh_collection.query(query)
else:
    results = en_collection.query(query)
```

---

### Q15: 如何构建生产级RAG系统？⭐⭐⭐⭐⭐

**参考答案**:

**架构设计**:
```
┌─────────────────────────────────────────────┐
│  API层 (FastAPI)                            │
├─────────────────────────────────────────────┤
│  服务层                                      │
│  ├─ 查询服务                                │
│  ├─ 索引服务                                │
│  └─ 评估服务                                │
├─────────────────────────────────────────────┤
│  存储层                                      │
│  ├─ ChromaDB (向量)                         │
│  ├─ PostgreSQL (元数据)                     │
│  └─ Redis (缓存)                            │
├─────────────────────────────────────────────┤
│  外部服务                                    │
│  ├─ OpenAI (LLM + Embedding)               │
│  └─ Reranker (Cohere)                      │
└─────────────────────────────────────────────┘
```

**关键组件**:

1. **API层**:
```python
from fastapi import FastAPI

app = FastAPI()

@app.post("/query")
async def query(request: QueryRequest):
    # 参数验证
    # 检索
    # 生成
    # 返回
    pass
```

2. **缓存**:
```python
import redis

cache = redis.Redis()

def cached_retrieve(query):
    cache_key = f"query:{hash(query)}"
    cached = cache.get(cache_key)

    if cached:
        return json.loads(cached)

    results = retrieve(query)
    cache.setex(cache_key, 3600, json.dumps(results))
    return results
```

3. **监控**:
```python
from prometheus_client import Counter

QUERY_COUNTER = Counter('rag_queries_total', 'Total queries')

@app.post("/query")
async def query(request: QueryRequest):
    QUERY_COUNTER.inc()
    # ...
```

4. **容错**:
```python
# 降级策略
def retrieve_with_fallback(query):
    try:
        return vector_search(query)
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return bm25_search(query)  # 降级到BM25
```

---

## 总结

**必会题目** (面试高频):
- Q1: 向量数据库vs传统数据库
- Q2: Embedding原理
- Q4: RAG vs Fine-tuning
- Q6: 混合检索
- Q10: RAG评估

**加分题目** (深入理解):
- Q5: 优化检索质量
- Q11: 多模态RAG
- Q14: 中英文混合检索
- Q15: 生产级架构

**建议**:
1. 理解向量空间和相似度计算
2. 掌握RAG完整流程
3. 了解不同检索策略的trade-off
4. 能够评估和优化RAG系统

---

**下一步**: [Day 24-25: Scaling Laws](../Day24-25_Scaling_Laws/quiz.md)
