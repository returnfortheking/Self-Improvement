# Day 22-23: 向量数据库与RAG基础

> **学习目标**: 掌握向量数据库原理，实现RAG基础流程，构建文档问答系统
> **时间分配**: 6小时（理论2h + 实践4h）
> **难度**: ⭐⭐⭐⭐
> **重要性**: ⭐⭐⭐⭐⭐ (RAG是当前AI应用的核心技术)

---

## 📚 核心概念

### 1. 向量数据库基础

#### 1.1 什么是向量数据库？

**传统数据库 vs 向量数据库**:

| 维度 | 传统数据库 (MySQL/MongoDB) | 向量数据库 (Chroma/Pinecone) |
|------|---------------------------|------------------------------|
| **查询方式** | 精确匹配 | 相似度匹配 |
| **索引** | B-Tree、Hash | HNSW、IVF |
| **数据类型** | 结构化数据 | 向量 (Embedding) |
| **应用场景** | 事务处理 | 语义搜索、推荐 |

**核心思想**:
```
文本 → Embedding模型 → 向量 → 向量数据库 → 相似度搜索
```

#### 1.2 Embedding（嵌入）

**定义**: 将高维数据（文本、图像）映射到低维向量空间

**特性**:
- **语义相似**: 相似文本的向量距离近
- **固定维度**: 如OpenAI text-embedding-3-small → 1536维
- **密集向量**: 每个维度都有意义

**示例**:
```python
# 文本
text1 = "机器学习是AI的分支"
text2 = "深度学习属于机器学习"
text3 = "今天天气很好"

# Embedding后
vec1 = [0.12, -0.34, 0.56, ...]  # 1536维
vec2 = [0.13, -0.32, 0.54, ...]  # 与vec1相似（距离近）
vec3 = [-0.45, 0.78, -0.23, ...]  # 与vec1不相似（距离远）
```

#### 1.3 相似度度量

**常用距离公式**:

1. **余弦相似度** (最常用):
   ```
   cos_sim(A, B) = (A · B) / (||A|| × ||B||)
   范围: [-1, 1]，越大越相似
   ```

2. **欧氏距离**:
   ```
   euclidean(A, B) = sqrt(Σ(Ai - Bi)²)
   范围: [0, +∞)，越小越相似
   ```

3. **点积**:
   ```
   dot(A, B) = Σ(Ai × Bi)
   范围: [-∞, +∞]，越大越相似
   ```

**选择建议**:
- 文本搜索: **余弦相似度**（归一化，不受长度影响）
- 图像检索: 欧氏距离
- 推荐系统: 点积（快速计算）

---

### 2. RAG (Retrieval-Augmented Generation) 原理

#### 2.1 为什么需要RAG？

**LLM的问题**:
1. **知识截止**: 训练数据有截止日期
2. **幻觉**: 可能编造错误信息
3. **私有数据**: 无法访问企业内部文档

**RAG解决方案**:
```
用户查询 → 检索相关文档 → LLM基于文档生成回答
```

#### 2.2 RAG架构

**完整流程**:
```
┌─────────────────────────────────────────────────────────┐
│                    RAG Pipeline                          │
├─────────────────────────────────────────────────────────┤
│  1. 文档加载 (Document Loading)                          │
│     ├─ PDF, Markdown, TXT                               │
│     └─ 网页抓取                                         │
│                                                          │
│  2. 文档分块 (Chunking)                                  │
│     ├─ 固定大小分块 (512 tokens)                         │
│     ├─ 语义分块 (按段落、章节)                           │
│     └─ 重叠分块 (overlap=50)                             │
│                                                          │
│  3. 向量化 (Embedding)                                   │
│     └─ OpenAI, BGE, MTEB模型                            │
│                                                          │
│  4. 存储到向量数据库 (Vector Store)                       │
│     └─ Chroma, Pinecone, Weaviate                        │
│                                                          │
│  5. 检索 (Retrieval)                                     │
│     ├─ 向量检索 (Vector Search)                          │
│     ├─ 混合检索 (Hybrid: Vector + BM25)                 │
│     └─ 重排序 (Rerank)                                   │
│                                                          │
│  6. 生成 (Generation)                                    │
│     └─ LLM基于检索到的上下文生成回答                     │
└─────────────────────────────────────────────────────────┘
```

---

### 3. Chroma向量数据库

#### 3.1 Chroma基础

**特点**:
- ✅ 开源、轻量级
- ✅ 本地部署（无需API key）
- ✅ Python原生支持
- ✅ 持久化存储

**安装**:
```bash
pip install chromadb
```

**基础操作**:
```python
import chromadb

# 1. 初始化客户端
client = chromadb.PersistentClient(path="./data/chroma")

# 2. 创建/获取集合
collection = client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
)

# 3. 添加文档
collection.add(
    documents=["这是第一段文本", "这是第二段文本"],
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],  # 可选，自动生成
    metadatas=[{"source": "doc1"}, {"source": "doc2"}],
    ids=["doc1", "doc2"]
)

# 4. 查询
results = collection.query(
    query_texts=["用户查询"],
    n_results=5  # 返回top 5
)

# 5. 删除
collection.delete(ids=["doc1"])
```

#### 3.2 高级功能

**元数据过滤**:
```python
# 只检索特定来源的文档
results = collection.query(
    query_texts=["查询"],
    where={"source": "doc1"},  # 精确匹配
    n_results=5
)

# 复杂条件
results = collection.query(
    query_texts=["查询"],
    where={
        "$and": [
            {"source": {"$ne": "tmp"}},  # source != "tmp"
            {"date": {"$gte": "2024-01-01"}}
        ]
    }
)
```

**自动Embedding**:
```python
import chromadb
from chromadb.utils import embedding_functions

# 使用OpenAI Embedding
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="your-key",
    model_name="text-embedding-3-small"
)

client = chromadb.Client()
collection = client.get_or_create_collection(
    name="docs",
    embedding_function=openai_ef
)

# 自动embed
collection.add(
    documents=["文本1", "文本2"],
    ids=["doc1", "doc2"]  # 无需手动提供embeddings
)
```

---

### 4. 文档处理

#### 4.1 文档加载

**使用LangChain**:
```python
from langchain.document_loaders import PyPDFLoader, DirectoryLoader

# 加载单个PDF
loader = PyPDFLoader("data/report.pdf")
pages = loader.load()

# 加载整个目录
loader = DirectoryLoader(
    "data/documents",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)
docs = loader.load()
```

**使用unstructured** (更强大):
```python
from unstructured.partition.pdf import partition_pdf

# 支持复杂布局（表格、图片）
elements = partition_pdf(
    filename="report.pdf",
    extract_images_in_pdf=True,
    infer_table_structure=True
)
```

#### 4.2 文档分块策略

**策略1: 固定长度分块**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,        # 每块512字符
    chunk_overlap=50,      # 重叠50字符（保持上下文）
    separators=["\n\n", "\n", "。", " ", ""]
)

chunks = splitter.split_documents(docs)
```

**策略2: 语义分块**
```python
from langchain_experimental.text_splitter import SemanticChunker

# 基于语义相似度分块
splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile"
)

chunks = splitter.split_text(text)
```

**策略3: 自定义分块**
```python
def custom_chunker(text, max_length=512, overlap=50):
    """自定义分块逻辑"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_length
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # 重叠
    return chunks
```

**分块最佳实践**:
| 场景 | chunk_size | overlap | 分隔符 |
|------|-----------|---------|--------|
| 长文档 | 1024-2048 | 128-256 | 段落 |
| 代码 | 512 | 50 | 函数/类 |
| QA对 | 256-512 | 0 | 问题 |

---

## 🔧 实战案例

### 案例1: 构建PDF文档RAG系统

**完整流程**:

```python
import chromadb
from chromadb.utils import embedding_functions
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai

# 1. 初始化组件
class PDFRAGSystem:
    def __init__(self):
        # ChromaDB
        self.client = chromadb.PersistentClient(path="./data/db")
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key="your-key",
            model_name="text-embedding-3-small"
        )
        self.collection = self.client.get_or_create_collection(
            name="pdf_docs",
            embedding_function=self.embedding_fn
        )

        # 文档分块
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50
        )

    def ingest_pdf(self, pdf_path):
        """加载PDF并索引"""
        # 加载PDF
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        # 分块
        chunks = []
        for page in pages:
            page_chunks = self.splitter.split_text(page.page_content)
            for i, chunk in enumerate(page_chunks):
                chunks.append({
                    "text": chunk,
                    "metadata": {
                        "source": pdf_path,
                        "page": page.metadata["page"],
                        "chunk": i
                    }
                })

        # 存储到Chroma
        self.collection.add(
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
            ids=[f"{pdf_path}_{i}" for i in range(len(chunks))]
        )

        print(f"✅ 索引完成: {len(chunks)}个chunks")

    def query(self, question, top_k=5):
        """查询并回答"""
        # 1. 检索相关文档
        results = self.collection.query(
            query_texts=[question],
            n_results=top_k
        )

        # 2. 构建prompt
        context = "\n\n".join(results["documents"][0])
        prompt = f"""
        基于以下文档回答问题。如果文档中没有相关信息，请说"我不知道"。

        文档:
        {context}

        问题: {question}
        答案:
        """

        # 3. LLM生成
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content

# 使用
rag = PDFRAGSystem()
rag.ingest_pdf("data/report.pdf")
answer = rag.query("报告的主要结论是什么？")
print(answer)
```

---

### 案例2: 混合检索（Vector + BM25）

```python
from rank_bm25 import BM25Okapi
import jieba

class HybridRAG:
    def __init__(self):
        # 向量检索
        self.client = chromadb.PersistentClient(path="./data/db")
        self.collection = self.client.get_or_create_collection("docs")

        # BM25检索
        self.bm25_index = None
        self.docs = []

    def index(self, documents):
        """构建混合索引"""
        self.docs = documents

        # 1. 向量索引
        self.collection.add(
            documents=documents,
            ids=[f"doc_{i}" for i in range(len(documents))]
        )

        # 2. BM25索引
        tokenized_docs = [list(jieba.cut(doc)) for doc in documents]
        self.bm25_index = BM25Okapi(tokenized_docs)

    def search(self, query, top_k=5, alpha=0.5):
        """
        混合检索

        Args:
            alpha: 向量检索权重 (0-1)
                   alpha=1: 纯向量检索
                   alpha=0: 纯BM25检索
                   alpha=0.5: 加权融合
        """
        # 1. 向量检索
        vector_results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        vector_scores = {id_: 1-i/top_k for i, id_ in enumerate(vector_results["ids"][0])}

        # 2. BM25检索
        tokenized_query = list(jieba.cut(query))
        bm25_results = self.bm25_index.get_top_n(tokenized_query, self.docs, n=top_k)
        bm25_scores = {doc: score for doc, score in bm25_results}

        # 3. 融合打分
        final_scores = {}
        for doc_id in vector_scores:
            final_scores[doc_id] = (
                alpha * vector_scores[doc_id] +
                (1-alpha) * bm25_scores.get(doc_id, 0)
            )

        # 4. 排序
        ranked_docs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

        return ranked_docs[:top_k]
```

---

## 💡 实现技巧

### 1. Embedding优化

**批量处理**:
```python
def batch_embed(texts, batch_size=100):
    """批量Embedding（加速）"""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = embedding_model.embed_documents(batch)
        embeddings.extend(batch_embeddings)
    return embeddings
```

**缓存Embedding**:
```python
import hashlib
import pickle

class CachedEmbedding:
    def __init__(self, model):
        self.model = model
        self.cache = {}

    def embed(self, text):
        # 生成hash
        text_hash = hashlib.md5(text.encode()).hexdigest()

        # 检查缓存
        if text_hash in self.cache:
            return self.cache[text_hash]

        # 计算并缓存
        embedding = self.model.embed_query(text)
        self.cache[text_hash] = embedding
        return embedding
```

### 2. 分块优化

**滑动窗口分块**:
```python
def sliding_window_chunks(text, window_size=512, stride=256):
    """滑动窗口分块（更多信息保留）"""
    chunks = []
    for i in range(0, len(text), stride):
        chunk = text[i:i+window_size]
        if len(chunk) > 0:
            chunks.append(chunk)
    return chunks
```

**语义边界检测**:
```python
def semantic_aware_chunk(text, max_length=512):
    """在语义边界处分块"""
    sentences = text.split("。")
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) > max_length:
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            current_chunk += sentence + "。"

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
```

### 3. 检索优化

**Rerank（重排序）**:
```python
from sentence_transformers import CrossEncoder

def retrieve_with_rerank(query, top_k=100, rerank_top=10):
    """两阶段检索"""
    # 1. 粗排（召回）
    candidates = vector_store.search(query, k=top_k)

    # 2. 精排（Rerank）
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    pairs = [[query, doc["text"]] for doc in candidates]
    scores = reranker.predict(pairs)

    # 3. 重新排序
    reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    return [doc for doc, score in reranked[:rerank_top]]
```

**查询扩展**:
```python
def query_expansion(query):
    """查询扩展（提高召回率）"""
    # 同义词扩展
    expansions = []
    if "机器学习" in query:
        expansions.append("AI")
        expansions.append("深度学习")

    # 重写
    expanded_query = f"{query} {' '.join(expansions)}"

    return expanded_query
```

---

## 📊 性能评估

### RAG评估指标

**检索质量**:
1. **召回率 (Recall)**: 检索到的相关文档 / 所有相关文档
2. **精确率 (Precision)**: 检索到的相关文档 / 检索到的总文档
3. **MRR (Mean Reciprocal Rank)**: 第一个相关文档的倒数排名

**生成质量**:
1. **忠实度 (Faithfulness)**: 答案是否基于检索到的文档
2. **答案相关性 (Answer Relevance)**: 答案是否回答了问题

**评估工具**:
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

results = evaluate(
    dataset=test_dataset,
    metrics=[faithfulness, answer_relevancy]
)

print(results)
```

---

## 🎯 学习检验

### 关键问题

1. **向量数据库**:
   - 向量数据库与传统数据库的区别？
   - 如何选择合适的相似度度量？
   - Chroma的高级功能有哪些？

2. **RAG架构**:
   - RAG的完整流程是什么？
   - 如何优化文档分块策略？
   - 混合检索的优势是什么？

3. **实现细节**:
   - 如何构建生产级RAG系统？
   - 如何评估RAG系统性能？
   - 常见问题及解决方案？

### 代码练习

完成 [examples.py](examples.py) 中的练习题。

---

## 📖 延伸阅读

**文档**:
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [RAGAS Evaluation](https://docs.ragas.io/)

**论文**:
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
- "Indexify: A Framework for Building RAG Applications" (2023)

**代码**:
- [MODULAR-RAG-MCP-SERVER](references/github/MODULAR-RAG-MCP-SERVER/) - 完整RAG实现

---

## ⚠️ 常见陷阱

1. **分块策略不当**:
   - ❌ chunk_size太小（丢失上下文）
   - ❌ chunk_size太大（检索不精确）
   - ✅ 根据文档类型调整（512-2048）

2. **Embedding模型选择**:
   - ❌ 使用通用模型处理专业文档
   - ✅ 使用领域微调的模型

3. **检索数量**:
   - ❌ top_k太大（引入噪声）
   - ❌ top_k太小（遗漏关键信息）
   - ✅ 根据任务调整（3-10）

4. **Prompt工程**:
   - ❌ 只提供问题，不提供上下文
   - ✅ 明确指示"基于以下文档回答"

---

**下一步**: [Day 24-25: Scaling Laws](../Day24-25_Scaling_Laws/README.md)
