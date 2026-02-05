"""
Day 22-23: 向量数据库与RAG基础 - 代码示例
涵盖：ChromaDB基础、Embedding、文档处理、RAG实现
"""

import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any
import numpy as np


# ============================================================================
# Part 1: ChromaDB 基础操作
# ============================================================================

def example_1_chroma_basic():
    """示例1: ChromaDB基础操作"""
    print("=" * 60)
    print("示例1: ChromaDB基础操作")
    print("=" * 60)

    # 1. 初始化客户端
    client = chromadb.PersistentClient(path="./data/chroma_db")

    # 2. 创建集合
    collection = client.get_or_create_collection(
        name="demo_collection",
        metadata={"hnsw:space": "cosine"}  # 余弦相似度
    )

    # 3. 添加文档
    documents = [
        "Python是一种高级编程语言",
        "机器学习是人工智能的一个分支",
        "深度学习使用神经网络进行学习",
        "自然语言处理处理文本数据"
    ]

    # 生成简单的embedding（实际应该使用模型）
    embeddings = [[np.random.rand() for _ in range(10)] for _ in range(len(documents))]

    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=[{"category": "编程"}, {"category": "AI"}, {"category": "AI"}, {"category": "AI"}],
        ids=["doc1", "doc2", "doc3", "doc4"]
    )

    print(f"✅ 添加了{len(documents)}个文档")

    # 4. 查询
    query_embedding = [[np.random.rand() for _ in range(10)]]
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=2
    )

    print(f"\n查询结果:")
    for i, (doc, distance) in enumerate(zip(results["documents"][0], results["distances"][0])):
        print(f"  {i+1}. {doc} (距离: {distance:.4f})")

    # 5. 统计
    count = collection.count()
    print(f"\n集合中文档总数: {count}")


def example_2_similarity_metrics():
    """示例2: 相似度计算"""
    print("\n" + "=" * 60)
    print("示例2: 相似度计算")
    print("=" * 60)

    import numpy as np

    def cosine_similarity(a, b):
        """余弦相似度"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def euclidean_distance(a, b):
        """欧氏距离"""
        return np.linalg.norm(a - b)

    def dot_product(a, b):
        """点积"""
        return np.dot(a, b)

    # 示例向量
    vec1 = np.array([1, 2, 3, 4])
    vec2 = np.array([1, 2, 3, 5])  # 相似
    vec3 = np.array([-1, -2, -3, -4])  # 相反

    print("向量1:", vec1)
    print("向量2:", vec2)
    print("向量3:", vec3)

    print("\n余弦相似度:")
    print(f"  vec1 vs vec2: {cosine_similarity(vec1, vec2):.4f}")
    print(f"  vec1 vs vec3: {cosine_similarity(vec1, vec3):.4f}")

    print("\n欧氏距离:")
    print(f"  vec1 vs vec2: {euclidean_distance(vec1, vec2):.4f}")
    print(f"  vec1 vs vec3: {euclidean_distance(vec1, vec3):.4f}")

    print("\n点积:")
    print(f"  vec1 · vec2: {dot_product(vec1, vec2):.4f}")
    print(f"  vec1 · vec3: {dot_product(vec1, vec3):.4f}")


def example_3_metadata_filtering():
    """示例3: 元数据过滤"""
    print("\n" + "=" * 60)
    print("示例3: 元数据过滤")
    print("=" * 60)

    client = chromadb.PersistentClient(path="./data/chroma_db")
    collection = client.get_or_create_collection("products")

    # 添加产品数据
    products = [
        "iPhone 15 Pro - 苹果最新旗舰手机",
        "MacBook Pro - 专业笔记本电脑",
        "iPad Air - 轻薄平板电脑",
        "AirPods Pro - 无线降噪耳机"
    ]

    embeddings = [[np.random.rand() for _ in range(10)] for _ in range(len(products))]

    collection.add(
        documents=products,
        embeddings=embeddings,
        metadatas=[
            {"category": "手机", "price": 7999},
            {"category": "电脑", "price": 12999},
            {"category": "平板", "price": 4799},
            {"category": "耳机", "price": 1999}
        ],
        ids=["p1", "p2", "p3", "p4"]
    )

    # 查询+过滤
    query_emb = [[np.random.rand() for _ in range(10)]]

    # 只查询电脑类别
    results = collection.query(
        query_embeddings=query_emb,
        where={"category": "电脑"},
        n_results=5
    )

    print("电脑类产品:")
    for doc in results["documents"][0]:
        print(f"  - {doc}")

    # 价格过滤
    results = collection.query(
        query_embeddings=query_emb,
        where={"price": {"$gte": 5000}},  # 价格>=5000
        n_results=5
    )

    print("\n价格>=5000的产品:")
    for doc in results["documents"][0]:
        print(f"  - {doc}")


# ============================================================================
# Part 2: 文档处理与分块
# ============================================================================

def example_4_document_chunking():
    """示例4: 文档分块"""
    print("\n" + "=" * 60)
    print("示例4: 文档分块")
    print("=" * 60)

    def fixed_size_chunk(text, chunk_size=100, overlap=20):
        """固定大小分块"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap  # 重叠
        return chunks

    def semantic_chunk(text):
        """语义分块（按段落）"""
        paragraphs = text.split("\n\n")
        return [p.strip() for p in paragraphs if p.strip()]

    # 示例文档
    document = """
    机器学习是人工智能的一个分支。

    它使计算机能够在没有明确编程的情况下学习。

    深度学习是机器学习的一个子领域，使用多层神经网络。

    自然语言处理（NLP）是AI的另一个重要应用领域。

    NLP技术包括文本分类、命名实体识别、机器翻译等。
    """

    print("固定大小分块:")
    chunks = fixed_size_chunk(document, chunk_size=50, overlap=10)
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(chunk[:50] + "...")

    print("\n" + "-" * 40)
    print("\n语义分块:")
    chunks = semantic_chunk(document)
    for i, chunk in enumerate(chunks):
        print(f"\n段落 {i+1}:")
        print(chunk.strip())


def example_5_sliding_window():
    """示例5: 滑动窗口分块"""
    print("\n" + "=" * 60)
    print("示例5: 滑动窗口分块")
    print("=" * 60)

    text = "这是一段很长的文本，需要被分割成多个chunk。" * 10

    def sliding_window(text, window_size=50, stride=25):
        """滑动窗口分块"""
        chunks = []
        for i in range(0, len(text), stride):
            chunk = text[i:i+window_size]
            if len(chunk) > 0:
                chunks.append((i, chunk))  # 记录起始位置
        return chunks

    chunks = sliding_window(text, window_size=80, stride=40)

    print(f"文档长度: {len(text)}")
    print(f"生成chunk数: {len(chunks)}")

    for i, (start, chunk) in enumerate(chunks[:3]):  # 只显示前3个
        print(f"\nChunk {i+1} (位置 {start}-{start+len(chunk)}):")
        print(chunk[:60] + "...")


# ============================================================================
# Part 3: RAG 实现
# ============================================================================

class SimpleRAG:
    """简单的RAG系统"""

    def __init__(self, collection_name="rag_collection"):
        # 初始化ChromaDB
        self.client = chromadb.PersistentClient(path="./data/rag_db")
        self.collection = self.client.get_or_create_collection(collection_name)

    def ingest(self, documents: List[str], metadatas: List[Dict] = None):
        """索引文档"""
        ids = [f"doc_{i}" for i in range(len(documents))]

        # 生成简单embedding
        embeddings = [[np.random.rand() for _ in range(384)] for _ in range(len(documents))]

        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas or [{} for _ in documents],
            ids=ids
        )

        print(f"✅ 索引了{len(documents)}个文档")

    def retrieve(self, query: str, top_k: int = 3):
        """检索相关文档"""
        # 生成查询embedding
        query_embedding = [[np.random.rand() for _ in range(384)]]

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )

        return results["documents"][0], results["metadatas"][0], results["distances"][0]

    def generate_answer(self, query: str, context: List[str]) -> str:
        """基于检索结果生成答案（模拟）"""
        # 实际应用中这里应该调用LLM
        context_text = "\n".join([f"- {doc}" for doc in context])

        answer = f"""
        基于检索到的文档，以下是对问题的回答：

        检索到的相关文档:
        {context_text}

        问题: {query}

        答案: 这是一个模拟的答案。在实际应用中，
        这里会调用LLM（如GPT-4）基于检索到的上下文生成答案。
        """

        return answer


def example_6_rag_pipeline():
    """示例6: 完整RAG流程"""
    print("\n" + "=" * 60)
    print("示例6: RAG流程")
    print("=" * 60)

    # 1. 创建RAG系统
    rag = SimpleRAG()

    # 2. 准备文档
    documents = [
        "Python是一种广泛使用的高级编程语言，由Guido van Rossum于1991年创建。",
        "机器学习是人工智能的一个分支，它使计算机能够从数据中学习。",
        "深度学习是机器学习的一个子领域，使用多层神经网络来模拟人脑。",
        "PyTorch是一个开源的机器学习库，由Facebook的人工智能研究团队开发。"
    ]

    metadatas = [
        {"topic": "编程语言", "year": 1991},
        {"topic": "AI", "type": "ML"},
        {"topic": "AI", "type": "DL"},
        {"topic": "AI", "type": "框架"}
    ]

    # 3. 索引文档
    rag.ingest(documents, metadatas)

    # 4. 查询
    query = "什么是深度学习？"

    print(f"\n用户查询: {query}")

    retrieved_docs, metadatas, distances = rag.retrieve(query, top_k=2)

    print(f"\n检索到{len(retrieved_docs)}个相关文档:")
    for i, (doc, meta, dist) in enumerate(zip(retrieved_docs, metadatas, distances)):
        print(f"\n  文档 {i+1} (距离: {dist:.4f}):")
        print(f"  内容: {doc}")
        print(f"  元数据: {meta}")

    # 5. 生成答案
    answer = rag.generate_answer(query, retrieved_docs)

    print(f"\n生成的答案:")
    print(answer)


# ============================================================================
# Part 4: 高级功能
# ============================================================================

def example_7_hybrid_search():
    """示例7: 混合检索（向量+关键词）"""
    print("\n" + "=" * 60)
    print("示例7: 混合检索")
    print("=" * 60)

    class HybridRetriever:
        """混合检索器"""

        def __init__(self):
            self.client = chromadb.PersistentClient(path="./data/hybrid_db")
            self.collection = self.client.get_or_create_collection("hybrid")

        def add_documents(self, docs):
            """添加文档"""
            embeddings = [[np.random.rand() for _ in range(384)] for _ in docs]
            self.collection.add(
                documents=docs,
                embeddings=embeddings,
                ids=[f"doc_{i}" for i in range(len(docs))]
            )

        def vector_search(self, query, top_k=3):
            """向量检索"""
            query_emb = [[np.random.rand() for _ in range(384)]]
            results = self.collection.query(
                query_embeddings=query_emb,
                n_results=top_k
            )
            return results["documents"][0]

        def keyword_search(self, query, docs, top_k=3):
            """关键词检索（简单的TF-IDF）"""
            query_terms = set(query.lower().split())

            scores = []
            for doc in docs:
                doc_terms = set(doc.lower().split())
                # 计算重叠度
                overlap = len(query_terms & doc_terms)
                scores.append((doc, overlap))

            # 排序
            scores.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, score in scores[:top_k]]

        def hybrid_search(self, query, docs, top_k=3, alpha=0.5):
            """混合检索"""
            # 向量检索
            vector_results = self.vector_search(query, top_k=top_k*2)
            vector_scores = {doc: 1-i/top_k for i, doc in enumerate(vector_results)}

            # 关键词检索
            keyword_results = self.keyword_search(query, docs, top_k=top_k*2)
            keyword_scores = {doc: 1-i/top_k for i, doc in enumerate(keyword_results)}

            # 融合
            final_scores = {}
            all_docs = set(vector_scores.keys()) | set(keyword_scores.keys())

            for doc in all_docs:
                vec_score = vector_scores.get(doc, 0)
                kw_score = keyword_scores.get(doc, 0)
                final_scores[doc] = alpha * vec_score + (1-alpha) * kw_score

            # 排序
            ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

            return [doc for doc, score in ranked[:top_k]]

    # 使用
    retriever = HybridRetriever()

    docs = [
        "Python是一种编程语言",
        "机器学习使用算法",
        "深度学习是机器学习的分支",
        "自然语言处理处理文本"
    ]

    retriever.add_documents(docs)

    query = "机器学习算法"

    print(f"查询: {query}\n")

    print("向量检索结果:")
    for doc in retriever.vector_search(query, top_k=2):
        print(f"  - {doc}")

    print("\n关键词检索结果:")
    for doc in retriever.keyword_search(query, docs, top_k=2):
        print(f"  - {doc}")

    print("\n混合检索结果 (alpha=0.5):")
    for doc in retriever.hybrid_search(query, docs, top_k=2, alpha=0.5):
        print(f"  - {doc}")


def example_8_evaluation():
    """示例8: RAG评估"""
    print("\n" + "=" * 60)
    print("示例8: RAG评估指标")
    print("=" * 60)

    def compute_recall(retrieved_docs, relevant_docs):
        """召回率"""
        retrieved_set = set(retrieved_docs)
        relevant_set = set(relevant_docs)

        if len(relevant_set) == 0:
            return 0.0

        return len(retrieved_set & relevant_set) / len(relevant_set)

    def compute_precision(retrieved_docs, relevant_docs):
        """精确率"""
        retrieved_set = set(retrieved_docs)
        relevant_set = set(relevant_docs)

        if len(retrieved_set) == 0:
            return 0.0

        return len(retrieved_set & relevant_set) / len(retrieved_set)

    def compute_mrr(ranked_list, relevant_docs):
        """Mean Reciprocal Rank"""
        for i, doc in enumerate(ranked_list):
            if doc in relevant_docs:
                return 1.0 / (i + 1)
        return 0.0

    # 示例
    all_docs = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant_docs = {"doc1", "doc3"}  # 真实相关的文档

    # 假设检索返回
    retrieved = ["doc1", "doc4", "doc2"]  # 按相关性排序

    recall = compute_recall(retrieved, relevant_docs)
    precision = compute_precision(retrieved, relevant_docs)
    mrr = compute_mrr(retrieved, relevant_docs)

    print(f"所有文档: {all_docs}")
    print(f"相关文档: {relevant_docs}")
    print(f"检索结果: {retrieved}")

    print(f"\n评估指标:")
    print(f"  召回率 (Recall): {recall:.2%}")
    print(f"  精确率 (Precision): {precision:.2%}")
    print(f"  MRR: {mrr:.4f}")


# ============================================================================
# 运行所有示例
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("向量数据库与RAG基础 - 代码示例")
    print("="*60)

    # ChromaDB基础
    example_1_chroma_basic()
    example_2_similarity_metrics()
    example_3_metadata_filtering()

    # 文档处理
    example_4_document_chunking()
    example_5_sliding_window()

    # RAG实现
    example_6_rag_pipeline()

    # 高级功能
    example_7_hybrid_search()
    example_8_evaluation()

    print("\n" + "="*60)
    print("所有示例运行完成！")
    print("="*60)

    print("\n关键要点:")
    print("1. ChromaDB是轻量级向量数据库，适合本地部署")
    print("2. 余弦相似度是最常用的向量相似度度量")
    print("3. 文档分块策略影响RAG效果")
    print("4. 混合检索结合向量和关键词检索")
    print("5. 评估RAG需要多个指标（召回率、精确率、MRR）")
