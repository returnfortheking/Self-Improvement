"""
Day 22-23: 向量数据库与RAG基础 - 练习题
难度：⭐⭐ ~ ⭐⭐⭐⭐
"""

import chromadb
import numpy as np
from typing import List, Dict, Any


# ============================================================================
# 练习1: 实现余弦相似度计算（⭐⭐）
# ============================================================================

def exercise_1_cosine_similarity():
    """
    任务：实现余弦相似度计算

    要求：
    1. 实现余弦相似度公式
    2. 处理零向量情况
    3. 测试正确性
    """
    def cosine_similarity(vec1, vec2):
        """
        计算两个向量的余弦相似度

        Args:
            vec1, vec2: 向量（list或np.array）

        Returns:
            similarity: 余弦相似度 [-1, 1]
        """
        # TODO: 实现余弦相似度
        # cos_sim = (vec1 · vec2) / (||vec1|| × ||vec2||)
        raise NotImplementedError

    # 测试
    vec1 = np.array([1, 2, 3])
    vec2 = np.array([4, 5, 6])
    vec3 = np.array([-1, -2, -3])

    sim12 = cosine_similarity(vec1, vec2)
    sim13 = cosine_similarity(vec1, vec3)

    print(f"vec1 vs vec2: {sim12:.4f}")
    print(f"vec1 vs vec3: {sim13:.4f}")

    assert 0 < sim12 < 1, "vec1和vec2应该相似"
    assert -1 < sim13 < 0, "vec1和vec3应该不相似"

    print("✅ 练习1完成: 余弦相似度计算正确")


# ============================================================================
# 练习2: 实现文档分块器（⭐⭐⭐）
# ============================================================================

def exercise_2_document_chunker():
    """
    任务：实现文档分块器

    要求：
    1. 固定大小分块
    2. 支持重叠
    3. 保留元数据
    """
    class DocumentChunker:
        def __init__(self, chunk_size=512, overlap=50):
            self.chunk_size = chunk_size
            self.overlap = overlap

        def chunk(self, text: str, metadata: Dict = None) -> List[Dict]:
            """
            分块文档

            Args:
                text: 输入文本
                metadata: 文档元数据

            Returns:
                chunks: 分块列表，每个包含text和metadata
            """
            # TODO: 实现分块逻辑
            chunks = []

            # 1. 按chunk_size分块
            # 2. 每块重叠overlap字符
            # 3. 为每块生成metadata（包含chunk_index）

            raise NotImplementedError

    # 测试
    text = "这是一段测试文本。" * 100
    chunker = DocumentChunker(chunk_size=200, overlap=50)

    chunks = chunker.chunk(text, metadata={"source": "test.txt"})

    print(f"生成了{len(chunks)}个chunks")
    print(f"第一个chunk: {chunks[0]['text'][:50]}...")
    print(f"第一个chunk的metadata: {chunks[0]['metadata']}")

    assert len(chunks) > 1, "应该生成多个chunks"
    assert "chunk_index" in chunks[0]["metadata"], "metadata应包含chunk_index"

    print("✅ 练习2完成: 文档分块器实现正确")


# ============================================================================
# 练习3: 实现简单RAG系统（⭐⭐⭐⭐）
# ============================================================================

def exercise_3_simple_rag():
    """
    任务：实现简单的RAG系统

    要求：
    1. 文档索引
    2. 向量检索
    3. 答案生成（模拟）
    """
    class SimpleRAG:
        def __init__(self):
            # TODO: 初始化ChromaDB客户端
            raise NotImplementedError

        def index_documents(self, documents: List[str]):
            """
            索引文档

            Args:
                documents: 文档列表
            """
            # TODO:
            # 1. 为每个文档生成embedding
            # 2. 存储到ChromaDB
            raise NotImplementedError

        def retrieve(self, query: str, top_k: int = 3) -> List[str]:
            """
            检索相关文档

            Args:
                query: 查询文本
                top_k: 返回文档数量

            Returns:
                documents: 检索到的文档列表
            """
            # TODO:
            # 1. 为查询生成embedding
            # 2. 在ChromaDB中查询
            # 3. 返回top_k个文档
            raise NotImplementedError

        def generate_answer(self, query: str, context: List[str]) -> str:
            """
            基于检索到的文档生成答案

            Args:
                query: 用户问题
                context: 检索到的相关文档

            Returns:
                answer: 生成的答案
            """
            # TODO: 构建prompt并模拟LLM生成
            # 在实际应用中，这里应该调用LLM API
            prompt = f"基于以下文档回答问题：\n\n{context}\n\n问题：{query}"
            raise NotImplementedError

    # 测试
    rag = SimpleRAG()

    docs = [
        "Python是一种编程语言",
        "机器学习是AI的分支",
        "深度学习使用神经网络"
    ]

    print("索引文档...")
    rag.index_documents(docs)

    query = "什么是深度学习？"
    print(f"\n查询: {query}")

    retrieved = rag.retrieve(query, top_k=2)
    print(f"检索到{len(retrieved)}个相关文档")

    answer = rag.generate_answer(query, retrieved)
    print(f"\n答案:\n{answer}")

    print("✅ 练习3完成: 简单RAG系统实现")


# ============================================================================
# 练习4: 实现混合检索（⭐⭐⭐⭐）
# ============================================================================

def exercise_4_hybrid_retrieval():
    """
    任务：实现混合检索（向量+关键词）

    要求：
    1. 向量检索
    2. 关键词检索（BM25）
    3. 结果融合
    """
    class HybridRetriever:
        def __init__(self):
            # TODO: 初始化
            pass

        def vector_search(self, query: str, top_k: int = 5) -> List[Dict]:
            """
            向量检索

            Returns:
                results: [{"doc": ..., "score": ...}, ...]
            """
            # TODO: 实现向量检索
            raise NotImplementedError

        def keyword_search(self, query: str, top_k: int = 5) -> List[Dict]:
            """
            关键词检索（简化版BM25）

            Returns:
                results: [{"doc": ..., "score": ...}, ...]
            """
            # TODO: 实现关键词检索
            # 提示：分词、计算TF-IDF、排序
            raise NotImplementedError

        def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[Dict]:
            """
            混合检索

            Args:
                query: 查询文本
                top_k: 返回结果数量
                alpha: 向量检索权重 (0-1)

            Returns:
                results: 融合后的结果
            """
            # TODO:
            # 1. 分别进行向量和关键词检索
            # 2. 融合打分: score = alpha * vec_score + (1-alpha) * kw_score
            # 3. 重新排序并返回top_k
            raise NotImplementedError

    # 测试
    retriever = HybridRetriever()

    # 添加测试文档
    docs = [
        "Python是一种编程语言",
        "Java也是一种编程语言",
        "机器学习使用数据训练模型",
        "深度学习是机器学习的分支"
    ]

    print("测试混合检索")
    query = "编程语言"

    results = retriever.hybrid_search(query, top_k=2, alpha=0.5)

    print(f"\n查询: {query}")
    print(f"结果:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['doc']} (score: {result['score']:.4f})")

    print("✅ 练习4完成: 混合检索实现")


# ============================================================================
# 练习5: 实现RAG评估（⭐⭐⭐）
# ============================================================================

def exercise_5_rag_evaluation():
    """
    任务：实现RAG评估指标

    要求：
    1. 召回率（Recall）
    2. 精确率（Precision）
    3. MRR (Mean Reciprocal Rank)
    """
    def compute_recall(retrieved_ids, relevant_ids):
        """
        计算召回率

        Args:
            retrieved_ids: 检索到的文档ID列表
            relevant_ids: 相关文档ID集合

        Returns:
            recall: 召回率 [0, 1]
        """
        # TODO: 实现召回率计算
        # recall = |retrieved ∩ relevant| / |relevant|
        raise NotImplementedError

    def compute_precision(retrieved_ids, relevant_ids):
        """
        计算精确率

        Args:
            retrieved_ids: 检索到的文档ID列表
            relevant_ids: 相关文档ID集合

        Returns:
            precision: 精确率 [0, 1]
        """
        # TODO: 实现精确率计算
        # precision = |retrieved ∩ relevant| / |retrieved|
        raise NotImplementedError

    def compute_mrr(ranked_ids, relevant_ids):
        """
        计算MRR

        Args:
            ranked_ids: 排序的文档ID列表
            relevant_ids: 相关文档ID集合

        Returns:
            mrr: Mean Reciprocal Rank
        """
        # TODO: 实现MRR计算
        # MRR = 1 / rank_of_first_relevant
        raise NotImplementedError

    # 测试
    retrieved = ["doc1", "doc3", "doc5", "doc7", "doc9"]
    relevant = {"doc1", "doc2", "doc5", "doc8"}

    recall = compute_recall(retrieved, relevant)
    precision = compute_precision(retrieved, relevant)
    mrr = compute_mrr(retrieved, relevant)

    print(f"检索结果: {retrieved}")
    print(f"相关文档: {relevant}")
    print(f"\n评估指标:")
    print(f"  召回率: {recall:.2%}")
    print(f"  精确率: {precision:.2%}")
    print(f"  MRR: {mrr:.4f}")

    # 验证
    assert 0 <= recall <= 1, "召回率应在[0,1]之间"
    assert 0 <= precision <= 1, "精确率应在[0,1]之间"
    assert 0 <= mrr <= 1, "MRR应在[0,1]之间"

    print("✅ 练习5完成: RAG评估指标计算正确")


# ============================================================================
# 运行所有练习
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("向量数据库与RAG基础 - 练习题")
    print("="*60)

    exercises = [
        ("练习1: 余弦相似度", exercise_1_cosine_similarity),
        ("练习2: 文档分块器", exercise_2_document_chunker),
        ("练习3: 简单RAG系统", exercise_3_simple_rag),
        ("练习4: 混合检索", exercise_4_hybrid_retrieval),
        ("练习5: RAG评估", exercise_5_rag_evaluation),
    ]

    for name, exercise_func in exercises:
        print(f"\n{'='*60}")
        print(f"{name}")
        print('='*60)
        try:
            exercise_func()
        except NotImplementedError as e:
            print(f"⚠️  待实现: {e}")
        except Exception as e:
            print(f"❌ 错误: {e}")

    print(f"\n{'='*60}")
    print("练习完成！")
    print('='*60)
