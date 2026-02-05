"""
Day 26-27: Assignment 4 - Data Processing - 代码示例
涵盖：文本处理、去重算法、内容过滤、数据管道
"""

import hashlib
import re
from datasketch import MinHashLSH, MinHash
from typing import List, Dict
import jieba


# ============================================================================
# Part 1: 文本处理
# ============================================================================

def example_1_text_cleaning():
    """示例1: 文本清洗"""
    print("=" * 60)
    print("示例1: 文本清洗")
    print("=" * 60)

    def clean_text(text):
        """清洗文本"""
        # 去除多余空白
        text = re.sub(r'\s+', ' ', text)

        # 去除特殊字符
        text = re.sub(r'[^\u4e00-\u9fff\u0041-\u007a\u0020-\u007e]', '', text)

        # 去除过短行
        lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 10]

        return '\n'.join(lines)

    # 测试
    dirty_text = """
    这是一段  文本。

    包含   多余   空白。

    点击这里查看更多！！！
    """

    cleaned = clean_text(dirty_text)
    print("原始文本:")
    print(dirty_text)
    print("\n清洗后:")
    print(cleaned)


def example_2_html_to_text():
    """示例2: HTML转文本"""
    print("\n" + "=" * 60)
    print("示例2: HTML转文本")
    print("=" * 60)

    # 模拟HTML（实际使用时用真实HTML）
    html = """
    <html>
        <head><title>测试页面</title></head>
        <body>
            <h1>这是标题</h1>
            <p>这是正文内容。</p>
            <script>alert('广告')</script>
        </body>
    </html>
    """

    def html_to_text_simple(html):
        """简单的HTML转文本"""
        # 移除script标签
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)

        # 移除style标签
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)

        # 移除所有HTML标签
        text = re.sub(r'<[^>]+>', '', html)

        # 清理空白
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    text = html_to_text_simple(html)
    print("提取的文本:")
    print(text)


# ============================================================================
# Part 2: 去重算法
# ============================================================================

def example_3_exact_deduplication():
    """示例3: 精确去重"""
    print("\n" + "=" * 60)
    print("示例3: 精确去重")
    print("=" * 60)

    def exact_deduplicate(documents):
        """精确去重（基于MD5哈希）"""
        seen = set()
        unique_docs = []

        for doc in documents:
            # 计算MD5哈希
            doc_hash = hashlib.md5(doc['text'].encode()).hexdigest()

            if doc_hash not in seen:
                seen.add(doc_hash)
                unique_docs.append(doc)

        return unique_docs

    # 测试数据
    docs = [
        {'id': 1, 'text': '这是第一段文档'},
        {'id': 2, 'text': '这是第一段文档'},  # 重复
        {'id': 3, 'text': '这是第二段文档'},
        {'id': 4, 'text': '这是第三段文档'},
    ]

    print(f"原始文档数: {len(docs)}")

    unique_docs = exact_deduplicate(docs)
    print(f"去重后: {len(unique_docs)}")
    print(f"去重率: {(1 - len(unique_docs)/len(docs))*100:.1f}%")


def example_4_minhash_deduplication():
    """示例4: MinHash LSH去重"""
    print("\n" + "=" * 60)
    print("示例4: MinHash LSH去重")
    print("=" * 60)

    def minhash_deduplicate(documents, threshold=0.8):
        """使用MinHash LSH去重"""
        # 创建LSH索引
        lsh = MinHashLSH(threshold=threshold, num_perm=128)

        # 计算MinHash签名
        minhashes = {}
        for idx, doc in enumerate(documents):
            # 分词（使用jieba中文分词）
            words = list(jieba.cut(doc['text']))

            # 计算MinHash
            m = MinHash(num_perm=128)
            for word in words:
                m.update(word.encode())

            lsh.insert(idx, m)
            minhashes[idx] = m

        # 查找重复
        duplicates = set()
        for idx in range(len(documents)):
            result = lsh.query(minhashes[idx])
            for similar_idx in result:
                if similar_idx != idx:
                    print(f"文档{idx}与文档{similar_idx}相似（阈值={threshold}）")
                    duplicates.add(similar_idx)

        # 返回去重后的文档
        return [doc for idx, doc in enumerate(documents) if idx not in duplicates]

    # 测试数据
    docs = [
        {'id': 1, 'text': '机器学习是人工智能的重要分支'},
        {'id': 2, 'text': '机器学习是人工智能的重要分支'},  # 完全相同
        {'id': 3, 'text': '深度学习使用神经网络进行学习'},
        {'id': 4, 'text': '自然语言处理处理文本数据'},
    ]

    print(f"原始文档数: {len(docs)}")

    unique_docs = minhash_deduplicate(docs, threshold=0.7)
    print(f"去重后: {len(unique_docs)}")


# ============================================================================
# Part 3: 内容过滤
# ============================================================================

def example_5_quality_scoring():
    """示例5: 质量评分"""
    print("\n" + "=" * 60)
    print("示例5: 质量评分")
    print("=" * 60)

    def quality_score(text):
        """计算文档质量分数（0-1）"""
        score = 0.0

        # 1. 长度评分
        length = len(text)
        if 50 <= length <= 5000:
            score += 0.3
        elif length > 5000:
            score += 0.2

        # 2. 句子数量
        sentences = [s for s in text.split('。') if len(s.strip()) > 0]
        if 2 <= len(sentences) <= 50:
            score += 0.3

        # 3. 标点符号比例
        punctuation_count = sum(1 for c in text if c in '。，！？；：')
        punctuation_ratio = punctuation_count / len(text) if len(text) > 0 else 0
        if 0.01 <= punctuation_ratio <= 0.1:
            score += 0.2

        # 4. 特殊词检测
        spam_keywords = ['点击', '广告', '推广']
        if not any(kw in text for kw in spam_keywords):
            score += 0.2

        return score

    # 测试
    texts = [
        "这是一段正常文本。它有合理的长度和标点符号。",
        "点击广告",
        "短",  # 太短
        "这是一段较长的文本内容，它包含了多个句子，每个句子都有明确的含义。这样的文本通常质量较高。" * 10,  # 长文本
    ]

    print(f"{'文本':<30} {'长度':<6} {'句子数':<6} {'质量分':<6}")
    print("-" * 50)

    for text in texts:
        score = quality_score(text)
        sentences = [s for s in text.split('。') if s.strip()]
        print(f"{text[:30]:<30} {len(text):<6} {len(sentences):<6} {score:<6.2f}")


# ============================================================================
# Part 4: 完整数据管道
# ============================================================================

def example_6_data_pipeline():
    """示例6: 完整数据处理管道"""
    print("\n" + "=" * 60)
    print("示例6: 完整数据处理管道")
    print("=" * 60)

    class DataPipeline:
        """数据处理管道"""

        def __init__(self):
            self.documents = []

        def load(self, raw_texts):
            """加载原始文本"""
            for idx, text in enumerate(raw_texts):
                self.documents.append({'id': idx, 'text': text})
            print(f"加载: {len(self.documents)}个文档")

        def clean(self):
            """清洗文本"""
            for doc in self.documents:
                # 清洗
                doc['text'] = re.sub(r'\s+', ' ', doc['text'])
                doc['text'] = re.sub(r'[^\u4e00-\u9fff\u0041-\u007a\u0020-\u007e]', '', doc['text'])

            print(f"清洗: {len(self.documents)}个文档")

        def deduplicate(self, method='exact'):
            """去重"""
            if method == 'exact':
                seen = set()
                unique = []
                for doc in self.documents:
                    doc_hash = hashlib.md5(doc['text'].encode()).hexdigest()
                    if doc_hash not in seen:
                        seen.add(doc_hash)
                        unique.append(doc)
                self.documents = unique

            print(f"去重后: {len(self.documents)}个文档")

        def filter_quality(self, min_score=0.6):
            """质量过滤"""
            filtered = []
            for doc in self.documents:
                score = self._quality_score(doc['text'])
                if score >= min_score:
                    doc['quality'] = score
                    filtered.append(doc)

            self.documents = filtered
            print(f"质量过滤后: {len(self.documents)}个文档")

        def _quality_score(self, text):
            """质量评分"""
            score = 0.0
            if 50 <= len(text) <= 5000:
                score += 0.5
            sentences = [s for s in text.split('。') if len(s.strip()) > 0]
            if 2 <= len(sentences) <= 50:
                score += 0.5
            return score

        def process(self, raw_texts):
            """完整处理流程"""
            print("\n数据处理流程:")
            print("-" * 30)
            self.load(raw_texts)
            self.clean()
            self.deduplicate(method='exact')
            self.filter_quality(min_score=0.6)

            return self.documents

    # 测试
    raw_texts = [
        "这是第一段文档。",
        "这是第一段文档。",  # 重复
        "第二段文档内容。",
        "点击广告",  # 低质量
        "x" * 10,  # 太短
        "这是一段较长的正常内容，包含多个句子。" * 3,
    ]

    pipeline = DataPipeline()
    documents = pipeline.process(raw_texts)

    print(f"\n最终结果: {len(documents)}个高质量文档")


# ============================================================================
# Part 5: 中文分词
# ============================================================================

def example_7_chinese_tokenization():
    """示例7: 中文分词"""
    print("\n" + "=" * 60)
    print("示例7: 中文分词")
    print("=" * 60)

    import jieba

    text = "机器学习是人工智能的重要分支"

    # 精确模式
    words_exact = list(jieba.cut(text, cut_all=False))
    print(f"精确模式: {' '.join(words_exact)}")

    # 全模式
    words_all = list(jieba.cut(text, cut_all=True))
    print(f"全模式: {' '.join(words_all)}")

    # 搜索引擎模式
    words_search = list(jieba.cut_for_search(text))
    print(f"搜索引擎模式: {' '.join(words_search)}")


# ============================================================================
# Part 6: 批量处理
# ============================================================================

def example_8_batch_processing():
    """示例8: 批量处理优化"""
    print("\n" + "=" * 60)
    print("示例8: 批量处理优化")
    print("=" * 60)

    from concurrent.futures import ProcessPoolExecutor

    def process_single_html(html_file):
        """处理单个HTML文件"""
        # 模拟处理
        return f"processed_{html_file}"

    def batch_process(html_files, batch_size=3):
        """批量处理"""
        results = []

        for i in range(0, len(html_files), batch_size):
            batch = html_files[i:i+batch_size]

            # 并行处理
            with ProcessPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(process_single_html, f) for f in batch]
                batch_results = [f.result() for f in futures]
                results.extend(batch_results)

            print(f"处理batch {i//batch_size + 1}: {len(batch)}个文件")

        return results

    # 测试
    files = [f"file_{i}.html" for i in range(10)]
    results = batch_process(files, batch_size=3)

    print(f"\n处理完成: {len(results)}个文件")


# ============================================================================
# 运行所有示例
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Assignment 4: Data Processing - 代码示例")
    print("=" * 60)

    example_1_text_cleaning()
    example_2_html_to_text()
    example_3_exact_deduplication()
    example_4_minhash_deduplication()
    example_5_quality_scoring()
    example_6_data_pipeline()
    example_7_chinese_tokenization()
    example_8_batch_processing()

    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)

    print("\n关键要点:")
    print("1. 文本清洗: 去除噪声，保留有用信息")
    print("2. 去重算法: 精确去重 vs MinHash LSH模糊去重")
    print("3. 质量评分: 长度、句子数、标点符号等指标")
    print("4. 数据管道: 加载→清洗→去重→过滤")
    print("5. 中文处理: 使用jieba分词")
    print("6. 批量处理: 并行处理提升效率")
