"""
Day 26-27: Assignment 4 - Data Processing - 练习题
难度：⭐⭐ ~ ⭐⭐⭐⭐
"""

import hashlib
import re
from datasketch import MinHashLSH, MinHash
from typing import List, Dict


# 练习1: 实现文本清洗函数（⭐⭐）
def exercise_1_text_cleaning():
    """实现文本清洗函数"""
    # TODO: 实现clean_text函数
    # 要求：去除多余空白、特殊字符、过短行
    raise NotImplementedError


# 练习2: 实现精确去重（⭐⭐）
def exercise_2_exact_dedup():
    """实现基于MD5的精确去重"""
    # TODO: 实现exact_deduplicate函数
    # 提示：使用hashlib.md5
    raise NotImplementedError


# 练习3: 实现MinHash去重（⭐⭐⭐⭐）
def exercise_3_minhash_dedup():
    """实现MinHash LSH去重"""
    # TODO: 实现minhash_deduplicate函数
    # 提示：
    # 1. 使用jieba分词
    # 2. 计算MinHash签名
    # 3. 使用LSH查找相似文档
    raise NotImplementedError


# 练习4: 实现质量评分（⭐⭐⭐）
def exercise_4_quality_score():
    """实现文档质量评分"""
    # TODO: 实现quality_score函数
    # 评分维度：
    # - 文本长度（50-5000字符）
    # - 句子数量（2-50个）
    # - 标点符号比例（1%-10%）
    raise NotImplementedError


# 练习5: 实现数据管道（⭐⭐⭐⭐⭐）
def exercise_5_data_pipeline():
    """实现完整数据处理管道"""
    class DataPipeline:
        def __init__(self):
            self.documents = []

        def load(self, raw_texts):
            # TODO: 加载原始文本
            raise NotImplementedError

        def clean(self):
            # TODO: 清洗文本
            raise NotImplementedError

        def deduplicate(self):
            # TODO: 去重（支持精确和MinHash）
            raise NotImplementedError

        def filter_quality(self, min_score=0.6):
            # TODO: 质量过滤
            raise NotImplementedError

    # 测试
    pipeline = DataPipeline()
    # TODO: 测试你的实现
    raise NotImplementedError


if __name__ == "__main__":
    print("Data Processing - 练习题")
    # 运行各练习
    print("完成所有练习！")
