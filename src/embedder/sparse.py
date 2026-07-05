"""BM25 sparse 向量(hybrid 的稀疏路)。jieba 中文分词 → token,稳定 hash 成 uint32 index,
词频 tf 作 values;打分交给 Qdrant 的 `Modifier.IDF`(它统计 collection 的 doc frequency 算 BM25)。
零模型、纯 CPU。doc 端 values=tf,query 端 values=1。

精确词是选 BM25 的初衷:jieba 会把 'GPT-4'/'v1.2'/编号切碎,所以额外用正则**完整保留**字母数字串,
doc/query 用同一套 token,精确串(型号/法条编号/数字)才能一字不差命中。"""
from __future__ import annotations

import re
from collections import Counter

import jieba
from qdrant_client import models

_MASK = (1 << 32) - 1
_PUNCT = re.compile(r"^[\s\W_]+$", re.UNICODE)
# 精确串:连续字母数字,允许内部 - _ . / 连接(gpt-4, v1.2, 42000000, no.42)
_ALNUM = re.compile(r"[A-Za-z0-9]+(?:[-_./][A-Za-z0-9]+)*")


def _tok_id(tok: str) -> int:
    """稳定 hash(FNV-1a 32-bit)。NOT Python hash()——后者跨进程随机化,doc/query 会对不上。"""
    h = 0x811C9DC5
    for ch in tok.encode("utf-8"):
        h = ((h ^ ch) * 0x01000193) & _MASK
    return h


def tokenize(text: str, stopwords: frozenset[str] = frozenset()) -> list[str]:
    """jieba 切分(中英混合)+ 额外完整保留字母数字精确串。去纯标点/空白/停用词。"""
    text = (text or "").lower()
    toks: list[str] = []
    seen_jieba: set[str] = set()
    for t in jieba.lcut(text, cut_all=False):
        t = t.strip()
        if t and not _PUNCT.match(t) and t not in stopwords:
            toks.append(t)
            seen_jieba.add(t)
    # 只补 jieba **没完整切出**的精确串(gpt-4 被切成 gpt/4 才补)。否则与 jieba 重复 append -> doc tf 双计;
    # 而纯中文 token 不被 _ALNUM 匹配不会双计 -> 字母数字 token 被不对称加权,扭曲 BM25(seal#5)。
    for m in _ALNUM.findall(text):
        if len(m) >= 2 and m not in stopwords and m not in seen_jieba:
            toks.append(m)
    return toks


def doc_sparse(text: str, stopwords: frozenset[str] = frozenset()) -> models.SparseVector | None:
    """文档端:词频 tf 作 values(Qdrant Modifier.IDF 再乘 IDF)。无有效 token -> None(只走稠密路)。"""
    counts = Counter(_tok_id(t) for t in tokenize(text, stopwords))
    if not counts:
        return None
    return models.SparseVector(indices=list(counts.keys()), values=[float(c) for c in counts.values()])


def query_sparse(text: str, stopwords: frozenset[str] = frozenset()) -> models.SparseVector | None:
    """查询端:命中即 1.0(BM25 分由文档 tf × IDF 决定,query 权重取 1)。"""
    ids = {_tok_id(t) for t in tokenize(text, stopwords)}
    if not ids:
        return None
    return models.SparseVector(indices=list(ids), values=[1.0] * len(ids))
