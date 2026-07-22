"""BM25 sparse 单测(纯 CPU)。重点验精确串保留——选 BM25 的初衷。"""


from embedder.sparse import _tok_id, doc_sparse, query_sparse, tokenize


def test_exact_string_preserved():
    # 精确串(型号/版本/编号/数字)不能被 jieba 切碎,否则 BM25 一字不差命中的优势就废了
    toks = tokenize("使用 GPT-4 和 v1.2,参考第42条,营收 42000000 元")
    assert "gpt-4" in toks, toks
    assert "v1.2" in toks, toks
    assert "42000000" in toks, toks


def test_doc_query_consistent_hash():
    # doc 与 query 同 token -> 同 hash index(否则 hybrid 的 sparse 路对不上)
    d = doc_sparse("公司营收 42000000 元人民币")
    q = query_sparse("42000000")
    assert d is not None and q is not None
    assert set(q.indices) & set(d.indices), "精确串 42000000 在 doc/query 对不上"


def test_empty_returns_none():
    assert doc_sparse("") is None
    assert doc_sparse("。,、 ") is None          # 纯标点 -> 无有效 token


def test_no_double_count_tf():
    # seal#5: jieba 已完整切出的字母数字 token 不被正则重复 append -> doc tf 不双计(否则字母数字相对中文被不对称加权)
    assert tokenize("model model").count("model") == 2   # 真实 2 次(jieba),不是 4(jieba+正则)
    # gpt-4 被 jieba 切碎(gpt/4),正则补回 gpt-4 -> tf=1(真实出现一次),不被额外加权
    d = doc_sparse("gpt-4")
    gid = _tok_id("gpt-4")
    assert d.values[list(d.indices).index(gid)] == 1.0, (d.indices, d.values)


if __name__ == "__main__":
    test_exact_string_preserved()
    test_doc_query_consistent_hash()
    test_empty_returns_none()
    test_no_double_count_tf()
    print("sparse tests OK")
    print("sample tokens:", tokenize("第42条 GPT-4 营收42000000 净利润"))
