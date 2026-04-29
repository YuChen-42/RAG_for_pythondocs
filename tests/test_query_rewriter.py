from src.query_rewriter import QueryRewriter


def test_rewrite():
    qw = QueryRewriter()
    result = qw.rewrite("asyncio usage")
    assert isinstance(result, list)
    assert len(result) > 0
