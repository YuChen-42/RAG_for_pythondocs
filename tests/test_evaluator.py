from src.evaluator import Evaluator


def test_evaluator():
    e = Evaluator()
    assert e.metrics is not None
