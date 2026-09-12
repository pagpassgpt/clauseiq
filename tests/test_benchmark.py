from eval.benchmark_v2 import build_dataset, metrics

def test_benchmark_dataset_has_holdout_and_paraphrases():
    clauses, cases = build_dataset()
    assert len(clauses) >= 50
    assert len(cases) >= 60
    assert any(c.split == 'holdout' for c in cases)
    assert all(len(c.query.split()) >= 2 for c in cases)

def test_metrics_known_rank():
    assert metrics(['x','target','z'], 'target') == (0.0, 1.0, 1.0, 0.5, 1.0/1.584962500721156)
