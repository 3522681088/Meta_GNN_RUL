import numpy as np
from evaluation.metrics import regression_metrics

def test_perfect_prediction():
    m=regression_metrics([1,2,3],[1,2,3])
    assert m["rmse"]==0 and m["mae"]==0 and m["nasa_score"]==0 and m["r2"]==1

def test_late_prediction_penalty_is_larger():
    late=regression_metrics([50],[60])["nasa_score"]
    early=regression_metrics([50],[40])["nasa_score"]
    assert late>early

