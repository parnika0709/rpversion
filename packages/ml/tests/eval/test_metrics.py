import numpy as np
import pytest
from braintumor.eval.metrics import dice, iou, sensitivity, precision

def test_metrics_perfect_overlap():
    """Test metrics when prediction and target are identical."""
    pred = np.array([0, 1, 0, 1], dtype=np.uint8)
    target = np.array([0, 1, 0, 1], dtype=np.uint8)
    region = [1]

    assert dice(pred, target, region) == 1.0
    assert iou(pred, target, region) == 1.0
    assert sensitivity(pred, target, region) == 1.0
    assert precision(pred, target, region) == 1.0

def test_metrics_no_overlap():
    """Test metrics when there is no overlap."""
    pred = np.array([0, 1, 0, 0], dtype=np.uint8)
    target = np.array([0, 0, 1, 0], dtype=np.uint8)
    region = [1]

    assert dice(pred, target, region) == 0.0
    assert iou(pred, target, region) == 0.0
    assert sensitivity(pred, target, region) == 0.0
    assert precision(pred, target, region) == 0.0

def test_metrics_empty_masks():
    """Test metrics when masks are empty."""
    pred = np.zeros(10, dtype=np.uint8)
    target = np.zeros(10, dtype=np.uint8)
    region = [1]

    assert np.isnan(dice(pred, target, region))
    assert np.isnan(iou(pred, target, region))
    assert np.isnan(sensitivity(pred, target, region))
    assert np.isnan(precision(pred, target, region))
