import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ml.model import compute_model_metrics, inference, train_model


def test_train_model_returns_random_forest():
    """
    Test that the train_model function returns a RandomForestClassifier
    instance.
    """
    X_train = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    y = np.array([0, 1, 0, 1])
    model = train_model(X_train, y)
    assert isinstance(model, RandomForestClassifier)


def test_inference_returns_expected_shape():
    """
    Test that the inference function returns a numpy array of predictions
    with one prediction per input row.
    """
    X_train = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    y_train = np.array([0, 1, 0, 1])
    model = train_model(X_train, y_train)
    X_test = np.array([[0, 0], [1, 1], [2, 2]])
    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X_test.shape[0]


def test_compute_model_metrics_perfect_predictions():
    """
    Test that compute_model_metrics returns precision, recall, and fbeta
    of 1.0 for perfect predictions.
    """
    y = np.array([0, 1, 1, 0, 1])
    pred = np.array([0, 1, 1, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y, pred)
    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0
