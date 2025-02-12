from itertools import product
from typing import Any, Dict, Iterable, List, Tuple, Union

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def _create_features(data: np.ndarray, n_lags: int) -> Tuple[np.ndarray]:
    X, y = [], []
    for i in range(n_lags, len(data)):
        X.append(data[i - n_lags : i])
        y.append(data[i])
    return np.array(X), np.array(y)


def _ensure_iterable(param: Union[int, float, Iterable]) -> Iterable:
    return param if isinstance(param, Iterable) else [param]


def learn_xgboost(
    data: np.ndarray,
    test_size: float = 0.5,
    n_lags: int = 10,
    num_boost_round: Union[int, Iterable[int]] = 100,
    max_depth: Union[int, Iterable[int]] = 20,
    eta: Union[float, Iterable[float]] = 0.3,
    subsample: Union[float, Iterable[float]] = 0.8,
    colsample_bytree: Union[float, Iterable[float]] = 0.8,
) -> xgb.Booster:
    """Learn an XGBoost model using the given data and a range of hyperparameters."""
    # Ensure all parameters are iterable
    num_boost_round = _ensure_iterable(num_boost_round)
    max_depth = _ensure_iterable(max_depth)
    eta = _ensure_iterable(eta)
    subsample = _ensure_iterable(subsample)
    colsample_bytree = _ensure_iterable(colsample_bytree)

    X, y = _create_features(data, n_lags)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, shuffle=False)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    evals = [(dtrain, "train"), (dval, "eval")]

    best_score = float("inf")
    best_model = None
    best_params = None
    for num_boost_round_val, max_depth_val, eta_val, subsample_val, colsample_bytree_val in product(
        num_boost_round, max_depth, eta, subsample, colsample_bytree
    ):
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": max_depth_val,
            "eta": eta_val,
            "subsample": subsample_val,
            "colsample_bytree": colsample_bytree_val,
        }
        model = xgb.train(
            params,
            dtrain,
            num_boost_round_val,
            evals=evals,
            early_stopping_rounds=10,
            verbose_eval=False,
        )
        preds = model.predict(dval)
        mse = mean_squared_error(y_val, preds)
        if mse < best_score:
            best_score = mse
            best_params = params
            best_model = model

    print(f"Best params: {best_params}")
    print(f"Best MSE: {best_score}")
    return best_model


def predict_n_steps_ahead(
    model: xgb.Booster, data: np.ndarray, n_steps: int, n_lags: int = 10
) -> np.ndarray:
    """Predict N steps ahead using the trained XGBoost model, feeding back its own predictions.

    Parameters:
    - model: Trained XGBoost model.
    - data: Original time series data as a NumPy array.
    - n_lags: Number of lags used in the model.
    - n_steps: Number of steps to predict ahead.

    Returns:
    - predictions: Predicted values as a NumPy array.
    """
    predictions = []
    data = data.tolist()  # Convert to list for easier manipulation

    for _ in range(n_steps):
        # Create the latest feature set
        X = np.array(data[-n_lags:]).reshape(1, -1)
        dmatrix = xgb.DMatrix(X)

        # Predict the next value
        next_pred = model.predict(dmatrix)[0]

        # Append the prediction to the results and the data
        predictions.append(next_pred)
        data.append(next_pred)

    return np.array(predictions)
