from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import xgboost as xgb


def create_features(data: np.ndarray, n_lags: int) -> Tuple[np.ndarray]:
    X, y = [], []
    for i in range(n_lags, len(data)):
        X.append(data[i - n_lags : i])
        y.append(data[i])
    return np.array(X), np.array(y)


def train_xgboost(
    data: np.ndarray,
    params: dict,#param_grid: List[Dict[str, Any]],
    num_boost_round: int = 100,
    test_size: float = 0.5,
    n_lags: int = 10,
) -> xgb.Booster:
    X, y = create_features(data, n_lags)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, shuffle=False)
    best_score = float("inf")
    best_model = None
    best_params = None

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    evals = [(dtrain, "train"), (dval, "eval")]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round,
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

    print(f"Best MSE: {best_score}")
    print(f"Best Params: {best_params}")
    return best_model


def predict_n_steps_ahead(model: xgb.Booster, data: np.ndarray, n_steps: int, n_lags: int = 10) -> np.ndarray:
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
    data = data.tolist() # Convert to list for easier manipulation

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