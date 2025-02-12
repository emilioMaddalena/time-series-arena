import warnings
from itertools import product
from typing import List, Union

import numpy as np
import statsmodels.api as sm


def train_arima(
    train_data: np.ndarray,
    p_values: Union[int, List],
    d_values: Union[int, List],
    q_values: Union[int, List],
) -> sm.tsa.ARIMA:
    """Fit ARIMA models and return the best model in terms of log likelihood."""
    warnings.filterwarnings("ignore")
    grid_results = []

    # Transform params into lists if necessary
    p_values = [p_values] if isinstance(p_values, int) else p_values
    d_values = [d_values] if isinstance(d_values, int) else d_values
    q_values = [q_values] if isinstance(q_values, int) else q_values

    # Try fitting models and collect log likelihoods
    for p, d, q in product(p_values, d_values, q_values):
        try:
            model = sm.tsa.ARIMA(train_data, order=(p, d, q))
            model_fit = model.fit()
            log_likelihood = model_fit.llf
            grid_results.append((p, d, q, log_likelihood))
        except Exception as e:
            print(f"Failed to fit ARIMA({p}, {d}, {q}): {e}")
            grid_results.append((p, d, q, -float('inf')))

    # Find the best model
    grid_results_array = np.array(grid_results, dtype=object)
    best_index = np.argmax(grid_results_array[:, 3].astype(float))
    best_p, best_d, best_q = grid_results_array[best_index, :3].astype(int)
    best_model = sm.tsa.ARIMA(train_data, order=(best_p, best_d, best_q)).fit()

    print(best_model.summary())
    return best_model
    
