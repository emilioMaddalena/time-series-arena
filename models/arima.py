import warnings
from itertools import product
from typing import Iterable, Union

import numpy as np
import statsmodels.api as sm

from .model import TimeSeriesModel


class ArimaModel(TimeSeriesModel):  # noqa: D101
    def __init__(self):  # noqa: D107
        super().__init__()
    
    def learn_model(
        self,
        train_data: np.ndarray,
        p_values: Union[int, Iterable[int]],
        d_values: Union[int, Iterable[int]],
        q_values: Union[int, Iterable[int]],
    ) -> sm.tsa.ARIMA:
        """Fit ARIMA models and register the best model in terms of log likelihood."""
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
        self.model = best_model

    def train_results(self, context: np.ndarray, steps: int) -> np.ndarray:

        return self.model.get_forecast(steps)
    
    # See ARIMA get_prediction

# #! To be refined
# def rolling_forecast_arima(
#     initial_train_data: np.ndarray,
#     test_data: np.ndarray,
#     best_model: sm.tsa.ARIMA,
#     steps: int
# ) -> np.ndarray:
#     """Perform rolling forecast with ARIMA model without retraining."""
#     history = list(initial_train_data)
#     predictions = []

#     # Perform rolling forecast
#     for i in range(0, len(test_data), steps):
#         # Forecast next 'steps' observations
#         forecast = best_model.get_forecast(steps=steps)
#         forecast_values = forecasts.predicted_mean
#         predictions.extend(forecast_values)
#         # Update history with new observations
#         history.extend(test_data[i:i + steps])
#         # Update the model with new observations
#         best_model = best_model.append(test_data[i:i + steps], refit=False)

#     return np.array(predictions)