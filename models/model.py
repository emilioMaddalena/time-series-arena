from abc import ABC, abstractmethod

import numpy as np


class TimeSeriesModel(ABC):
    """A generic class that represents a time series model.

    N.B. The model object owns the train and test data.
    """

    def __init__(self, train: np.ndarray, test: np.ndarray):
        """Store the training and test data."""
        self.train = train
        self.test = test
        self.model = None

    @abstractmethod
    def learn_model(self, train_data: np.ndarray, **kwargs):
        """Learn a model based on some training data and hyperparameters search."""
        pass

    @abstractmethod
    def predict_training_set(self, frac: float) -> np.ndarray:
        """Predict the last frac % of the training time series (in-sample)."""
        pass

    @abstractmethod
    def predict_test_set(self, frac: float) -> np.ndarray:
        """Predict the last frac % of the test time series (out-of-sample)."""
        pass

    @abstractmethod
    def predict(self, context: np.ndarray, steps: int) -> np.ndarray:
        """Predict (generic) N steps ahead given some tim-series context."""
        pass