from abc import ABC, abstractmethod

import numpy as np


class TimeSeriesModel(ABC):
    """A generic class that represents a time series model.

    N.B. The model object only owns the train data.
    """

    def __init__(self, train_series: np.ndarray):
        """Store the training data."""
        self.train_series = train_series
        self.model = None

    @abstractmethod
    def learn_model(self, **kwargs):
        """Learn a model based on the training data (attribute) and hyperparameters."""
        pass

    @abstractmethod
    def predict_training_set(self):
        """Predict on the training set (in-sample)."""
        pass

    @abstractmethod
    def predict_test_set(self, test_series: np.ndarray) -> np.ndarray:
        """Predict on the test set (out-of-sample)."""
        pass