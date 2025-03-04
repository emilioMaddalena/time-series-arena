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
    def predict_training_set(self) -> np.ndarray:
        """Predict on the training set (in-sample).
        
        N.B. These are closed-loop predictions, i.e., the model always has access to 
        the ground-truth values. Its predictions are never fed back into the model.
        """
        pass

    @abstractmethod
    def predict(self, n_steps: int, context: np.ndarray, use_training_context: bool = False) -> np.ndarray:
        """Predict the next n_steps steps given the context.
        
        N.B. These are open-loop predictions, i.e., the model does not have access to
        any new information while predicting, only the provided context. As a result
        its predictions are fed back into the model.

        If use_training_context is True, the model will use the training series as the context.
        """
        pass