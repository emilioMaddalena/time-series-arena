import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from .model import TimeSeriesModel


class FeedForwardNeuralNetwork(TimeSeriesModel):
    """A simple recurrent neutral network.

    This class relies on BaseFeedForwardNeuralNetwork and BaseTimeSeriesDataset.
    """

    def __init__(self, train_series: np.ndarray):
        """Store the training and test data."""
        self.train_series = train_series
        self.model = None

    def learn_model(
        self,
        window_size: int,
        hidden_size: int,
        dropout_rate: float = 0.2,
        batch_size: int = 50,
        num_epochs: int = 2000,
        lr: float = 0.01,
        seed: int = None,
    ):
        """Use mini-batch stochastic gradient descent to learn the neural net."""
        if seed:
            torch.manual_seed(seed)

        # Instantiate a fresh model
        model = BaseFeedForwardNeuralNetwork(window_size, hidden_size, dropout_rate)

        # Loss and optimizer
        loss_function = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        best_loss = float("inf")
        temp_model_file = "temp.pt"

        # Create a DataLoader
        self.train_dataset = BaseTimeSeriesDataset(self.train_series, window_size)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for batch_idx, (x, y) in enumerate(self.train_dataloader):
                x = x.reshape(-1, window_size)
                optimizer.zero_grad()
                pred = model(x)
                loss = loss_function(pred, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_dataloader)
            if avg_loss < best_loss:
                torch.save(model.state_dict(), temp_model_file)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}")

        model.load_state_dict(torch.load(temp_model_file))
        print("Finished Training!")

        torch.set_grad_enabled(False) # gradients not needed anymore
        self.model = model

    def predict_training_set(self):
        """Predict the last frac % of the training time series (in-sample)."""
        #! in the future, just return the predicted values
        y_pred = np.array([])
        y_true = np.array([])

        self.model.eval()
        with torch.no_grad():
            for idx in range(len(self.train_dataset)):
                x, y = self.train_dataset[idx]
                output = self.model(x.reshape(-1, self.model.input_size))
                y_pred = np.concatenate((y_pred, output.numpy().flatten()))
                y_true = np.concatenate((y_true, y.numpy().flatten()))
                
        # Plot the predictions and ground-truth values
        plt.figure(figsize=(12, 6))
        plt.plot(y_pred, label='Predictions')
        plt.plot(y_true, 'k--', label='Ground Truth')
        plt.legend()
        plt.show()

    def predict_test_set(self, test_series: float) -> np.ndarray:
        """Predict the last frac % of the test time series (out-of-sample)."""
        pass


class BaseFeedForwardNeuralNetwork(nn.Module):
    """A simple FF neutral network."""

    def __init__(self, input_size, hidden_size=25, dropout_rate=0.5):
        super().__init__()
        self.input_size = input_size
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        output = self.linear_relu_stack(x)
        return output


class BaseTimeSeriesDataset(Dataset):
    def __init__(self, data, num_lags):
        """
        data: a 1D numpy array or list of time series values.
        num_lags: the number of previous time steps to use as input features.
        """
        self.data = data
        self.num_lags = num_lags

    def __len__(self):
        # Each sample consists of seq_length inputs and 1 target value.
        return len(self.data) - self.num_lags

    def __getitem__(self, idx):
        # Input sequence: from idx to idx+seq_length.
        x = self.data[idx : idx + self.num_lags]
        # Target: the value immediately after the input sequence.
        y = self.data[idx + self.num_lags]
        # Reshape x to (num_lags, 1) and y to (1,)
        x = np.array(x, dtype=np.float32).reshape(-1, 1)
        y = np.array([y], dtype=np.float32)
        return torch.tensor(x), torch.tensor(y)
