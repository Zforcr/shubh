import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, input_size=2049, hidden_size=128, num_layers=2, num_classes=3):
        super(RNNModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, hidden):
        # RNN forward pass
        out, hidden = self.rnn(x, hidden)

        # Take the last output in the sequence
        out = out[:, -1, :]  # Shape: (batch, hidden_size)

        # Fully connected layer
        out = self.fc(out)  # Shape: (batch, num_classes)

        return out, hidden

    def initHidden(self, batch_size):
        # Initialize hidden state with zeros
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

