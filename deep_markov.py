import torch
import torch.nn as nn


class DeepMarkovModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.rnn = nn.RNN(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden):
        # Propagate data through the recurrent neural network
        rnn_out, hidden = self.rnn(x, hidden)

        # Apply a dense layer to predict the output
        output = self.fc(rnn_out.squeeze(0))

        return output, hidden

    def init_hidden(self):
        # Initialize the hidden state with zeros
        return torch.zeros(1, 1, self.hidden_dim)
        