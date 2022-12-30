import torch
import unittest
from deep_markov import DeepMarkovModel


class TestDeepMarkovModel(unittest.TestCase):
    def test_model_output(self):
        # Define input, hidden and output dimensions
        input_dim = 5
        hidden_dim = 10
        output_dim = 3

        # Create the model object
        model = DeepMarkovModel(input_dim, hidden_dim, output_dim)

        # Initialize the hidden state
        hidden = model.init_hidden()

        # Prepare the input data (a sequence of length 3 with input vectors of dimension 5)
        x = torch.randn(3, 1, input_dim)

        # Propagate the data through the model
        print(x.shape)
        print(hidden.shape)
        output, hidden = model(x, hidden)

        self.assertEqual(output.shape, torch.Size([3, 1, 3]))
        self.assertEqual(hidden.shape, torch.Size([1, 1, 10]))
        