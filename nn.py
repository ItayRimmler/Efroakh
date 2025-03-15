"""

Script name: "nn.py"\n
Goal of the script: Contains functions used to build and train a NN.\n
Part of project: "Efroakh"\n
Description of project: A video game that programs itself.\n
Ways to contact me if something went wrong in the code: itay.rimmler@gmail.com\n
Uploaded to GitHub in the link: https://github.com/ItayRimmler?tab=repositories\n
Deez: Nuts\n

"""


# Imports
import torch
from torch import nn

class RNN_Model(nn.Module):
    """
    Our RNN model.\n
    """

    def __init__(self, input_size, output_size, hidden_size, number_of_layers):
        """
        Initialization of the RNN.\n
        """
        super(RNN_Model, self).__init__()
        self.hidden_size = hidden_size
        self.number_of_layers = number_of_layers
        self.rnn = nn.LSTM(input_size, hidden_size, number_of_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.do = nn.Dropout(p=0.5)

    def forward(self, x):
        """
        Forward propagation.
        """
        # Apply dropout
        x = self.do(x)

        # Initialize hidden and cell states
        h0 = torch.zeros(self.number_of_layers, self.hidden_size)
        c0 = torch.zeros(self.number_of_layers, self.hidden_size)

        # Pass through LSTM
        out, _ = self.rnn(x, (h0, c0))

        # Pass the output through the fully connected layer
        out = self.fc(out)

        return out
