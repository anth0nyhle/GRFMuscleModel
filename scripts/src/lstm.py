import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        # call the parent class constructor
        super(LSTMModel, self).__init__()

        # define the LSTM layer
        # nn.LSTM(input_size, hidden_size, num_layers=1, bias=True, batch_first=False,
        # dropout=0.0, bidirectional=False, proj_size=0, device=None, dtype=None)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)

        # fully connected layer to map from hidden state to output features
        # nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # pass through LSTM layer
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch, seq_length, hidden_size)

        # pass through fully connected layer to get the output
        output = self.fc(lstm_out)  # output shape: (batch, seq_length, output_size)

        return output
