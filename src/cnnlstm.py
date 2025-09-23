import torch
import torch.nn as nn
import torch.optim as optim


class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(CNNLSTMModel, self).__init__()

        # define the CNN
        self.cnn = nn.Sequential(
            # nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
            nn.Conv1d(input_size, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=1),

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1)
        )

        # define the LSTM
        self.lstm = nn.LSTM(128, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # define the fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # x shape: (batch_size, grf_dim, seq_len)

        # input: (N, C_in, L_in)
        # output: (N, C_out, L_out)
        cnn_out = self.cnn(x)  # cnn_out shape: (batch_size, feature_map_dim, seq_len)

        # permute the dimensions of the output of the CNN
        # (N, C_out, L_out) -> (N, L_out, C_out)
        cnn_out = cnn_out.permute(0, 2, 1)  # cnn_out shape: (batch_size, seq_len, feature_map_dim)

        # batch_first=True, so the input and output tensors are provided as (batch, seq, feature)
        # input: (N, L, H_in), H_in = input_size
        # output: (N, L, H_out), H_out = hidden_size
        lstm_out, _ = self.lstm(cnn_out)  # lstm_out shape: (batch_size, seq_len, hidden_size)

        # input: (N, *, H_in)
        # output: (N, *, H_out)
        output = self.fc(lstm_out)  # output shape: (batch_size, seq_len, output_size)

        return output
