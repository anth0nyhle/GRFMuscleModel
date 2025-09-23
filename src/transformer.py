import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout):
        super(TransformerModel, self).__init__()

        self.input_embedding = nn.Linear(input_dim, d_model)
        # self.positional_encoding = nn.Parameter(self._generate_positional_encoding(d_model, max_len=100))
        self.register_buffer("positional_encoding", self._generate_positional_encoding(d_model, max_len=100))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, input_dim)
        seq_len = x.size(1)
        x = self.input_embedding(x)  # (batch_size, sequence_length, d_model)

        x = x + self.positional_encoding[:, :seq_len, :].to(x.device)  # Add positional encoding

        x = self.transformer_encoder(x)  # (sequence_length, batch_size, d_model)

        output = self.output_layer(x)  # (batch_size, sequence_length, output_dim)

        return output

    @staticmethod
    def _generate_positional_encoding(d_model, max_len=100):
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log1p(torch.tensor(10000.0)) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)  # Add batch dimension
