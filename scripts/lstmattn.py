import torch
import torch.nn as nn
import torch.optim as optim


class LSTMAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, output_size, lstm_dropout=0.0, attn_dropout=0.0):
        super(LSTMAttentionModel, self).__init__()
        
        # define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=lstm_dropout)
        
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=attn_dropout)
        
        # fully connected layer to map from hidden state to output features
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # pass through LSTM layer
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch, seq_length, hidden_size)
        lstm_out = lstm_out.permute(1, 0, 2)  # lstm_out shape: (seq_length, batch, hidden_size)
        
        # Self-attention forward pass
        attention_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attention_out = attention_out.permute(1, 0, 2) # attention_out shape: (batch, seq_length, hidden_size)
        
        # pass through fully connected layer to get the output
        output = self.fc(attention_out) # output shape: (batch, seq_length, output_size)        
        return output
