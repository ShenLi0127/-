import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_dim = input_dim
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim * 2, num_layers + 1, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):

        h_dim = self.hidden_dim * 2
        h0 = torch.zeros(self.num_layers + 1, 
                         x.size(0), h_dim).to(x.device)
        c0 = torch.zeros(self.num_layers + 1, 
                         x.size(0), h_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        out = self.dropout1(out[:, -1, :])
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        
        return out
