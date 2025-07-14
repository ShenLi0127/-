import torch
import torch.nn as nn
import math

class Transformer(nn.Module):
    def __init__(self, input_size=1, output_size=1, hidden_size=128, 
                 num_layers=3, nhead=4, dim_feedforward=512, dropout=0.2, 
                 num_features=1):  

        super(Transformer, self).__init__()
        self.output_size = output_size
        self.num_features = num_features 
        
        self.embedding = nn.Linear(input_size, hidden_size)
        
        self.position_embedding = PositionalEncoding(hidden_size, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            #  output_size * num_features
            nn.Linear(64, output_size * 1)
        ) 
    
    def forward(self, x):
        # x.shape: (batch_size, seq_len, input_size)     
        batch_size = x.size(0)
        
        x = self.embedding(x)  # (batch_size, seq_len, hidden_size)
        x = self.position_embedding(x)
        encoded = self.transformer_encoder(x)  # (batch_size, seq_len, hidden_size)
        last_output = encoded[:, -1, :]  # (batch_size, hidden_size)
        output = self.fc(last_output)  # (batch_size, output_size * num_features)
        
        # (batch_size, output_size, num_features)
        return output.view(batch_size, self.output_size)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *  
            (-math.log(10000.0) / d_model)
        )
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x.shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:x.size(1)].transpose(0, 1) 
        return self.dropout(x)
