import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, lstm_output):
        scores = self.attn(lstm_output)  # [B, Seq, 1]
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * lstm_output, dim=1)  # [B, Hidden*2]
        return context

class BiLSTMAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers, dropout):
        super(BiLSTMAttentionModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.ln = nn.LayerNorm(hidden_size*2)
        self.attention = Attention(hidden_size)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), 
            nn.LayerNorm(hidden_size),            
            nn.ReLU(),                               
            nn.Dropout(dropout),                      
            nn.Linear(hidden_size, num_classes)       
        )

    def forward(self, x):
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)  # [B, Seq, Hidden*2]
        out = self.ln(out)
        context = self.attention(out)
        output = self.classifier(context)
        return output
