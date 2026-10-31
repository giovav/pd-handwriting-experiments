"""
baseline_models.py: le reti del classificatore di partenza (src/models.py), invariate.

Il benchmark usa solo CNN_BiLSTMModel (classifier_adapter.py); CNN1D, BiLSTM e
CNN_BiLSTM_MLP restano come nel file originale. Ingresso [B, C, T], uscita [B]: un logit
per finestra.
"""

import torch
import torch.nn as nn

class CNN1DModel(nn.Module):
    def __init__(self, input_channels):
        super(CNN1DModel, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1) 
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1) 
        x = self.fc(x)
        return x.squeeze(1)


class BiLSTMModel(nn.Module):
    def __init__(self, input_features, hidden_size=64, num_layers=2):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_features, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.3
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        out, (hn, cn) = self.lstm(x)
        
        hidden_fwd = hn[-2, :, :]
        hidden_bwd = hn[-1, :, :]
        hidden_concat = torch.cat([hidden_fwd, hidden_bwd], dim=1)
        
        out = self.fc(hidden_concat)
        return out.squeeze(1)
    
    
class CNN_BiLSTMModel(nn.Module):
    def __init__(self, input_channels, hidden_size=64, num_layers=1):
        super(CNN_BiLSTMModel, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.lstm = nn.LSTM(
            input_size=64, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.conv_blocks(x) 
        x = x.transpose(1, 2) 
        
        lstm_out, (hn, cn) = self.lstm(x)
        
        hidden_fwd = hn[-2, :, :]
        hidden_bwd = hn[-1, :, :]
        hidden_concat = torch.cat([hidden_fwd, hidden_bwd], dim=1)
        
        out = self.fc(hidden_concat)
        return out.squeeze(1)


class CNN_BiLSTM_MLPModel(nn.Module):
    def __init__(self, input_channels, hidden_size=64, num_layers=1):
        super(CNN_BiLSTM_MLPModel, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.lstm = nn.LSTM(
            input_size=64, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0.0
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.conv_blocks(x) 
        x = x.transpose(1, 2) 
        
        lstm_out, (hn, cn) = self.lstm(x)
        
        hidden_fwd = hn[-2, :, :]
        hidden_bwd = hn[-1, :, :]
        hidden_concat = torch.cat([hidden_fwd, hidden_bwd], dim=1)
        
        out = self.mlp(hidden_concat)
        return out.squeeze(1)