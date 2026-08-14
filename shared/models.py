"""
shared/models.py
================
PyTorch implementations of all model architectures, including:
- Baseline cCNN & 6 cCNN ablation variants (A1-A6)
- Baseline cRNN & 6 cRNN ablation variants (B1-B6)
- Existing models: AlexNet, VGGNet, LSTM
- New models: CNN_LSTM, CNN_BiLSTM, Swin Transformer
- ResBiLSTM: Residual + Bidirectional LSTM (Zhao et al., 2024)
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _same_pad_3d(kernel: int) -> int:
    return kernel // 2


# ===========================================================================
# cCNN Baseline & Ablations (A1 - A6)
# ===========================================================================

# Baseline cCNN
class CCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1   = nn.LazyConv3d(32, kernel_size=3)
        self.bn1     = nn.LazyBatchNorm3d()
        self.pool1   = nn.MaxPool3d(kernel_size=2, stride=2)
        self.drop1   = nn.Dropout(0.5)
        self.fc1     = nn.LazyLinear(128)
        self.drop2   = nn.Dropout(0.5)
        self.fc_out  = nn.LazyLinear(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        return F.softmax(self.fc_out(x), dim=1)


# A1: Remove MaxPooling
class CCNN_A1_NoPooling(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1   = nn.LazyConv3d(32, kernel_size=3)
        self.bn1     = nn.LazyBatchNorm3d()
        self.drop1   = nn.Dropout(0.5)
        self.fc1     = nn.LazyLinear(128)
        self.drop2   = nn.Dropout(0.5)
        self.fc_out  = nn.LazyLinear(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        return F.softmax(self.fc_out(x), dim=1)


# A2: Remove Dropout
class CCNN_A2_NoDropout(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1   = nn.LazyConv3d(32, kernel_size=3)
        self.bn1     = nn.LazyBatchNorm3d()
        self.pool1   = nn.MaxPool3d(kernel_size=2, stride=2)
        self.fc1     = nn.LazyLinear(128)
        self.fc_out  = nn.LazyLinear(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc_out(x), dim=1)


# A3: Dense=64 instead of 128
class CCNN_A3_Dense64(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1   = nn.LazyConv3d(32, kernel_size=3)
        self.bn1     = nn.LazyBatchNorm3d()
        self.pool1   = nn.MaxPool3d(kernel_size=2, stride=2)
        self.drop1   = nn.Dropout(0.5)
        self.fc1     = nn.LazyLinear(64)
        self.drop2   = nn.Dropout(0.5)
        self.fc_out  = nn.LazyLinear(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        return F.softmax(self.fc_out(x), dim=1)


# A4: Conv Filters=16
class CCNN_A4_Filters16(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1   = nn.LazyConv3d(16, kernel_size=3)
        self.bn1     = nn.LazyBatchNorm3d()
        self.pool1   = nn.MaxPool3d(kernel_size=2, stride=2)
        self.drop1   = nn.Dropout(0.5)
        self.fc1     = nn.LazyLinear(128)
        self.drop2   = nn.Dropout(0.5)
        self.fc_out  = nn.LazyLinear(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        return F.softmax(self.fc_out(x), dim=1)


# A5: Conv Filters=64
class CCNN_A5_Filters64(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1   = nn.LazyConv3d(64, kernel_size=3)
        self.bn1     = nn.LazyBatchNorm3d()
        self.pool1   = nn.MaxPool3d(kernel_size=2, stride=2)
        self.drop1   = nn.Dropout(0.5)
        self.fc1     = nn.LazyLinear(128)
        self.drop2   = nn.Dropout(0.5)
        self.fc_out  = nn.LazyLinear(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        return F.softmax(self.fc_out(x), dim=1)


# A6: Kernel 5x5x5 instead of 3x3x3
class CCNN_A6_Kernel5(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1   = nn.LazyConv3d(32, kernel_size=5)
        self.bn1     = nn.LazyBatchNorm3d()
        self.pool1   = nn.MaxPool3d(kernel_size=2, stride=2)
        self.drop1   = nn.Dropout(0.5)
        self.fc1     = nn.LazyLinear(128)
        self.drop2   = nn.Dropout(0.5)
        self.fc_out  = nn.LazyLinear(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        return F.softmax(self.fc_out(x), dim=1)


# ===========================================================================
# cRNN Baseline & Ablations (B1 - B6)
# ===========================================================================

# Baseline cRNN
class RNNModel(nn.Module):
    def __init__(self, num_classes: int, time_steps: int, input_dim: int):
        super().__init__()
        self.time_steps = time_steps
        self.input_dim  = input_dim
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=256, num_layers=4,
                          batch_first=True, nonlinearity="tanh", dropout=0.3)
        self.drop1 = nn.Dropout(0.3)
        self.fc1 = nn.LazyLinear(128)
        self.drop2 = nn.Dropout(0.3)
        self.fc_out = nn.LazyLinear(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), self.time_steps, self.input_dim)
        x, _ = self.rnn(x)           
        x = x[:, -1, :]              # take last time step
        x = torch.flatten(x, 1)
        x = self.drop1(x)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        return F.softmax(self.fc_out(x), dim=1)


# B1: One RNN layer only
class CRNN_B1_OneLayer(nn.Module):
    def __init__(self, num_classes: int, time_steps: int, input_dim: int):
        super().__init__()
        self.time_steps = time_steps
        self.input_dim  = input_dim
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=256, num_layers=1,
                          batch_first=True, nonlinearity="tanh", dropout=0.0)
        self.drop1 = nn.Dropout(0.3)
        self.fc1 = nn.LazyLinear(128)
        self.drop2 = nn.Dropout(0.3)
        self.fc_out = nn.LazyLinear(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), self.time_steps, self.input_dim)
        x, _ = self.rnn(x)           
        x = x[:, -1, :]
        x = torch.flatten(x, 1)
        x = self.drop1(x)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        return F.softmax(self.fc_out(x), dim=1)


# B2: Two RNN layers
class CRNN_B2_TwoLayers(nn.Module):
    def __init__(self, num_classes: int, time_steps: int, input_dim: int):
        super().__init__()
        self.time_steps = time_steps
        self.input_dim  = input_dim
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=256, num_layers=2,
                          batch_first=True, nonlinearity="tanh", dropout=0.3)
        self.drop1 = nn.Dropout(0.3)
        self.fc1 = nn.LazyLinear(128)
        self.drop2 = nn.Dropout(0.3)
        self.fc_out = nn.LazyLinear(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), self.time_steps, self.input_dim)
        x, _ = self.rnn(x)           
        x = x[:, -1, :]
        x = torch.flatten(x, 1)
        x = self.drop1(x)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        return F.softmax(self.fc_out(x), dim=1)


# B3: No Dropout
class CRNN_B3_NoDropout(nn.Module):
    def __init__(self, num_classes: int, time_steps: int, input_dim: int):
        super().__init__()
        self.time_steps = time_steps
        self.input_dim  = input_dim
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=256, num_layers=4,
                          batch_first=True, nonlinearity="tanh", dropout=0.0)
        self.fc1 = nn.LazyLinear(128)
        self.fc_out = nn.LazyLinear(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), self.time_steps, self.input_dim)
        x, _ = self.rnn(x)           
        x = x[:, -1, :]
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc_out(x), dim=1)


# B4: 1024 units instead of 256
class CRNN_B4_Hidden1024(nn.Module):
    def __init__(self, num_classes: int, time_steps: int, input_dim: int):
        super().__init__()
        self.time_steps = time_steps
        self.input_dim  = input_dim
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=1024, num_layers=4,
                          batch_first=True, nonlinearity="tanh", dropout=0.3)
        self.drop1 = nn.Dropout(0.3)
        self.fc1 = nn.LazyLinear(128)
        self.drop2 = nn.Dropout(0.3)
        self.fc_out = nn.LazyLinear(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), self.time_steps, self.input_dim)
        x, _ = self.rnn(x)           
        x = x[:, -1, :]
        x = torch.flatten(x, 1)
        x = self.drop1(x)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        return F.softmax(self.fc_out(x), dim=1)


# B5: 512 units
class CRNN_B5_Hidden512(nn.Module):
    def __init__(self, num_classes: int, time_steps: int, input_dim: int):
        super().__init__()
        self.time_steps = time_steps
        self.input_dim  = input_dim
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=512, num_layers=4,
                          batch_first=True, nonlinearity="tanh", dropout=0.3)
        self.drop1 = nn.Dropout(0.3)
        self.fc1 = nn.LazyLinear(128)
        self.drop2 = nn.Dropout(0.3)
        self.fc_out = nn.LazyLinear(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), self.time_steps, self.input_dim)
        x, _ = self.rnn(x)           
        x = x[:, -1, :]
        x = torch.flatten(x, 1)
        x = self.drop1(x)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        return F.softmax(self.fc_out(x), dim=1)


# B6: Replace SimpleRNN with GRU
class CRNN_B6_GRU(nn.Module):
    def __init__(self, num_classes: int, time_steps: int, input_dim: int):
        super().__init__()
        self.time_steps = time_steps
        self.input_dim  = input_dim
        self.gru = nn.GRU(input_size=input_dim, hidden_size=256, num_layers=4,
                          batch_first=True, dropout=0.3)
        self.drop1 = nn.Dropout(0.3)
        self.fc1 = nn.LazyLinear(128)
        self.drop2 = nn.Dropout(0.3)
        self.fc_out = nn.LazyLinear(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), self.time_steps, self.input_dim)
        x, _ = self.gru(x)           
        x = x[:, -1, :]
        x = torch.flatten(x, 1)
        x = self.drop1(x)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        return F.softmax(self.fc_out(x), dim=1)


# ===========================================================================
# Existing Models: AlexNet, VGGNet, LSTM
# ===========================================================================

class AlexNet3D(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1  = nn.LazyConv3d(96, kernel_size=11, stride=4)
        self.bn1    = nn.LazyBatchNorm3d()
        self.pool1  = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.drop1  = nn.Dropout(0.25)

        self.conv2  = nn.LazyConv3d(256, kernel_size=5, stride=1, padding=_same_pad_3d(5))
        self.bn2    = nn.LazyBatchNorm3d()
        self.pool2  = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.drop2  = nn.Dropout(0.25)

        self.conv3  = nn.LazyConv3d(384, kernel_size=3, stride=1, padding=_same_pad_3d(3))
        self.bn3    = nn.LazyBatchNorm3d()
        self.conv4  = nn.LazyConv3d(256, kernel_size=3, stride=1, padding=_same_pad_3d(3))
        self.bn4    = nn.LazyBatchNorm3d()
        self.pool3  = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.drop3  = nn.Dropout(0.25)

        self.fc1    = nn.LazyLinear(4096)
        self.drop4  = nn.Dropout(0.5)
        self.fc2    = nn.LazyLinear(4096)
        self.drop5  = nn.Dropout(0.5)
        self.fc_out = nn.LazyLinear(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x))); x = self.pool1(x); x = self.drop1(x)
        x = F.relu(self.bn2(self.conv2(x))); x = self.pool2(x); x = self.drop2(x)
        x = F.relu(self.bn3(self.conv3(x))); x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool3(x); x = self.drop3(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x)); x = self.drop4(x)
        x = F.relu(self.fc2(x)); x = self.drop5(x)
        return F.softmax(self.fc_out(x), dim=1)


class VGGNet3D(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1_1 = nn.LazyConv3d(64, kernel_size=3, padding=1)
        self.bn1_1   = nn.LazyBatchNorm3d()
        self.conv1_2 = nn.LazyConv3d(64, kernel_size=3, padding=1)
        self.bn1_2   = nn.LazyBatchNorm3d()
        self.pool1   = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv2_1 = nn.LazyConv3d(128, kernel_size=3, padding=1)
        self.bn2_1   = nn.LazyBatchNorm3d()
        self.conv2_2 = nn.LazyConv3d(128, kernel_size=3, padding=1)
        self.bn2_2   = nn.LazyBatchNorm3d()
        self.pool2   = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3_1 = nn.LazyConv3d(256, kernel_size=3, padding=1)
        self.bn3_1   = nn.LazyBatchNorm3d()
        self.conv3_2 = nn.LazyConv3d(256, kernel_size=3, padding=1)
        self.bn3_2   = nn.LazyBatchNorm3d()
        self.conv3_3 = nn.LazyConv3d(256, kernel_size=3, padding=1)
        self.bn3_3   = nn.LazyBatchNorm3d()
        self.pool3   = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv4_1 = nn.LazyConv3d(512, kernel_size=3, padding=1)
        self.bn4_1   = nn.LazyBatchNorm3d()
        self.conv4_2 = nn.LazyConv3d(512, kernel_size=3, padding=1)
        self.bn4_2   = nn.LazyBatchNorm3d()
        self.conv4_3 = nn.LazyConv3d(512, kernel_size=3, padding=1)
        self.bn4_3   = nn.LazyBatchNorm3d()
        self.pool4   = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv5_1 = nn.LazyConv3d(512, kernel_size=3, padding=1)
        self.bn5_1   = nn.LazyBatchNorm3d()
        self.conv5_2 = nn.LazyConv3d(512, kernel_size=3, padding=1)
        self.bn5_2   = nn.LazyBatchNorm3d()
        self.conv5_3 = nn.LazyConv3d(512, kernel_size=3, padding=1)
        self.bn5_3   = nn.LazyBatchNorm3d()
        self.pool5   = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.fc1    = nn.LazyLinear(4096)
        self.drop1  = nn.Dropout(0.5)
        self.fc2    = nn.LazyLinear(4096)
        self.drop2  = nn.Dropout(0.5)
        self.fc_out = nn.LazyLinear(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1_1(self.conv1_1(x))); x = F.relu(self.bn1_2(self.conv1_2(x))); x = self.pool1(x)
        x = F.relu(self.bn2_1(self.conv2_1(x))); x = F.relu(self.bn2_2(self.conv2_2(x))); x = self.pool2(x)
        x = F.relu(self.bn3_1(self.conv3_1(x))); x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = F.relu(self.bn3_3(self.conv3_3(x))); x = self.pool3(x)
        x = F.relu(self.bn4_1(self.conv4_1(x))); x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = F.relu(self.bn4_3(self.conv4_3(x))); x = self.pool4(x)
        x = F.relu(self.bn5_1(self.conv5_1(x))); x = F.relu(self.bn5_2(self.conv5_2(x)))
        x = F.relu(self.bn5_3(self.conv5_3(x))); x = self.pool5(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x)); x = self.drop1(x)
        x = F.relu(self.fc2(x)); x = self.drop2(x)
        return F.softmax(self.fc_out(x), dim=1)


class LSTMModel(nn.Module):
    def __init__(self, num_classes: int, time_steps: int, input_dim: int):
        super().__init__()
        self.time_steps = time_steps
        self.input_dim  = input_dim
        self.lstm   = nn.LSTM(input_size=input_dim, hidden_size=256, num_layers=4,
                               batch_first=True, bidirectional=True, dropout=0.3)
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=2, batch_first=True)
        self.drop1  = nn.Dropout(0.3)
        self.fc1    = nn.LazyLinear(128)
        self.drop2  = nn.Dropout(0.3)
        self.fc_out = nn.LazyLinear(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), self.time_steps, self.input_dim)
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        x = attn_out.mean(dim=1)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        return F.softmax(self.fc_out(x), dim=1)


# ===========================================================================
# New Models: CNN_LSTM, CNN_BiLSTM, Swin Transformer
# ===========================================================================

class CNN_LSTM(nn.Module):
    """
    3D CNN Feature Extractor followed by LSTM sequence processing.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.LazyConv3d(32, kernel_size=3, padding=1)
        self.bn1   = nn.LazyBatchNorm3d()
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        
        self.conv2 = nn.LazyConv3d(64, kernel_size=3, padding=1)
        self.bn2   = nn.LazyBatchNorm3d()
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        
        self.lstm  = nn.LSTM(input_size=64, hidden_size=128, num_layers=2,
                             batch_first=True, bidirectional=False, dropout=0.2)
        self.drop  = nn.Dropout(0.3)
        self.fc1   = nn.LazyLinear(64)
        self.fc_out= nn.LazyLinear(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, 1, D, H, W)
        b, c, d, h, w = x.shape
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # Spatial average pool -> (B, C, D, 1, 1) -> (B, D, C)
        x = F.adaptive_avg_pool3d(x, (d, 1, 1)).squeeze(-1).squeeze(-1) # (B, C, D)
        x = x.permute(0, 2, 1) # (B, D, C) = (B, sequence_length, features)
        
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :] # (B, 128)
        
        x = self.drop(F.relu(self.fc1(last_step)))
        return F.softmax(self.fc_out(x), dim=1)


class CNN_BiLSTM(nn.Module):
    """
    3D CNN Feature Extractor followed by Bidirectional LSTM.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.LazyConv3d(32, kernel_size=3, padding=1)
        self.bn1   = nn.LazyBatchNorm3d()
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        
        self.conv2 = nn.LazyConv3d(64, kernel_size=3, padding=1)
        self.bn2   = nn.LazyBatchNorm3d()
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        
        self.bilstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2,
                              batch_first=True, bidirectional=True, dropout=0.2)
        self.drop   = nn.Dropout(0.3)
        self.fc1    = nn.LazyLinear(64)
        self.fc_out = nn.LazyLinear(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.adaptive_avg_pool3d(x, (d, 1, 1)).squeeze(-1).squeeze(-1)
        x = x.permute(0, 2, 1) # (B, D, C)
        
        lstm_out, _ = self.bilstm(x) # (B, D, 256)
        last_step = lstm_out[:, -1, :]
        
        x = self.drop(F.relu(self.fc1(last_step)))
        return F.softmax(self.fc_out(x), dim=1)


# Clean, self-contained Vision/Swin Transformer Architecture
class SwinBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class Swin_Transformer(nn.Module):
    """
    Patch-based Vision/Swin Transformer tailored for 3D spectrogram volume frames.
    Extracts patch tokens per frame, applies Swin Transformer blocks, and pools across time/space.
    """
    def __init__(self, num_classes: int, embed_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_conv = nn.LazyConv2d(embed_dim, kernel_size=4, stride=4)
        self.block1 = SwinBlock(embed_dim, num_heads)
        self.block2 = SwinBlock(embed_dim, num_heads)
        self.drop = nn.Dropout(0.3)
        self.fc1 = nn.LazyLinear(128)
        self.fc_out = nn.LazyLinear(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape can be 3D volume (B, 1, D, H, W) or 2D image (B, C, H, W)
        if x.ndim == 5:
            # (B, C, D, H, W) -> merge batch and depth frames
            b, c, d, h, w = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
            is_volume = True
        else:
            b = x.shape[0]
            d = 1
            is_volume = False

        # Patch Embed: (B*D, embed_dim, H/4, W/4)
        x = self.patch_conv(x)
        bd, emb, h_p, w_p = x.shape
        x = x.flatten(2).permute(0, 2, 1) # (B*D, N_patches, embed_dim)

        x = self.block1(x)
        x = self.block2(x)
        x = x.mean(dim=1) # Pool patch tokens -> (B*D, embed_dim)

        if is_volume:
            x = x.view(b, d, emb).mean(dim=1) # Temporal pool across depth frames -> (B, embed_dim)

        x = self.drop(F.relu(self.fc1(x)))
        return F.softmax(self.fc_out(x), dim=1)


# Alias for Swin Transformer name variants
SwinTransformer = Swin_Transformer


# ===========================================================================
# ResBiLSTM – Residual + Bidirectional LSTM
# Zhao W, Wang W F, Patnaik L M, et al.
# "Residual and bidirectional LSTM for epileptic seizure detection"
# Frontiers in Computational Neuroscience, 2024, 18: 1415967.
# Source: https://github.com/snailpt/ResBiLSTM
# ===========================================================================

class ResnetBasicBlock1D(nn.Module):
    """
    1-D Residual Block with kernel-5 convolutions.
    Applies a 1×1 conv shortcut whenever stride > 1 or channel count changes.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.stride       = stride

        self.conv1   = nn.Conv1d(in_channels, out_channels, kernel_size=5,
                                 padding=2, stride=stride, bias=False)
        self.bn1     = nn.BatchNorm1d(out_channels)
        self.conv2   = nn.Conv1d(out_channels, out_channels, kernel_size=5,
                                 padding=2, stride=1, bias=False)
        self.bn2     = nn.BatchNorm1d(out_channels)
        # Identity / projection shortcut
        self.conv1x1 = nn.Conv1d(in_channels, out_channels, kernel_size=1,
                                 stride=stride, bias=False)
        self.bn1x1   = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.stride != 1 or self.in_channels != self.out_channels:
            residual = self.bn1x1(self.conv1x1(x))
        else:
            residual = x
        out = out + residual
        return torch.relu(out)


class ResBiLSTM(nn.Module):
    """
    ResBiLSTM: three 1-D residual blocks → BiLSTM → two FC layers.

    Input tensor shape: (B, 1, L) where L is the 1-D signal / feature length.

    Architecture (paper defaults):
        Block 1: ResnetBasicBlock1D(1  → 64,  stride=2) + Dropout(0.2)
        Block 2: ResnetBasicBlock1D(64 → 64,  stride=1) + Dropout(0.2)
        Block 3: ResnetBasicBlock1D(64 → 128, stride=2) + Dropout(0.2)
        BiLSTM : LSTM(128, rnn_cells=128, num_layers=1, bidirectional=True)
        FC1    : Linear(rnn_cells*2, fc1_units=64)  + Dropout(0.5)
        FC2    : Linear(fc1_units, num_classes)
    """
    def __init__(
        self,
        num_classes: int,
        rnn_cells: int = 128,
        fc1_units: int = 64,
    ):
        super().__init__()

        self.block1 = nn.Sequential(
            ResnetBasicBlock1D(1,   64,  stride=2),
            nn.Dropout(p=0.2),
        )
        self.block2 = nn.Sequential(
            ResnetBasicBlock1D(64,  64,  stride=1),
            nn.Dropout(p=0.2),
        )
        self.block3 = nn.Sequential(
            ResnetBasicBlock1D(64,  128, stride=2),
            nn.Dropout(p=0.2),
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=rnn_cells,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.fc1    = nn.Linear(rnn_cells * 2, fc1_units)
        self.fc2    = nn.Linear(fc1_units, num_classes)
        self._rnn_cells = rnn_cells

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, L)
        x = self.block1(x)                          # (B, 64,  L/2)
        x = self.block2(x)                          # (B, 64,  L/2)
        x = self.block3(x)                          # (B, 128, L/4)

        # (B, 128, T) → (B, T, 128) for LSTM
        x = x.permute(0, 2, 1)
        x_out, (h, c) = self.lstm(x)               # h: (2, B, rnn_cells)

        # Concatenate forward & backward final hidden states
        x = torch.cat([h[0], h[1]], dim=1)         # (B, rnn_cells*2)
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.dropout(F.relu(self.fc1(x)), p=0.5, training=self.training)
        return F.softmax(self.fc2(x), dim=1)


# ===========================================================================
# Factory Function
# ===========================================================================
def build_model(
    model_name: str,
    num_classes: int,
    time_steps: int = 32,
    input_dim: int = 32,
    lr: float = 1e-3,
) -> Tuple[nn.Module, torch.optim.Optimizer]:
    """Instantiate a model, move to DEVICE, return (model, optimizer)."""
    
    # cCNN Baseline & Ablations
    if model_name == "cCNN":
        model = CCNN(num_classes)
    elif model_name in ["cCNN_A1_NoPooling", "A1"]:
        model = CCNN_A1_NoPooling(num_classes)
    elif model_name in ["cCNN_A2_NoDropout", "A2"]:
        model = CCNN_A2_NoDropout(num_classes)
    elif model_name in ["cCNN_A3_Dense64", "A3"]:
        model = CCNN_A3_Dense64(num_classes)
    elif model_name in ["cCNN_A4_Filters16", "A4"]:
        model = CCNN_A4_Filters16(num_classes)
    elif model_name in ["cCNN_A5_Filters64", "A5"]:
        model = CCNN_A5_Filters64(num_classes)
    elif model_name in ["cCNN_A6_Kernel5", "A6"]:
        model = CCNN_A6_Kernel5(num_classes)

    # cRNN Baseline & Ablations
    elif model_name in ["cRNN", "RNN"]:
        model = RNNModel(num_classes, time_steps, input_dim)
    elif model_name in ["cRNN_B1_OneLayer", "B1"]:
        model = CRNN_B1_OneLayer(num_classes, time_steps, input_dim)
    elif model_name in ["cRNN_B2_TwoLayers", "B2"]:
        model = CRNN_B2_TwoLayers(num_classes, time_steps, input_dim)
    elif model_name in ["cRNN_B3_NoDropout", "B3"]:
        model = CRNN_B3_NoDropout(num_classes, time_steps, input_dim)
    elif model_name in ["cRNN_B4_Hidden1024", "B4"]:
        model = CRNN_B4_Hidden1024(num_classes, time_steps, input_dim)
    elif model_name in ["cRNN_B5_Hidden512", "B5"]:
        model = CRNN_B5_Hidden512(num_classes, time_steps, input_dim)
    elif model_name in ["cRNN_B6_GRU", "B6"]:
        model = CRNN_B6_GRU(num_classes, time_steps, input_dim)

    # Existing models
    elif model_name == "AlexNet":
        model = AlexNet3D(num_classes)
    elif model_name == "VGGNET":
        model = VGGNet3D(num_classes)
    elif model_name == "LSTM":
        model = LSTMModel(num_classes, time_steps, input_dim)

    # New models
    elif model_name == "CNN_LSTM":
        model = CNN_LSTM(num_classes)
    elif model_name in ["CNN_BiLSTM", "CNN-BiLSTM"]:
        model = CNN_BiLSTM(num_classes)
    elif model_name in ["Swin_Transformer", "Swin Transformer", "SwinTransformer"]:
        model = Swin_Transformer(num_classes)

    # ResBiLSTM
    elif model_name in ["ResBiLSTM", "ResBiLSTM_Net"]:
        model = ResBiLSTM(num_classes)
    else:
        raise ValueError(f"Unknown model name: '{model_name}'")

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer
