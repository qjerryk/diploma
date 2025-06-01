import torch.nn as nn
import torch
import math
from einops import rearrange

class BaseLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(BaseLSTM, self).__init__()

        self.lstm_layer = nn.LSTM(input_size, hidden_size1, dropout=0.2, batch_first=True, num_layers=1)
        self.feedforward1 = nn.Linear(hidden_size1, hidden_size2)
        self.bn1 = nn.BatchNorm1d(hidden_size2)
        self.feedforward2 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        out, _ = self.lstm_layer(x)
        x = self.feedforward1(out[:, -1, :])
        x = nn.ReLU()(x)
        x = self.bn1(x)
        x = self.feedforward2(x)
        return x
    
class ConvLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, conv_hidden, output_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, conv_hidden, kernel_size=3, stride=1),
            nn.GELU(),
            nn.BatchNorm1d(conv_hidden),
            # nn.Conv1d(conv_hidden, conv_hidden, kernel_size=3, stride=2),
            # nn.GELU(),
            # nn.BatchNorm1d(conv_hidden),
            # nn.Conv1d(conv_hidden, conv_hidden, kernel_size=3, stride=2),
        )
        self.lstm_layer = nn.LSTM(conv_hidden, hidden_size1, dropout=0.2, batch_first=True, num_layers=1)
        self.feedforward1 = nn.Linear(hidden_size1, hidden_size2)
        self.bn1 = nn.BatchNorm1d(hidden_size2)
        self.feedforward2 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        # x.shape b seq_len feats
        x = x.transpose(1, 2) 
        x = self.conv(x) # b feats new_seq_len
        x = x.transpose(1, 2)
        out, _ = self.lstm_layer(x)
        x = self.feedforward1(out[:, -1, :])
        x = nn.ReLU()(x)
        x = self.bn1(x)
        x = self.feedforward2(x)
        return x

class ConvLSTMNoEM(torch.nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, conv_hidden, output_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, conv_hidden, kernel_size=3, stride=1),
            nn.GELU(),
            nn.BatchNorm1d(conv_hidden),
            # nn.Conv1d(conv_hidden, conv_hidden, kernel_size=3, stride=2),
            # nn.GELU(),
            # nn.BatchNorm1d(conv_hidden),
            # nn.Conv1d(conv_hidden, conv_hidden, kernel_size=3, stride=2),
        )
        self.lstm_layer = nn.LSTM(conv_hidden, hidden_size1, dropout=0.2, batch_first=True, num_layers=1)
        self.feedforward1 = nn.Linear(hidden_size1, hidden_size2)
        self.bn1 = nn.BatchNorm1d(hidden_size2)
        self.feedforward2 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = x[:, :, -1:]
        x = x.transpose(1, 2) 
        x = self.conv(x) # b feats new_seq_len
        x = x.transpose(1, 2)
        out, _ = self.lstm_layer(x)
        x = self.feedforward1(out[:, -1, :])
        x = nn.ReLU()(x)
        x = self.bn1(x)
        x = self.feedforward2(x)
        return x
    
class NoEMFeaturesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(NoEMFeaturesLSTM, self).__init__()
        self.lstm_layer = nn.LSTM(input_size, hidden_size1, dropout=0.2, batch_first=True, num_layers=1)
        self.feedforward1 = nn.Linear(hidden_size1, hidden_size2)
        self.bn1 = nn.BatchNorm1d(hidden_size2)
        self.feedforward2 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = x[:, :, -1:]
        out, _ = self.lstm_layer(x)
        x = self.feedforward1(out[:, -1, :])
        x = nn.ReLU()(x)
        x = self.bn1(x)
        x = self.feedforward2(x)
        return x


class EmbedsEM(nn.Module):
    def __init__(self, embed_dim, n_components, num_layers=4):
        super(EmbedsEM, self).__init__()
        self.n_components = n_components

        self.cls_token = nn.Parameter(torch.zeros(1, 3))
        nn.init.xavier_uniform_(self.cls_token)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=3,
            nhead=3,
            dim_feedforward=12,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.avg_pool = nn.AvgPool1d(kernel_size=n_components)
        self.max_pool = nn.MaxPool1d(kernel_size=n_components)
        self.fc = nn.Linear(3 * 3, embed_dim)

    def forward(self, x):
        x = torch.transpose(x.view(x.shape[0], 3, self.n_components), 1, 2)
        x = torch.cat([
            self.cls_token.expand(x.shape[0], 1, 3),
            x
        ], dim=1)

        transformer_output = self.transformer_encoder(x)

        cls_token_output = transformer_output[:, 0, :]
        features = torch.transpose(transformer_output[:, 1:, :], 1, 2)

        pooled = torch.cat([
            self.avg_pool(features),
            self.max_pool(features),
            cls_token_output.view(x.size(0), 3, 1)
        ], dim=2)

        return self.fc(pooled.view(x.size(0), -1))

class EncodeEMLSTM(torch.nn.Module):
    def __init__(self, input_size, n_components, embed_size, hidden_size1, hidden_size2, output_size):
        super(EncodeEMLSTM, self).__init__()
        self.encoder = EmbedsEM(embed_size, n_components)
        self.n_components = n_components
        self.embed_size = embed_size
        self.lstm = BaseLSTM(
            input_size=input_size - n_components * 3 + embed_size,
            hidden_size1=hidden_size1,
            hidden_size2=hidden_size2,
            output_size=output_size
        )

    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        em_features = x[:, :, : self.n_components * 3].view(-1, self.n_components * 3)
        x = x[:, :, self.n_components * 3 :]
        em_features = self.encoder(em_features)
        em_features = em_features.view(batch_size, seq_len, self.embed_size)
        x = torch.concatenate([em_features, x], dim=-1)
        return self.lstm(x)
    
class LinearRegressionEM(torch.nn.Module):
    def __init__(self, input_size, seq_size):
        super().__init__()
        self.fc = nn.Linear(input_size * seq_size, 1)
    
    def forward(self, x):
        x = rearrange(x, 'b s i -> b (s i)')
        return self.fc(x)

class LinearRegressionNoEM(torch.nn.Module):
    def __init__(self, input_size, seq_size):
        super().__init__()
        self.fc = nn.Linear(seq_size, 1)
    
    def forward(self, x):
        x = x[:, :, 11:]
        x = rearrange(x, 'b s i -> b (s i)')
        return self.fc(x)
    
class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(LSTM, self).__init__()

        self.lstm_layer = nn.LSTM(input_size, hidden_size1, dropout=0.2, batch_first=True, num_layers=1)
        self.feedforward1 = nn.Linear(hidden_size1, hidden_size2)
        # self.bn1 = nn.BatchNorm1d(hidden_size2)
        self.feedforward2 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        out, _ = self.lstm_layer(x) # [b, seq, hidden_size]
        x = self.feedforward1(out)
        x = nn.ReLU()(x)
        # x = self.bn1(x)
        x = self.feedforward2(x)
        return x # [b, seq, 1]

class NoEmLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(NoEmLSTM, self).__init__()

        self.lstm_layer = nn.LSTM(input_size, hidden_size1, dropout=0.2, batch_first=True, num_layers=1)
        self.feedforward1 = nn.Linear(hidden_size1, hidden_size2)
        # self.bn1 = nn.BatchNorm1d(hidden_size2)
        self.feedforward2 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = x[:, :, 9:]
        out, _ = self.lstm_layer(x) # [b, seq, hidden_size]
        x = self.feedforward1(out)
        x = nn.ReLU()(x)
        # x = self.bn1(x)
        x = self.feedforward2(x)
        return x

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModelEM(torch.nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, output_size):
        super(TransformerModelEM, self).__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, input_size))
        nn.init.xavier_uniform_(self.cls_token)
        self.pe = PositionalEncoding(input_size, max_len=350)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=1,
            dim_feedforward=input_size * 4,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
    
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.pe(x)  # [batch, seq, input_size]
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # [batch, 1, input_size]
        x = torch.cat((cls_tokens, x), dim=1)  # [batch, seq+1, input_size]
        x = self.transformer_encoder(x)
        cls_output = x[:, 0, :]  # [batch, input_size]
        x = self.fc1(cls_output)
        x = self.bn(x)
        x = torch.relu(x)
        x = self.fc2(x)  # [batch, output_size]
        return x

    

# class TransformerModelEM(torch.nn.Module):
#     def __init__(self, input_size, num_layers, hidden_size, output_size):
#         super(TransformerModelEM, self).__init__()
#         self.cls_token = nn.Parameter(torch.zeros(1, input_size))
#         nn.init.xavier_uniform_(self.cls_token)
#         self.pe = PositionalEncoding(input_size, max_len=350)
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=input_size,
#             nhead=1,
#             dim_feedforward=input_size * 4,
#             activation='gelu',
#             batch_first=True
#         )
#         self.transformer_encoder = nn.TransformerEncoder(
#             encoder_layer,
#             num_layers=num_layers
#         )
    
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         # self.bn = nn.BatchNorm1d(hidden_size)
#         self.fc2 = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         x = self.pe(x.transpose(0, 1)).transpose(0, 1)
#         # cls_tokens = self.cls_token.expand(x.size(0), 1, -1)  # [batch_size, 1, input_size]
#         # x = torch.cat((cls_tokens, x), dim=1)  # [batch_size, seq_len+1, input_size]
        
#         # Pass through transformer encoder
#         x = self.transformer_encoder(x)  # [batch_size, seq_len+1, input_size]
        
#         # Get only the CLS token's output (first position)
#         # cls_output = x[:, 0, :]  # [batch_size, input_size]
#         cls_output = x
        
#         # Pass through the classification head
#         x = self.fc1(cls_output)  # [batch_size, hidden_size]
#         # x = self.bn(x)
#         x = torch.relu(x)
#         x = self.fc2(x)  # [batch_size, output_size]
        
#         return x
    
    
class TransformerModel(torch.nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, output_size):
        super(TransformerModel, self).__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, input_size))
        nn.init.xavier_uniform_(self.cls_token)
        self.pe = PositionalEncoding(input_size, max_len=350)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=1,
            dim_feedforward=input_size * 4,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
    
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x[:, :, 9:]
        x = self.pe(x.transpose(0, 1)).transpose(0, 1)
        cls_tokens = self.cls_token.expand(x.size(0), 1, -1)  # [batch_size, 1, input_size]
        x = torch.cat((cls_tokens, x), dim=1)  # [batch_size, seq_len+1, input_size]
        
        x = self.transformer_encoder(x)  # [batch_size, seq_len+1, input_size]
        
        cls_output = x[:, 0, :]  # [batch_size, input_size]
        
        x = self.fc1(cls_output)  # [batch_size, hidden_size]
        x = self.bn(x)
        x = torch.relu(x)
        x = self.fc2(x)  # [batch_size, output_size]
        
        return x
