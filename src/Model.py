import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm
from collections import OrderedDict


class CNNModel(nn.Module):
    '''
    CNN Model for ICU dataset
    '''
    def __init__(self, vitals_input_dim, labs_input_dim):
        super(CNNModel, self).__init__()

        # Vitals input branch
        self.vitals_conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)  # Change in_channels to 1
        self.vitals_pool1 = nn.MaxPool1d(kernel_size=2)
        self.vitals_conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.vitals_pool2 = nn.MaxPool1d(kernel_size=2)
        self.vitals_bn = nn.BatchNorm1d(64)

        # Labs input branch
        self.labs_conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)  # Change in_channels to 1
        self.labs_pool1 = nn.MaxPool1d(kernel_size=2)
        self.labs_conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.labs_pool2 = nn.MaxPool1d(kernel_size=2)
        self.labs_bn = nn.BatchNorm1d(64)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 2, 64)  # Concatenated input
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, vitals, labs):
        # Vitals branch
        vitals = vitals.unsqueeze(1)  # Add a channel dimension for vitals
        vitals = F.relu(self.vitals_conv1(vitals))
        vitals = self.vitals_pool1(vitals)
        vitals = F.relu(self.vitals_conv2(vitals))
        vitals = self.vitals_pool2(vitals)
        vitals = torch.mean(vitals, dim=-1)  # Global average pooling
        vitals = self.vitals_bn(vitals)

        # Labs branch
        labs = labs.unsqueeze(1)  # Add a channel dimension for labs
        labs = F.relu(self.labs_conv1(labs))
        labs = self.labs_pool1(labs)
        labs = F.relu(self.labs_conv2(labs))
        labs = self.labs_pool2(labs)
        labs = torch.mean(labs, dim=-1)  # Global average pooling
        labs = self.labs_bn(labs)

        # Concatenate branches
        merged = torch.cat([vitals, labs], dim=1)

        # Fully connected layers
        x = F.relu(self.fc1(merged))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        output = torch.sigmoid(self.output(x))

        return output


class RNNModel(nn.Module):
    '''
    RNN Model for ICU dataset
    '''
    def __init__(self, vitals_input_dim, labs_input_dim):
        super(RNNModel, self).__init__()

        # Vital channel
        self.mask_value = -2.0
        self.vitals_gru1 = nn.GRU(vitals_input_dim, 16, batch_first=True)
        self.vitals_gru2 = nn.GRU(16, 16, batch_first=True)
        self.vitals_gru3 = nn.GRU(16, 16, batch_first=True)
        self.vitals_bn = nn.BatchNorm1d(16)

        # Labs channel
        self.labs_gru1 = nn.GRU(labs_input_dim, 16, batch_first=True)
        self.labs_gru2 = nn.GRU(16, 16, batch_first=True)
        self.labs_gru3 = nn.GRU(16, 16, batch_first=True)
        self.labs_bn = nn.BatchNorm1d(16)

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 2, 16)
        self.fc2 = nn.Linear(16, 16)
        self.output = nn.Linear(16, 1)

    def forward(self, vitals, labs):
        # Masking values for vitals
        vitals = torch.where(vitals == self.mask_value, torch.zeros_like(vitals), vitals)

        # Vital channel GRU layers
        vitals, _ = self.vitals_gru1(vitals)
        vitals, _ = self.vitals_gru2(vitals)
        vitals, _ = self.vitals_gru3(vitals)

        # Check if vitals is 2-dimensional and add a sequence length dimension if necessary
        if vitals.dim() == 2:
            vitals = vitals.unsqueeze(1)

        vitals = vitals[:, -1, :]  # Get the last hidden state
        vitals = self.vitals_bn(vitals)

        # Masking values for labs
        labs = torch.where(labs == self.mask_value, torch.zeros_like(labs), labs)

        # Labs channel GRU layers
        labs, _ = self.labs_gru1(labs)
        labs, _ = self.labs_gru2(labs)
        labs, _ = self.labs_gru3(labs)

        # Check if labs is 2-dimensional and add a sequence length dimension if necessary
        if labs.dim() == 2:
            labs = labs.unsqueeze(1)

        labs = labs[:, -1, :]  # Get the last hidden state
        labs = self.labs_bn(labs)

        # Concatenation of both branches
        merged = torch.cat([vitals, labs], dim=1)

        # Fully connected layers
        x = F.relu(self.fc1(merged))
        x = F.relu(self.fc2(x))
        output = torch.sigmoid(self.output(x))

        return output


class Hypernetwork(nn.Module):
    def __init__(self, target_model, n_nodes, embedding_dim, hidden_dim=100, spec_norm=False, n_hidden=1):
        super(Hypernetwork, self).__init__()
        self.target_model = target_model
        self.embeddings = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)

        layers = [
            spectral_norm(nn.Linear(embedding_dim, hidden_dim)) if spec_norm else nn.Linear(embedding_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim), )

        self.mlp = nn.Sequential(*layers)

        self.hyper_layers = self.create_hyper_layers(hidden_dim, spec_norm)  # Pass hidden_dim instead of embedding_dim

    def create_hyper_layers(self, embedding_dim, spec_norm):
        """
        Sinh các lớp hypernetwork để tạo tham số tương ứng với target model.
        """
        state_dict = self.target_model.state_dict()
        hyper_layers = nn.ModuleDict()

        for name, param in state_dict.items():
            output_dim = param.numel()
            safe_name = name.replace(".", "__")  # Thay "." bằng "_"
            layer = nn.Linear(embedding_dim, output_dim)
            if spec_norm:
                layer = nn.utils.spectral_norm(layer)  # Áp dụng spectral norm nếu cần
            hyper_layers[safe_name] = layer  # Lưu với tên an toàn

        return hyper_layers

    def forward(self, idx):
        emd = self.embeddings(idx)  # Nhúng ID thành vector embedding
        features = self.mlp(emd)  # Qua MLP để tạo đặc trưng

        # Tạo trọng số từ hyper_layers
        weights = OrderedDict()

        for safe_name, layer in self.hyper_layers.items():
            original_name = safe_name.replace("__", ".")  # Chuyển từ tên an toàn về tên gốc
            weight = layer(features)

            # Kiểm tra nếu layer là bias
            if 'bias' in original_name:
                weights[original_name] = weight.view(-1)  # Flatten nếu là bias
            else:
                # Giữ nguyên cấu trúc ban đầu giống target model
                target_weight_shape = self.target_model.state_dict()[original_name].shape
                weights[original_name] = weight.view(target_weight_shape)

        return weights


class TransformerBlock(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout_rate)
        self.attention_norm = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.ffn = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_dim, input_dim)
        )
        self.ffn_norm = nn.LayerNorm(input_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Multi-head self-attention
        attn_output, _ = self.attention(x, x, x)
        x = self.attention_norm(x + self.dropout1(attn_output))  # Skip connection

        # Feed-forward network
        ffn_output = self.ffn(x)
        x = self.ffn_norm(x + self.dropout2(ffn_output))  # Skip connection

        return x


class TransformerModel(nn.Module):
    '''
    Transformer Model for ICU dataset
    '''
    def __init__(self, vitals_input_dim, labs_input_dim, num_heads, ff_dim):
        super(TransformerModel, self).__init__()

        # Vitals branch
        self.vitals_dense = nn.Linear(vitals_input_dim, 64)
        self.vitals_transformer = TransformerBlock(64, num_heads, ff_dim)
        self.vitals_pool = nn.AdaptiveAvgPool1d(1)
        # Change num_features to 1 to match the output of global average pooling
        self.vitals_bn = nn.BatchNorm1d(1)

        # Labs branch
        self.labs_dense = nn.Linear(labs_input_dim, 64)
        self.labs_transformer = TransformerBlock(64, num_heads, ff_dim)
        self.labs_pool = nn.AdaptiveAvgPool1d(1)
        # Change num_features to 1 to match the output of global average pooling
        self.labs_bn = nn.BatchNorm1d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(2, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, vitals, labs):
        # Vitals branch
        vitals = F.relu(self.vitals_dense(vitals))
        vitals = vitals.unsqueeze(1)
        vitals = self.vitals_transformer(vitals)
        vitals = vitals.squeeze(1)  # squeeze the seq_len
        vitals = self.vitals_pool(vitals.unsqueeze(1)).squeeze(-1)  # Global average pooling
        vitals = self.vitals_bn(vitals)

        # Labs branch (apply the same changes for labs)
        labs = F.relu(self.labs_dense(labs))
        labs = labs.unsqueeze(1)
        labs = self.labs_transformer(labs)
        labs = labs.squeeze(1)  # squeeze the seq_len
        labs = self.labs_pool(labs.unsqueeze(1)).squeeze(-1)  # Global average pooling
        labs = self.labs_bn(labs)

        # Concatenate both branches
        merged = torch.cat([vitals, labs], dim=1)

        # Fully connected layers
        x = F.relu(self.fc1(merged))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        output = torch.sigmoid(self.output(x))

        return output
