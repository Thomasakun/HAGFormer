import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  # 添加这个导入

class SelfAttention(nn.Module):
    """
    Self-Attention mechanism for input sequences.

    Args:
        input_dim (int): The number of expected features in the input.
    """
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        """
        Forward pass for self-attention mechanism.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, input_dim).
        """
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(x.size(-1))
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        weighted_values = torch.matmul(attention_weights, V)

        return weighted_values

class CustomGRU(nn.Module):
    """
    Custom implementation of GRU using PyTorch's nn.GRU.

    Args:
        input_size (int): The number of expected features in the input.
        hidden_size (int): The number of features in the hidden state.
        num_layers (int): Number of recurrent layers.
        bias (bool): If False, then the layer does not use bias weights.
        batch_first (bool): If True, then the input and output tensors are provided as (batch, seq, feature).
        dropout (float): If non-zero, introduces a Dropout layer on the outputs of each GRU layer except the last layer.
        bidirectional (bool): If True, becomes a bidirectional GRU.
    """
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False):
        super(CustomGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x, h=None):
        """
        Forward pass for GRU.

        Args:
            x (Tensor): Input tensor.
            h (Optional[Tensor]): Initial hidden state.

        Returns:
            Tuple[Tensor, Tensor]: Output tensor and final hidden state.
        """
        return self.gru(x, h)

class CustomTransformerEncoderLayer(nn.Module):
    """
    Custom implementation of a Transformer Encoder Layer.

    Args:
        d_model (int): The number of expected features in the input.
        nhead (int): The number of heads in the multiheadattention models.
        dim_feedforward (int): The dimension of the feedforward network model.
        dropout (float): The dropout value.
        activation (str): The activation function of the intermediate layer, can be 'relu' or 'gelu'.
        layer_norm_eps (float): The eps value in layer normalization components.
        batch_first (bool): If True, then the input and output tensors are provided as (batch, seq, feature).
        norm_first (bool): If True, layer norm is done prior to attention and feedforward operations.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', layer_norm_eps=1e-5, batch_first=False, norm_first=False):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Forward pass for Transformer Encoder Layer.

        Args:
            src (Tensor): Input tensor.
            src_mask (Optional[Tensor]): Source mask.
            src_key_padding_mask (Optional[Tensor]): Source key padding mask.

        Returns:
            Tensor: Output tensor.
        """
        if self.norm_first:
            src = src + self._sa_block(self.norm1(src), src_mask, src_key_padding_mask)
            src = src + self._ff_block(self.norm2(src))
        else:
            src = self.norm1(src + self._sa_block(src, src_mask, src_key_padding_mask))
            src = self.norm2(src + self._ff_block(src))
        return src

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """
        Self-attention block.

        Args:
            x (Tensor): Input tensor.
            attn_mask (Optional[Tensor]): Attention mask.
            key_padding_mask (Optional[Tensor]): Key padding mask.

        Returns:
            Tensor: Output tensor.
        """
        x, _ = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        return self.dropout1(x)

    def _ff_block(self, x):
        """
        Feed-forward block.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

class CustomTransformerEncoder(nn.Module):
    """
    Custom implementation of a Transformer Encoder.

    Args:
        encoder_layer (nn.Module): An instance of the TransformerEncoderLayer() class.
        num_layers (int): The number of sub-encoder-layers in the encoder.
        norm (Optional[nn.Module]): The layer normalization component.
    """
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(CustomTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        """
        Forward pass for Transformer Encoder.

        Args:
            src (Tensor): Input tensor.
            mask (Optional[Tensor]): Source mask.
            src_key_padding_mask (Optional[Tensor]): Source key padding mask.

        Returns:
            Tensor: Output tensor.
        """
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output

class TransformerClassifier(nn.Module):
    """
    Transformer-based classifier.

    Args:
        input_dim (int): The number of expected features in the input.
        num_classes (int): The number of output classes.
        dim_feedforward (int): The dimension of the feedforward network model.
        nhead (int): The number of heads in the multiheadattention models.
        num_layers (int): The number of sub-encoder-layers in the encoder.
        dropout (float): The dropout value.
    """
    def __init__(self, input_dim, num_classes, dim_feedforward=512, nhead=3, num_layers=2, dropout=0.2):
        super(TransformerClassifier, self).__init__()
        self.self_attention = SelfAttention(input_dim)
        self.gru = CustomGRU(input_dim, input_dim, batch_first=True)
        encoder_layer = CustomTransformerEncoderLayer(d_model=2*input_dim, nhead=nhead,
                                                  dim_feedforward=dim_feedforward,
                                                  dropout=dropout,
                                                  batch_first=True)
        self.transformer_encoder = CustomTransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(2*input_dim, num_classes)

    def forward(self, src):
        """
        Forward pass for Transformer-based classifier.

        Args:
            src (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        attention_output = self.self_attention(src)
        gru_output, _ = self.gru(src)
        concat_output = torch.cat((attention_output, gru_output), dim=-1)
        transformed = self.transformer_encoder(concat_output)
        output = self.classifier(transformed)
        return output

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy loss.

    Args:
        smoothing (float): The smoothing parameter.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        """
        Forward pass for Label Smoothing Cross Entropy loss.

        Args:
            input (Tensor): Input tensor.
            target (Tensor): Target tensor.

        Returns:
            Tensor: Loss value.
        """
        log_probs = F.log_softmax(input, dim=-1)
        n_classes = input.size(-1)
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (n_classes - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))
