from src.models.attention import PositionalEncoding,MultiHeadAttention,PositionwiseFeedForward
import torch.nn as nn
import math

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 自注意力子层
        attn_output = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        # 前馈神经网络子层
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)
        return x



class Encoder(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, d_ff, num_layers, dropout):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # src: [batch_size, src_len]
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x
