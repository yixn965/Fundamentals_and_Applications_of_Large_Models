from src.models.encoder import Encoder
from src.models.decoder import Decoder
import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, num_heads, d_ff, num_layers, dropout):
        super().__init__()
        self.encoder = Encoder(input_dim, d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(output_dim, d_model, num_heads, d_ff, num_layers, dropout)

    def make_src_mask(self, src, en_vocab):
        # 生成源序列的掩码，屏蔽填充位置
        src_mask = (src != en_vocab['<pad>']).unsqueeze(1)
        return src_mask  # [batch_size, 1, 1, src_len]

    def make_trg_mask(self, trg, zh_vocab):
        # 生成目标序列的掩码，包含填充位置和未来信息
        trg_pad_mask = (trg != zh_vocab['<pad>']).unsqueeze(1)  # [batch_size, 1, 1, trg_len]
        trg_len = trg.size(1)
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=trg.device)).bool()  # [trg_len, trg_len]
        trg_mask = trg_pad_mask & trg_sub_mask  # [batch_size, 1, trg_len, trg_len]
        return trg_mask

    def forward(self, src, trg, en_vocab, zh_vocab):
        src_mask = self.make_src_mask(src, en_vocab)
        trg_mask = self.make_trg_mask(trg, zh_vocab)
        enc_output = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_output, src_mask, trg_mask)
        return output
