import torch
import torch.nn as nn
import torch.optim as optim
from src.models.model import Transformer
# 定义损失函数和优化器
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR


def noam_lambda(step,d_model=512, warmup_steps=4000):
    step = max(1, step)
    return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))

# 定义训练函数
def train(model, dataloader, device, en_vocab, zh_vocab):
    criterion = nn.CrossEntropyLoss(ignore_index=zh_vocab['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # scheduler = LambdaLR(optimizer, lr_lambda=noam_lambda)
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    for src, trg in dataloader:
        src = src.to(device)
        trg = trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg[:, :-1], en_vocab, zh_vocab)  # 输入不包括最后一个词
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)  # 目标不包括第一个词
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        # scheduler.step()
        learning_rate = optimizer.param_groups[0]['lr']
        
        # 计算准确率
        _, predicted = torch.max(output, dim=1)
        correct += (predicted == trg).sum().item()
        total += trg.size(0)

        epoch_loss += loss.item()
    accuracy = correct / total
    return epoch_loss / len(dataloader), accuracy, learning_rate