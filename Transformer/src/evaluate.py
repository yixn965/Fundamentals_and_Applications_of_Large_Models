import torch
import csv
import numpy as np
import torch.nn as nn

def evaluate(model, dataloader, device, en_vocab, zh_vocab):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=zh_vocab['<pad>'])
    val_loss = 0
    correct = 0
    total = 0

    # 用于保存预测准确率较高的10个例子
    high_accuracy_examples = []

    with torch.no_grad():
        for src, trg in dataloader:
            src = src.to(device)
            trg = trg.to(device)
            output = model(src, trg[:, :-1], en_vocab, zh_vocab) 
            
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            # 计算损失
            loss = criterion(output, trg)
            val_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(output, dim=1)
            correct += (predicted == trg).sum().item()
            total += trg.size(0)


            # 保存每个样本的预测值、真实值和准确率
            trg = trg.view(-1, trg.size(0) // src.size(0))

            for i in range(src.size(0)):  

                src_sentence = ' '.join([en_vocab.itos[idx] for idx in src[i].cpu().numpy()])
                trg_sentence = ' '.join([zh_vocab.itos[idx] for idx in trg[i].cpu().numpy()])
                predicted = predicted.view(-1, predicted.size(0) // src.size(0))
                pred_sentence = ' '.join([zh_vocab.itos[idx] for idx in predicted[i].cpu().numpy()])

                # 计算准确率
                accuracy = (predicted[i] == trg[i]).sum().item()

                high_accuracy_examples.append({
                    'src': src_sentence,
                    'trg': trg_sentence,
                    'pred': pred_sentence,
                    'accuracy': accuracy
                })

    val_accuracy = correct / total
    return val_loss / len(dataloader), val_accuracy, high_accuracy_examples
