import torch
import torch.optim as optim
import torch.nn as nn
from src.models.model import Transformer
from src.dataset.tokenizer import tokenizer_zh, tokenizer_en, build_vocab, process_sentence
from src.dataset.dataloader import get_dataloaders
from src.utils.logger import setup_stdout_stderr_tee
from src.utils.utils import save_model, plot_metrics,plot_metrics_for_eval, save_model_statistics
from src.train import train
from src.evaluate import evaluate
import os
import matplotlib.pyplot as plt
import pandas as pd
import torchsummary
import yaml
import csv

# 参数设置
with open('configs/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# 代码日志
save_dir = config['save_dir']
log_file = setup_stdout_stderr_tee(save_dir)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# --------------------------------------预处理----------------------------------------

# 加载句子
with open('data/english_sentences.txt', 'r', encoding='utf-8') as f:
    english_sentences = [line.strip() for line in f]
with open('data/chinese_sentences.txt', 'r', encoding='utf-8') as f:
    chinese_sentences = [line.strip() for line in f]

# 构建词汇表
zh_vocab = build_vocab(chinese_sentences, tokenizer_zh)
en_vocab = build_vocab(english_sentences, tokenizer_en)
# torch.save(en_vocab, 'results/vocab/en_vocab.pt')
# torch.save(zh_vocab, 'results/vocab/zh_vocab.pt')
print(f'英文词汇表大小：{len(en_vocab)}')
print(f'中文词汇表大小：{len(zh_vocab)}')

# -----------------------------------创建数据加载器----------------------------------

# 处理所有句子
en_sequences = [process_sentence(sentence, tokenizer_en, en_vocab) for sentence in english_sentences]
zh_sequences = [process_sentence(sentence, tokenizer_zh, zh_vocab) for sentence in chinese_sentences]
print("示例英文句子索引序列：", en_sequences[0])
print("示例中文句子索引序列：", zh_sequences[0])

batch_size = config['batch_size']
train_dataloader, val_dataloader = get_dataloaders(en_sequences, zh_sequences, batch_size, en_vocab, zh_vocab)

# ------------------------------------实例化模型------------------------------------

# 定义模型参数
input_dim = len(en_vocab)
output_dim = len(zh_vocab)
d_model = config['d_model']
num_heads = config['num_heads']
d_ff = config['d_ff']
num_layers = config['num_layers']
dropout = config['dropout']
warmup_steps = 4000

model = Transformer(input_dim, output_dim, d_model, num_heads, d_ff, num_layers, dropout).to(device)


# ---------------------------------------训练--------------------------------------
n_epochs = config['n_epochs']

# 初始化变量用于保存曲线
train_losses = []
train_accuracies = []
learning_rates = []
eval_losses = []
eval_accuracies = []

for epoch in range(n_epochs):
    train_loss, train_accuracy,learning_rate = train(model, train_dataloader, device, en_vocab, zh_vocab)
    val_loss, val_accuracy,_ = evaluate(model, val_dataloader, device, en_vocab, zh_vocab)
    print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.3f}, Accuracy: {train_accuracy:.3f}')
    print(f'Epoch {epoch+1}/{n_epochs}, Evaluate Loss: {train_loss:.3f}, Accuracy: {val_accuracy:.3f}')
    
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    learning_rates.append(learning_rate)
    eval_losses.append(val_loss)
    eval_accuracies.append(val_accuracy)

# 保存模型和绘制曲线
plot_metrics(train_losses, train_accuracies, learning_rates)
plot_metrics_for_eval(eval_losses, eval_accuracies)
save_model_statistics(model)
# save_model(model, epoch+1)

# ---------------------------------------评估--------------------------------------

if config['evaluate']:

    # model_path = 'results/checkpoints/model_epoch_100.pt'
    # if os.path.exists(model_path):
    #     model.load_state_dict(torch.load(model_path))
    #     print(f"模型权重从 {model_path} 加载成功！")
    # else:
    #     print(f"未找到模型权重文件，路径：{model_path}")

    val_loss, val_accuracy, high_accuracy_examples = evaluate(model, val_dataloader, device, en_vocab, zh_vocab)
    print(f'Validation Loss: {val_loss:.3f}, Validation Accuracy: {val_accuracy:.3f}')

    # 按准确率排序并挑选出准确率较高的10个例子
    high_accuracy_examples_sorted = sorted(high_accuracy_examples, key=lambda x: x['accuracy'], reverse=True)[:10]

    # 保存到CSV文件
    output_path = 'results/output/evaluate_examples.csv'
    with open(output_path, mode='w', newline='', encoding='utf-8') as file:

        writer = csv.writer(file)
        
        # 写入列名
        writer.writerow(['src', 'trg', 'pred', 'accuracy'])
        
        for example in high_accuracy_examples_sorted:
            # 清理 src, trg, pred 中的 <bos>, <eos>, <pad> 标记
            src_sentence = ' '.join([word for word in example['src'].split() if word not in ['<bos>', '<eos>', '<pad>']])
            trg_sentence = ' '.join([word for word in example['trg'].split() if word not in ['<bos>', '<eos>', '<pad>']])
            pred_sentence = ' '.join([word for word in example['pred'].split() if word not in ['<bos>', '<eos>', '<pad>']])
            
            # 获取准确率
            accuracy = example['accuracy']
            
            # 写入每条翻译结果，每句结果占四行
            writer.writerow([src_sentence, trg_sentence, pred_sentence, accuracy])
            writer.writerow([])

    print(f'Top 10 high accuracy examples saved to {output_path}')
