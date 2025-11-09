import torch
import torch.optim as optim
import torch.nn as nn
from src.models.model import Transformer
from src.dataset.tokenizer import tokenizer_zh, tokenizer_en, build_vocab, process_sentence
from src.dataset.dataloader import get_dataloaders
from src.utils.logger import setup_stdout_stderr_tee
from src.utils.utils import save_model, plot_metrics, save_model_statistics
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


en_vocab = torch.load('results/vocab/en_vocab.pt')
zh_vocab = torch.load('results/vocab/zh_vocab.pt')

# 定义模型参数
input_dim = len(en_vocab)
output_dim = len(zh_vocab)
d_model = config['d_model']
num_heads = config['num_heads']
d_ff = config['d_ff']
num_layers = config['num_layers']
dropout = config['dropout']

criterion = nn.CrossEntropyLoss(ignore_index=zh_vocab['<pad>'])

# 实例化模型
model = Transformer(input_dim, output_dim, d_model, num_heads, d_ff, num_layers, dropout).to(device)

model_path = 'results/checkpoints/model_epoch_5.pt'

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print(f"模型权重从 {model_path} 加载成功！")
else:
    print(f"未找到模型权重文件，路径：{model_path}")

val_loss, val_accuracy, high_accuracy_examples = evaluate(model, val_dataloader, criterion, device, en_vocab, zh_vocab)
print(f'Validation Loss: {val_loss:.3f}, Validation Accuracy: {val_accuracy:.3f}')

# 按准确率排序并挑选出准确率较高的10个例子
high_accuracy_examples_sorted = sorted(high_accuracy_examples, key=lambda x: x['accuracy'], reverse=True)[:10]

# 保存到CSV文件
output_path = 'results/output/evaluate_examples.csv'
with open(output_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=['src', 'trg', 'pred', 'accuracy'])
    writer.writeheader()
    writer.writerows(high_accuracy_examples_sorted)

print(f'Top 10 high accuracy examples saved to {output_path}')
