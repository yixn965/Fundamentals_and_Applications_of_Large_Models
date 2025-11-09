import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math
import os

# 保存模型权重
def save_model(model, epoch, save_path='results/checkpoints'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, f'model_epoch_{epoch}.pt'))

# 绘制图形
def plot_metrics(train_losses, accuracies, learning_rates, output_dir='results/output'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plot Loss
    plt.figure()
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(output_dir, 'train_loss_curve.png'))
    
    # Plot Accuracy
    plt.figure()
    plt.plot(accuracies)
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(output_dir, 'train_accuracy_curve.png'))
    
    # Plot Learning Rate
    plt.figure()
    plt.plot(learning_rates)
    plt.title('Learning Rate')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.savefig(os.path.join(output_dir, 'train_learning_rate_curve.png'))

def plot_metrics_for_eval(eval_losses, eval_accuracies, output_dir='results/output'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plot Loss
    plt.figure()
    plt.plot(eval_losses)
    plt.title('Evaluating Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(output_dir, 'eval_loss_curve.png'))
    
    # Plot Accuracy
    plt.figure()
    plt.plot(eval_accuracies)
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(output_dir, 'eval_accuracy_curve.png'))
    

# 计算并保存模型大小和参数统计
def save_model_statistics(model, output_file='results/output/model_stats.txt'):
    model_size = sum(p.numel() for p in model.parameters())
    model_info = f"Model Size: {model_size} parameters\n"
    model_info += "Detailed Layer Info:\n"
    
    for name, param in model.named_parameters():
        model_info += f"{name}: {param.size()}\n"

    with open(output_file, 'w') as f:
        f.write(model_info)
    
    print(f"Model saved to {output_file}")
