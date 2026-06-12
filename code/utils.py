# utils.py
import torch
import matplotlib.pyplot as plt
import numpy as np
import os


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # 添加进度提示
    from tqdm import tqdm
    loop = tqdm(dataloader, desc='Training')

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 更新进度条显示
        loop.set_postfix(loss=loss.item(), acc=100. * correct / total)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """
    验证/测试
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def save_metrics(history, model_name, save_path='../results/'):
    """
    保存训练指标到文件
    """
    os.makedirs(save_path, exist_ok=True)

    # 保存为numpy文件
    np.save(f'{save_path}/{model_name}_history.npy', history)

    # 保存为文本文件
    with open(f'{save_path}/{model_name}_metrics.txt', 'w') as f:
        f.write(f"=== {model_name} 训练结果 ===\n")
        f.write(f"最佳验证准确率: {max(history['val_acc']):.2f}%\n")
        f.write(f"最终训练准确率: {history['train_acc'][-1]:.2f}%\n")
        f.write(f"最终验证准确率: {history['val_acc'][-1]:.2f}%\n")

    print(f"指标已保存到 {save_path}")


def plot_training_curves(history_resnet, history_swin, save_path='../results/figures/'):
    """
    绘制训练曲线对比图
    """
    os.makedirs(save_path, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 损失曲线
    axes[0].plot(history_resnet['train_loss'], label='ResNet18 Train', color='blue')
    axes[0].plot(history_resnet['val_loss'], label='ResNet18 Val', color='blue', linestyle='--')
    axes[0].plot(history_swin['train_loss'], label='Swin Train', color='red')
    axes[0].plot(history_swin['val_loss'], label='Swin Val', color='red', linestyle='--')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('训练损失曲线对比')
    axes[0].legend()
    axes[0].grid(True)

    # 准确率曲线
    axes[1].plot(history_resnet['train_acc'], label='ResNet18 Train', color='blue')
    axes[1].plot(history_resnet['val_acc'], label='ResNet18 Val', color='blue', linestyle='--')
    axes[1].plot(history_swin['train_acc'], label='Swin Train', color='red')
    axes[1].plot(history_swin['val_acc'], label='Swin Val', color='red', linestyle='--')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('训练准确率曲线对比')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(f'{save_path}/training_curves.png', dpi=300)
    plt.show()
    print(f"训练曲线图已保存到 {save_path}/training_curves.png")
