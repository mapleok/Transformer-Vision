# 2_train_resnet.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import train_epoch, validate, save_metrics  # 现在可以导入了
from data_utils import create_dataloaders  # 注意：这里需要根据你的文件名调整


def create_resnet18(num_classes=10):
    """
    创建ResNet18模型，加载ImageNet预训练权重
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_resnet18(train_loader, val_loader, num_epochs=10, device='cuda'):
    """
    训练ResNet18模型
    """
    model = create_resnet18().to(device)

    # 第一阶段：冻结主干网络（只训练分类头）
    print("第一阶段：冻结主干网络，仅训练分类头...")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=2e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=5)

    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(5):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        scheduler.step()
        print(f"阶段1 Epoch {epoch + 1}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")

    # 第二阶段：全局微调
    print("\n第二阶段：全局微调...")
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    patience_counter = 0
    os.makedirs('../models', exist_ok=True)

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        scheduler.step()
        print(f"阶段2 Epoch {epoch + 1}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), '../models/best_resnet18.pth')
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 5:
            print(f"早停触发，最佳验证准确率: {best_val_acc:.2f}%")
            break

    # 保存训练历史
    save_metrics(history, 'resnet18')

    return model, history


if __name__ == "__main__":
    # 注意：这里需要根据你的数据加载文件调整
    # 如果你的数据加载代码在 1_data_preprocess.py 中
    from data_utils import create_dataloaders  # 或者 from 1_data_preprocess import create_dataloaders

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    train_loader, val_loader, test_loader = create_dataloaders(batch_size=16)

    model, history = train_resnet18(train_loader, val_loader, num_epochs=10, device=device)
    print("ResNet18训练完成！")
