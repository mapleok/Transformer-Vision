# code/3_train_swin.py 搭建Swin Transformer（实验组）
import torch
import torch.nn as nn
import torch.optim as optim
import timm  # PyTorch Image Models库
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import train_epoch, validate


def create_swin_transformer(num_classes=10):
    """
    创建Swin Transformer模型，加载ImageNet预训练权重
    """
    # 使用timm库加载预训练的Swin Transformer
    # 'swin_tiny_patch4_window7_224' 是Swin-Tiny版本，参数量适中
    model = timm.create_model('swin_tiny_patch4_window7_224',
                              pretrained=True,
                              num_classes=num_classes)

    return model


def train_swin(train_loader, val_loader, num_epochs=10, device='cuda'):
    """
    训练Swin Transformer模型
    采用与ResNet18完全相同的训练策略，保证对比公平
    """
    model = create_swin_transformer().to(device)

    # 第一阶段：冻结主干，训练分类头
    print("第一阶段：冻结主干网络...")
    for param in model.parameters():
        param.requires_grad = False
    # Swin的分类头叫'head'，解冻它
    for param in model.head.parameters():
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
        print(f"Epoch {epoch + 1}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")

    # 第二阶段：全局微调
    print("\n第二阶段：全局微调...")
    for param in model.parameters():
        param.requires_grad = True

    # 使用更小的学习率进行微调
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    patience_counter = 0
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        scheduler.step()
        print(f"Epoch {epoch + 1}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), './models/best_swin.pth')
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 5:
            print(f"早停触发，最佳验证准确率: {best_val_acc:.2f}%")
            break

    return model, history
