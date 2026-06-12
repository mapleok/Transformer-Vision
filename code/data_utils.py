# data_utils.py
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
import numpy as np
import os


def prepare_cifar10_data(data_root='../data'):
    """
    下载CIFAR-10并按照8:1:1划分训练集、验证集、测试集
    """
    # 基础预处理
    base_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 训练集的数据增强
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 加载整个CIFAR-10数据集（训练集+测试集）
    print("正在加载CIFAR-10数据集...")

    # 加载原始训练集（50000张）
    full_train = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=False, transform=None
    )

    # 加载原始测试集（10000张）
    full_test = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=False, transform=None
    )

    # 合并所有数据（60000张）
    from torch.utils.data import ConcatDataset
    all_dataset = ConcatDataset([full_train, full_test])

    total_size = len(all_dataset)  # 60000张
    train_size = int(0.8 * total_size)  # 48000张
    val_size = int(0.1 * total_size)  # 6000张
    test_size = total_size - train_size - val_size  # 6000张

    print(f"总数据量: {total_size} 张")
    print(f"按8:1:1划分: 训练集={train_size}, 验证集={val_size}, 测试集={test_size}")

    # 随机划分
    indices = list(range(total_size))
    np.random.seed(42)
    np.random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # 创建带不同预处理的子集
    class DatasetWithTransform:
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            img, label = self.dataset[self.indices[idx]]
            return self.transform(img), label

    train_dataset = DatasetWithTransform(all_dataset, train_indices, train_transform)
    val_dataset = DatasetWithTransform(all_dataset, val_indices, base_transform)
    test_dataset = DatasetWithTransform(all_dataset, test_indices, base_transform)

    print(f"\n数据集划分完成（8:1:1）：")
    print(f"  训练集: {len(train_dataset)} 张 (80%)")
    print(f"  验证集: {len(val_dataset)} 张 (10%)")
    print(f"  测试集: {len(test_dataset)} 张 (10%)")

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(batch_size=16):
    train_dataset, val_dataset, test_dataset = prepare_cifar10_data()

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = create_dataloaders(batch_size=16)
    print("\n数据加载器创建成功！")

    for images, labels in train_loader:
        print(f"批次图像形状: {images.shape}")
        print(f"批次标签形状: {labels.shape}")
        break
