# code/4_evaluate.py 计算所有评估指标
import torch
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, test_loader, device, model_name):
    """
    计算Accuracy、Precision、Recall、F1-score
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 计算各项指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"\n========== {model_name} 评估结果 ==========")
    print(f"Accuracy (准确率):  {accuracy:.4f}")
    print(f"Precision (精确率): {precision:.4f}")
    print(f"Recall (召回率):    {recall:.4f}")
    print(f"F1-Score:          {f1:.4f}")

    return {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': all_preds,
        'labels': all_labels
    }


def plot_confusion_matrix(labels, predictions, class_names, model_name, save_path='./results/figures/'):
    """
    绘制混淆矩阵
    """
    cm = confusion_matrix(labels, predictions)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} 混淆矩阵')
    plt.tight_layout()
    plt.savefig(f'{save_path}/confusion_matrix_{model_name.lower()}.png', dpi=300)
    plt.show()

    return cm


def compare_models(result_resnet, result_swin, save_path='./results/figures/'):
    """
    对比两个模型的评估指标
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    resnet_scores = [result_resnet[m] for m in metrics]
    swin_scores = [result_swin[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width / 2, resnet_scores, width, label='ResNet18', color='blue')
    rects2 = ax.bar(x + width / 2, swin_scores, width, label='Swin Transformer', color='red')

    ax.set_ylabel('Score')
    ax.set_title('Swin Transformer vs ResNet18 性能对比')
    ax.set_xticks(x)
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score'])
    ax.legend()
    ax.set_ylim(0, 1)

    # 在柱状图上显示数值
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f'{save_path}/model_comparison.png', dpi=300)
    plt.show()


def show_misclassified_examples(model, test_loader, device, class_names, num_examples=5):
    """
    展示错分的图像样例
    """
    model.eval()
    misclassified = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)

            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    misclassified.append({
                        'image': images[i].cpu(),
                        'true_label': labels[i].item(),
                        'pred_label': preds[i].item()
                    })
            if len(misclassified) >= num_examples:
                break

    # 显示错分样例
    fig, axes = plt.subplots(1, num_examples, figsize=(15, 3))
    for i, example in enumerate(misclassified[:num_examples]):
        img = example['image'].permute(1, 2, 0)
        # 反归一化
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        img = img * std + mean
        img = torch.clamp(img, 0, 1)

        axes[i].imshow(img)
        axes[i].set_title(f"True: {class_names[example['true_label']]}\nPred: {class_names[example['pred_label']]}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('./results/figures/misclassified_examples.png', dpi=300)
    plt.show()
