# code/run_evaluation.py 运行完整评估
from data_utils import create_dataloaders
from evaluate import evaluate_model, plot_confusion_matrix, compare_models, show_misclassified_examples
import torch

# CIFAR-10类别名称
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    _, _, test_loader = create_dataloaders(batch_size=16)

    # 加载训练好的模型
    from train_resnet import create_resnet18
    from train_swin import create_swin_transformer

    resnet_model = create_resnet18().to(device)
    resnet_model.load_state_dict(torch.load('./models/best_resnet18.pth'))

    swin_model = create_swin_transformer().to(device)
    swin_model.load_state_dict(torch.load('./models/best_swin.pth'))

    # 评估ResNet18
    result_resnet = evaluate_model(resnet_model, test_loader, device, "ResNet18")
    plot_confusion_matrix(result_resnet['labels'], result_resnet['predictions'],
                          CLASS_NAMES, "ResNet18")

    # 评估Swin Transformer
    result_swin = evaluate_model(swin_model, test_loader, device, "Swin Transformer")
    plot_confusion_matrix(result_swin['labels'], result_swin['predictions'],
                          CLASS_NAMES, "Swin Transformer")

    # 对比展示
    compare_models(result_resnet, result_swin)

    # 展示错分样例
    show_misclassified_examples(swin_model, test_loader, device, CLASS_NAMES)


if __name__ == "__main__":
    main()
