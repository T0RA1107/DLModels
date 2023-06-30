from hydra import initialize_config_dir, compose
from ResNet.model import resnet101
from ViT.model import ViT
from Dataset.CIFAR10 import dataloader
from train import train
import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim

config_path = "/Users/tora/Desktop/DL/DLModels/ImageRecognition"

# Global Contrast Normalization による前処理
class gcn():
    def __init__(self):
        pass

    def __call__(self, x):
        mean = torch.mean(x)
        std = torch.std(x)
        return (x - mean)/(std + 10**(-6))


GCN = gcn()

# 適当なData Augmentationを加えてTransformを定義する
transform_train = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomCrop(32, padding=(4, 4, 4, 4), padding_mode='constant'),
                                    transforms.ToTensor(),
                                    GCN])

transform_valid = transforms.Compose([transforms.ToTensor(),
                                    GCN])

def main():
    initialize_config_dir(config_dir=config_path, version_base=None)
    config = compose(config_name="config.yaml")
    model = ViT(
        32,
        4,
        10,
        config.model.dim_model,
        config.model.depth,
        config.model.n_head,
        config.model.dim_mlp
    )
    dataloader_train, dataloader_valid = dataloader(config.dataset.batch_size, transform_train, transform_valid)
    loss_function = nn.CrossEntropyLoss()
    train(model, dataloader_train, dataloader_valid, loss_function, optim.AdamW, config)
    
if __name__ == "__main__":
    main()