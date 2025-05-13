import time
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from utils import *
from torch.utils.data import DataLoader, random_split
import wandb


batch_size = 128
dropout_rate = 0.4
num_epochs = 200
train_set_ratio = 0.5
test_set_ratio = 1 - train_set_ratio
pretrained = True

hyperparameters = {
    'model': 'convnext_tiny',
    'batch_size': batch_size,
    'dropout_rate': dropout_rate,
    'num_epochs': num_epochs,
    'train_set_ratio': train_set_ratio,
    'test_set_ratio': 1 - train_set_ratio,
    'pretrained': pretrained
}

print("hyperparameters = ", hyperparameters)


wandb.init(
    project="CS444-Final Project",
    name="Coin_detection_convnext_128",
    config=hyperparameters
)


weights = ConvNeXt_Tiny_Weights.DEFAULT

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


all_dataset = CoinDataset(
    csv_file='data/train.csv',
    img_dir='data/train',
    transform=None
)

train_size = int(train_set_ratio * len(all_dataset))
test_size = len(all_dataset) - train_size

train_dataset_no_transform, test_dataset_no_transform = random_split(
    all_dataset, [train_size, test_size],
    generator=torch.Generator().manual_seed(0)
)

train_dataset = TransformedDataset(train_dataset_no_transform, transform=train_transform)
test_dataset = TransformedDataset(test_dataset_no_transform, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

print("Training and testing datasets ready")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_classes = all_dataset.get_num_classes()

net = convnext_tiny(weights=weights)

in_features = net.classifier[-1].in_features
net.classifier[-1] = nn.Sequential(
    nn.Dropout(p=dropout_rate),
    nn.Linear(in_features, num_classes)
)
net = net.to(device)
print("model = ", net.__class__.__name__)


criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)


wandb.config.update({"net_summary": str(net)})
wandb.watch(net, log="all")


train(
    net=net,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=num_epochs,
    init_lr=1e-2,
    device=device,
    trainloader=train_loader,
    test_loader=test_loader
)

torch.save(net.state_dict(), f"convnext_tiny_coin_{num_epochs}epochs_{int(time.time())}.pth")

