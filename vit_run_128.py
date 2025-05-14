import time
import torch

import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vit_b_16, ViT_B_16_Weights
from utils import *
from torch.utils.data import DataLoader, random_split

model = 'vit_b_16'     
batch_size = 128
dropout_rate = 0
num_epochs = 200
train_set_ratio = 0.8
test_set_ratio = 1 - train_set_ratio
pretrained = True

hyperparameters = {
    'model': model,
    'batch_size': batch_size,
    'dropout_rate': dropout_rate,
    'num_epochs': num_epochs,
    'train_set_ratio': train_set_ratio,
    'test_set_ratio': test_set_ratio,
    'pretrained': pretrained
}

print("hyperparameters = ", hyperparameters)

wandb.init(
    project="CS444-Final Project",  # Project name
    name="Coin_detection_vit_128",               # Run name
    config={
        'model': model,
        'batch_size': batch_size, 
        'dropout_rate': dropout_rate, 
        'num_epochs': num_epochs, 
        'train_set_ratio': train_set_ratio, 
        'test_set_ratio': test_set_ratio, 
        'pretrained': pretrained
    }
)

# weights = ViT_B_16_Weights.DEFAULT
# test_transform = weights.transforms()
# normalize = test_transform.transforms[-1]

# train_transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomRotation(15),
#     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
#     transforms.ToTensor(),
#     normalize,
# ])
weights = ViT_B_16_Weights.DEFAULT
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std  = (0.229, 0.224, 0.225)

test_transform = transforms.Compose([
    transforms.ToPILImage(),          
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

train_transform = transforms.Compose([
    transforms.ToPILImage(),           
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.1,0.1,0.1,0.05),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

all_dataset = CoinDataset(
    csv_file='data/train.csv',
    img_dir='data/train',
    transform=None
)

train_dataset_no_transform, test_dataset_no_transform = random_split(
    all_dataset,
    [0.9, 0.1],
    generator=torch.Generator().manual_seed(0)
)

train_dataset = TransformedDataset(train_dataset_no_transform, transform=train_transform)
test_dataset  = TransformedDataset(test_dataset_no_transform, transform=test_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4
)

print("Training and testing datasets ready")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_classes = all_dataset.get_num_classes()

if model == 'vit_b_16':
    net = vit_b_16(weights=weights)
else:
    raise ValueError("Invalid model name. Choose 'vit_b_16'.")

hidden_dim = net.heads.head.in_features
net.heads = nn.Sequential(
    # nn.Dropout(p=dropout_rate),
    nn.Linear(hidden_dim, num_classes)
)
net = net.to(device)
print("model = ", net.__class__.__name__)

criterion = nn.CrossEntropyLoss()
# optimizer = optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.01)
optimizer = optim.Adam(net.parameters(), lr=0.001)



print("Network ready to train")

wandb.config.update({"net_summary": str(net)})
wandb.watch(net, log="all")


train(
    net=net,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=num_epochs,
    init_lr=0.01,
    device=device,
    trainloader=train_loader,
    test_loader=test_loader
)


torch.save(
    net.state_dict(),
    f"vit_b16_coin_{num_epochs}epochs_{int(time.time())}.pth"
)
print("训练完成，模型已保存。")