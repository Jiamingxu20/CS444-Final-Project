import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet18
from utils import *
from torch.utils.data import DataLoader, random_split

# Load the data

batch_size = 64

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((225, 225)),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


all_dataset = CoinDataset(
    csv_file='data/train.csv',
    img_dir='data/train',
    transform=transform
)

# Split the entire train datasets into train and test sets, 0.8 for train and 0.2 for test
train_dataset, test_dataset = random_split(all_dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(0))


train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=batch_size,
    shuffle=False, # not to shuffle because we want consistent performance evaluation
    num_workers=2
)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

get_num_classes = all_dataset.get_num_classes()

net = resnet18(num_classes=get_num_classes)
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Trian 
train(net, criterion, optimizer, num_epochs=45, decay_epochs=15, init_lr=0.01,
       device=device, trainloader=train_loader, test_loader=test_loader)

# Save the model
torch.save(net.state_dict(), 'coin_clas.pth')