import torch
import wandb
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, densenet121, ResNet18_Weights, DenseNet121_Weights, resnet50, ResNet50_Weights
from utils import *
from torch.utils.data import DataLoader, random_split

# Hyperparameters:
model = 'densenet121'
batch_size = 128
dropout_rate = 0.4
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


# Initialize wandb
wandb.init(
    project="CS444-Final Project",  # Project name
    name="densenet_adaface_0.8_128",               # Run name
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




test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((225, 225)),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((250, 250)), 
    transforms.RandomCrop((225, 225)),  # crop to target size
    transforms.RandomHorizontalFlip(p=0.5),  
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


all_dataset = CoinDataset(
    csv_file='data/train.csv',
    img_dir='data/train',
    transform=None
)

# # Split the entire train datasets into train and test sets, 0.9 for train and 0.1 for test
train_dataset_no_transform, test_dataset_no_transform = random_split(all_dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(0))

train_dataset = TransformedDataset(train_dataset_no_transform, transform=train_transform)
test_dataset = TransformedDataset(test_dataset_no_transform, transform=test_transform)

train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size,
    shuffle=True,
    num_workers=4
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=batch_size,
    shuffle=False, # not to shuffle because we want consistent performance evaluation
    num_workers=4
)

print("Training and testing datasets ready")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

get_num_classes = all_dataset.get_num_classes()


backbone = densenet121(weights=DenseNet121_Weights.DEFAULT)
feat_dim = backbone.classifier.in_features
backbone.classifier = nn.Identity()


arc_face = AdaFace(
    in_features=feat_dim,
    out_features=get_num_classes,
    s=32.0,      
    m=0.4,       
    h=0.333      
)
backbone = backbone.to(device)
arc_face = arc_face.to(device)




criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(backbone.parameters()) + list(arc_face.parameters()), lr=0.001)
print("Network ready to train")


wandb.config.update({"net_summary": str(backbone)})
wandb.watch(backbone, log="all")
wandb.watch(arc_face, log="all")


# Trian 
train_arcface(backbone, arc_face, criterion, optimizer,
      num_epochs, init_lr=0.001, device = device, train_loader = train_loader, test_loader = test_loader)

# Save the model
torch.save(backbone.state_dict(), f'densenet_adaface_{time.time()}.pth')

