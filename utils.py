import os
import torch
import wandb
import pandas as pd
import numpy as np
import time
import math
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import imageio.v3 as iio
import torch.nn.functional as F

# Datasets
class CoinDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.coin_data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.labelencoder = LabelEncoder()
        self.labels = self.coin_data['Class']
        self.encoded_labels = torch.tensor(self.labelencoder.fit_transform(self.labels))
        
    def __len__(self):
        return len(self.coin_data)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, f"{self.coin_data.iloc[idx, 0]}")

        # PIL.Image and torchvision.io.decode_image do not work for some files
        try:
            image = iio.imread(f"{img_name}.jpg")
        except:
            try:
                image = iio.imread(f"{img_name}.png")
            except:
                image = iio.imread(f"{img_name}.webp")

        label = self.encoded_labels[idx].clone().detach().long()

        if len(image.shape) == 4:
            image = image[0]

        # if the image is 1 channel, grey scale
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=2)

        # if the image is 4 channels, rgba
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_original_label(self, encoded_label):
        return self.labelencoder.inverse_transform([encoded_label])[0]

    def get_num_classes(self):
        return len(self.labelencoder.classes_)

    def get_class_mapping(self):
        classes = self.labelencoder.classes_
        indices = self.labelencoder.transform(classes)
        return dict(zip(classes, indices))

class TransformedDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.dataset)




def adjust_learning_rate(optimizer, epoch, init_lr, total_epochs):
    lr = init_lr * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def run_test(net, testloader, criterion, device, epoch):
    correct = 0
    total = 0
    avg_test_loss = 0.0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # loss
            avg_test_loss += criterion(outputs, labels)  / len(testloader)

    print('TESTING:')
    print(f'Accuracy of the network on test images: {100 * correct / total:.2f} %')
    print(f'Average loss on test images: {avg_test_loss:.3f}')

    # log test loss and acc in wandb
    wandb.log({
        "test_loss": avg_test_loss,
        "test_acc": 100 * correct / total,
        "epoch": epoch
    })


# Both the self-supervised rotation task and supervised CIFAR10 classification are
# trained with the CrossEntropyLoss, so we can use the training loop code.
def train(net, criterion, optimizer, num_epochs, init_lr, device, trainloader, test_loader):
    print("Training begins")
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        running_correct = 0.0
        running_total = 0.0
        counter = 0
        start_time = time.time()

        net.train()
        for i, (imgs, labels) in enumerate(trainloader):
            
            adjust_learning_rate(optimizer, epoch, init_lr, num_epochs)
            
            device = torch.device("cuda")

            inputs = imgs.clone().detach().to(device)
            labels = labels.clone().detach().to(device)

            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            print_freq = 60
            running_loss += loss.item()

            # calc acc
            running_total += labels.size(0)
            running_correct += (predicted == labels).sum().item()

            counter += 1
            
        # log train loss and acc
        print('TRAINING:')
        print(f'epoch: {epoch} loss: {running_loss / counter:.3f} acc: {100*running_correct / running_total:.2f} time: {time.time() - start_time:.2f}')
        
        # log train loss and acc in wandb
        wandb.log({
            "train_loss": running_loss / counter,
            "train_acc": 100 * running_correct / running_total,
            "epoch": epoch
        })

        running_loss, running_correct, running_total = 0.0, 0.0, 0.0
        start_time = time.time()


        net.eval()
        run_test(net, test_loader, criterion, device, epoch)

    print('Finished Training')



class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))

        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m


    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight)) #[N, C]

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)+1e-6)
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)

        ouput = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        return ouput * self.s
    
class AdaFace(nn.Module):
    def __init__(self, in_features, out_features, m=0.4, h=0.333, s=64.0, t_alpha=1.0, eps=1e-3):
        super(AdaFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.h = h
        self.s = s
        self.eps = eps
        self.t_alpha = t_alpha

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.register_buffer("batch_moments", torch.zeros(2))  # mean, std of norms

    def forward(self, x, label):
        norm_x = torch.norm(x, dim=1, keepdim=True).clamp(min=self.eps)
        x_normalized = x / norm_x
        w_normalized = F.normalize(self.weight)

        cosine = F.linear(x_normalized, w_normalized)

        with torch.no_grad():
            batch_mean = norm_x.mean()
            batch_std = norm_x.std()
            self.batch_moments[0] = (1 - 0.01) * self.batch_moments[0] + 0.01 * batch_mean
            self.batch_moments[1] = (1 - 0.01) * self.batch_moments[1] + 0.01 * batch_std

        margin_scaler = ((norm_x - self.batch_moments[0]) / (self.batch_moments[1] + self.eps)).clamp(-1, 1) * self.h
        theta = torch.acos(cosine.clamp(-1.0 + self.eps, 1.0 - self.eps))
        final_theta = theta + self.m * margin_scaler
        target_cos = torch.cos(final_theta)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)

        output = (one_hot * target_cos) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        return output


def train_arcface(backbone, arc_face, criterion, optimizer,
          num_epochs, init_lr, device, train_loader, test_loader):
    print("Training begins")
    for epoch in range(num_epochs):
        backbone.train()
        arc_face.train()
        running_loss, running_correct, running_total = 0.0, 0.0, 0.0
        counter = 0

        start_time = time.time()

        for idx, (imgs, labels) in enumerate(train_loader):
            if idx == 0:
                print("Training batch: ", idx)
            adjust_learning_rate(optimizer, epoch, init_lr, num_epochs)

            imgs   = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()


            features = backbone(imgs)               # [N, feat_dim]

            logits   = arc_face(features, labels)   # [N, num_classes]

            loss     = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss    += loss.item()
            running_total   += labels.size(0)
            running_correct += (logits.argmax(1) == labels).sum().item()
            counter         += 1


        train_loss = running_loss / counter
        train_acc  = 100.0 * running_correct / running_total
        elapsed    = time.time() - start_time

        print(f'TRAINING: epoch {epoch} | '
              f'loss: {train_loss:.3f} | acc: {train_acc:.2f}% | time: {elapsed:.2f}s')
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc
        })

        backbone.eval()
        arc_face.eval()
        val_loss, val_correct, val_total = 0.0, 0.0, 0.0
        val_counter = 0

        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs   = imgs.to(device)
                labels = labels.to(device)

                feats  = backbone(imgs)
                logits = arc_face(feats, labels)
                loss   = criterion(logits, labels)

                val_loss    += loss.item()
                val_total   += labels.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_counter += 1

        val_loss = val_loss / val_counter
        val_acc  = 100.0 * val_correct / val_total

        print(f'VALIDATION: epoch {epoch} | '
              f'loss: {val_loss:.3f} | acc: {val_acc:.2f}%')
        wandb.log({
            "test_loss": val_loss,
            "test_acc": val_acc
        })

    print('Finished Training')

