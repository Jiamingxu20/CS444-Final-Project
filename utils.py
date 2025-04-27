import os
import torch
import pandas as pd
import torchvision.transforms as transforms
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import imageio.v3 as iio


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


def run_test(net, testloader, criterion, device):
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
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f} %')
    print(f'Average loss on the 10000 test images: {avg_test_loss:.3f}')


# Both the self-supervised rotation task and supervised CIFAR10 classification are
# trained with the CrossEntropyLoss, so we can use the training loop code.
def train(net, criterion, optimizer, num_epochs, decay_epochs, init_lr, device, trainloader, test_loader):

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        running_correct = 0.0
        running_total = 0.0
        start_time = time.time()

        net.train()
        for i, (imgs, labels) in enumerate(trainloader):
            # adjust_learning_rate(optimizer, epoch, init_lr, decay_epochs)
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
            print_freq = 100
            running_loss += loss.item()

            # calc acc
            running_total += labels.size(0)
            running_correct += (predicted == labels).sum().item()

            if i % print_freq == (print_freq - 1):    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / print_freq:.3f} acc: {100*running_correct / running_total:.2f} time: {time.time() - start_time:.2f}')
                running_loss, running_correct, running_total = 0.0, 0.0, 0.0
                start_time = time.time()

        net.eval()
        run_test(net, test_loader, criterion, device)

    print('Finished Training')



def adjust_learning_rate(optimizer, epoch, init_lr, decay_epochs=30):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // decay_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



