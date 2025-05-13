import os
import torch
import pandas as pd
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.models import densenet121
from torch.utils.data import Dataset, DataLoader
import imageio.v3 as iio
import numpy as np
import torch.optim as optim
from utils import AdaFace
from torchvision.models.densenet import DenseNet121_Weights


# Dataset class for the test data
class TestDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_id = self.data.iloc[idx, 0]
        img_name = os.path.join(self.img_dir, f"{img_id}")

        # Try different image formats
        try:
            image = iio.imread(f"{img_name}.jpg")
        except:
            try:
                image = iio.imread(f"{img_name}.png")
            except:
                image = iio.imread(f"{img_name}.webp")

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

        return image, img_id

def main():
    # Path to model checkpoint
    model_path = 'checkpoints/adaface_densenet.pth'  
    
    # Paths for test data
    test_csv = 'data/test.csv' 
    test_img_dir = 'data/test' 
    
    # Path for output file
    output_file = 'submission.csv'
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define transform for test images
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((225, 225)),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load the class mapping from the training data
    # This is necessary to map numeric predictions back to class labels
    train_csv = 'data/train.csv'  # Update with your train CSV path
    train_data = pd.read_csv(train_csv)
    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    labelencoder.fit(train_data['Class'])
    num_classes = len(labelencoder.classes_)
    
    print(f"Number of classes: {num_classes}")
    
    # Create test dataset and dataloader
    test_dataset = TestDataset(
        csv_file=test_csv,
        img_dir=test_img_dir,
        transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    backbone = densenet121(weights=DenseNet121_Weights.DEFAULT)
    feat_dim = backbone.classifier.in_features
    backbone.classifier = nn.Identity()

    arc_face = AdaFace(
        in_features=feat_dim,
        out_features=num_classes,
        s=32.0,      
        m=0.4,       
        h=0.333      
    )

    backbone.load_state_dict(torch.load(model_path, map_location=device))
    backbone = backbone.to(device)
    arc_face = arc_face.to(device)

    # Set model to evaluation mode
    backbone.eval()
    arc_face.eval()
    
    # Make predictions
    print("Making predictions...")
    predictions = []
    
    with torch.no_grad():
        for images, img_ids in test_loader:
            images = images.to(device)
            
            # Extract features
            features = backbone(images)
            
            # Get logits (we don't have labels in test set, so we use a modified version of AdaFace)
            logits = arc_face(features)
            
            # Get predicted class indices
            _, predicted = torch.max(logits, 1)
            
            # Convert predicted indices to original class labels
            predicted_classes = labelencoder.inverse_transform(predicted.cpu().numpy())
            
            # Save predictions
            for img_id, pred_class in zip(img_ids, predicted_classes):
                predictions.append({'Id': img_id, 'Class': pred_class})
    
    # Create submission DataFrame
    submission_df = pd.DataFrame(predictions)
    
    # Save to CSV
    submission_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    main()