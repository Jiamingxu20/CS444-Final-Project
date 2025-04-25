import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from zipfile import ZipFile
import matplotlib.pyplot as plt
import cv2
import random
import os
from collections import Counter


"""## 1. Load Data

We'll read the CSV files: `train.csv` and `test.csv`. The `train.csv` includes both an `Id` (our unique image identifier) and a `Class` label.
"""

train_df = pd.read_csv('kaggle/train.csv')
test_df = pd.read_csv('kaggle/test.csv')

print('Train shape:', train_df.shape)
print('Test shape:', test_df.shape)

train_df.head()

"""## 2. Basic Exploratory Data Analysis
We'll look at how many classes we have, and how many images per class.
"""

num_classes = train_df['Class'].nunique()
print(f'Number of classes: {num_classes}')

class_counts = train_df['Class'].value_counts()
class_counts.head(10)

"""### Distribution of the Top 10 Classes

Let's visualize the 10 most common classes in the training set.
"""

top_10 = class_counts.head(10)
plt.figure(figsize=(8, 4))
top_10.plot(kind='bar')
plt.title('Top 10 Classes by Image Count')
plt.xlabel('Class Name')
plt.ylabel('Count')
plt.show()


"""## 3. Challenges & Caveats
Below are some important details about this dataset:

- **Class Imbalances**: Some coins have many samples, while others are relatively underrepresented.
- **Mint Variations**: Coins of the same denomination might differ subtly due to different mints or yearly revisions.
- **Type Variations**: Coins of the same denomination might have different types.
- **Varied Conditions**: Tarnish, rust, and scratches can alter a coin’s appearance. Lighting and angle variations further increase complexity.
- **Subtle Differences**: Denominations of euro coins can sometimes only be distinguished by edge patterns.

## Examples of mint and type variations

All images are 50 Cents,Euro,netherlands class, but have substantial differences.
"""



fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 12))

for i, image_id in enumerate([11525, 11522, 11534]):
  image_path = os.path.join('kaggle/train', f'{image_id}.jpg')
  img = cv2.imread(image_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  axes[i].imshow(img)
  axes[i].set_title(f'ID: {image_id}, 50 Cents,Euro,netherlands')
  axes[i].axis('off')

plt.tight_layout()
plt.show()



"""## Subtle Differences
Lets take a look at 20, 50 cents and 1 euro from Germany. Note that the only difference between 20 and 50 cents is the pattern on the edge of the coin.
"""



fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 12))

for i, image_id in enumerate([8780, 11308, 8829]):
  image_path = os.path.join('kaggle/train', f'{image_id}.jpg')
  img = cv2.imread(image_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  axes[i].imshow(img)
  axes[i].set_title(f'ID: {image_id}, Euro,Germany')
  axes[i].axis('off')

plt.tight_layout()
plt.show()


"""## 4. Generate a Random Submission
As a quick baseline, we can create a ‘random classifier’ that:
1. Collects the list of possible classes from `train.csv`.
2. For each row in `test.csv`, randomly picks a class from our list.
3. Writes out a `submission.csv` file with `[Id, Class]` columns.

Obviously, this won't be accurate, but it provides a simple template for how you’ll submit predictions to Kaggle.
"""



classes = train_df['Class'].unique()
test_ids = test_df['Id'].values

preds = np.random.choice(classes, size=len(test_ids))

submission_df = pd.DataFrame({
    'Id': test_ids,
    'Class': preds
})

submission_df.to_csv('submission.csv', index=False)
submission_df.head(10)

"""## 5. Submission"""

