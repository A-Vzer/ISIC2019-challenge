from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
import numpy as np


# Create dataset from ISIC images
class ISICDatset(Dataset):
    def __init__(self, img_path, transform, csv_path=None, test=False):
        self.targets = pd.read_csv(csv_path)
        self.img_path = img_path
        self.transform = transform
        self.test = test

    def __getitem__(self, index):
        img_name = os.path.join(self.img_path, f'{self.targets.iloc[index, 0]}.jpg')
        img = Image.open(img_name)
        img = self.transform(img)
        if not self.test:
            targets = self.targets.iloc[index, 1:]
            targets = np.array([targets])
            targets = targets.astype('float').reshape(-1, 9)
            return {'image': img, 'label': targets}
        else:
            return {'image': img}

    def __len__(self):
        return len(self.targets)