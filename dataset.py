#Custom dataset for SatelliteSet
import cv2
import pandas as pd
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset


# Custom dataset class to load deep globe dataset
class SatelliteSet(Dataset):
    def __init__(self,
                 # mpandas dataframe loaded with a meta data csv containing image file names
                 meta_data, 
                 # Class dictionary
                 class_dict,
                 # directory where the data is stored
                 data_dir, 
                 # albumentations transform
                 transform=None):
        self.meta_data = meta_data
        self.data_dir = data_dir
        self.transform = transform
        self.class_dict = class_dict

    # number of samples in dataset
    def __len__(self):
        return len(self.meta_data)

    # load and return sample at index idx
    def __getitem__(self, idx):
        # Read image
        img_path = os.path.join(self.data_dir, self.meta_data.iloc[idx]['sat_image_path'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Read target mask
        mask_path = os.path.join(self.data_dir, self.meta_data.iloc[idx]['mask_path'])
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        if self.transform:
            # In the transform from albumentation we pass both the image and the mask together to make sure
            # they undergo the same transformation, e.g. this ensure both have the same random crop
            transformed = self.transform(image = image, mask = mask)
            image = transformed['image'].to(torch.float32)
            mask = transformed['mask']
            mask = torch.tensor(np.apply_along_axis(lambda k: self.class_dict[tuple(k)], 2, mask))
            mask_onehot = torch.zeros((len(self.class_dict), mask.shape[0], mask.shape[1]))
            for w,h in mask.nonzero(as_tuple=False):
                mask_onehot[mask[w,h], w, h] = 1
            mask = mask_onehot.to(torch.float32)

        #image.require_grad = True

        return image, mask
    
