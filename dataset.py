#Custom dataset for SatelliteSet
import cv2
import pandas as pd
import os

from torch.utils.data import Dataset, DataLoader


class SatelliteSet(Dataset):
    def __init__(self, meta_data_file, data_dir, transform=None):
        self.meta_data = pd.read_csv(meta_data_file)
        self.data_dir = data_dir
        self.transform = transform

    #number of samples in dataset
    def __len__(self):
        return len(self.meta_data)

    #load and return sample at index idx
    def __getitem__(self, idx):
        # Read image
        img_path = os.path.join(self.data_dir, self.meta_data.iloc[0]['sat_image_path'])
        image = cv2.imread(img_path)
        # Read target mask
        mask_path = os.path.join(self.data_dir, self.meta_data.iloc[0]['mask_path'])
        mask = cv2.imread(mask_path)
        if self.transform:
            # In the transform from albumentation we pass both the image and the mask together to make sure
            # they undergo the same transformation, e.g. this ensure both have the same random crop
            image, mask = self.transform(image = image, mask = mask)
        return image, mask
    
torch.utils.data.random_split(dataset, lengths)

#Dataloader
t_dataloader = DataLoader(t_data, batch_size=64, shuffle=True, num_workers = 2)



#albu to import
#t_transform = alb.Compose(
#   [...]
#)

#t_dataset = SatelliteSet(images_filepaths = FILEPATH TRAINING, transform = t_transform)