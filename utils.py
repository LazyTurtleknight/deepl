import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import pandas as pd

from PIL import Image
from dataset import SatelliteSet
from torch.utils.data import SubsetRandomSampler

def plot(sample, data_dir):

    plt.figure(figsize=(5,4))
    ax = plt.subplot(2,2,1)
    plt.imshow(np.asarray(Image.open(os.path.join(data_dir, sample['sat_image_path'].iloc[0]))))
    plt.gray()
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)

    ax = plt.subplot(2,2,2)
    plt.imshow(np.asarray(Image.open(os.path.join(data_dir, sample['mask_path'].iloc[0]))))
    plt.gray()
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)

    plt.show()

# TODO: make something like a dictionary for parametere to pass to data loader
def get_data_loaders(data_dir, transform, shuffle_dataset, test_split, random_seed, batch_size):

    # Load metadata csv
    metadata = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))

    # We need to filter for row where 'split' is 'train' because samples where
    # 'split' is 'valid' or 'test' have no target mask
    metadata = metadata[metadata['split'] == 'train']

    class_dict = pd.read_csv(os.path.join(data_dir, 'class_dict.csv'))

    classes = {}
    c = 0
    for i in class_dict.index:
        classes[tuple(class_dict.iloc[i,1:])] = c
        c += 1

    # We need to filter for row where 'split' is 'train' because samples where
    # 'split' is 'valid' or 'test' have no target mask

    dataset = SatelliteSet(meta_data=metadata, class_dict=classes, data_dir=data_dir, transform=transform)

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    #TODO:Implement validation

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                            sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=test_sampler)
    return train_loader, test_loader

def evaluate(model, writer, dataloader, epoch):
    iou_score = 0
    dice_score = 0
    for x, y in dataloader:
        pred = model(x)
        # Calculate IOU
        iou_score += get_iou_score(pred, y)
        # Calculate DICE
        dice_score += get_dice_score(pred, y)

    writer.add_scalar('DICE Score', dice_score / len(dataloader), epoch)
    writer.add_scalar('IOU Score', iou_score / len(dataloader), epoch)

def get_iou_score(pred, y):
    intersection = np.logical_and(y, pred)
    union = np.logical_or(y, pred)
    iou_score += np.sum(intersection) / np.sum(union)
    return iou_score

def get_dice_score(pred, y):
    intersection = np.logical_and(y, pred)
    dice_score += np.sum(intersection) / (y.size() + pred.size())
    return dice_score