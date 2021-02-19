#!/bin/env python3

import argparse
import os
from collections import namedtuple
from math import sqrt

import torch
import torchvision
from torch import nn
from torchvision import models, transforms
from torch.utils.data import Dataset, TensorDataset
import numpy as np

import cv2 as cv

import split_imgs

# to be sure that we don't mix them, use this instead of a tuple
TestResult = namedtuple('TestResult', 'truth predictions')

class NumpyArrays(Dataset):
    """
    Generic numpy array data loader
    """

    def __init__(self, arrays: np.array, transform=None):
        self.arrays = arrays
        self.transform = transform

    def __len__(self):
        return len(self.arrays)

    def __getitem__(self, idx):
        image = self.arrays[idx]
        if self.transform is not None:
            image = self.transform(image)
        # WARNING -1 indicates no target, it's useful to keep the same interface as torchvision
        return image, -1

def map_to_color(predicted_class):
    color_map = {
        0: (240, 216, 62), # anual crop
        1: (23, 94, 33), # forest
        2: (114, 186, 124), # herbaceous vegetation
        3: (97, 97, 97), # highway
        4: (120, 86, 0), # industrial
        5: (119, 255, 0), # pasture
        6: (255, 217, 0), # permanent crop
        7: (255, 255, 255), # residential
        8: (92, 250, 255), # river
        9: (30, 26, 171) # sea or lake
    }
    return np.array(color_map.get(predicted_class,(0,0,0))).astype('float32')

@torch.no_grad()
def predict_mask(image, chunks_num, model_path, batch_size=8, workers=4):
    """
    Run the model on the specified data.
    Automatically moves the samples to the same device as the model.
    """
    save = torch.load(model_path, map_location='cpu')
    normalization = save['normalization']
    model = models.resnet50(num_classes=save['model_state']['fc.bias'].numel())
    model.load_state_dict(save['model_state'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    tr = transforms.Compose([transforms.ToTensor(), transforms.Normalize(**normalization)])
    splitted = split_imgs.split(image, chunks_num)
    test = NumpyArrays(splitted, transform=tr)
    test_dl = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True
    )

    device = next(model.parameters()).device

    model.eval()
    preds = []
    for images, _ in test_dl:
        images = images.to(device, non_blocking=True)
        p = model(images).argmax(1).tolist()
        preds += p

    chunks_dim = int(sqrt(chunks_num))
    preds_np = np.array(preds).reshape(chunks_dim, chunks_dim)
    g = np.vectorize(map_to_color, signature='()->(n)')
    color_mask = g(preds_np)

    resized = cv.resize(color_mask, image.shape[:-1], cv.INTER_NEAREST)

    return resized
