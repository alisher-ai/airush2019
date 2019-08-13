import os
import numpy as np
import pathlib
import pandas as pd
import pickle as pkl
from PIL import Image
import torch
from numpy import random
from torchvision import transforms as tr
from keras_preprocessing.image import ImageDataGenerator
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from nsml import DATASET_PATH


def normalization():
    train_label_path_ = os.path.join(DATASET_PATH, 'train', 'train_label')
    labels = np.load(train_label_path_)
    print(labels.shape)

    weights = np.ones((350, 1))
    maxx = np.argmax(labels, axis=1)
    # minn = 999999
    epsilon = 1e-1
    for ii in range(350):
        print("{} --- {}".format(ii, np.sum(maxx==ii)))
        if np.sum(maxx==ii) < 500:
            weights[ii] = 1
        else:
            weights[ii] = 500 / np.sum(maxx==ii)
    return weights


def train_dataloader(input_size=128, batch_size=64, num_workers=0,):
    image_dir = os.path.join(DATASET_PATH, 'train', 'train_data', 'images')
    train_label_path_ = os.path.join(DATASET_PATH, 'train', 'train_label')
    labels = np.load(train_label_path_)
    train_meta_path = os.path.join(DATASET_PATH, 'train', 'train_data', 'train_with_valid_tags.csv')
    train_meta_data_ = pd.read_csv(train_meta_path, delimiter=',', header=0)
    val_num = int(train_meta_data_.shape[0]*0.01)

    train_meta_data = train_meta_data_.iloc[val_num:, :]
    val_meta_data = train_meta_data_.iloc[:val_num, :]

    train_labels = labels[val_num:, :]
    val_labels = labels[:val_num, :]

    train_dataloader = DataLoader(
        AIRushDataset(image_dir, train_meta_data, label_path=train_label_path_,
                      transform=tr.Compose(
                          [
                              tr.Resize((input_size, input_size)),
                              tr.RandomChoice(
                                  [
                                      tr.RandomAffine(30),
                                      tr.Grayscale(3),
                                      tr.RandomHorizontalFlip(),
                                      tr.RandomRotation(degrees=15),
                                      tr.CenterCrop(size=input_size),
                                      tr.RandomResizedCrop(input_size),
                                      tr.ColorJitter(.4, .4, .4)
                                      ]),
                              tr.ToTensor(),
                              tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                          ]
                      ),
                      labels=train_labels),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True)

    val_dataloader = DataLoader(
        AIRushDataset(image_dir, val_meta_data, label_path=train_label_path_,
                      transform=tr.Compose([tr.Resize((input_size, input_size)), tr.ToTensor(),
                                            tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                            ]),
                      labels=val_labels),
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True)

    return train_dataloader, val_dataloader


class AIRushDataset(Dataset):
    def __init__(self, image_data_path, meta_data, label_path=None, transform=None, labels=None):
        self.meta_data = meta_data
        self.image_dir = image_data_path
        self.label_path = label_path
        self.transform = transform
        self.labels = labels

        if self.label_path is not None:
            if self.labels is not None:
                # self.label_matrix = np.load(label_path)
                self.label_matrix = self.labels

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, str(self.meta_data['package_id'].iloc[idx]), str(self.meta_data['sticker_id'].iloc[idx]) + '.png')
        png = Image.open(img_name).convert('RGBA')
        png.load()  # required for png.split()

        new_img = Image.new("RGB", png.size, (255, 255, 255))
        new_img.paste(png, mask=png.split()[3])  # 3 is the alpha channel

        if self.transform:
            new_img = self.transform(new_img)

        if self.label_path is not None:
            tags = torch.tensor(np.argmax(self.label_matrix[idx]))
            return new_img, tags
        else:
            return new_img

