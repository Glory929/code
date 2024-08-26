# coding=gbk
import os
from torch.utils.data import Dataset
import random
import numpy as np
from torchvision.transforms import transforms
import h5py
import torch
import torch.nn.functional as F


class BraTS(Dataset):
    def __init__(self, data_path, file_path, transform=None):
        with open(file_path, 'r') as f:
            self.paths = [os.path.join(data_path, x.strip()) for x in f.readlines()]
        self.transform = transform

    def __getitem__(self, item):
        h5f = h5py.File(self.paths[item], 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        # [0,1,2,4] -> [0,1,2,3]
        label[label == 4] = 3
        # print(image.shape)
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample['image'], sample['label']

    def __len__(self):
        return len(self.paths)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]


class RandomCrop(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        (c, w, h, d) = image.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[:, w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return {'image': image, 'label': label}


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        (c, w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[:, w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


class RandomRotFlip(object):

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.stack([np.rot90(x, k) for x in image], axis=0)
        label = np.rot90(label, k)
        axis = np.random.randint(1, 4)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis - 1).copy()

        return {'image': image, 'label': label}


def augment_gaussian_noise(data_sample, noise_variance=(0, 0.1)):
    if noise_variance[0] == noise_variance[1]:
        variance = noise_variance[0]
    else:
        variance = random.uniform(noise_variance[0], noise_variance[1])
    data_sample = data_sample + np.random.normal(0.0, variance, size=data_sample.shape)
    return data_sample


class GaussianNoise(object):
    def __init__(self, noise_variance=(0, 0.1), p=0.5):
        self.prob = p
        self.noise_variance = noise_variance

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if np.random.uniform() < self.prob:
            image = augment_gaussian_noise(image, self.noise_variance)
        return {'image': image, 'label': label}


class ToTensor(object):

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        return {'image': image, 'label': label}


if __name__ == '__main__':
    data_path = '../datasets/BraTSout'
    train_txt = '../datasets/BraTSout/train.txt'
    patch_size = (160, 160, 128)
    train_dataset = BraTS(data_path, train_txt, transform=transforms.Compose([
        RandomRotFlip(),
        RandomCrop(patch_size),
        GaussianNoise(p=0.1),
        ToTensor()
    ]))
    # print(train_dataset[0][0].shape)
    # print(train_dataset[0][1].shape)

    sample = train_dataset[0]  #
    image = sample[0]
    label = sample[1]
    print(image.shape)
    print(label.shape)
    label_2 = F.one_hot(label, 4)
    print(label_2.shape)
