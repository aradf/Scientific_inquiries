import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
#from plotcm import plot_confusion_matrix

import pdb


# class OHLC(Dataset):
#     def __init__(self, csv_file):
#         self.data = pd.read_csv(csv_file)
    
#     def __getitem__(self, index):
#         r = self.data.iloc[index]
#         label = torch.tensor(r.is_up_day, dtype=torch.long)
#         sample = self.normalize(torch.tensor([r.open, r.high, r.low, r.close]))
#         return sample, label
    
#     def __len__(self):
#         return len(self.data)
    
    

# class FashionMNIST(MNIST):
#     """`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.

#     Args:
#         root (string): Root directory of dataset where ``processed/training.pt``
#             and  ``processed/test.pt`` exist.
#         train (bool, optional): If True, creates dataset from ``training.pt``,
#             otherwise from ``test.pt``.
#         download (bool, optional): If true, downloads the dataset from the internet and
#             puts it in root directory. If dataset is already downloaded, it is not
#             downloaded again.
#         transform (callable, optional): A function/transform that  takes in an PIL image
#             and returns a transformed version. E.g, ``transforms.RandomCrop``
#         target_transform (callable, optional): A function/transform that takes in the
#             target and transforms it.
#     """
#     urls = ['http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
#             'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
#             'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
#             'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',]

def main():
    print ("Hello World")
    torch.set_printoptions(linewidth=120)
    train_set = torchvision.datasets.FashionMNIST(root='./data', 
                                                  train=True,
                                                  download=True,
                                                  transform=transforms.Compose([transforms.ToTensor()]))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=10, shuffle=True)
    print(len(train_set))
    print(train_set.train_labels)
    print(train_set.targets)
    print(train_set.train_labels.bincount())
    print(train_set.targets.bincount())
    sample = next(iter(train_set))
    print(type(sample))
    image, label = sample
    print(type(label))
    print(image.shape)
    # print(label.shape)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.show()
    print('label : ', label)

    print(image.squeeze().shape)
    
    print(torch.tensor(label))
    display_loader = torch.utils.data.DataLoader(train_set, batch_size=10)
    batch = next(iter(display_loader))
    print('len: ', len(batch))
    images, labels = batch
    print('types:', type(images), type(labels))
    print('shapes:', images.shape, labels.shape)
    print(images[0].shape)
    print(labels.shape)
    
    grid = torchvision.utils.make_grid(images, nrow=10)
    plt.figure(figsize=(15,15))
    plt.imshow(np.transpose(grid, (1,2,0)))
    plt.show()
    print('labels:', labels)
    grid = torchvision.utils.make_grid(images, nrow=10)
    plt.figure(figsize=(15,15))
    plt.imshow(grid.permute(1,2,0))
    plt.show()
    print('labels:', labels)

    how_many_to_plot = 20
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
    plt.figure(figsize=(50,50))
    for i, batch in enumerate(train_loader, start=1):
        image, label = batch
        plt.subplot(10,10,i)
        plt.imshow(image.reshape(28,28), cmap='gray')
        plt.axis('off')
        plt.title(train_set.classes[label.item()], fontsize=28)
        if (i >= how_many_to_plot): break

    plt.show()
    

if __name__ == "__main__":
    main()
    print("Got this far.")