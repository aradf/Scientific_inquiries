import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import math

from collections import OrderedDict

torch.set_printoptions(linewidth=150)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 12, 5)

        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):

        t = F.relu(self.conv1(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)

        return t

def main():
    print("hello world")
    torch.manual_seed(50)
    train_set = torchvision.datasets.FashionMNIST(root='./data',
                                                  train=True,
                                                  download=True,
                                                  transform=transforms.Compose([transforms.ToTensor()]))
    image, label = train_set[0]
    print (image.shape)
    in_features = image.numel()
    print (in_features)

    plt.imshow(image.squeeze(), cmap='gray')
    plt.show()
    print(train_set.classes)


    out_features = math.floor(in_features / 2)
    print (out_features)
    out_classes = len(train_set.classes)
    print (out_classes)

    network1 = nn.Sequential(nn.Flatten(start_dim=1), 
                             nn.Linear(in_features, out_features),
                             nn.Linear(out_features, out_classes))
    
    print (network1(image))
    
    layers = OrderedDict([('flat', nn.Flatten(start_dim=1)),
                          ('hidden', nn.Linear(in_features, out_features)),
                          ('output', nn.Linear(out_features, out_classes))])
    network2 = nn.Sequential(layers)

    print(network2(image))

    network3 = nn.Sequential()
    network3.add_module('flat', nn.Flatten(start_dim=1))
    network3.add_module('hidden', nn.Linear(in_features, out_features))
    network3.add_module('output', nn.Linear(out_features, out_classes))

    print(network3(image))

    network4 = Network()

    sequential = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5), 
                              nn.ReLU(), 
                              nn.MaxPool2d(kernel_size=2, stride=2), 
                              nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5), 
                              nn.ReLU(), 
                              nn.MaxPool2d(kernel_size=2, stride=2), 
                              nn.Flatten(start_dim=1), 
                              nn.Linear(in_features=12*4*4, out_features=120), 
                              nn.ReLU(), 
                              nn.Linear(in_features=120, out_features=60), 
                              nn.ReLU(), 
                              nn.Linear(in_features=60, out_features=10))
    
    

if __name__ == "__main__":
    main()
    print("The end is nigh")

