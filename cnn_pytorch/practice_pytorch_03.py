import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True) 
print ( torch.__version__)
print (torchvision.__version__)

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

class Lizard:
    def __init__(self, name):
        self.name = name 
    
    def set_name(self, name):
        self.name = name 


class Network(nn.Module): 
    def __init__(self):
        super(Network, self).__init__() 
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        # (1) input layer 
        # print (t.shape)
        t = t

        # (2) hidden conv layer 
        # print (self.conv1.weight.shape) - print (t.min().item()) - print (t.shape)
        t = self.conv1(t)
        t = F.relu(t)           
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (3) hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (4) hidden linear layer
        t = t.reshape(-1, 12 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)

        # (5) hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)

        # (6) output layer
        t = self.out(t)
        #t = F.softmax(t, dim=1)
        return t

    def __repr__(self):
        return "lizardnet"

def main():
    print("Hello World")
    lizard = Lizard('hello')
    lizard.set_name('deep')
    print (lizard)
    type (lizard )

    network = Network()
    print(network)
    print(network.conv1)
    print(network.fc1)
    print(network.out)
    print(network.conv1.weight)
    print(network.conv1.weight.shape)
    print(network.conv2.weight.shape)
    print(network.fc1.weight.shape)
    print(network.fc2.weight.shape)
    print(network.out.weight.shape)
    print(network.conv2.weight[0].shape)
    in_features = torch.tensor([1, 2, 3, 4],dtype=torch.float32)
    weight_matrix = torch.tensor([[1,2,3,4],
                                  [2,3,4,5],
                                  [3,4,5,6]],dtype=torch.float32)

    print(weight_matrix.matmul(in_features))

    for name, param in network.named_parameters():
        print(name, '\t\t', param.shape)

    for param in network.parameters():
        print(param.shape)

    fc = nn.Linear(in_features=4, out_features=3, bias=True)
    fc(in_features)
    fc.weight = nn.Parameter(weight_matrix)
    fc = nn.Linear(in_features=4, out_features=3, bias=False)
    fc.weight = nn.Parameter(weight_matrix)
    fc(in_features)
    fc = nn.Linear(in_features=4, out_features=3, bias=True)
    t = torch.tensor([1,2,3,4],dtype=torch.float32)
    output = fc(t)


    train_set = torchvision.datasets.FashionMNIST(root='./data', 
                                                  train=True,
                                                  download=True,
                                                  transform=transforms.Compose([transforms.ToTensor()]))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=10, shuffle=True)

    network = Network()
    sample = next(iter(train_set))
    image, label = sample
    print (image.shape)
    image.unsqueeze(0).shape
    # image shape needs to be (batch_size × in_channels × H × W)
    pred = network(image.unsqueeze(0))
    print( pred.shape )
    print( pred.argmax(dim=1))
    print ( F.softmax ( pred, dim=1) )
    print ( F.softmax ( pred, dim=1).sum() )

    data_loader = torch.utils.data.DataLoader(train_set, batch_size=10)
    batch = next(iter(data_loader))
    images, labels = batch
    print ( images.shape )
    print ( labels.shape ) 
    preds = network(images)
    print ( preds.shape )
    print ( preds )
    print ( preds.argmax(dim=1) )
    print (labels) 
    print (preds.argmax(dim=1).eq(labels))
    print (preds.argmax(dim=1).eq(labels).sum())
    get_num_correct(preds, labels)

    train_set = torchvision.datasets.FashionMNIST(root='./data', 
                                                  train=True,
                                                  download=True,
                                                  transform=transforms.Compose([transforms.ToTensor()]))
    network = Network()

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01)

    batch = next(iter(train_loader)) # Get Batch
    images, labels = batch

    preds = network(images) # Pass Batch
    loss = F.cross_entropy(preds, labels) # Calculate Loss

    loss.backward() # Calculate Gradients
    optimizer.step() # Update Weights

    print('loss1:', loss.item())
    preds = network(images)
    loss = F.cross_entropy(preds, labels)
    print('loss2:', loss.item())

    network = Network()

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01)

    total_loss = 0
    total_correct = 0

    for batch in train_loader: # Get Batch
        images, labels = batch 

        preds = network(images) # Pass Batch
        loss = F.cross_entropy(preds, labels) # Calculate Loss

        optimizer.zero_grad()
        loss.backward() # Calculate Gradients
        optimizer.step() # Update Weights

        total_loss += loss.item()
        total_correct += get_num_correct(preds, labels)
        print('loss:', loss.item())
        print('total_loss:',total_loss)


    print("epoch:", 0, "total_correct:", total_correct, "loss:", total_loss)

    print("Got this far.")

    
if __name__ == "__main__":
    main()
    print("Got this far.")
