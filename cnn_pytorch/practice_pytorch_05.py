import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from collections import OrderedDict
from collections import namedtuple
from itertools import product

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True) 
print ( torch.__version__)
print (torchvision.__version__)

class RunBuilder():
    @staticmethod
    def get_runs(params):

        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs

params = OrderedDict(lr = [.01, .001], batch_size = [1000, 10000])
# runs = RunBuilder.get_runs(params)

Run = namedtuple('Run', params.keys())

print (params.keys())
print (params.values())

runs = []
for v in product(*params.values()):
    runs.append(Run(*v))

print(runs)

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

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
        t = t.flatten(start_dim=1)
        t = self.fc1(t)
        t = F.relu(t)

        # (5) hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)

        # (6) output layer
        t = self.out(t)
        #t = F.softmax(t, dim=1)
        return t

def main():
    print("hello world")
    network = Network()
    train_set = torchvision.datasets.FashionMNIST(root='./data', 
                                                  train=True,
                                                  download=True,
                                                  transform=transforms.Compose([transforms.ToTensor()]))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01)

    images, labels = next(iter(train_loader))
    grid = torchvision.utils.make_grid(images)

    tb = SummaryWriter()
    tb.add_image('images', grid)
    tb.add_graph(network, images)

    for epoch in range(10):

        total_loss = 0
        total_correct = 0

        for batch in train_loader: # Get Batch
            images, labels = batch
            preds = network(images)
            loss = F.cross_entropy( preds, labels)

            optimizer.zero_grad()
            loss.backward() # Calculate Gradients
            optimizer.step() # Update Weights

            total_loss += loss.item()
            total_correct += get_num_correct( preds, labels)


        tb.add_scalar('Loss', total_loss, epoch)
        tb.add_scalar('Number Correct', total_correct, epoch)
        tb.add_scalar('Accuracy', total_correct / len(train_set), epoch)

        tb.add_histogram('conv1.bias', network.conv1.bias, epoch)
        tb.add_histogram('conv1.weight', network.conv1.weight, epoch)
        tb.add_histogram('conv1.weight.grad', network.conv1.weight.grad, epoch)

        print("epoch", epoch, "total_correct:", total_correct, "loss:", total_loss)

    tb.close()

if __name__ == "__main__":
    main()
    print("Got this far.")
