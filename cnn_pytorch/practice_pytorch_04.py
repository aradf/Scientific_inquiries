import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import tensorflow as tf

import torchvision
import torchvision.transforms as transforms

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True) 
print ( torch.__version__)
print (torchvision.__version__)

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

@torch.no_grad()
def get_all_preds(model, loader):
    all_preds = torch.tensor([])
    for batch in loader:
        images, labels = batch

        preds = model(images)
        all_preds = torch.cat(
            (all_preds, preds)
            ,dim=0
        )
    return all_preds

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

def main():
    network = Network()

    train_set = torchvision.datasets.FashionMNIST(root='./data', 
                                                  train=True,
                                                  download=True,
                                                  transform=transforms.Compose([transforms.ToTensor()]))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01)

    for epoch in range(10):

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

        print("epoch", epoch, "total_correct:", total_correct, "loss:", total_loss)

def concat_stack():
    import torch
    t1 = torch.tensor([1,1,1])
    print(t1)
    print(t1.unsqueeze(dim=0))
    print(t1.unsqueeze(dim=1))

    print(t1.shape)
    print(t1.unsqueeze(dim=0).shape)
    print(t1.unsqueeze(dim=1).shape)

    t1 = torch.tensor([1,1,1])
    t2 = torch.tensor([2,2,2])
    t3 = torch.tensor([3,3,3])

    print (torch.cat((t1,t2,t3), dim=0))
    print (torch.stack((t1,t2,t3), dim=0))
    
    print (torch.cat((t1.unsqueeze(0), t2.unsqueeze(0), t3.unsqueeze(0)),dim=0))
    print (torch.cat((t1.unsqueeze(0), t2.unsqueeze(0), t3.unsqueeze(0)),dim=1))

    print (t1.unsqueeze(1))
    print (t2.unsqueeze(1))
    print (t3.unsqueeze(1))

    t1 = tf.constant([1,1,1])
    t2 = tf.constant([2,2,2])
    t3 = tf.constant([3,3,3])
    print(t1, t2, t3)
    print(tf.concat((t1,t2,t3), axis=0))
    print(tf.stack((t1,t2,t3), axis=0))
    print(tf.concat((tf.expand_dims(t1, 0), tf.expand_dims(t2, 0), tf.expand_dims(t3, 0)), axis=0))
    
    print(tf.stack((t1,t2,t3), axis=1))
    print(tf.concat((tf.expand_dims(t1, 1),
                        tf.expand_dims(t2, 1),
                        tf.expand_dims(t3, 1)), axis=1))

    import numpy as np
    t1 = np.array([1,1,1])
    t2 = np.array([2,2,2])
    t3 = np.array([3,3,3])
    np.concatenate((t1,t2,t3), axis=0)
    np.stack((t1,t2,t3), axis=0)
    np.concatenate((np.expand_dims(t1, 0),
                    np.expand_dims(t2, 0),
                    np.expand_dims(t3, 0)),axis=0)
    np.stack((t1,t2,t3), axis=1)
    np.concatenate((np.expand_dims(t1, 1),
                    np.expand_dims(t2, 1),
                    np.expand_dims(t3, 1)), axis=1)
    
    import torch
    t1 = torch.zeros(3,28,28)
    t2 = torch.zeros(3,28,28)
    t3 = torch.zeros(3,28,28)
    torch.stack((t1,t2,t3), dim=0).shape

    import torch
    t1 = torch.zeros(1,3,28,28)
    t2 = torch.zeros(1,3,28,28)
    t3 = torch.zeros(1,3,28,28)
    torch.cat((t1,t2,t3), dim=0).shape

    import torch
    batch = torch.zeros(3,3,28,28)
    t1 = torch.zeros(3,28,28)
    t2 = torch.zeros(3,28,28)
    t3 = torch.zeros(3,28,28)

    torch.cat((batch, torch.stack((t1,t2,t3), dim=0)), dim=0).shape
    import torch
    batch = torch.zeros(3,3,28,28)
    t1 = torch.zeros(3,28,28)
    t2 = torch.zeros(3,28,28)
    t3 = torch.zeros(3,28,28)

    torch.cat((batch, t1.unsqueeze(0), t2.unsqueeze(0), t3.unsqueeze(0)), dim=0).shape
    print("Got this far")
                    
if __name__ == "__main__":
    concat_stack()
    main()
    print("Got this far.")
