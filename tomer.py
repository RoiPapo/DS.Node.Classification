
# # Commented out IPython magic to ensure Python compatibility.
# # Install required packages.
# import os
# import torch
# os.environ['TORCH'] = torch.__version__
# print(torch.__version__)


from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

from dataset import create_ds

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, data):
        super().__init__()
        # torch.manual_seed(666)
        self.num_features = data['x'].shape[1]
        self.num_classes =40 # len(set(data['y'].squeeze(1).numpy()))
        self.conv1 = GCNConv(self.num_features, hidden_channels[0])
        self.conv2 = GCNConv(hidden_channels[0], hidden_channels[1])
        self.conv3 = GCNConv(hidden_channels[1], self.num_classes)

        self.softmax = nn.Softmax(dim=1)                   

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        # x = self.conv2(x, edge_index)
        # x = x.relu()
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return self.softmax(x)



def train(model,optimizer, criterion, data):
      model.train()
      optimizer.zero_grad()  # Clear gradients.
      out = model(data.x, data.edge_index)  # Perform a single forward pass.
      loss = criterion(out[data.train_mask], data.y[data.train_mask].squeeze())  # Compute the loss solely based on the training nodes.
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      return loss

def val(model, criterion, data):
      model.eval()
      out = model(data.x, data.edge_index)
      val_loss = criterion(out[data.val_mask], data.y[data.val_mask].squeeze())  # Compute the loss solely based on the training nodes.

      pred = out.argmax(dim=1)
      val_correct = pred[data.val_mask] == data.y[data.val_mask].squeeze()  # Check against ground-truth labels.
      val_acc = int(val_correct.sum()) / len(data.val_mask)  # Derive ratio of correct predictions.
      return val_acc, val_loss





def train_aux(data):
    model = GCN(hidden_channels=[96, 128], data=data)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # weight_decay=5e-4
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(1, 600):
        train_loss = train(model,optimizer, criterion, data)
        val_acc, val_loss = val(model,criterion, data)

        print(f'Epoch: {epoch:03d}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, val acc: {val_acc:.4f}')

def main(): 
    dataset = create_ds()
    train_aux(dataset)

if __name__=="__main__":
    main()
