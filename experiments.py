from dataset import create_ds
import os
import torch
import numpy as np
from torch_geometric.nn import GCNConv , GATConv
import torch.nn.functional as F
from dataset import HW3Dataset
data = HW3Dataset(root='data/hw3/')


class GAT(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        torch.manual_seed(1234567)
        self.config=config
        self.num_features = data[0]['x'].shape[1] +1
        self.num_classes = len(set(data[0]['y'].squeeze(1).numpy()))
        self.conv1 = GATConv(self.num_features, config["hidden_channels"], heads=config["heads"])
        self.conv2 = GATConv(config["hidden_channels"] * 7, config["hidden_channels"], heads=6)
        self.conv3 = GATConv(config["hidden_channels"] * 6, data.num_classes, heads=6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv3(x, edge_index)
        return x


min_value=data[0].node_year[data[0].train_mask].min()

diff = data[0].node_year[data[0].train_mask].max() - min_value 

best_params={
        'heads':7,
        'hidden_channels':64,
        'lr': 0.005,
        'weight_decay': 0,
        'dropout_parm':0.1,
        'act':'relu'
    }


device = "cuda" if torch.cuda.is_available() else "cpu"

model = GAT(best_params)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005) #, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()


def train():
    model.train()
    optimizer.zero_grad()
    out = model(
        torch.concatenate((data[0].x, (data[0].node_year - min_value) / diff), axis=1).to(device), 
        data[0].edge_index.to(device)
    )
    loss = criterion(
        out[data[0].train_mask].to(device),
        data[0].y[data[0].train_mask].reshape(-1,).to(device)
    )

    loss.backward()
    optimizer.step()
    return loss


def test():
    model.eval()
    
    out = model(
        torch.concatenate((data[0].x, (data[0].node_year - min_value) / diff), axis=1).to(device), 
        data[0].edge_index.to(device)
    )
    pred = out.argmax(dim=1).detach().cpu()
    test_correct = pred[data[0]
                        .val_mask] == data[0].y[data[0].val_mask].reshape(-1,)
    test_acc = int(test_correct.sum()) / len(test_correct)
    
    return test_acc, pred[data[0].val_mask], data[0].y[data[0].val_mask].reshape(-1,)


for epoch in range(1, 610):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    test_acc, test_pred, test_true = test()
    print(f'Test Accuracy: {test_acc:.4f}')
torch.save(model.state_dict(), 'Model.pt')

