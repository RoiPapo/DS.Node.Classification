from dataset import create_ds
import os
import torch
import numpy as np
from torch_geometric.nn import GCNConv , GATConv
import torch.nn.functional as F
from dataset import HW3Dataset
# import wandb

# start a new wandb run to track this script
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="my-awesome-project",

#     # track hyperparameters and run metadata
#     config={
#         "learning_rate": 0.02,
#         "architecture": "CNN",
#         "dataset": "CIFAR-100",
#         "epochs": 10,
#     }
# )


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, dataset):
        super().__init__()
        torch.manual_seed(666)
        self.num_features = data['x'].shape[1]
        self.num_classes = len(set(data['y'].squeeze(1).numpy()))
        self.conv1 = GCNConv(self.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, self.num_classes)
        # self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        # x= self.softmax(x)
        return x




# class GAT(torch.nn.Module):
#     def __init__(self, hidden_channels, heads):
#         super().__init__()
#         torch.manual_seed(1234567)
#         self.num_features = data['x'].shape[1]
#         self.num_classes = len(set(data['y'].squeeze(1).numpy()))
#         self.conv1 = GATConv(self.num_features, hidden_channels, heads=heads,
#                              dropout=0.1)
#         self.conv2 = GATConv(hidden_channels * heads, self.num_classes, heads=1, concat=False,
#                              dropout=0.1)

#     def forward(self, x, edge_index):
#         # x = F.dropout(x, p=0.6, training=self.training)
#         x = self.conv1(x, edge_index)
#         x = F.elu(x)
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.conv2(x, edge_index)
#         return x


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels,heads):
        super().__init__()
        torch.manual_seed(1234567)
        self.num_features = data[0]['x'].shape[1]
        self.num_classes = len(set(data[0]['y'].squeeze(1).numpy()))
        self.conv1 = GATConv(self.num_features, hidden_channels, heads=7)
        self.conv2 = GATConv(hidden_channels * 7, hidden_channels, heads=6)
        self.conv3 = GATConv(hidden_channels * 6, self.num_classes, heads=6)

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


def add_date_to_data():
    data = HW3Dataset(root='data/hw3/')
    min_value = data[0].node_year[data[0].train_mask].min()
    max_value = data[0].node_year[data[0].train_mask].max()
    diff = max_value - min_value
    
    data[0].x = torch.concatenate((data[0].x, (data[0].node_year - min_value) / diff), axis=1) 
    
    # year_feature = (data.node_year-torch.min(data.node_year))/(torch.max(data.node_year)-torch.min(data.node_year))
    # data.x= torch.concat((data.x,year_feature),1)
    return data

    
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # data = create_ds()
    data=add_date_to_data()
    data[0].x.to(device)
    data[0].edge_index.to(device)
    # model = GCN(hidden_channels=96, dataset=data)
    model = GAT(hidden_channels=64, heads=8)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    optimizer.zero_grad()
    for epoch in range(1, 800):
        optimizer.zero_grad()  # Clear gradients.
        out = model(data[0].x, data[0].edge_index)  # Perform a single forward pass.
        loss = criterion(out[data[0].train_mask],
                         data[0].y[data[0].train_mask].squeeze())  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.

        # model.eval()
        out = model(data[0].x, data[0].edge_index)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        val_correct = pred[data[0].val_mask] == data[0].y[data[0].val_mask].squeeze()  # Check against ground-truth labels.
        val_acc = int(val_correct.sum()) / len(data[0].y[data[0].val_mask].squeeze())  # Derive ratio of correct predictions.
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}')
        # wandb.log({"acc": val_acc, "loss": loss})
        # return acc

    # wandb.finish()




    # for epoch in range(1, 600):
    #     optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    #     criterion = torch.nn.CrossEntropyLoss()
    #     model.train()
    #     optimizer.zero_grad()  # Clear gradients.
    #     out = model(data.x, data.edge_index)  # Perform a single forward pass.
    #     loss = criterion(out[data.train_mask],
    #                      data.y[data.train_mask].squeeze())  # Compute the loss solely based on the training nodes.
    #     loss.backward()  # Derive gradients.
    #     optimizer.step()  # Update parameters based on gradients.
    #     print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    # model.eval()
    # out = model(data.x, data.edge_index)
    # pred = out.argmax(dim=1)  # Use the class with highest probability.
    # val_correct = pred[data.val_mask] == data.y[data.val_mask].squeeze()  # Check against ground-truth labels.
    # val_acc = int(val_correct.sum()) / len(data.y[data.val_mask].squeeze())  # Derive ratio of correct predictions.
    #
    # print(f'Test Accuracy: {val_acc:.4f}')

    # print(model)
