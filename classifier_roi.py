from dataset import create_ds
import os
import torch
import numpy as np
from torch_geometric.nn import GCNConv , GATConv
import torch.nn.functional as F
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
        self.num_features = data['x'].shape[1]
        self.num_classes = len(set(data['y'].squeeze(1).numpy()))
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


def add_date_to_data(data):
    year_feature = (data.node_year-torch.min(data.node_year))/(torch.max(data.node_year)-torch.min(data.node_year))
    data.x= torch.concat((data.x,year_feature),1)
    return data

    
if __name__ == '__main__':
    data = create_ds()
    data=add_date_to_data(data)
    # model = GCN(hidden_channels=96, dataset=data)
    model = GAT(hidden_channels=64, heads=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(1, 800):
        optimizer.zero_grad()  # Clear gradients.
        out = model(data.x, data.edge_index)  # Perform a single forward pass.
        loss = criterion(out[data.train_mask],
                         data.y[data.train_mask].squeeze())  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.

        # model.eval()
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        val_correct = pred[data.val_mask] == data.y[data.val_mask].squeeze()  # Check against ground-truth labels.
        val_acc = int(val_correct.sum()) / len(data.y[data.val_mask].squeeze())  # Derive ratio of correct predictions.
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
