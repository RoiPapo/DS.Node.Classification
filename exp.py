from dataset import create_ds
import os
import torch
import numpy as np
from torch_geometric.nn import GCNConv , GATConv
import torch.nn.functional as F
# import wandb
from dataset import HW3Dataset
# wandb.login()

data = HW3Dataset(root='data/hw3/')
min_value = data[0].node_year[data[0].train_mask].min()
max_value = data[0].node_year[data[0].train_mask].max()
diff = max_value - min_value

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
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

def train(model,device,optimizer,criterion):
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
    # loss = criterion(
    #     out[torch.arange(8500)].to(device),
    #     data[0].y[torch.arange(8500)].reshape(-1,).to(device)
    # )
    loss.backward()
    optimizer.step()
    return loss


def test(model,device):
    model.eval()
    out = model(
        torch.concatenate((data[0].x, (data[0].node_year - min_value) / diff), axis=1).to(device), 
        data[0].edge_index.to(device)
    )
    pred = out.argmax(dim=1).detach().cpu()
    test_correct = pred[data[0]
                        .val_mask] == data[0].y[data[0].val_mask].reshape(-1,)
    test_acc = int(test_correct.sum()) / len(test_correct)

    train_correct = pred[data[0]
                        .train_mask] == data[0].y[data[0].train_mask].reshape(-1,)
    train_acc = int(train_correct.sum()) / len(train_correct)    
    return test_acc, pred[data[0].val_mask], data[0].y[data[0].val_mask].reshape(-1,), train_acc, pred





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
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        hidden_channels=hidden_channels["hidden_channels"]
        self.num_features = data[0]['x'].shape[1] +1
        self.num_classes = len(set(data[0]['y'].squeeze(1).numpy()))
        self.conv1 = GATConv(data.num_features + 1, hidden_channels, heads=7)
        self.conv2 = GATConv(hidden_channels * 7, hidden_channels, heads=6)
        self.conv3 = GATConv(hidden_channels * 6, data.num_classes, heads=6)

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


# class GAT(torch.nn.Module):
#     def __init__(self,config):
#         super().__init__()
#         torch.manual_seed(1234567)
#         self.config=config
#         self.num_features = data[0]['x'].shape[1] +1
#         self.num_classes = len(set(data[0]['y'].squeeze(1).numpy()))
#         self.conv1 = GATConv(self.num_features, config['hidden_channels'], heads=config["heads"])
#         self.conv2 = GATConv(config['hidden_channels'] * config["heads"], config['hidden_channels'], heads=config["heads"]-1)
#         self.conv3 = GATConv(config['hidden_channels'] * (config["heads"]-1), self.num_classes, heads=config["heads"]-1)

#     def forward(self, x, edge_index):
#         x = F.dropout(x, p=self.config["dropout_parm"], training=self.training)
#         x = self.conv1(x, edge_index)
#         if self.config["act"]=='relu':
#             x = F.relu(x)
#         else:
#             x= F.gelu(x)
#         x = F.dropout(x, p=0.1, training=self.training)
#         x = self.conv2(x, edge_index)
#         if self.config["act"]=='relu':
#             x = F.relu(x)
#         else:
#             x= F.gelu(x)
#         x = F.dropout(x, p=0.1, training=self.training)
#         x = self.conv3(x, edge_index)
#         return x


def add_date_to_data(data):
    year_feature = (data.node_year-torch.min(data.node_year))/(torch.max(data.node_year)-torch.min(data.node_year))
    data.x= torch.concat((data.x,year_feature),1)
    return data

def network(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GAT(config)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],weight_decay=config["weight_decay"])
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(1, 620):
        loss = train(model,device,optimizer,criterion)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        test_acc, test_pred, test_true,train_acc,preds = test(model,device)
        print(f'Test Accuracy: {test_acc:.4f} "Train Accuracy": {train_acc:.4f}')
        # wandb.log({"validation Accuracy": test_acc, "Train loss": loss,"Train Accuracy": train_acc})
    torch.save(model.state_dict(), 'Model.py')
    print(preds)
    return test_acc
        
def sweeper():
    # wandb.init(project='my-Best-sweep')
    score = network(best_params)
    # wandb.log({'accuracy': score})





    
    

if __name__ == '__main__':
        # 2: Define the search space

    best_params={
            'heads':7,
            'hidden_channels':96,
            'lr': 0.0058,
            'weight_decay': 0,
            'dropout_parm':0.2,
            'act':'relu'
        }

    

    sweep_configuration = {
        'method': 'random',
        'metric': 
        {
            'goal': 'maximize', 
            'name': 'score'
            },
        # 'parameters': 
        # {
        #     'heads': {'values': [6,7, 8, 12]},
        #     'hidden_channels': {'values': [128,64, 96]},
        #     'lr': {'max': 0.01, 'min': 0.005},
        #     'weight_decay': {'values': [5e-4,0]},
        #     'dropout_parm':{'values':[0.0,0.1,0.2]},
        #     'act':{'values':['relu','gelu']}
        # }
        'parameters': 
        {
            'heads': {'values': [6]},
            'hidden_channels': {'values': [128]},
            'lr': {'values': [0.0062]},
            'weight_decay': {'values': [0]},
            'dropout_parm':{'values':[0.1]},
            'act':{'values':['gelu']}
        }
    }
    sweeper()
    # # 3: Start the sweep
    # sweep_id = wandb.sweep(
    #     sweep=sweep_configuration, 
    #     project='my-Best-sweep'
    #     )

    # wandb.agent(sweep_id, function=sweeper, count=1)

        # model = GCN(hidden_channels=96, dataset=data)
        
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
