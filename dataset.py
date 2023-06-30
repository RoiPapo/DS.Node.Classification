# %%
import numpy as np
import requests
import os
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from torch_geometric.data import Dataset
import torch


class HW3Dataset(Dataset):
    url = 'https://technionmail-my.sharepoint.com/:u:/g/personal/ploznik_campus_technion_ac_il/EUHUDSoVnitIrEA6ALsAK1QBpphP5jX3OmGyZAgnbUFo0A?download=1'

    def __init__(self, root, transform=None, pre_transform=None):
        super(HW3Dataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ['data.pt']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        file_url = self.url.replace(' ', '%20')
        response = requests.get(file_url)

        if response.status_code != 200:
            raise Exception(f"Failed to download the file, status code: {response.status_code}")

        with open(os.path.join(self.raw_dir, self.raw_file_names[0]), 'wb') as f:
            f.write(response.content)

    def process(self):
        raw_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        data = torch.load(raw_path)
        torch.save(data, self.processed_paths[0])

    def len(self):
        return 1

    def get(self, idx):
        return torch.load(self.processed_paths[0])


def visualize_latent_space(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())
    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()


def plot_label_distribution(labels):
    # Convert labels to a one-dimensional numpy array
    labels = np.ravel(labels)

    # Count the occurrences of each label
    label_counts = np.bincount(labels)

    # Get the unique labels
    unique_labels = np.unique(labels)

    # Calculate the percentage values
    total_count = len(labels)
    label_percentages = (label_counts / total_count) * 100

    # Set the figure size
    plt.figure(figsize=(10, 6))  # Adjust the width and height as desired

    # Create a bar chart to visualize the label distribution
    plt.bar(unique_labels, label_percentages)
    plt.xlabel('Labels')
    plt.ylabel('Percentage')
    plt.title('Label Distribution')

    # Add ticks to each bar
    plt.xticks(unique_labels)

    plt.show()


def create_ds():
    dataset = HW3Dataset(root='data/hw3/')
    data = dataset[0]
    return data


if __name__ == '__main__':
    dataset = HW3Dataset(root='data/hw3/')
    data = dataset[0]
    plot_label_distribution(data['y'])
    # visualize_latent_space(data['x'][:10000],color=data['y'][:10000])
    print(data)

# %%
