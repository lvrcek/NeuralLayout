import os
from abc import abstractmethod

import torch
from torch_geometric.data import Dataset


class AlgoDatasetBase(Dataset):

    def __init__(self, root, device='cpu', split='train', transform=None, pre_transform=None):
        super(AlgoDatasetBase, self).__init__(root, transform, pre_transform)
        self.device = device
        self.split = split

    def len(self):
        return len(os.listdir(self.processed_dir)) - 2

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, str(idx) + '.pt'))

    @property
    def raw_file_names(self):
        dirname = self.raw_dir
        raw_files = os.listdir(dirname)
        return raw_files

    @property
    def processed_file_names(self):
        processed_names = [str(n) + '.pt' for n in range(self.len())]
        return processed_names

    def download(self):
        if not os.path.isdir(self.raw_dir):
            os.system('python graph_generator.py')

    @staticmethod
    def get_graph_attr(file):
        graph_type, num_nodes = file.readline().split()
        num_nodes = int(num_nodes)
        edge_index = [[], []]
        edge_index[0] = list(map(int, file.readline().split()))
        edge_index[1] = list(map(int, file.readline().split()))
        edge_index = torch.tensor(edge_index)
        edge_attr = torch.tensor(list(map(float, file.readline().split()))).unsqueeze_(-1)
        return num_nodes, edge_index, edge_attr

    @abstractmethod
    def process(self):
        pass
