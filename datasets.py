import os
import torch
from torch_geometric.data import Dataset, Data


class BFSDataset(Dataset):

    def __init__(self, root, device='cpu', split='train', transform=None, pre_transform=None):
        super(BFSDataset, self).__init__(root, transform, pre_transform)
        self.device = device
        self.split = split

    def len(self):
        return len(os.listdir(self.processed_dir)) - 2

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, str(idx)+'.pt'))

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
        edge_index[1] = list(map(int, file.readline().split()))  # this reading should be a separate function
        edge_index = torch.tensor(edge_index)
        edge_attr = torch.tensor(list(map(float, file.readline().split()))).unsqueeze_(-1)
        return num_nodes, edge_index, edge_attr

    @staticmethod
    def get_bfs_step(file):
        line = file.readline()
        if 'TERMINATE' in line:
            return 'TERMINATE'
        return torch.tensor(list(map(int, line.split()))).unsqueeze_(-1)

    # This is for a single algorithm, make it more general
    @staticmethod
    def get_algo_record(file):
        x = []
        for line in file:
            try:
                x.append(list(map(int, line.strip().split())))
            except:
                return x
        return x

    def process(self):
        cnt = 0
        for raw_file in self.raw_file_names:
            raw_path = os.path.join(self.raw_dir, raw_file)
            with open(raw_path) as file:
                num_nodes, edge_index, edge_attr = self.get_graph_attr(file)
                record = self.get_algo_record(file)
                x = torch.tensor(record[:-1]).t().contiguous().unsqueeze(-1)
                y = torch.tensor(record[1:]).t().contiguous().unsqueeze(-1)
                try:
                    data = Data(num_nodes=num_nodes, edge_index=edge_index, edge_attr=edge_attr, x=x, y=y)
                except:
                    print(raw_path)
                processed_file = str(cnt) + '.pt'
                torch.save(data, os.path.join(self.processed_dir, processed_file))
                cnt += 1


class MultiAlgoDataset(Dataset):

    def __init__(self, root, device='cpu', split='train', transform=None, pre_transform=None):
        super(MultiAlgoDataset, self).__init__(root, transform, pre_transform)
        self.device = device
        self.split = split

    def len(self):
        return len(os.listdir(self.processed_dir)) - 2

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, str(idx)+'.pt'))

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
        edge_index[1] = list(map(int, file.readline().split()))  # this reading should be a separate function
        edge_index = torch.tensor(edge_index)
        edge_attr = torch.tensor(list(map(float, file.readline().split()))).unsqueeze_(-1)
        return num_nodes, edge_index, edge_attr

    @staticmethod
    def get_multi_algo_record(file):
        x_trans, x_tips, x_bubbles = [], [], []
        for line in file:
            if 'TERMINATE_TRANS' in line:
                break
            else:
                x_trans.append(list(map(int, line.strip().split())))
        for line in file:
            if 'TERMINATE_TIPS' in line:
                break
            else:
                x_tips.append(list(map(int, line.strip().split())))
        for line in file:
            if 'TERMINATE_BUBBLES' in line:
                break
            else:
                x_bubbles.append(list(map(int, line.strip().split())))
        return x_trans, x_tips, x_bubbles

    def process(self):
        cnt = 0
        for raw_file in self.raw_file_names:
            raw_path = os.path.join(self.raw_dir, raw_file)
            with open(raw_path) as file:
                num_nodes, edge_index, edge_attr = self.get_graph_attr(file)
                record_trans, record_tips, record_bubbles = self.get_multi_algo_record(file)
                x_trans = torch.tensor(record_trans[:-1]).t().contiguous().unsqueeze(-1)
                y_trans = torch.tensor(record_trans[1:]).t().contiguous().unsqueeze(-1)
                x_tips = torch.tensor(record_tips[:-1]).t().contiguous().unsqueeze(-1)
                y_tips = torch.tensor(record_tips[1:]).t().contiguous().unsqueeze(-1)
                x_bubbles = torch.tensor(record_bubbles[:-1]).t().contiguous().unsqueeze(-1)
                y_bubbles = torch.tensor(record_bubbles[1:]).t().contiguous().unsqueeze(-1)
                try:
                    data = Data(num_nodes=num_nodes, edge_index=edge_index, edge_attr=edge_attr, x_trans=x_trans,
                                y_trans=y_trans, x_tips=x_tips, y_tips=y_tips, x_bubbles=x_bubbles, y_bubbles=y_bubbles)
                except:
                    print(raw_path)
                processed_file = str(cnt) + '.pt'
                torch.save(data, os.path.join(self.processed_dir, processed_file))
                cnt += 1