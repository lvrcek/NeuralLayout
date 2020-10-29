import os

import torch
from torch_geometric.data import Data

from datasets import AlgoDatasetBase


class SingleAlgoDataset(AlgoDatasetBase):

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
                data = Data(num_nodes=num_nodes, edge_index=edge_index, edge_attr=edge_attr, x=x, y=y)
                processed_file = str(cnt) + '.pt'
                torch.save(data, os.path.join(self.processed_dir, processed_file))
                cnt += 1