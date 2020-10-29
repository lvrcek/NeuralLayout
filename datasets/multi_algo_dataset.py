import os

import torch
from torch_geometric.data import Data

from datasets import AlgoDatasetBase


class MultiAlgoDataset(AlgoDatasetBase):

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
                data = Data(num_nodes=num_nodes, edge_index=edge_index, edge_attr=edge_attr, x_trans=x_trans,
                            y_trans=y_trans, x_tips=x_tips, y_tips=y_tips, x_bubbles=x_bubbles, y_bubbles=y_bubbles)
                processed_file = str(cnt) + '.pt'
                torch.save(data, os.path.join(self.processed_dir, processed_file))
                cnt += 1
