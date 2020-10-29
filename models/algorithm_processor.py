import torch
import torch.nn as nn

import models.bfs_network
from layers import MPNN
from hyperparameters import get_hyperparameters


class AlgorithmProcessor(nn.Module):

    def __init__(self, latent_features, processor_type='MPNN'):
        super(AlgorithmProcessor, self).__init__()
        if processor_type == 'MPNN':
            self.processor = MPNN(latent_features, latent_features, latent_features, bias=get_hyperparameters()['bias'])
        self.algorithms = nn.ModuleDict()

    def add_algorithm(self, name, algorithm):
        self.algorithms[name] = algorithm

    def update_weights(self, optimizer):
        loss = 0
        for name, algorithm in self.algorithms.items():
            loss += algorithm.get_total_loss()
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

    def update_weights_step(self, optimizer):
        loss = 0
        for name, algorithm in self.algorithms.items():
            loss += algorithm.get_last_step_loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def add_algorithms(self, algo_list):
        hyperparameters = get_hyperparameters()
        device = hyperparameters['device']
        node_features = hyperparameters['dim_nodes']
        edge_features = hyperparameters['dim_edges']
        latent_features = hyperparameters['dim_latent']
        for algo in algo_list:
            if algo == 'BFS':
                self.algorithms[algo] = models.bfs_network.BFSNetwork(node_features, edge_features,
                                                                      latent_features, self).to(device)
            elif algo in ['TRANS', 'TIPS', 'BUBBLES']:
                self.algorithms[algo] = models.traversal_network.TraversalNetwork(node_features, edge_features,
                                                                                    latent_features, self).to(device)
            else:
                # For other algorithms
                pass

    def process_graph(self, graph, optimizer, loss_list, accuracy_list, train=True, device='cpu'):
        num_steps = graph.x.shape[1]  # See if you can get something nicer
        edge_features = graph.edge_attr.clone().to(device)
        edge_index = graph.edge_index.clone().to(device)
        self.algorithms['TRANS'].init_losses()
        self.processor.zero_hidden(graph.num_nodes)
        last_latent = self.processor.hidden.detach()
        # torch.autograd.set_detect_anomaly(True)
        for step in range(num_steps):
            node_features = graph.x[:, step].clone().detach().float().to(device)
            next_step = graph.y[:, step].clone().detach().float().to(device)
            for algo, algo_net in self.algorithms.items():
                output, last_latent, term_pred = algo_net(node_features, edge_features, edge_index, last_latent)
                correct = (output.sigmoid().round() == next_step).sum().item()
                accuracy_list.append((correct, output.size(0)))
                terminate = torch.tensor((1 if step == num_steps - 1 else 0,), dtype=torch.float)
                algo_net.calculate_losses(output, next_step, term_pred, terminate)

            if train:
                self.update_weights_step(optimizer)

        loss_list.append(self.algorithms['TRANS'].get_total_loss())  # returns list of all the losses per step

    def process_graph_all(self, graph, optimizer, loss_list, accuracy_list, train=True, device='cpu', last_step=None):
        num_steps_trans = graph.x_trans.shape[1]
        num_steps_tips = graph.x_tips.shape[1]
        num_steps_bubbles = graph.x_bubbles.shape[1]
        num_steps = min(num_steps_trans, num_steps_tips, num_steps_bubbles)  # See if you can get something nicer
        edge_features = graph.edge_attr.clone().to(device)
        edge_index = graph.edge_index.clone().to(device)
        for algo, algo_net in self.algorithms.items():
            algo_net.init_losses()
        self.processor.zero_hidden(graph.num_nodes)
        last_latent = self.processor.hidden.detach()
        for step in range(num_steps):

            for algo, algo_net in self.algorithms.items():
                if algo == 'TRANS':
                    node_features = graph.x_trans[:, step].clone().detach().float().to(device)
                    next_step = graph.y_trans[:, step].clone().detach().float().to(device)
                elif algo == 'TIPS':
                    node_features = graph.x_tips[:, step].clone().detach().float().to(device)
                    next_step = graph.y_tips[:, step].clone().detach().float().to(device)
                elif algo == 'BUBBLES':
                    node_features = graph.x_bubbles[:, step].clone().detach().float().to(device)
                    next_step = graph.y_bubbles[:, step].clone().detach().float().to(device)

                output, last_latent, term_pred = algo_net(node_features, edge_features, edge_index, last_latent)
                correct = (output.sigmoid().round() == next_step).sum().item()
                accuracy_list[algo].append((correct, output.size(0)))
                if last_step is not None and step == num_steps - 1:
                    last_step[algo] = accuracy_list[algo][-1]

                terminate = torch.tensor((1 if step == num_steps - 1 else 0,), dtype=torch.float)
                algo_net.calculate_losses(output, next_step, term_pred, terminate)

            if train:
                self.update_weights_step(optimizer)

        loss_list.append(self.algorithms['TRANS'].get_total_loss())
