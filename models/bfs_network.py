from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

import layers
import models
import utils


class BFSNetwork(nn.Module):

    def __init__(self, node_features, edge_features, latent_features, algo_processor, bias=False):
        super().__init__()
        self.node_encoder = nn.Sequential(nn.Linear(node_features+latent_features, latent_features, bias=bias),
                                          nn.LeakyReLU())
        self.edge_encoder = nn.Sequential(nn.Linear(edge_features, latent_features, bias=bias),
                                          nn.LeakyReLU())
        self.processor = algo_processor.processor
        self.decoder = layers.DecoderNetwork(2*latent_features, node_features, bias=bias)
        self.termination = layers.TerminationNetwork(latent_features, 1, bias=bias)
        self.init_losses()

    def init_losses(self):
        self.losses = {'reachability': [], 'termination': []}

    def get_loss_dict(self):
        return self.losses

    def calculate_losses(self, output, next_step, term_pred, terminate):
        reach_loss = self.calculate_reach_loss(output, next_step)
        term_loss = self.calculate_term_loss(term_pred, terminate)

    def calculate_reach_loss(self, prediction, real):
        loss = F.binary_cross_entropy_with_logits(prediction, real)
        self.losses['reachability'].append(loss)
        return loss

    def calculate_term_loss(self, prediciton, real):
        loss = F.binary_cross_entropy_with_logits(prediciton, real)
        self.losses['termination'].append(loss)
        return loss

    def get_last_step_loss(self):
        reach_loss = self.losses['reachability'][-1]
        term_loss = self.losses['termination'][-1]
        return reach_loss + term_loss

    def get_total_loss(self):
        # No need for a list comprehension here
        reach_loss = sum([l for l in self.losses['reachability']]) / len(self.losses['reachability'])
        term_loss = sum([l for l in self.losses['termination']]) / len(self.losses['termination'])
        return reach_loss + term_loss

    def forward(self, node_features, edge_features, edge_index, last_latent):
        node_enc = self.node_encoder(torch.cat((node_features, last_latent), dim=1))
        edge_enc = self.edge_encoder(edge_features)
        latent_features = self.processor(node_enc, edge_enc, edge_index).clone()
        output = self.decoder(torch.cat((node_enc, latent_features), dim=1))
        term = self.termination(latent_features.mean(dim=0))
        return output, latent_features, term
