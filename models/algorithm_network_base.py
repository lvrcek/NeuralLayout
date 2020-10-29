import torch
import torch.nn as nn

import layers


class AlgorithmNetworkBase(nn.Module):

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

    def forward(self, node_features, edge_features, edge_index, last_latent):
        node_enc = self.node_encoder(torch.cat((node_features, last_latent), dim=1))
        edge_enc = self.edge_encoder(edge_features)
        latent_features = self.processor(node_enc, edge_enc, edge_index).clone()
        output = self.decoder(torch.cat((node_enc, latent_features), dim=1))
        term = self.termination(latent_features.mean(dim=0))
        return output, latent_features, term
