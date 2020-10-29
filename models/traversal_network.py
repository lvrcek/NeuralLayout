import torch.nn.functional as F

from models import AlgorithmNetworkBase


class TraversalNetwork(AlgorithmNetworkBase):

    # def __init__(self, node_features, edge_features, latent_features, algo_processor, bias=False):
    #     super().__init__()
    #     self.node_encoder = nn.Sequential(nn.Linear(node_features+latent_features, latent_features, bias=bias),
    #                                       nn.LeakyReLU())
    #     self.edge_encoder = nn.Sequential(nn.Linear(edge_features, latent_features, bias=bias),
    #                                       nn.LeakyReLU())
    #     self.processor = algo_processor.processor
    #     self.decoder = layers.DecoderNetwork(2*latent_features, node_features, bias=bias)
    #     self.termination = layers.TerminationNetwork(latent_features, 1, bias=bias)
    #     self.init_losses()

    def init_losses(self):
        self.losses = {'removal': [], 'termination': []}

    def get_loss_dict(self):
        return self.losses

    def calculate_losses(self, output, next_step, term_pred, terminate):
        reach_loss = self.calculate_removal_loss(output, next_step)
        term_loss = self.calculate_term_loss(term_pred, terminate)

    def calculate_removal_loss(self, prediction, real):
        loss = F.binary_cross_entropy_with_logits(prediction, real)
        self.losses['removal'].append(loss)
        return loss

    def calculate_term_loss(self, prediction, real):
        loss = F.binary_cross_entropy_with_logits(prediction, real)
        self.losses['termination'].append(loss)
        return loss

    def get_last_step_loss(self):
        remove_loss = self.losses['removal'][-1]
        term_loss = self.losses['termination'][-1]
        return remove_loss + term_loss

    def get_total_loss(self):
        remove_loss = sum([loss for loss in self.losses['removal']]) / len(self.losses['removal'])
        term_loss = sum([loss for loss in self.losses['termination']]) / len(self.losses['termination'])
        return remove_loss + term_loss
