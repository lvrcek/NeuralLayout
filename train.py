import copy
import argparse

import torch
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt

import models
from datasets import MultiAlgoDataset
from hyperparameters import get_hyperparameters


class AlgorithmError(Exception):
    pass


def get_algo_list(algos):
    args = algos.lower()
    if args == 'all':
        return ['TRANS', 'TIPS', 'BUBBLES']
    elif args == 'trans' or args == 'transitive':
        return ['TRANS']
    elif args == 'tips':
        return ['TIPS']
    elif args == ['bubbles']:
        return ['BUBBLES']
    else:
        raise AlgorithmError('Algorithm undefined. Choose between (trans, tips, bubbles).')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--algos', type=str, default='all', help='algorithm to learn (default: all)')
    args = parser.parse_args()
    algo_list = get_algo_list(args.algos)

    hyperparameters = get_hyperparameters()

    device = hyperparameters['device']
    dim_latent = hyperparameters['dim_latent']
    batch_size = hyperparameters['batch_size']
    patience = hyperparameters['patience']

    NUM_EPOCHS = 10

    mode = 'test'
    print(algo_list)

    processor = models.AlgorithmProcessor(dim_latent).to(device)
    processor.add_algorithms(algo_list)
    params = list(processor.parameters())
    input()
    model_path = 'processor.pt'

    ds = MultiAlgoDataset('data/train')
    ds_test = MultiAlgoDataset('data/test')

    num_graphs = len(ds)
    valid_fraction = 0.3
    valid_size = int(round(num_graphs * valid_fraction))
    train_size = num_graphs - valid_size
    ds_train, ds_valid = random_split(ds, [train_size, valid_size])

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_valid = DataLoader(ds_valid, batch_size=batch_size, shuffle=False)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(params, lr=1e-5)

    best_model = models.AlgorithmProcessor(dim_latent)
    best_model.algorithms = nn.ModuleDict(processor.algorithms.items())
    best_model.load_state_dict(copy.deepcopy(processor.state_dict()))

    # TRAINING
    if mode == 'train':
        loss_per_epoch_train = []
        loss_per_epoch_valid = []
        accuracy_per_epoch_train = {'TRANS': [],
                                    'TIPS': [],
                                    'BUBBLES': []}
        accuracy_per_epoch_valid = {'TRANS': [],
                                    'TIPS': [],
                                    'BUBBLES': []}

        for epoch in range(NUM_EPOCHS):
            print(f'Epoch: {epoch}')
            processor.train()
            loss_per_graph = []
            accuracy_per_graph = {'TRANS': [],
                                  'TIPS': [],
                                  'BUBBLES': []}
            for data in dl_train:
                # processor.process_graph(data, optimizer, loss_per_graph, accuracy_per_graph, train=True, device=device)
                processor.process_graph_all(data, optimizer, loss_per_graph, accuracy_per_graph, train=True,
                                            device=device)

            loss_per_epoch_train.append(sum(loss_per_graph) / len(loss_per_graph))
            accuracy_per_epoch_train['TRANS'].append(
                sum([c for c, l in accuracy_per_graph['TRANS']]) / sum([l for c, l in accuracy_per_graph['TRANS']])
            )
            accuracy_per_epoch_train['TIPS'].append(
                sum([c for c, l in accuracy_per_graph['TIPS']]) / sum([l for c, l in accuracy_per_graph['TIPS']])
            )
            accuracy_per_epoch_train['BUBBLES'].append(
                sum([c for c, l in accuracy_per_graph['BUBBLES']]) / sum([l for c, l in accuracy_per_graph['BUBBLES']])
            )

            # VALIDATION
            with torch.no_grad():
                processor.eval()
                loss_per_graph = []
                accuracy_per_graph = {'TRANS': [],
                                      'TIPS': [],
                                      'BUBBLES': []}
                for data in dl_valid:
                    processor.process_graph_all(data, optimizer, loss_per_graph, accuracy_per_graph, train=False)
                    # print(loss_per_graph)
                current_loss = sum(loss_per_graph) / len(loss_per_graph)
                if len(loss_per_epoch_valid) > 0 and current_loss < min(loss_per_epoch_valid):
                    best_model.load_state_dict(copy.deepcopy(processor.state_dict()))
                    torch.save(best_model.state_dict(), model_path)
                loss_per_epoch_valid.append(current_loss)
                accuracy_per_epoch_valid['TRANS'].append(
                    sum([c for c, l in accuracy_per_graph['TRANS']]) / sum([l for c, l in accuracy_per_graph['TRANS']])
                )
                accuracy_per_epoch_valid['TIPS'].append(
                    sum([c for c, l in accuracy_per_graph['TIPS']]) / sum([l for c, l in accuracy_per_graph['TIPS']])
                )
                accuracy_per_epoch_valid['BUBBLES'].append(
                    sum([c for c, l in accuracy_per_graph['BUBBLES']]) / sum(
                        [l for c, l in accuracy_per_graph['BUBBLES']])
                )

        counter = 0

        plt.figure()
        plt.plot(loss_per_epoch_train, label='train')
        plt.plot(loss_per_epoch_valid, label='validation')
        plt.title('Loss over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'figs/loss_all_{counter}.png')
        plt.show()

        plt.figure()
        plt.plot(accuracy_per_epoch_train['TRANS'], label='trans')
        plt.plot(accuracy_per_epoch_train['TIPS'], label='tips')
        plt.plot(accuracy_per_epoch_train['BUBBLES'], label='bubbles')
        plt.title('Training accuracy over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f'figs/train_accuracy_all_{counter}.png')
        plt.show()

        plt.figure()
        # print(accuracy_per_epoch_valid['TRANS'][-1])
        # print(accuracy_per_epoch_valid['TIPS'][-1])
        # print(accuracy_per_epoch_valid['BUBBLES'][-1])
        plt.plot(accuracy_per_epoch_valid['TRANS'], label='trans')
        plt.plot(accuracy_per_epoch_valid['TIPS'], label='tips')
        plt.plot(accuracy_per_epoch_valid['BUBBLES'], label='bubbles')
        plt.title('Validation accuracy over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f'figs/valid_accuracy_all_{counter}.png')
        plt.show()

        torch.save(processor.state_dict(), model_path)

    processor.load_state_dict(torch.load(model_path))

    # TESTING
    accuracy_test = []
    with torch.no_grad():
        processor.eval()

        loss_per_graph = []
        accuracy = {'TRANS': [],
                    'TIPS': [],
                    'BUBBLES': []}
        last_step = {'TRANS': [],
                     'TIPS': [],
                     'BUBBLES': []}
        for data in dl_test:
            processor.process_graph_all(data, optimizer, loss_per_graph, accuracy, train=False, last_step=last_step)
        print("\nACCURACY TRANS:", sum([c for c, l in accuracy['TRANS']]) / sum([l for c, l in accuracy['TRANS']]))
        print("\nACCURACY TIPS:", sum([c for c, l in accuracy['TIPS']]) / sum([l for c, l in accuracy['TIPS']]))
        print("\nACCURACY BUBBLES:",
              sum([c for c, l in accuracy['BUBBLES']]) / sum([l for c, l in accuracy['BUBBLES']]))
        print("\nLAST STEP ACC TRANS:\t", last_step['TRANS'][0] / last_step['TRANS'][1])
        print("\nLAST STEP ACC TIPS:\t", last_step['TIPS'][0] / last_step['TIPS'][1])
        print("\nLAST STEP ACC BUBBLES:\t", last_step['BUBBLES'][0] / last_step['BUBBLES'][1])
