import os
import copy
import argparse
from datetime import datetime

import torch
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt

import models
from datasets import SingleAlgoDataset, MultiAlgoDataset
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


def draw_loss_plot(train_loss, valid_loss, timestamp):
    plt.figure()
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='validation')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'figures/loss_{timestamp}.png')
    plt.show()


def draw_accuracy_plots(train_acc, valid_acc, algo_list, timestamp):
    plt.figure()
    for algo in algo_list:
        plt.plot(train_acc[algo], label=algo)
    plt.title('Training accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'figures/train_accuracy_{timestamp}.png')
    plt.show()

    plt.figure()
    for algo in algo_list:
        plt.plot(valid_acc[algo], label=algo)
    plt.title('Validation accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'figures/valid_accuracy_{timestamp}.png')
    plt.show()


def append_accuracy_list(accuracy_list, accuracy_per_graph, algo_list):
    for algo in algo_list:
        accuracy_list[algo].append(
            sum([c for c, l in accuracy_per_graph[algo]]) / sum([l for c, l in accuracy_per_graph[algo]])
        )


def print_mean_accuracy(accuracy, algo_list):
    for algo in algo_list:
        try:
            print(f"\nACCURACY {algo}:", sum([c for c, l in accuracy[algo]]) / sum([l for c, l in accuracy[algo]]))
        except ZeroDivisionError:
            print(f"\nACCURACY {algo}:", 0)


def print_last_step_accuracy(last_step, algo_list):
    for algo in algo_list:
        print(f"\nLAST STEP ACC {algo}:\t", last_step[algo][0] / last_step[algo][1])


def main(algo_list, test, train_path, test_path):

    hyperparameters = get_hyperparameters()
    num_epochs = hyperparameters['num_epochs']
    device = hyperparameters['device']
    dim_latent = hyperparameters['dim_latent']
    batch_size = hyperparameters['batch_size']
    patience_limit = hyperparameters['patience_limit']

    mode = 'test' if test else 'train'
    time_now = datetime.now().strftime('%Y-%b-%d-%H-%M')

    processor = models.AlgorithmProcessor(dim_latent).to(device)
    processor.add_algorithms(algo_list)
    params = list(processor.parameters())
    model_path = f'trained_models/processor_{time_now}.pt'

    if not os.path.isdir(os.path.join(train_path, 'processed')):
        os.mkdir(os.path.join(train_path, 'processed'))
    if not os.path.isdir(os.path.join(test_path, 'processed')):
        os.mkdir(os.path.join(test_path, 'processed'))

    ds = MultiAlgoDataset(train_path) if len(algo_list) > 1 else SingleAlgoDataset(train_path)
    ds_test = MultiAlgoDataset(test_path) if len(algo_list) > 1 else SingleAlgoDataset(test_path)

    num_graphs = len(ds)
    valid_fraction = 0.3
    valid_size = int(round(num_graphs * valid_fraction))
    train_size = num_graphs - valid_size
    ds_train, ds_valid = random_split(ds, [train_size, valid_size])

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_valid = DataLoader(ds_valid, batch_size=batch_size, shuffle=False)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(params, lr=1e-5)

    patience = 0
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

        for epoch in range(num_epochs):
            print(f'Epoch: {epoch}')
            processor.train()
            patience += 1
            loss_per_graph = []
            accuracy_per_graph = {'TRANS': [],
                                  'TIPS': [],
                                  'BUBBLES': []}
            for data in dl_train:
                # processor.process_graph(data, optimizer, loss_per_graph, accuracy_per_graph, train=True,
                #                         device=device)
                processor.process_graph_all(data, optimizer, loss_per_graph, accuracy_per_graph, train=True,
                                            device=device)

            loss_per_epoch_train.append(sum(loss_per_graph) / len(loss_per_graph))
            append_accuracy_list(accuracy_per_epoch_train, accuracy_per_graph, algo_list)

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
                    patience = 0
                    best_model.load_state_dict(copy.deepcopy(processor.state_dict()))
                    torch.save(best_model.state_dict(), model_path)
                elif patience > patience_limit:
                    break
                loss_per_epoch_valid.append(current_loss)
                append_accuracy_list(accuracy_per_epoch_valid, accuracy_per_graph, algo_list)

        draw_loss_plot(loss_per_epoch_train, loss_per_epoch_valid, time_now)
        draw_accuracy_plots(accuracy_per_epoch_train, accuracy_per_epoch_valid, algo_list, time_now)

        torch.save(processor.state_dict(), model_path)

    processor.load_state_dict(torch.load(model_path))

    # TESTING
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

        print_mean_accuracy(accuracy, algo_list)
        # print_last_step_accuracy(last_step, algo_list)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--algos', type=str, default='all', help='algorithm to learn (default: all)')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--train_path', type=str, default='data/train', help='path to the training data')
    parser.add_argument('--test_path', type=str, default='data/test', help='path to testing data')
    arguments = parser.parse_args()
    algorithm_list = get_algo_list(arguments.algos)
    is_test = arguments.test
    train_pth = arguments.train_path
    test_pth = arguments.test_path
    main(algorithm_list, is_test, train_pth, test_pth)
