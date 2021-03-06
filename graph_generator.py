import argparse
import os
import random
import subprocess

import torch
import torch_geometric
from torch_geometric.utils import from_networkx
import networkx as nx

from deterministic import bfs, remove_transitive, remove_tips, find_bubbles


def initialize_node_attr(graph):
    graph.x = torch.zeros((graph.num_nodes, 1))


def randomize_edge_attr(graph, a=0.2, b=1.):
    temp = [random.uniform(a, b) for _ in range(graph.num_edges)]  # For undirected put "// 2"
    edge_attr = []
    for t in temp:
        edge_attr.append(t)
        # Uncomment for undirected graphs
        # edge_attr.append(t)
    graph.edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze_(-1)


def append_edge_index(edge_index, src, dst):
    edge_index[0].append(src)
    edge_index[1].append(dst)


def sort_edge_index(edge_index):
    new_edge_index = [[], []]
    edge_set = set()
    for src, dst in zip(edge_index[0], edge_index[1]):
        src, dst = src.item(), dst.item()
        if (src, dst) in edge_set:
            continue
        else:
            append_edge_index(new_edge_index, src, dst)
            # Uncomment for undirected graphs
            # append_edge_index(new_edge_index, dst, src)
            edge_set.add((src, dst))
            edge_set.add((dst, src))
    return torch.tensor(new_edge_index)


def from_nx_to_torch(graph_nx, n=None):
    graph_torch = from_networkx(graph_nx)
    graph_torch.edge_index = sort_edge_index(graph_torch.edge_index)
    initialize_node_attr(graph_torch)
    randomize_edge_attr(graph_torch)
    return graph_torch


def generate_ladder(n):
    graph_nx = nx.ladder_graph(n)
    return from_nx_to_torch(graph_nx)


def generate_grid(m, n):
    graph_nx = nx.grid_2d_graph(m, n)
    return from_nx_to_torch(graph_nx)


def generate_erdos_renyi_graph(n, p):
    graph_nx = nx.erdos_renyi_graph(n, p)
    return from_nx_to_torch(graph_nx)


def generate_barabasi_albert_graph(n, m):
    graph_nx = nx.barabasi_albert_graph(n, m)
    return from_nx_to_torch(graph_nx)


def generate_balanced_tree(r, h):
    graph_nx = nx.balanced_tree(r, h)
    return from_nx_to_torch(graph_nx)


def generate_transitive_chain(n):
    graph_nx = nx.DiGraph()
    graph_nx.add_nodes_from(range(n))
    graph_nx.add_edges_from([(i-1, i) for i in range(1, n)])
    single_hop, multi_trans, multi_hop = 0.2, 0.1, 0.2
    multi_min, multi_max = 2, 3
    hop_min, hop_max = 2, 3
    num_nodes = n

    for _ in range(round(n * single_hop)):
        start_node = random.randint(0, n - 2)
        end_node = start_node + 1
        new_node = num_nodes
        graph_nx.add_node(new_node)
        graph_nx.add_edges_from([(start_node, new_node), (new_node, end_node)])
        num_nodes += 1
    for _ in range(round(n * multi_trans)):
        multi = random.randint(multi_min, multi_max)
        start_node = random.randint(0, n - 2)
        end_node = start_node + 1
        for m in range(multi):
            new_node = num_nodes
            graph_nx.add_node(new_node)
            graph_nx.add_edges_from([(start_node, new_node), (new_node, end_node)])
            start_node = new_node
            num_nodes += 1
    for _ in range(round(n * multi_hop)):
        hop = random.randint(hop_min, hop_max)
        start_node = random.randint(0, n - hop - 1)
        end_node = start_node + hop
        new_node = num_nodes
        graph_nx.add_node(new_node)
        graph_nx.add_edges_from([(start_node, new_node), (new_node, end_node)])
        num_nodes += 1

    edge_index = [[], []]
    for edge in graph_nx.edges:
        edge_index[0].append(edge[0])
        edge_index[1].append(edge[1])
    edge_index = torch.tensor(edge_index)
    num_nodes = len(graph_nx.nodes)
    graph_torch = torch_geometric.data.Data(num_nodes=num_nodes, edge_index=edge_index)
    initialize_node_attr(graph_torch)
    randomize_edge_attr(graph_torch)
    return graph_torch


def generate_tip_chain(n):
    graph_nx = nx.DiGraph()
    graph_nx.add_nodes_from(range(n))
    graph_nx.add_edges_from([(i-1, i) for i in range(1, n)])
    short_tips, long_tips = 0.2, 0.1
    num_nodes = n

    for _ in range(round(n * short_tips)):
        curr_node = random.randint(0, n - 1)
        tip_length = random.randint(1, 5)
        for i in range(tip_length):
            new_node = num_nodes
            graph_nx.add_node(new_node)
            graph_nx.add_edge(curr_node, new_node)
            num_nodes += 1
            curr_node = new_node

    for _ in range(round(n * long_tips)):
        curr_node = random.randint(0, n - 1)
        tip_length = random.randint(6, 10)
        for i in range(tip_length):
            new_node = num_nodes
            graph_nx.add_node(new_node)
            graph_nx.add_edge(curr_node, new_node)
            num_nodes += 1
            curr_node = new_node

    edge_index = [[], []]
    for edge in graph_nx.edges:
        edge_index[0].append(edge[0])
        edge_index[1].append(edge[1])
    edge_index = torch.tensor(edge_index)
    num_nodes = len(graph_nx.nodes)
    graph_torch = torch_geometric.data.Data(num_nodes=num_nodes, edge_index=edge_index)
    initialize_node_attr(graph_torch)
    randomize_edge_attr(graph_torch)
    return graph_torch


def generate_bubble_chain(n):
    graph_nx = nx.DiGraph()
    graph_nx.add_nodes_from(range(n))
    graph_nx.add_edges_from([(i-1, i) for i in range(1, n)])
    bubbles = 0.1
    num_nodes = n

    for _ in range(round(num_nodes * bubbles)):
        min_bubble_len, max_bubble_len = 2, 5
        bubble_length = random.randint(min_bubble_len, max_bubble_len)
        start_node = random.randint(0, num_nodes - max_bubble_len - 1)
        curr_node = start_node
        for i in range(bubble_length):
            new_node = num_nodes
            graph_nx.add_node(new_node)
            graph_nx.add_edge(curr_node, new_node)
            num_nodes += 1
            curr_node = new_node
            if random.random() > 0.9:
                in_node = num_nodes
                graph_nx.add_node(in_node)
                graph_nx.add_edge(in_node, curr_node)
                num_nodes += 1
            if random.random() < 0.1:
                out_node = num_nodes
                graph_nx.add_node(out_node)
                graph_nx.add_edge(curr_node, out_node)
                num_nodes += 1
        skip = random.randint(min_bubble_len, max_bubble_len)
        graph_nx.add_edge(curr_node, start_node + skip)

    edge_index = [[], []]
    for edge in graph_nx.edges:
        edge_index[0].append(edge[0])
        edge_index[1].append(edge[1])
    edge_index = torch.tensor(edge_index)
    num_nodes = len(graph_nx.nodes)
    graph_torch = torch_geometric.data.Data(num_nodes=num_nodes, edge_index=edge_index)
    initialize_node_attr(graph_torch)
    randomize_edge_attr(graph_torch)
    return graph_torch


def generate_training_graph(n):
    graph_nx = nx.DiGraph()
    graph_nx.add_nodes_from(range(n))
    graph_nx.add_edges_from([(i-1, i) for i in range(1, n)])

    # Multi trans are wrong
    single_hop, multi_trans, multi_hop = 0.1, 0.1, 0.1
    multi_min, multi_max = 2, 3
    hop_min, hop_max = 2, 3
    short_tips, long_tips = 0.1, 0.1
    bubbles = 0.1
    num_nodes = n

    for _ in range(round(n * single_hop)):
        start_node = random.randint(0, n - 2)
        end_node = start_node + 1
        new_node = num_nodes
        graph_nx.add_node(new_node)
        graph_nx.add_edges_from([(start_node, new_node), (new_node, end_node)])
        num_nodes += 1
    for _ in range(round(n * multi_trans)):
        multi = random.randint(multi_min, multi_max)
        start_node = random.randint(0, n - 2)
        end_node = start_node + 1
        for m in range(multi):
            new_node = num_nodes
            graph_nx.add_node(new_node)
            graph_nx.add_edges_from([(start_node, new_node), (new_node, end_node)])
            start_node = new_node
            num_nodes += 1
    for _ in range(round(n * multi_hop)):
        hop = random.randint(hop_min, hop_max)
        start_node = random.randint(0, n - hop - 1)
        end_node = start_node + hop
        new_node = num_nodes
        graph_nx.add_node(new_node)
        graph_nx.add_edges_from([(start_node, new_node), (new_node, end_node)])
        num_nodes += 1

    for _ in range(round(n * short_tips)):
        curr_node = random.randint(0, n - 1)
        tip_length = random.randint(1, 5)
        for i in range(tip_length):
            new_node = num_nodes
            graph_nx.add_node(new_node)
            graph_nx.add_edge(curr_node, new_node)
            num_nodes += 1
            curr_node = new_node

    for _ in range(round(n * long_tips)):
        curr_node = random.randint(0, n - 1)
        tip_length = random.randint(6, 10)
        for i in range(tip_length):
            new_node = num_nodes
            graph_nx.add_node(new_node)
            graph_nx.add_edge(curr_node, new_node)
            num_nodes += 1
            curr_node = new_node

    for _ in range(round(num_nodes * bubbles)):
        min_bubble_len, max_bubble_len = 2, 5
        bubble_length = random.randint(min_bubble_len, max_bubble_len)
        start_node = random.randint(0, num_nodes - max_bubble_len - 1)
        curr_node = start_node
        for i in range(bubble_length):
            new_node = num_nodes
            graph_nx.add_node(new_node)
            graph_nx.add_edge(curr_node, new_node)
            num_nodes += 1
            curr_node = new_node
            if random.random() > 0.9:
                in_node = num_nodes
                graph_nx.add_node(in_node)
                graph_nx.add_edge(in_node, curr_node)
                num_nodes += 1
            if random.random() < 0.1:
                out_node = num_nodes
                graph_nx.add_node(out_node)
                graph_nx.add_edge(curr_node, out_node)
                num_nodes += 1
        skip = random.randint(min_bubble_len, max_bubble_len)
        graph_nx.add_edge(curr_node, start_node + skip)

    edge_index = [[], []]
    for edge in graph_nx.edges:
        edge_index[0].append(edge[0])
        edge_index[1].append(edge[1])
    edge_index = torch.tensor(edge_index)
    num_nodes = len(graph_nx.nodes)
    graph_torch = torch_geometric.data.Data(num_nodes=num_nodes, edge_index=edge_index)
    initialize_node_attr(graph_torch)
    randomize_edge_attr(graph_torch)
    return graph_torch


def read_csv(graph_path):
    graph_nx = nx.DiGraph()
    node_set = set()
    with open(graph_path) as f:
        for line in f.readlines():
            src, dst = map(int, line.strip().split(','))
            if src not in node_set:
                graph_nx.add_node(src)
                node_set.add(src)
            if dst not in node_set:
                graph_nx.add_node(dst)
                node_set.add(dst)
            graph_nx.add_edge(src, dst)

    # graphs = list(nx.connected_component_subgraphs(graph_nx))

    edge_index = [[], []]
    for edge in graph_nx.edges:
        edge_index[0].append(edge[0])
        edge_index[1].append(edge[1])
    edge_index = torch.tensor(edge_index)
    num_nodes = max(graph_nx.nodes) + 1
    graph_torch = torch_geometric.data.Data(num_nodes=num_nodes, edge_index=edge_index)
    initialize_node_attr(graph_torch)
    randomize_edge_attr(graph_torch)
    return graph_torch


def write_to_file(filename, graph, graph_type, bfs_steps):
    with open(filename, 'w') as f:
        f.write(f'{graph_type} {graph.num_nodes}\n')
        f.write(' '.join(list(map(str, map(int, graph.edge_index[0])))) + '\n')
        f.write(' '.join(list(map(str, map(int, graph.edge_index[1])))) + '\n')
        # f.write(' '.join(list(map(str, [int(el.item()) for el in graph.x]))) + '\n')
        f.write(' '.join(list(map(str, [round(float(el.item()), 3) for el in graph.edge_attr]))) + '\n')
        # Each step is a new update in node attributes until the termination of an algorithm
        for step in bfs_steps:
            f.write(' '.join(list(map(str, [int(el.item()) for el in step]))) + '\n')
        f.write('TERMINATE\n')


def write_all_to_file(filename, graph, graph_type, all_steps):
    steps_trans, steps_tips, steps_bubbles = all_steps[0], all_steps[1], all_steps[2]
    with open(filename, 'w') as f:
        f.write(f'{graph_type} {graph.num_nodes}\n')
        f.write(' '.join(list(map(str, map(int, graph.edge_index[0])))) + '\n')
        f.write(' '.join(list(map(str, map(int, graph.edge_index[1])))) + '\n')
        f.write(' '.join(list(map(str, [round(float(el.item()), 3) for el in graph.edge_attr]))) + '\n')
        # Each step is a new update in node attributes until the termination of an algorithm
        for step in steps_trans:
            f.write(' '.join(list(map(str, [int(el.item()) for el in step]))) + '\n')
        f.write('TERMINATE_TRANS\n')
        for step in steps_tips:
            f.write(' '.join(list(map(str, [int(el.item()) for el in step]))) + '\n')
        f.write('TERMINATE_TIPS\n')
        for step in steps_bubbles:
            f.write(' '.join(list(map(str, [int(el.item()) for el in step]))) + '\n')
        f.write('TERMINATE_BUBBLES\n')


def from_fastq(fastq_path, store_path, threads=1):
    fastq_path = os.path.abspath(fastq_path)
    tmp_path = os.path.abspath(os.path.join(store_path, '../tmp'))
    raven = os.path.abspath('vendor/raven/build/bin/raven')
    if not os.path.isdir(tmp_path):
        os.mkdir(tmp_path)
    subprocess.run(f'{raven} --weaken -t{threads} -p0 {fastq_path} > assembly.fasta', shell=True, cwd=tmp_path)
    csv_path = os.path.join(tmp_path, 'graph_1.csv')
    return csv_path


def clean_raw_csv(raw_csv, fastq_type, data_path):
    clean_csv = os.path.join(data_path, fastq_type + '.csv')
    with open(clean_csv, 'w') as f_out, open(raw_csv) as f_in:
        for line in f_in.readlines():
            line = line.strip().split(',')
            if line[2] == '0':
                continue
            out_info = line[0].split()[0] + ',' + line[1].split()[0] + '\n'
            f_out.write(out_info)
    return clean_csv


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Types of graph to generate.')
    parser.add_argument('--ladder', action='store_true', help='generate ladder graphs')
    parser.add_argument('--grid', action='store_true', help='generate grid graphs')
    parser.add_argument('--erdos', action='store_true', help='generate Erdos-Renyi graphs')
    parser.add_argument('--barabasi', action='store_true', help='generate Barabasi-Albert graphs')
    parser.add_argument('--tree', action='store_true', help='generate balanced tree graphs')
    parser.add_argument('--transitive', action='store_true', help='generate path graphs with transitive edges')
    parser.add_argument('--tips', action='store_true', help='generate path graphs with tips')
    parser.add_argument('--bubbles', action='store_true', help='generate path graphs with bubbles')
    parser.add_argument('--training', action='store_true', help='generate path graphs for training')
    parser.add_argument('--testing', action='store_true', help='generate path graphs for testing')
    parser.add_argument('--from_csv', action='store_true', help='parse CSV file to generate graph')
    parser.add_argument('--csv_path', type=str, help='path to CSV file')
    parser.add_argument('--csv_type', type=str, help='organism on whose graph the model will be run, e.g. ecoli',
                        default='unknown')
    parser.add_argument('--from_fastq', action='store_true', help='run raven to generate graphs in CSV')
    parser.add_argument('--fastq_path', type=str, help='path to FASTQ file')
    parser.add_argument('--fastq_type', type=str, help='type of the genomic data, e.g., ecoli',
                        default='unknown')
    parser.add_argument('--threads', type=int, default=1, help='number of threads used for raven')
    parser.add_argument('store_dir', type=str, help='Directory to where the data should be stored')

    args = parser.parse_args()
    data_path = os.path.abspath(args.store_dir)
    if not os.path.isdir(data_path):
        os.makedirs(data_path)

    if args.ladder:
        # GENERATE LADDER GRAPHS
        graph_type = 'ladder'
        ladder_min, ladder_max = 5, 30
        for i in range(ladder_min, ladder_max+1):
            graph = generate_ladder(i)
            start = random.randint(0, graph.num_nodes // 2 - 1)
            bfs_steps = bfs(graph, start)
            filename = os.path.join(data_path, f'{graph_type}_{i}.txt')
            write_to_file(filename, graph, graph_type, bfs_steps)

    if args.grid:
        # GENERATE GRID GRAPHS
        graph_type = 'grid'
        grid_min, grid_max = 3, 10
        for i in range(grid_min, grid_max+1):
            for j in range(grid_min, grid_max+1):
                graph = generate_grid(i, j)
                start = random.randint(0, graph.num_nodes // 2 - 1)
                bfs_steps = bfs(graph, start)
                filename = os.path.join(data_path, f'{graph_type}_{i}_{j}.txt')
                write_to_file(filename, graph, graph_type, bfs_steps)

    if args.erdos:
        # GENERATE ERDOS-RENYI GRAPHS
        graph_type = 'erdos_renyi'
        nodes_min, nodes_max = 5, 20
        probabilities = [0.2, 0.4, 0.6]
        for i in range(nodes_min, nodes_max+1):
            for p in probabilities:
                graph = generate_erdos_renyi_graph(i, p)
                if len(graph.edge_index[0]) == 0:
                    continue
                start = random.randint(0, graph.num_nodes // 2 - 1)
                bfs_steps = bfs(graph, start)
                if len(bfs_steps) <= 1:
                    continue
                filename = os.path.join(data_path, f'{graph_type}_{i}_{p}.txt')
                write_to_file(filename, graph, graph_type, bfs_steps)

    if args.barabasi:
        # GENERATE BARABASI-ALBERT GRAPHS
        graph_type = 'barabasi_albert'
        nodes_min, nodes_max = 5, 20
        connections = [1, 2, 3]
        for i in range(nodes_min, nodes_max):
            for c in connections:
                graph = generate_barabasi_albert_graph(i, c)
                start = random.randint(0, graph.num_nodes // 2 - 1)
                bfs_steps = bfs(graph, start)
                filename = os.path.join(data_path, f'{graph_type}_{i}_{c}.txt')
                write_to_file(filename, graph, graph_type, bfs_steps)

    if args.tree:
        # GENERATE BALANCED TREE - testing
        graph_type = 'balanced_tree'
        tree_deg = [3, 4]
        height_min, height_max = 3, 7
        for i in tree_deg:
            for j in range(height_min, height_max+1):
                graph = generate_balanced_tree(i, j)
                bfs_steps = bfs(graph, 0)
                filename = os.path.join(data_path, f'{graph_type}_{i}_{j}.txt')
                write_to_file(filename, graph, graph_type, bfs_steps)

    if args.transitive:
        # GENERATE TRANSITIVE CHAINS
        graph_type = 'transitive_chain'
        num_graphs = 100
        chain_length = 1000
        for i in range(num_graphs):
            graph = generate_transitive_chain(chain_length)
            steps = remove_transitive(graph)
            filename = os.path.join(data_path, f'{graph_type}_{i}.txt')
            write_to_file(filename, graph, graph_type, steps)

    if args.tips:
        # GENERATE TIP CHAINS
        graph_type = 'tip_chain'
        num_graphs = 100
        chain_length = 400
        for i in range(num_graphs):
            graph = generate_tip_chain(chain_length)
            steps = remove_tips(graph)
            filename = os.path.join(data_path, f'{graph_type}_{i}.txt')
            write_to_file(filename, graph, graph_type, steps)

    if args.bubbles:
        # GENERATE BUBBLE CHAINS
        graph_type = 'bubble_chain'
        num_graphs = 100
        chain_length = 400
        for i in range(num_graphs):
            graph = generate_bubble_chain(chain_length)
            steps = find_bubbles(graph)
            filename = os.path.join(data_path, f'{graph_type}_{i}.txt')
            write_to_file(filename, graph, graph_type, steps)

    if args.training:
        # GENERATE TRAINING CHAINS
        graph_type = 'training_chain'
        num_graphs = 100
        chain_length = 50
        for i in range(num_graphs):
            graph = generate_bubble_chain(chain_length)
            steps_trans = remove_transitive(graph)
            steps_tips = remove_tips(graph)
            steps_bubbles = find_bubbles(graph)
            filename = os.path.join(data_path, f'{graph_type}_{i}.txt')
            write_all_to_file(filename, graph, graph_type, [steps_trans, steps_tips, steps_bubbles])

    if args.testing:
        # GENERATE TRAINING CHAINS
        graph_type = 'testing_chain'
        num_graphs = 10
        chain_length = 50
        for i in range(num_graphs):
            graph = generate_bubble_chain(chain_length)
            steps_trans = remove_transitive(graph)
            steps_tips = remove_tips(graph)
            steps_bubbles = find_bubbles(graph)
            filename = os.path.join(data_path, f'{graph_type}_{i}.txt')
            write_all_to_file(filename, graph, graph_type, [steps_trans, steps_tips, steps_bubbles])

    if args.from_csv:
        # CREATE GRAPHS FROM CSV
        graph_type = args.csv_type
        graph_path = args.csv_path
        if not os.path.isfile(graph_path):
            raise OSError('CSV graph file does not exist!')
        graph = read_csv(graph_path)
        steps_trans = remove_transitive(graph)
        steps_tips = remove_tips(graph)
        steps_bubbles = find_bubbles(graph)
        filename = os.path.join(data_path, f'{graph_type}.txt')
        write_all_to_file(filename, graph, graph_type, [steps_trans, steps_tips, steps_bubbles])

    if args.from_fastq:
        fastq_path = args.fastq_path
        threads = args.threads
        fastq_type = args.fastq_type
        if not os.path.isfile(fastq_path):
            raise OSError('FASTQ file does not exist!')
        raw_csv = from_fastq(fastq_path, data_path, threads)
        clean_csv = clean_raw_csv(raw_csv, fastq_type, data_path)

