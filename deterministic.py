from collections import deque
import torch


def create_adj_list(graph):
    adj_list = {n:[] for n in range(graph.num_nodes)}
    for src, dst in zip(graph.edge_index[0], graph.edge_index[1]):
        src, dst = src.item(), dst.item()
        adj_list[src].append(dst)
    return adj_list


def create_predecessor_list(graph):
    pred_list = {n:[] for n in range(graph.num_nodes)}
    for src, dst in zip(graph.edge_index[0], graph.edge_index[1]):
        src, dst = src.item(), dst.item()
        pred_list[dst].append(src)
    return pred_list


def find_edge_index(graph, src, dst):
    idx_src = set([el.item() for el in (graph.edge_index[0] == src).nonzero()])
    idx_dst = set([el.item() for el in (graph.edge_index[1] == dst).nonzero()])
    index = idx_src & idx_dst
    return index.pop()


def bfs(graph, s):
    graph.adj_list = create_adj_list(graph)
    current, next_step = deque(), deque()
    current.append(s)
    step = 0
    step_list = []
    while current:
        for curr in current:
            if graph.x[curr] == 1:
                continue
            # print(f"Step {step}: visiting node {curr}")
            graph.x[curr] = 1
            next_step.extend(graph.adj_list[curr])
        step += 1
        current = next_step
        next_step = deque()
        step_list.append(graph.x.clone())
        if not current:
            break

    return step_list


def remove_transitive(graph):
    graph.x = torch.zeros_like(graph.x)
    graph.adj_list = create_adj_list(graph)
    graph.transitive = torch.zeros_like(graph.edge_attr)
    for node in range(graph.num_nodes):
        neighbors = graph.adj_list[node]
        for neighbor in neighbors:
            graph.x[neighbor] = 1
        for neighbor in neighbors:
            for trans in graph.adj_list[neighbor]:
                if graph.x[trans] == 1:
                    index = find_edge_index(graph, node, trans)
                    graph.transitive[index] = 1.
        for neighbor in neighbors:
            graph.x[neighbor] = 0.
    step_list = walk_transitive(graph)
    return step_list


def walk_transitive(graph):
    start = 0
    current, next_step = deque(), deque()
    current.append(start)
    step_list = []
    while current:
        for curr in current:
            if graph.x[curr] == 1:
                continue
            graph.x[curr] = 1
            neighbors = graph.adj_list[curr]
            if len(neighbors) == 0:
                break
            for neighbor in neighbors:
                edge = find_edge_index(graph, curr, neighbor)
                if graph.transitive[edge] == 1:
                    continue
                else:
                    next_step.append(neighbor)
        current = next_step
        next_step = deque()
        step_list.append(graph.x.clone())
        if not current:
            break
    return step_list


def remove_tips(graph):
    graph.x = torch.zeros_like(graph.x)
    graph.adj_list = create_adj_list(graph)
    graph.pred_list = create_predecessor_list(graph)
    graph.tips = torch.zeros_like(graph.edge_attr)
    tips = [k for k, v in graph.adj_list.items() if len(v) == 0]
    for node in tips:
        num = 0
        next_node = node
        while True:
            preds = graph.pred_list[next_node]
            if len(preds) == 0:
                break
            next_node = preds[0]
            if len(graph.adj_list[next_node]) > 1 or len(graph.pred_list[next_node]) > 1:
                break
            num += 1
        if len(graph.adj_list[next_node]) > 1 and num <= 5:
            while node != next_node:
                w = graph.pred_list[node][0]
                index = find_edge_index(graph, w, node)
                graph.tips[index] = 1.
                node = w

    step_list = walk_tips(graph)
    return step_list


def walk_tips(graph):
    start = 0
    current, next_step = deque(), deque()
    current.append(start)
    step_list = []
    while current:
        for curr in current:
            if graph.x[curr] == 1:
                continue
            graph.x[curr] = 1
            neighbors = graph.adj_list[curr]
            if len(neighbors) == 0:
                break
            for neighbor in neighbors:
                edge = find_edge_index(graph, curr, neighbor)
                if graph.tips[edge] == 1:
                    continue
                else:
                    next_step.append(neighbor)
        current = next_step
        next_step = deque()
        step_list.append(graph.x.clone())
        if not current:
            break
    return step_list


def find_bubbles(graph):
    graph.x = torch.zeros_like(graph.x)
    graph.adj_list = create_adj_list(graph)
    graph.pred_list = create_predecessor_list(graph)
    graph.bubbles = torch.zeros_like(graph.edge_attr)
    start_nodes = [k for k, v in graph.adj_list.items() if len(v) > 1]
    for start in start_nodes:
        graph.dist = torch.zeros_like(graph.x)
        graph.pred = torch.zeros_like(graph.x) - 1
        B = None
        Q = deque()
        Q.append(start)
        while Q:
            v = Q.popleft()
            for w in graph.adj_list[v]:
                idx = find_edge_index(graph, v, w)
                if w == start:
                    continue
                # if graph.dist[v] + graph.edge_attr[idx] > 5:  # bubble too long
                #     continue
                if graph.pred[w] != -1:
                    B = (v, w)
                    break
                graph.dist[w] = graph.dist[v] + graph.edge_attr[idx]
                graph.pred[w] = v
                Q.append(w)
        if B is None:
            continue

        v, vv = B
        P = [find_edge_index(graph, v, vv)]

        while True:
            if graph.pred[v] == -1:
                break
            else:
                P.append(find_edge_index(graph, graph.pred[v], v))
                v = int(graph.pred[v].item())
        PP = []
        while True:
            if graph.pred[vv] == -1:
                break
            else:
                PP.append(find_edge_index(graph, graph.pred[vv], vv))
                vv = int(graph.pred[vv].item())
        if len(set(P) & set(PP)) > 2:
            continue
        P = P[::-1]
        PP = PP[::-1]
        if not pop_bubble(graph, P):
            pop_bubble(graph, PP)

    return walk_bubbles(graph)


def pop_bubble(graph, P):
    i = -1
    for k in range(len(P)):
        idx = P[k]
        v, w = graph.edge_index[0][idx].item(), graph.edge_index[1][idx].item()
        if len(graph.pred_list[w]) > 1:
            i = k + 1
            break
    j = -1
    for k in range(len(P)):
        idx = P[k]
        v, w = graph.edge_index[0][idx].item(), graph.edge_index[1][idx].item()
        if len(graph.pred_list[w]) > 1 and len(graph.adj_list[w]) > 1:
            return False
        if len(graph.adj_list[w]) > 1:
            j = k + 1
    if i == j == -1:
        for idx in P:
            graph.bubbles[idx] = 1
    elif i == -1:
        for idx in P[j:]:
            graph.bubbles[idx] = 1
    elif j == -1:
        for idx in P[:i]:
            graph.bubbles[idx] = 1
    elif j < i:
        for idx in P[j:i]:
            graph.bubbles[idx] = 1
    else:
        return False
    return True


def walk_bubbles(graph):
    start = 0
    current, next_step = deque(), deque()
    current.append(start)
    step_list = []
    while current:
        for curr in current:
            if graph.x[curr] == 1:
                continue
            graph.x[curr] = 1
            neighbors = graph.adj_list[curr]
            if len(neighbors) == 0:
                break
            for neighbor in neighbors:
                edge = find_edge_index(graph, curr, neighbor)
                if graph.bubbles[edge] == 1:
                    continue
                else:
                    next_step.append(neighbor)
        current = next_step
        next_step = deque()
        step_list.append(graph.x.clone())
        if not current:
            break
    return step_list