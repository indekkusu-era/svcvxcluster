import networkx as nx

def is_separable(graph: nx.Graph, alpha=0.9):
    degrees = list(nx.degree(graph))
    maxdeg = max(degrees, key=lambda x: x[1])[1]
    cutoff = maxdeg * alpha
    deg_filtered = list(map(lambda x: x[0], filter(lambda x: x[1] >= cutoff, degrees)))
    new_graph = nx.Graph(graph.subgraph(deg_filtered))
    if nx.number_connected_components(new_graph) > 1:
        return True
    r = nx.radius(new_graph)
    degrees = list(nx.degree(new_graph))
    node_with_max_degree, degree_max = max(degrees, key=lambda x: x[1])
    shortest_path_lengths = nx.single_source_shortest_path_length(new_graph, node_with_max_degree)
    filtered_nodes = {key: value for (key, value) in shortest_path_lengths.items() if value >= r}
    filter_less = {key: value for (key, value) in shortest_path_lengths.items() if value <= r}
    maxnode, maxdeg = max(filter(lambda x: x[0] in filtered_nodes.keys(), degrees), key=lambda x: x[1])
    minnode, mindeg = min(filter(lambda x: x[0] in filter_less.keys(), degrees), key=lambda x: x[1])
    dist_maxnode = filtered_nodes[maxnode]
    dist_minnode = filter_less[minnode]
    if dist_maxnode == dist_minnode:
        return False
    return True

def separate_clusters(graph: nx.Graph, alpha=0.75):
    degrees = list(nx.degree(graph))
    maxdeg = max(degrees, key=lambda x: x[1])[1]
    cutoff = maxdeg * alpha
    deg_filtered = list(map(lambda x: x[0], filter(lambda x: x[1] >= cutoff, degrees)))
    components = sorted(list(nx.connected_components(graph.subgraph(deg_filtered))), key=lambda x: -len(x))
    labels = [[] for _ in components]
    visited = set()
    bfs = [nx.bfs_layers(graph, component) for component in components]
    nexts = [next(b) for b in bfs]
    while len(visited) < len(graph.nodes):
        try:
            for i, nex in enumerate(nexts):
                nodes_left = set(nex).difference(visited)
                if not len(nodes_left):
                    continue
                labels[i].extend(nodes_left)
                visited = visited.union(nodes_left)
                nexts[i] = next(bfs[i])
        except StopIteration:
            continue
    return labels

def separate(graph: nx.Graph, alpha=0.75, recursive_lvl=2):
    not_separated = [graph]
    recursion = [0]
    separated = []
    while not_separated:
        g = not_separated.pop()
        re = recursion.pop()
        if not is_separable(g, alpha) or re >= recursive_lvl:
            separated.append(list(g.nodes))
        else:
            labels = separate_clusters(g, alpha)
            for lbl in labels:
                not_separated.append(nx.Graph(g.subgraph(lbl)))
                recursion.append(re + 1)
    return separated

