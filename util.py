import torch
import numpy as np
import argparse
import networkx as nx
from graphviz import Digraph
from torch.autograd import Variable

# ？问题，argparse.ArgumnetParser的值可以通过import共享？还是说每次导入都会重新读
cmd_par = argparse.ArgumentParser(description='Argparser for graph classification experiments')
cmd_par.add_argument('-data', default='MUTAG', help='data folder name')
cmd_par.add_argument('-fold', type=int, default=1, help='Test data fold 1-10')
cmd_par.add_argument('-latent_dim', type=str, default='32 32 32 32 32', help='dimension(s) of attention layers')
cmd_par.add_argument('-num_epochs', type=int, default=200, help='number of epochs')
cmd_par.add_argument('-sortpool_k', type=float, default=0.6, help='Percentage of nodes kept after SortPooling')
cmd_par.add_argument('-lr', type=float, default=0.01, help='init learning_rate')
cmd_par.add_argument('-hidden', type=int, default=32, help='dimension of regression')
cmd_par.add_argument('-batch_size', type=int, default=128, help='minibatch size')
cmd_par.add_argument('-model', type=str, default='gin', help='concat, separate, fusion, no-att, gin')
cmd_par.add_argument('-embedding', type=bool, default=False, help='True or false')
# 0 presents Attention in concat features, 1 presents Attention separate, 2 presents add fusion function
cmd_par.add_argument('-concat', type=int, default=1, help='0 presents 1D-CNN feature concat, 1 presents not concat')
cmd_par.add_argument('-ff', type=str, default='mul', help='fusion function - max, sum, mul')


args = cmd_par.parse_args()
args.latent_dim = [int(x) for x in args.latent_dim.split(' ')]
if len(args.latent_dim) == 1:
    args.latent_dim = args.latent_dim[0]

class Graph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a numpy array of continuous node features
        '''

        self.num_nodes = len(node_tags)
        self.node_tags = node_tags
        self.label = label
        self.node_features = node_features  # numpy array [node_num, feature_dim]
        self.degs = list(dict(g.degree).values())

        if len(g.edges()) != 0:
            x, y = zip(*g.edges())
            # *用以传入多个参数，并以数组的形式传入
            # x[i] y[i]为一条边
            self.num_edges = len(x)
            self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
            self.edge_pairs[:, 0] = x
            self.edge_pairs[:, 1] = y
            self.edge_pairs = self.edge_pairs.flatten()
            # flatten in row major, which means flattened row by row
            # edge_pairs: 1 * 2n
            # [:n]是边的第一个点，[n+1:]是边的第二个点
        else:
            self.num_edges = 0
            self.edge_pairs = np.array([])


def load_data():
    print("\033[31m--loading data...\033[0m")
    g_list = []
    # graph list
    label_dict = {}
    # the set of graph label, key: node_tag, value: order
    # 因为判断a in dict，只会查询dict的key
    # 所以直接以label为key，并且将每个图的label转换成特定顺序的数值
    feat_dict = {}
    # the set of node tag, key: node_tag, value: order

    with open('data/%s/%s.txt' % (args.data, args.data)) as f:
        n_g = int(f.readline().strip())
        # 1st line is number of graph
        # following 1 line of n, l
        for i in range(n_g):
            line = f.readline().strip().split()
            n, l = [int(_) for _ in line]
            # n is the number of nodes in current graph
            # l is the label of current graph
            if not l in label_dict:
                # add new graph label in set label_dict
                _ = len(label_dict)
                label_dict[l] = _
            g = nx.Graph()
            node_tags = []
            node_features = []
            # iterate all node in current graph
            for j in range(n):
                g.add_node(j)
                line = f.readline().strip().split()
                # following t, m
                # t is the node tag, and m is the number of node's neighbours
                # then following m neighbours' indices
                node_tag, n_of_neighbours = int(line[0]), int(line[1])
                tmp = n_of_neighbours + 2
                if tmp == len(line):
                    # current node hasn't node features(attributes)
                    line = [int(_) for _ in line]
                    attr = None
                else:
                    line, attr = [int(_) for _ in line[:tmp]], np.array([float(_) for _ in line[tmp:]])
                if not line[0] in feat_dict:
                    _ = len(feat_dict)
                    feat_dict[line[0]] = _
                node_tags.append(feat_dict[line[0]])

                # 若node有attr
                if tmp < len(line):
                    node_features.append(attr)

                for k in range(2, len(line)):
                    g.add_edge(j, line[k])

            # 若node_features不为空集
            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n
            g_list.append(Graph(g, l, node_tags, node_features))

    for g in g_list:
        g.label = label_dict[g.label]

    args.num_class = len(label_dict)
    args.feat_dim = len(feat_dict)  # equals to maximum node label (tag)
    if node_feature_flag == True:
        args.attr_dim = node_features.shape[1]  # dim of node features (attributes)
    else:
        args.attr_dim = 0

    print('# classes: %d' % args.num_class)
    print('# maximum node tag: %d' % args.feat_dim)

    return g_list


def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(var.grad_fn)
    return dot


def print_params(params, state_dict):
    k = 0
    for name, param in params:
        l = 1
        print(name)
        print('Requires_grad: ', param.requires_grad)
        print("该层的结构：" + str(list(param.size())))
        for j in param.size():
            l *= j
        print("该层参数和：" + str(l))
        k = k + l
    print("总参数数量和：" + str(k))


if __name__ == '__main__':
    train_graphs, test_graphs = load_data()
    print(len(train_graphs))
    print(len(test_graphs))
    # len(train_graphs) = 1061
    # len(test_graphs) = 117
