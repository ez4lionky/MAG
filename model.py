import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from util import args
from pytorch_util import gnn_spmm, weights_init
import os
import sys

sys.path.append('%s/s2v_lib' % os.path.dirname(os.path.realpath(__file__)))
from s2v_lib import S2VLIB

class Model(nn.Module):
    def __init__(self, num_node_feats, latent_dim=[32, 32, 32, 1], k=30, conv1d_channels=[16, 32], conv1d_kws=[0, 5]):
        print('\033[31m--Initializing Model...\033[0m\n')
        super(Model, self).__init__()
        self.num_node_feats = num_node_feats
        self.latent_dim = latent_dim
        self.k = k
        self.total_latent_dim = sum(latent_dim)
        conv1d_kws[0] = self.total_latent_dim

        self.conv_params = nn.ModuleList()
        self.conv_params.append(nn.Linear(num_node_feats, latent_dim[0]))
        # X * W => N * F'
        # latent_dim是图卷积层的channel数
        # 添加GCN的weights参数
        for i in range(1, len(latent_dim)):
            self.conv_params.append(nn.Linear(latent_dim[i - 1], latent_dim[i]))

        # sort pooling最后的输出是batch_size, 1, k * 97
        # 所以kernel_size为97，stride为97，对图的每一个顶点的feature（即WL signature）进行一次卷积操作
        self.conv1d_params1 = nn.Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        # batch_size * channels[0] * k
        self.maxpool1d = nn.MaxPool1d(2, 2)
        # batch_size * channels[0] * ((k-2) / 2 + 1)
        self.conv1d_params2 = nn.Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1])
        # 最后一层卷积的kernel_size为5，stride为1，输出维度为 batch_size * channels[1] * ((k-2) / 2 + 1)
        # ？这里的1D卷积的维度问题？

        dense_dim = int((k - 2) / 2 + 1)
        self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        # MLP层的输入维度，即logits=conv1d_res.view(len(graph_sizes), -1)的维度

        weights_init(self)


    # forward的参数是什么，在创建了模型时，就要把对应的参数传入 =》 model(*inputs)
    def forward(self, graph_list, node_feat):
        graph_sizes = [graph_list[i].num_nodes for i in range(len(graph_list))]
        # graph_sizes是一个存储每个图的顶点数的list

        node_degs = [torch.Tensor(graph_list[i].degs) + 1 for i in range(len(graph_list))]
        # 每个顶点的度，并且因为原来没有self-loop，所以现在要+1
        node_degs = torch.cat(node_degs).unsqueeze(1)
        #　一个batch的图的度矩阵的拼接

        n2n_sp, _, __ = S2VLIB.PrepareMeanField(graph_list)
        # n2n_sp matrix，[total_num_nodes, total_num_nodes]
        # 一个batch中所有图的顶点数总和

        node_feat = Variable(node_feat)

        n2n_sp = Variable(n2n_sp) # Adjacent matrix
        node_degs = Variable(node_degs) # D^-1

        h = self.sortpooling_embedding(node_feat, n2n_sp, graph_sizes, node_degs)

        return h

    def sortpooling_embedding(self, node_feat, n2n_sp, graph_sizes, node_degs):

        ''' graph convolution layers '''
        lv = 0
        batch_size = len(graph_sizes)
        cur_message_layer = node_feat
        cat_message_layers = []
        while lv < len(self.latent_dim):
            # n2n_sp即为邻接矩阵，一个batch所有图的邻接矩阵
            n2npool = gnn_spmm(n2n_sp, cur_message_layer) + cur_message_layer  # Y = (A + I) * X
            node_linear = self.conv_params[lv](n2npool)  # Y = Y * W => shape N * F'
            normalized_linear = node_linear.div(node_degs)  # Y = D^-1 * Y
            cur_message_layer = F.tanh(normalized_linear)
            cat_message_layers.append(cur_message_layer)
            lv += 1

        cur_message_layer = torch.cat(cat_message_layers, 1)
        # total_node * 97
        ''' sortpooling layer '''
        sort_channel = cur_message_layer[:, -1]
        # sort_channel：　total_node * 1
        # 只对最后一个channel的feature进行sort
        batch_sortpooling_graphs = torch.zeros(len(graph_sizes), self.k, self.total_latent_dim)
        # 每一个图的顶点数都变为K
        batch_sortpooling_graphs = Variable(batch_sortpooling_graphs)
        accum_count = 0
        for i in range(batch_size):
            to_sort = sort_channel[accum_count: accum_count + graph_sizes[i]]
            k = self.k if self.k <= graph_sizes[i] else graph_sizes[i]  #　下面只需要判断是否pad
            _, topk_indices = to_sort.topk(k)
            # 返回K个最大值元组，(values, indices)
            topk_indices += accum_count
            # 因为是to_sort的indices，在原来的feature中提取出来还需要加上count
            sortpooling_graph = cur_message_layer.index_select(0, topk_indices)
            # 判断是否需要pad
            if k < self.k:
                to_pad = torch.zeros(self.k-k, self.total_latent_dim)

                to_pad = Variable(to_pad)
                sortpooling_graph = torch.cat((sortpooling_graph, to_pad), 0)
            batch_sortpooling_graphs[i] = sortpooling_graph
            accum_count += graph_sizes[i]
            # 每次对一个batch的feature进行sort

        ''' traditional 1d convlution and dense layers '''
        to_conv1d = batch_sortpooling_graphs.view((-1, 1, self.k * self.total_latent_dim))
        # pytorch的卷积输入是： 图片 => channels * width * height;句子 => sentence_length * embedding_size
        # 1D卷积只在一个方向上做卷积，2D卷积要在两个方向上做卷积，唯一的不同是stride参数可能会有些不一样
        # 一次对一个图的特征进行1D-CNN
        # （1D卷积kernel_size=3 <==> 2D kernel_size=3, embedding_size）
        # 即1D卷积默认只是朝一个方向，默认另外一个方向上的维度为全部
        # view => reshape, shape = batch_size, 1, 顶点数 * 97
        conv1d_res = self.conv1d_params1(to_conv1d)
        conv1d_res = F.relu(conv1d_res)
        conv1d_res = self.maxpool1d(conv1d_res)
        conv1d_res = self.conv1d_params2(conv1d_res)
        conv1d_res = F.relu(conv1d_res)
        logits = conv1d_res.view(len(graph_sizes), -1)

        return F.relu(logits)

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, with_dropout=False):
        super(MLPClassifier, self).__init__()
        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, num_class)
        self.with_dropout = with_dropout

        weights_init(self)

    def forward(self, x, y = None):
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, training=self.training)

        logits = self.h2_weights(h1)
        logits = F.log_softmax(logits, dim=1)

        if y is not None:
            y = Variable(y)
            loss = F.nll_loss(logits, y)

            pred = logits.data.max(1, keepdim=True)[1]
            acc = pred.eq(y.data.view_as(pred)).cpu().sum().item() / float(y.size()[0])
            return logits, loss, acc
        else:
            return logits


if __name__ == '__main__':
    print(args)
    model = Model(100, 1)
