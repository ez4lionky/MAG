import os
import sys
import torch
import numpy as np
import torch.nn as nn
from util import args
import torch.nn.functional as F
from torch.autograd import Variable
from pytorch_util import gnn_spmm, weights_init

sys.path.append('%s/s2v_lib' % os.path.dirname(os.path.realpath(__file__)))
from s2v_lib import S2VLIB

class Model(nn.Module):
    def __init__(self, num_node_feats, node_feature_size, n_heads=4, latent_dim=[32, 32, 32, 32],
                 k=30, conv1d_channels=[16, 32], conv1d_kws=[0, 5], lstm_hidden=20):
        print('\033[31m--Initializing Model...\033[0m\n')
        super(Model, self).__init__()
        self.num_node_feats = num_node_feats
        self.latent_dim = latent_dim
        self.total_latent_dim = sum(self.latent_dim)
        # self.total_latent_dim = self.latent_dim
        self.k = k
        conv1d_kws[0] = self.total_latent_dim

        # self.attentions = [SpGraphAttentionLayer(in_features=node_feature_size, out_features=latent_dim, concat=True)]
        # for i in range(len(self.attentions)):
        #     self.attentions[i].cuda()

        self.attentions = []
        for i in range(len(latent_dim)):
            if i==0:
                attention = [SpGraphAttentionLayer(in_features=node_feature_size, out_features=latent_dim[0], concat=True)
                 for _ in range(n_heads)]
            else:
                attention = [SpGraphAttentionLayer(in_features=latent_dim[i-1], out_features=latent_dim[i], concat=True)
                 for _ in range(n_heads)]
            self.attentions.append(attention)

        for i in range(len(latent_dim)):
            for j in range(n_heads):
                self.attentions[i][j].cuda()

        # 最后两层1D卷积和全连接层用于分类
        # sort pooling最后的输出是batch_size, 1, k * 97
        # 所以kernel_size为97，stride为97，对图的每一个顶点的feature（即WL signature）进行一次卷积操作
        self.conv1d_params1 = nn.Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        # batch_size * channels[0] * k
        self.maxpool1d = nn.MaxPool1d(2, 2)
        # batch_size * channels[0] * ((k-2) / 2 + 1)
        self.conv1d_params2 = nn.Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1])
        # 最后一层卷积的kernel_size为5，stride为1，输出维度为 batch_size * channels[1] * ((k-2) / 2 + 1)
        # ？这里的1D卷积的维度问题？


        self.lstm = torch.nn.LSTM(self.total_latent_dim, lstm_hidden, 2)

        dense_dim = int((k - 2) / 2 + 1)
        self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]

        # self.dense_dim = self.k * lstm_hidden
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

        non_zero, n2n_sp, _, __ = S2VLIB.PrepareMeanField(graph_list)
        # n2n_sp matrix，[total_num_nodes, total_num_nodes]
        # non_zero是有邻边的索引 2 * E
        # 一个batch中所有图的顶点数总和

        if isinstance(node_feat, torch.cuda.FloatTensor):
            n2n_sp = n2n_sp.cuda()
            node_degs = node_degs.cuda()

        node_feat = Variable(node_feat)

        n2n_sp = Variable(n2n_sp) # Adjacent matrix
        node_degs = Variable(node_degs) # D^-1

        h = self.sortpooling_embedding(non_zero, node_feat, n2n_sp, graph_sizes, node_degs)

        return h

    def sortpooling_embedding(self, non_zero, node_feat, n2n_sp, graph_sizes, node_degs):

        ''' graph convolution layers '''
        batch_size = len(graph_sizes)
        cat_message_layers = []
        cur_message_layer = None
        for i in range(len(self.latent_dim)):
            if i==0:
                cur_message_layer = np.sum([att(non_zero, node_feat) for att in self.attentions[i]])/len(self.attentions[i])
            else:
                cur_message_layer = np.sum([att(non_zero, cur_message_layer) for att in self.attentions[i]])/len(self.attentions[i])
            cat_message_layers.append(cur_message_layer)

        cur_message_layer = torch.cat(cat_message_layers, 1)
        ''' sortpooling layer '''
        sort_channel = cur_message_layer[:, -1]
        # sort_channel：　total_node * 1
        # 只对最后一个channel的feature进行sort
        batch_sortpooling_graphs = torch.zeros(len(graph_sizes), self.k, self.total_latent_dim)
        # 每一个图的顶点数都变为K
        batch_sortpooling_graphs = Variable(batch_sortpooling_graphs)
        if isinstance(node_feat.data, torch.cuda.FloatTensor):
            batch_sortpooling_graphs = batch_sortpooling_graphs.cuda()
        accum_count = 0
        # sort pool操作只对node_feat进行操作
        for i in range(batch_size):
            to_sort = sort_channel[accum_count: accum_count + graph_sizes[i]]
            k = self.k if self.k <= graph_sizes[i] else graph_sizes[i]  #　下面只需要判断是否pad
            _, topk_indices = to_sort.topk(k)
            # 返回K个最大值元组，(values, indices)
            topk_indices += accum_count
            # 因为是to_sort的indices，在原来的feature中提取出来还需要加上count
            sortpooling_graph = cur_message_layer.index_select(0, topk_indices).cuda()
            # 判断是否需要pad
            if k < self.k:
                to_pad = torch.zeros(self.k-k, self.total_latent_dim)
                if isinstance(node_feat.data, torch.cuda.FloatTensor):
                    to_pad = to_pad.cuda()

                to_pad = Variable(to_pad)
                sortpooling_graph = torch.cat((sortpooling_graph, to_pad), 0)
            batch_sortpooling_graphs[i] = sortpooling_graph
            accum_count += graph_sizes[i]
            # 每次对一个batch的feature进行sort

        ''' traditional 1d convlution and dense layers '''
        # batch_size * self.k * total_latent_dim
        # 图的数量 * 每个图固定的顶点数 * 最终维度
        res = []
        to_conv1d = batch_sortpooling_graphs.view((-1, 1, self.k * self.total_latent_dim))
        conv1d_res = self.conv1d_params1(to_conv1d)
        conv1d_res = F.relu(conv1d_res)
        # print("conv1d_res.shape", conv1d_res.shape) # 50 * 16 * 291
        conv1d_res = self.maxpool1d(conv1d_res)
        res.append(conv1d_res.view(len(graph_sizes), -1))
        # print("conv1d_res.shape", conv1d_res.shape) # 50 * 16 * 145
        conv1d_res = self.conv1d_params2(conv1d_res)
        conv1d_res = F.relu(conv1d_res)
        res.append(conv1d_res.view(len(graph_sizes), -1))
        # print("conv1d_res.shape", conv1d_res.shape) # 50 * 32 * 141

        # to_dense = torch.cat(res, 1)
        to_dense = conv1d_res.view(len(graph_sizes), -1)

        return F.relu(to_dense)

        # to_lstm = batch_sortpooling_graphs.view((self.k, -1, self.total_latent_dim))
        # lstm_features, self_hidden = self.lstm(to_lstm)
        # logits = lstm_features.view(len(graph_sizes), -1)
        #
        # return F.relu(logits)


class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_class):
        super(MLPClassifier, self).__init__()
        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, num_class)

        weights_init(self)

    def forward(self, x, y = None):
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, training=True)

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


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        b = b.cuda()
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape).cuda()
        # 使用COO格式存储稀疏矩阵，即三元组的形式
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(0.6)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.special_spmm = SpecialSpmm()

    def forward(self, non_zero, input):
        N = input.size()[0]
        edge = torch.LongTensor(non_zero)
        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2 x E

        print(edge.shape)
        print(edge_h.shape)
        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1)))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


if __name__ == '__main__':
    print(args)
    model = Model(100, 1)
