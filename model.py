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
    def __init__(self, num_node_feats, n_heads=4, latent_dim=[32, 32, 32], k=30, conv1d_channels=[16, 32],
                 conv1d_kws=[0, 5], lstm_hidden=20, embedding_size=3):
        print('\033[31m--Initializing Model...\033[0m\n')
        super(Model, self).__init__()
        self.num_node_feats = num_node_feats
        self.latent_dim = latent_dim
        if args.model=='concat' or args.model=='no-att':
            # Attention in concat feature
            self.k = k
            self.att_in_size = sum(latent_dim)
        elif args.model=='separate':
            # Separate attention each layer
            self.k = k * len(latent_dim)
            self.att_in_size = latent_dim[0]
        elif args.model == 'fusion':
            # Fusion attention multi scale
            self.k = k
            self.att_in_size = latent_dim[0]

        self.embedding = nn.Embedding(num_node_feats, embedding_size)
        ''' GCN weights '''
        self.conv_params = nn.ModuleList()
        if args.embedding==True:
            self.conv_params.append(nn.Linear(embedding_size, latent_dim[0]))
        else:
            self.conv_params.append(nn.Linear(num_node_feats, latent_dim[0]))
        # X * W => N * F'
        # latent_dim是图卷积层的channel数
        # 添加GCN的weights参数
        for i in range(1, len(latent_dim)):
            self.conv_params.append(nn.Linear(latent_dim[i-1], latent_dim[i]))

        att_out_size = latent_dim[-1]
        ''' Sort pool attention '''
        self.attention = [SpGraphAttentionLayer(in_features=self.att_in_size, out_features=att_out_size,
                                                layer=len(latent_dim),concat=True) for _ in range(n_heads)]
        for i in range(n_heads):
            self.attention[i] = self.attention[i].cuda()

        if args.model=='fusion':
            self.att_out_size = att_out_size
        elif args.model=='no-att':
            self.att_out_size = sum(latent_dim)
        else:
            # fusion 不需要对attention进行concat
            self.att_out_size = att_out_size * n_heads
        conv1d_kws[0] = self.att_out_size
        ''' 2layers 1DCNN for classification '''
        # 最后两层1D卷积和全连接层用于分类
        # sort pooling最后的输出是batch_size, 1, k * 97
        # 所以kernel_size为97，stride为97，对图的每一个顶点的feature（即WL signature）进行一次卷积操作
        self.conv1d_params1 = nn.Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.conv1d_params1 = self.conv1d_params1.cuda()
        # batch_size * channels[0] * k
        self.maxpool1d = nn.MaxPool1d(2, 2)
        # batch_size * channels[0] * ((k-2) / 2 + 1)
        self.conv1d_params2 = nn.Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1])
        # 最后一层卷积的kernel_size为5，stride为1，输出维度为 batch_size * channels[1] * ((k-2) / 2 + 1)

        dense_dim = int((self.k - 2) / 2 + 1)
        if args.concat==0:
            # not concat
            self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        elif args.concat==1:
            # concat 1DCNN
            self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1] + dense_dim * conv1d_channels[0]

        # self.lstm = torch.nn.LSTM(self.total_latent_dim, lstm_hidden, 2)
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
        lv = 0
        N = n2n_sp.size(0)
        batch_size = len(graph_sizes)
        n2n_sp = n2n_sp.cuda()
        node_degs = node_degs.cuda()
        if args.embedding==True:
            node_feat = self.embedding(node_feat.type(torch.LongTensor).cuda()).squeeze()
        cur_message_layer = node_feat
        cat_message_layers = []
        while lv < len(self.latent_dim):
            # n2n_sp即为邻接矩阵，一个batch所有图的邻接矩阵
            n2npool = gnn_spmm(n2n_sp, cur_message_layer) + cur_message_layer  # Y = (A + I) * X
            node_linear = self.conv_params[lv](n2npool)  # Y = Y * W => shape N * F'
            normalized_linear = node_linear
            # normalized_linear = node_linear.div(node_degs)  # Y = D^-1 * Y
            cur_message_layer = torch.tanh(normalized_linear)
            cat_message_layers.append(cur_message_layer)
            lv += 1

        # Attention in concat feature
        if args.model=='concat':
            cur_message_layer = torch.cat(cat_message_layers, 1)
            cur_message_layer = torch.cat([att(non_zero, cur_message_layer)[0] for att in self.attention], dim=1)
            # (total_node, sum(latent_dim))

        # Separate attention each layer
        elif args.model=='separate':
            cur_message_layer = torch.cat(cat_message_layers, 0)
            cur_message_layer = torch.cat([att(non_zero, cur_message_layer)[0] for att in self.attention], dim=1)
            # (total_node * k, latent_dim)

        # Fusion attention multi scale
        elif args.model=='fusion':
            a = []
            for f in cat_message_layers:
                tmp = torch.cat([att(non_zero, f)[1].view(-1, 1) for att in self.attention], dim=1)
                a.append((torch.sum(tmp, 1) / len(self.attention)).view(-1, 1))
            # Attention fusion
            if args.ff=='max':
                a = torch.cat(a, dim=1)
                a_r = torch.max(a, 1)
            elif args.ff=='sum':
                a = torch.cat(a, dim=1)
                a_r = torch.sum(a, 1)
            elif args.ff=='mul':
                a_r = None
                for i in range(len(cat_message_layers)):
                    if i==0:
                        a_r = a[0]
                    else:
                        a_r = torch.mul(a_r, a[i])

            # M = AX
            a_r = a_r.view(-1)
            special_spmm = SpecialSpmm()
            non_zero = torch.LongTensor(non_zero)
            cur_message_layer = special_spmm(non_zero, a_r, torch.Size([N, N]), cat_message_layers[-1])
            # (total_node, latent_dim)
        # No attention and just concat standard GNN
        elif args.model=='no-att':
            cur_message_layer = torch.cat(cat_message_layers, 1)
        ''' sortpooling layer '''
        sort_channel = cur_message_layer[:, -1]
        # sort_channel：　total_node * 1
        # 只对最后一个channel的feature进行sort
        batch_sortpooling_graphs = torch.zeros(len(graph_sizes), self.k, self.att_out_size)
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
                to_pad = torch.zeros(self.k-k, self.att_out_size)
                if isinstance(node_feat.data, torch.cuda.FloatTensor):
                    to_pad = to_pad.cuda()

                to_pad = Variable(to_pad)
                sortpooling_graph = torch.cat((sortpooling_graph, to_pad), 0)
            batch_sortpooling_graphs[i] = sortpooling_graph
            accum_count += graph_sizes[i]
            # 每次对一个batch的feature进行sort

        ''' traditional 1d convlution and dense layers '''
        # batch_size * self.k * att_out_dim
        # 图的数量 * 每个图固定的顶点数 * 最终维度
        res = []
        to_conv1d = batch_sortpooling_graphs.view((-1, 1, self.k * self.att_out_size))
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

        if args.concat==0:
            to_dense = conv1d_res.view(len(graph_sizes), -1)
        elif args.concat==1:
            to_dense = torch.cat(res, 1)


        return F.relu(to_dense)

        # to_lstm = batch_sortpooling_graphs.view((self.k, -1, self.att_out_dim))
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

    def __init__(self, in_features, out_features, layer, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        self.layer = layer

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(0.3)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.special_spmm = SpecialSpmm()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, non_zero, input):
        # Additive attention layer
        N = input.size()[0]

        if args.model=='separate':
            # Separate attention each GCN layer
            n = int(N / self.layer)
            non_zero = list(non_zero)
            tmp1 = []
            tmp2 = []
            for i in range(self.layer):
                tmp1.append(np.add(non_zero[0], n*i))
                tmp2.append(np.add(non_zero[1], n*i))
            non_zero[0] = np.concatenate(tmp1)
            non_zero[1] = np.concatenate(tmp2)

        edge = torch.LongTensor(non_zero)
        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # mask
        # edge: 2 x E
        # edge_h: E × 2F'

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E × 1

        edge_e = self.dropout(edge_e)

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        # 自动mask,因为其他是0
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime), edge_e
        else:
            # if this layer is last layer,
            return h_prime, edge_e


    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


if __name__ == '__main__':
    print(args)
    model = Model(100, 1)