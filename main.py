import os
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from model import Model, MLPClassifier
from util import args, load_data
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.dgcnn = Model(latent_dim=args.latent_dim,
                         num_node_feats=args.feat_dim + args.attr_dim,
                         # label one hot vector dimension + att dimension
                         k=args.sortpool_k)

        out_dim = self.dgcnn.dense_dim
        self.mlp = MLPClassifier(input_size=out_dim, hidden_size=args.hidden, num_class=args.num_class)

    def PrepareData(self, batch_graph):
        labels = torch.LongTensor(len(batch_graph))
        n_nodes = 0
        # 一个batch的图的所有顶点数总和
        # node_degs = []

        if batch_graph[0].node_tags is not None:
            node_tag_flag = True
            concat_tag = []
        else:
            node_tag_flag = False

        if batch_graph[0].node_features is not None:
            node_feat_flag = True
            concat_feat = []
        else:
            node_feat_flag = False

        for i in range(len(batch_graph)):
            labels[i] = batch_graph[i].label
            n_nodes += batch_graph[i].num_nodes
            if node_tag_flag == True:
                concat_tag += batch_graph[i].node_tags
            if node_feat_flag == True:
                tmp = torch.from_numpy(batch_graph[i].node_features).type('torch.FloatTensor')
                concat_feat.append(tmp)
            # node_degs += batch_graph[i].degs

        if node_tag_flag == True:
            concat_tag = torch.LongTensor(concat_tag).view(-1, 1)
            if args.embedding==True:
                node_tag = concat_tag
            else:
                node_tag = torch.zeros(n_nodes, args.feat_dim)
                node_tag.scatter_(1, concat_tag, 1)

        if node_feat_flag == True:
            node_feat = torch.cat(concat_feat, 0)

        if node_feat_flag and node_tag_flag:
            # concatenate one-hot embedding of node tags (node labels) with continuous node features
            node_feat = torch.cat([node_tag.type_as(node_feat), node_feat], 1)
        elif node_feat_flag == False and node_tag_flag == True:
            node_feat = node_tag
        elif node_feat_flag == True and node_tag_flag == False:
            pass
        else:
            # node_feat = torch.LongTensor(node_degs).view(-1, 1)
            node_feat = torch.ones(n_nodes, 1)  # use all-one vector as node features

        node_feat = node_feat.cuda()
        labels = labels.cuda()
        return node_feat, labels


    def forward(self, batch_graph):
        length = len(batch_graph)
        batch_graph = np.array(batch_graph)
        index1 = list(range(0, int(length/2)))
        index2 = list(range(int(length/2), length))
        batch_graph1 = batch_graph[index1]
        batch_graph2 = batch_graph[index2]

        node_feat1, labels1 = self.PrepareData(batch_graph1)
        node_feat2, labels2 = self.PrepareData(batch_graph2)


        embed1 = self.dgcnn(batch_graph1, node_feat1)
        embed2 = self.dgcnn(batch_graph2, node_feat2)

        labels = torch.cat([labels1, labels2], 0)
        embed = torch.cat([embed1, embed2], 0)
        return self.mlp(embed, labels)


def loop_dataset(g_list, clf, sample_idxes, optimizer=None, bsize=args.batch_size):
    total_loss = []
    total_iters = (len(sample_idxes) + (bsize - 1) * (optimizer is None)) // bsize
    # 返回一个可迭代total_iters次的tqdm进度条迭代器
    all_targets = []
    all_pred = []

    n_samples = 0
    inputs = None
    for pos in range(total_iters):
        selected_idx = sample_idxes[pos * bsize: (pos + 1) * bsize]

        batch_graph = [g_list[idx] for idx in selected_idx]
        if pos == 1:
            inputs = batch_graph
        targets = [g_list[idx].label for idx in selected_idx]
        all_targets += targets
        logits, loss, acc = clf(batch_graph)
        all_pred.append(logits[:, 1].detach())  # for binary classification

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.data.cpu().numpy()
        if pos == total_iters - 1:
            print('loss: %0.5f acc: %0.5f' % (loss, acc))

        total_loss.append(np.array([loss, acc]) * len(selected_idx))
        n_samples += len(selected_idx)

    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples
    all_pred = torch.cat(all_pred).cpu().numpy()

    # np.savetxt('test_scores.txt', all_pred)  # output test predictions

    all_targets = np.array(all_targets)

    fpr, tpr, _ = metrics.roc_curve(all_targets, all_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    avg_loss = np.concatenate((avg_loss, [auc]))
    return avg_loss


if __name__ == '__main__':
    if os.path.exists('results/acc_results.txt'):
        os.remove('results/acc_results.txt')
        print('Delete')
    if os.path.exists('results/auc_results.txt'):
        os.remove('results/auc_results.txt')
        print('Delete')
    # train_graphs, test_graphs = load_data()
    graphs = load_data()
    labels = [graphs[idx].label for idx in range(len(graphs))]
    print(args)
    random.seed(1)
    if args.sortpool_k <= 1:
        num_nodes_list = sorted([g.num_nodes for g in graphs])
        args.sortpool_k = num_nodes_list[int(math.ceil(args.sortpool_k * len(num_nodes_list))) - 1]
        args.sortpool_k = max(10, args.sortpool_k)
        print('k used in SortPooling is: ' + str(args.sortpool_k))
    skf = StratifiedKFold(n_splits=10)

    accs = []
    aucs = []
    for train_index, test_index in skf.split(graphs, labels):
        print('Training samples', len(train_index))
        print('Testing samples', len(test_index))
        train_graphs, test_graphs = np.array(graphs)[train_index], np.array(graphs)[test_index]
        clf = Classifier()
        clf = clf.cuda()
        optimizer = optim.Adam(clf.parameters(), lr=args.lr)
        train_idxes = list(range(len(train_graphs)))
        for epoch in range(args.num_epochs):
            random.shuffle(train_idxes)
            clf.train()

            # 模型进入训练模式，train和eval方法，主要是drop out与batch_norm层的操作不同
            avg_loss = loop_dataset(train_graphs, clf, train_idxes, optimizer=optimizer)
            print('\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (epoch, avg_loss[0], avg_loss[1], avg_loss[2]))

            clf.eval()
            # 固定住model的参数
            test_loss = loop_dataset(test_graphs, clf, list(range(len(test_graphs))))
            print('\033[93maverage test of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (epoch, test_loss[0], test_loss[1], test_loss[2]))

        print("Test result：")
        test_loss = loop_dataset(test_graphs, clf, list(range(len(test_graphs))))
        print('acc ', test_loss[1], ' auc ', test_loss[2])
        with open('results/acc_results.txt', 'a+') as f:
            f.write(str(test_loss[1]) + '\n')
        accs.append(test_loss[1])

        with open('results/auc_results.txt', 'a+') as f:
            f.write(str(test_loss[2]) + '\n')
        aucs.append(test_loss[2])

    accs = np.array(accs)
    mean_acc = np.sum(accs) / len(accs)
    with open('results/acc_results.txt', 'a+') as f:
        f.write('Mean_acc：\n')
        f.write(str(mean_acc) + '\n')

    aucs = np.array(aucs)
    mean_auc = np.sum(aucs) / len(aucs)
    with open('results/auc_results.txt', 'a+') as f:
        f.write('Mean_auc：\n')
        f.write(str(mean_auc) + '\n')