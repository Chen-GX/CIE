import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import random

# from gcnconv import GCNConv


class CausalSAGE(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, args, dropout_pct) -> None:
        super(CausalSAGE, self).__init__()
        # out_dim = n_class
        num_conv_layers = args.layers
        self.dropout_pct = dropout_pct
        self.num_classes = out_dim
        self.args = args
        self.with_random = args.with_random
        self.without_node_attention = args.without_node_attention

        # self.bn_feat = nn.BatchNorm1d(in_dim)  # 第一层的标准化
        self.conv_feat = SAGEConv(in_dim, hid_dim)  # 第一层的GCN
        self.bns_conv = nn.ModuleList()
        self.convs = nn.ModuleList()
        for i in range(num_conv_layers):
            self.bns_conv.append(nn.BatchNorm1d(hid_dim))
            self.convs.append(SAGEConv(hid_dim, hid_dim))
        # 这上面都是GCN提取特征过程

        self.node_att_mlp = nn.Linear(hid_dim, 2)  # soft mask

        # self.bnc = nn.BatchNorm1d(hid_dim)
        # self.bno = nn.BatchNorm1d(hid_dim)
        self.context_convs = SAGEConv(hid_dim, out_dim)
        self.objects_convs = SAGEConv(hid_dim, out_dim)

        # random mlp
        if self.args.cat_or_add == "cat":
            assert False

        elif self.args.cat_or_add == "add":
            self.context_objects_convs = SAGEConv(hid_dim, out_dim)
        else:
            assert False

        # BN initialization.  # 网络层缺少初始化，后面记得加上
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, x, edge_index, eval_random=True):
        # x, edge_index = data.x, data.edge_index

        # GCN提取特征模块
        # x = self.bn_feat(x)
        # x = F.dropout(x, self.dropout_pct, training=self.training)
        x = F.relu(self.conv_feat(x, edge_index))
        x = F.dropout(x, self.dropout_pct, training=self.training)
        for i, conv in enumerate(self.convs):
            x = self.bns_conv[i](x)
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, self.dropout_pct, training=self.training)

        # Soft Mask 模块
        if self.without_node_attention:
            node_att = 0.5 * torch.ones(x.shape[0], 2).cuda()
        else:  # x [num_nodes, hid_dim]  node_att [num_nodes, 2] 给每个节点一个att值
            node_att = F.softmax(self.node_att_mlp(x), dim=-1)  # 同上
        # node_att[:, 0].view(-1, 1) [num_nodes, 1]
        xo = node_att[:, 0].view(-1, 1) * x
        xc = node_att[:, 1].view(-1, 1) * x  # view是为了增加一个维度，xc[num_nodes, hid_dim]

        # xc = F.relu(self.context_convs(self.bnc(xc), edge_index))
        # xc = F.dropout(xc, self.dropout_pct, training=self.training)
        # xo = F.relu(self.objects_convs(self.bno(xo), edge_index))
        # xo = F.dropout(xo, self.dropout_pct, training=self.training)
        xo_logis = self.objects_readout_layer(xo, edge_index)
        xc_logis = self.context_readout_layer(xc, edge_index)
        xco_logis = self.random_readout_layer(xc, xo, edge_index, eval_random=eval_random)

        return xo_logis, xc_logis, xco_logis, F.relu(xo), F.relu(xc)

    def context_readout_layer(self, x, edge_index):

        x = self.context_convs(x, edge_index)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def objects_readout_layer(self, x, edge_index):

        x = self.objects_convs(x, edge_index)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def random_readout_layer(self, xc, xo, edge_index, eval_random):

        num = xc.shape[0]
        l = [i for i in range(num)]
        if self.with_random:
            if eval_random:
                random.shuffle(l)
        random_idx = torch.tensor(l)
        if self.args.cat_or_add == "cat":
            x = torch.cat((xc[random_idx], xo), dim=1)
        else:
            x = xc[random_idx] + xo

        x = self.context_objects_convs(x, edge_index)  # 一层GNN
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis
