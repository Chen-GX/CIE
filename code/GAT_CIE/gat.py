import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from torch_geometric.nn import MessagePassing
from gat_layer import GATConv, Spmm_GATConv


class GATNet(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, head1, head2, dropout_pct) -> None:
        super().__init__()
        self.dropout_pct = dropout_pct
        self.conv1 = GATConv(in_dim, hid_dim, heads=head1, dropout=dropout_pct)
        self.conv2 = GATConv(hid_dim * head1, out_dim, heads=head2, concat=False, dropout=dropout_pct)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout_pct, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout_pct, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


class CausalGAT(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, args, head1, head2, dropout_pct) -> None:
        super(CausalGAT, self).__init__()
        # out_dim = n_class
        num_conv_layers = args.layers
        self.dropout_pct = dropout_pct
        self.num_classes = out_dim
        self.args = args
        self.with_random = args.with_random
        self.without_node_attention = args.without_node_attention

        # self.bn_feat = nn.BatchNorm1d(in_dim)  # 第一层的标准化
        self.conv_feat = GATConv(in_dim, hid_dim, heads=head1, dropout=dropout_pct)  # 第一层的GCN
        # self.bns_conv = nn.ModuleList()
        self.convs = nn.ModuleList()
        for i in range(num_conv_layers):
            # self.bns_conv.append(nn.BatchNorm1d(hid_dim))
            self.convs.append(GATConv(hid_dim, hid_dim, heads=head1, dropout=dropout_pct))
        # 这上面都是GCN提取特征过程

        self.node_att_mlp = nn.Linear(hid_dim * head1, 2)  # soft mask

        # self.bnc = nn.BatchNorm1d(hid_dim)
        # self.bno = nn.BatchNorm1d(hid_dim)
        self.objects_convs = GATConv(hid_dim * head1, out_dim, heads=head2, concat=False, dropout=dropout_pct)
        self.context_convs = GATConv(hid_dim * head1, out_dim, heads=head2, concat=False, dropout=dropout_pct)

        # random mlp
        if self.args.cat_or_add == "cat":
            assert False
        elif self.args.cat_or_add == "add":
            self.context_objects_convs = GATConv(hid_dim * head1, out_dim, heads=head2, concat=False, dropout=dropout_pct)
        else:
            assert False

        # # BN initialization.
        # for m in self.modules():
        #     if isinstance(m, (torch.nn.BatchNorm1d)):
        #         torch.nn.init.constant_(m.weight, 1)
        #         torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, x, edge_index, eval_random=True):
        # x, edge_index = data.x, data.edge_index

        # GAT提取特征模块
        # x = self.bn_feat(x)
        x = F.dropout(x, p=self.dropout_pct, training=self.training)
        x = F.elu(self.conv_feat(x, edge_index))
        x = F.dropout(x, self.dropout_pct, training=self.training)
        for i, conv in enumerate(self.convs):
            # x = self.bns_conv[i](x)
            assert False
            x = F.relu(conv(x, edge_index))  # 损失函数有问题要看看
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

        return xo_logis, xc_logis, xco_logis, F.elu(xo), F.elu(xc)

    def objects_readout_layer(self, xo, edge_index):

        xo = self.objects_convs(xo, edge_index)
        xo_logis = F.log_softmax(xo, dim=-1)
        return xo_logis

    def context_readout_layer(self, xc, edge_index):

        xc = self.context_convs(xc, edge_index)
        xc_logis = F.log_softmax(xc, dim=-1)
        return xc_logis

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


class Spmm_GAT(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, head1, head2, dropout_pct):
        super().__init__()
        self.dropout_pct = dropout_pct
        self.conv1 = Spmm_GATConv(in_dim, hid_dim, heads=head1, dropout=self.dropout_pct)
        self.conv2 = Spmm_GATConv(hid_dim * head1, out_dim, heads=head2, concat=False, dropout=self.dropout_pct)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # x = F.dropout(x, p=self.dropout_pct, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout_pct, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


class PPI_GAT(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, head1, head2, dropout_pct) -> None:
        super().__init__()
        self.dropout_pct = dropout_pct
        self.conv1 = GATConv(in_dim, hid_dim, heads=head1, dropout=self.dropout_pct, residual=False)
        self.conv2 = GATConv(hid_dim * head1, hid_dim, heads=head1, dropout=self.dropout_pct, residual=True)
        self.conv3 = GATConv(hid_dim * head1, out_dim, heads=head2, concat=False, dropout=self.dropout_pct, residual=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return torch.sigmoid(x)
