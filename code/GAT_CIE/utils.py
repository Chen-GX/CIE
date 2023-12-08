import numpy as np
import torch
import os, time
import os.path as osp
import argparse
import random
import scipy.sparse as sp

# from gcn_pyg import CausalGCN
from gat import GATNet, CausalGAT
from torch_geometric.datasets import Planetoid, NELL, PPI
from torch_geometric.utils import add_self_loops
import torch_geometric.transforms as T
from torch_geometric.data import Data


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v == 'True':
        return True
    if v == 'False':
        return False


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def parse_args():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
    parser = argparse.ArgumentParser(description='Node CLF')

    parser.add_argument('--model', type=str, default='CausalGAT', help='GNN model')  # GAT CausalGAT
    parser.add_argument('--bias_type', type=str, default='mixture', help='bias_type, node or struc')
    parser.add_argument('--level', type=str, default='0.4+0.4', help='bias_grade, node(unbiased,0.7,0.8,0.9), struc(unbiased, 0.1, 0.2)')
    parser.add_argument('--timestamp', type=str, default=timestamp, help='timestamp')
    parser.add_argument('--save_result', type=str2bool, default=False)
    parser.add_argument('--with_random', type=str2bool, default=True)
    parser.add_argument('--without_node_attention', type=str2bool, default='False')
    parser.add_argument('--layers', type=int, default=0)
    parser.add_argument('--cat_or_add', type=str, default="add")
    parser.add_argument('--o', type=float, default=1.0)  # object
    parser.add_argument('--idp', type=float, default=0.5)  # context
    parser.add_argument('--co', type=float, default=0.5)
    # 独立的向量
    parser.add_argument('--idp_type', type=str, default='xo', help='xo/o_logs')
    # 特征标准化
    parser.add_argument('--feat_normalize', type=str2bool, default=False)
    # 存储文件
    parser.add_argument('--save_file_name', type=str, default='result')
    # TensorboardX
    parser.add_argument('--tensorboard', type=str2bool, default=True)
    # kernel
    parser.add_argument('--kernel', type=str, default='linear', help='rbf, linear')
    # 最优模型尊则crite
    parser.add_argument('--crite', type=str, default='acc', help='acc/loss')
    # 训练参数
    parser.add_argument('--use_scheduler', type=str2bool, default=False)
    parser.add_argument('--scheduler_type', type=int, default=1, help='1:每n步更新一点学习率, 2: 到达一定程度后, 验证集不变化n步更新学习率')
    # scheduler type 1
    parser.add_argument('--step_size', type=int, default=2, help='每n步更新学习率')
    parser.add_argument('--gamma', type=float, default=0.99, help='每次更新学习率多少')
    # scheduler type 2
    parser.add_argument('--scheduler_begin', type=int, default=50, help='第几个epoch开始调整参数')
    parser.add_argument('--scheduler_threh', type=int, default=50, help='多久没有最优结果就调整学习率')
    ######################## baseline set ##################################
    parser.add_argument('--dataset', type=str, default='Cora', help='dataset')  # Cora Citeseer Pubmed NELL
    parser.add_argument('--data_root', type=str, default="../../generate_bias_data")
    parser.add_argument('--device', type=str, default=device)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--hidden', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--lr', type=float, default=0.015)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1)  # 指的是图个数的batchsize
    parser.add_argument('--train_print_steps', type=int, default=2)
    parser.add_argument('--early_stopping', type=int, default=200)
    parser.add_argument("--log_save_path", type=str, default="log")

    # GAT参数
    parser.add_argument('--head1', type=int, default=8)
    parser.add_argument('--head2', type=int, default=8)

    args = parser.parse_args()
    if args.bias_type == 'node':
        args.data_root = osp.join(args.data_root, 'node_selection_bias_data')
    elif args.bias_type == 'struc':
        args.data_root = osp.join(args.data_root, 'structure_bias_data')
    elif args.bias_type == 'mixture':
        args.data_root = osp.join(args.data_root, 'mixture_bias_data')
    else:
        print(f'no {args.bias_type}')
        assert False
    if args.dataset == 'NELL' and args.model != "SPGCN":
        assert False
    # print(type(args.seed))
    if args.seed != -1:
        setup_seed(args.seed)
    return args


def get_model(args):
    def model_func1(num_features, num_classes):
        return GATNet(num_features, args.hidden, num_classes, head1=args.head1, head2=args.head2, dropout_pct=args.dropout)

    def model_func2(num_features, num_classes):
        return CausalGAT(num_features, args.hidden, num_classes, args, head1=args.head1, head2=args.head2, dropout_pct=args.dropout)

    if args.model == 'GAT':
        assert False
        model_func = model_func1
    elif args.model == "CausalGAT":
        model_func = model_func2
    else:
        assert False

    return model_func


def load_data(args, name, root=None, bias_type='node', level='unbiased'):
    if root is None or root == '':
        path = os.path.join(os.getcwd(), 'pyg_data')
    else:
        path = os.path.join(root)
    if name == 'Cora':
        if bias_type == 'node':
            data = load_node_selec_bias_data(args, name, root=root, level=level)
        elif bias_type == 'struc':
            data = load_struc_bias_data(args, name, root=root, level=level)
        elif bias_type == 'mixture':
            data = load_mixture_bias_data(args, name, root=root, level=level)
        else:
            assert False
    elif name == "citeseer":
        if bias_type == 'node':
            data = load_node_selec_bias_data(args, name, root=root, level=level)
        elif bias_type == 'struc':
            data = load_struc_bias_data(args, name, root=root, level=level)
        else:
            assert False
    elif name == "pubmed":
        if bias_type == 'node':
            data = load_node_selec_bias_data(args, name, root=root, level=level)
        elif bias_type == 'struc':
            data = load_struc_bias_data(args, name, root=root, level=level)
        else:
            assert False
    elif name == "NELL":
        dataset = NELL(root=os.path.join(path, name))
        dataset.data.x = dataset.data.x.to_torch_sparse_coo_tensor()
    elif name == "PPI":
        dataset = PPI(root=os.path.join(path, name))
    else:
        print(f'error no {name} dataset')
        assert False

    print(f"********** dataset {name} info *************")
    print(f"num nodes: {data.num_nodes}")
    print(f"num edges: {data.num_edges}")
    print(f"node features: {data.num_node_features}")
    print(f"edge features: {data.num_edge_features}")
    print(f"class: {data.num_classes}")
    print(f"contains_self_loops:  {data.has_self_loops()}")
    print(f"contains_isolated_nodes:  {data.has_isolated_nodes()}")
    print(f"directed: {data.is_directed()}")
    if name != "PPI":
        print(f"Train: {data.train_mask.sum().item()}")
        print(f"Valid: {data.val_mask.sum().item()}")
        print(f"Test: {data.test_mask.sum().item()}")
    return data


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def load_node_selec_bias_data(args, dataset_name, root, level):
    """加载node selection bias 的数据"""
    train_edge_index = np.load(osp.join(root, '{}_edge_index_train.npy'.format(dataset_name)))
    test_edge_index = np.load(osp.join(root, '{}_edge_index_test.npy'.format(dataset_name)))
    features = np.load(osp.join(root, '{}_features.npy'.format(dataset_name)))
    if args.feat_normalize:
        features = preprocess_features(features)
    label = np.load(osp.join(root, '{}_label.npy'.format(dataset_name)))
    valid_mask = np.load(osp.join(root, '{}_valid_mask.npy'.format(dataset_name)))
    test_mask = np.load(osp.join(root, '{}_test_mask.npy'.format(dataset_name)))
    num_classes = len(set(label))
    # 有区别的数据
    train_mask = np.load(osp.join(root, '{}_train_mask_{}.npy'.format(dataset_name, level)))
    # 转换为torch
    train_edge_index = torch.tensor(train_edge_index, dtype=torch.long)
    test_edge_index = torch.tensor(test_edge_index, dtype=torch.long)
    features = torch.tensor(features, dtype=torch.float)
    label = torch.tensor(label)
    train_mask = torch.tensor(train_mask, dtype=torch.bool)
    valid_mask = torch.tensor(valid_mask)
    test_mask = torch.tensor(test_mask)

    data = Data(x=features, edge_index=train_edge_index, y=label)
    data.test_edge_index = test_edge_index
    data.train_mask = train_mask
    data.val_mask = valid_mask
    data.test_mask = test_mask
    data.num_classes = num_classes
    return data


def load_struc_bias_data(args, dataset_name, root, level):
    """加载structure bias 的数据"""
    features = np.load(osp.join(root, '{}_features.npy'.format(dataset_name)))
    if args.feat_normalize:
        features = preprocess_features(features)
    label = np.load(osp.join(root, '{}_label.npy'.format(dataset_name)))
    train_mask = np.load(osp.join(root, '{}_train_mask.npy'.format(dataset_name)))
    valid_mask = np.load(osp.join(root, '{}_valid_mask.npy'.format(dataset_name)))
    test_mask = np.load(osp.join(root, '{}_test_mask.npy'.format(dataset_name)))
    num_classes = len(set(label))
    # 有区别的数据
    edge_index = np.load(osp.join(root, '{}_edge_index_{}.npy'.format(dataset_name, level)))
    test_edge_index = np.load(osp.join(root, '{}_edge_index_{}.npy'.format(dataset_name, 'test')))

    # 转换为torch
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    test_edge_index = torch.tensor(test_edge_index, dtype=torch.long)
    features = torch.tensor(features, dtype=torch.float)
    label = torch.tensor(label)
    train_mask = torch.tensor(train_mask, dtype=torch.bool)
    valid_mask = torch.tensor(valid_mask)
    test_mask = torch.tensor(test_mask)

    data = Data(x=features, edge_index=edge_index, y=label)
    data.test_edge_index = test_edge_index
    data.train_mask = train_mask
    data.val_mask = valid_mask
    data.test_mask = test_mask
    data.num_classes = num_classes
    return data

def load_mixture_bias_data(args, dataset_name, root, level):
    """加载mixture bias 的数据"""
    select_level, struc_level = level.split('+')
    train_edge_index = np.load(osp.join(root, '{}_edge_index_{}.npy'.format(dataset_name, level)))
    test_edge_index = np.load(osp.join(root, '{}_edge_index_test.npy'.format(dataset_name)))
    features = np.load(osp.join(root, '{}_features.npy'.format(dataset_name)))
    if args.feat_normalize:
        features = preprocess_features(features)
    label = np.load(osp.join(root, '{}_label.npy'.format(dataset_name)))
    valid_mask = np.load(osp.join(root, '{}_valid_mask.npy'.format(dataset_name)))
    test_mask = np.load(osp.join(root, '{}_test_mask.npy'.format(dataset_name)))
    num_classes = len(set(label))
    # 有区别的数据
    train_mask = np.load(osp.join(root, '{}_train_mask_{}.npy'.format(dataset_name, select_level)))
    # 转换为torch
    train_edge_index = torch.tensor(train_edge_index, dtype=torch.long)
    test_edge_index = torch.tensor(test_edge_index, dtype=torch.long)
    features = torch.tensor(features, dtype=torch.float)
    label = torch.tensor(label)
    train_mask = torch.tensor(train_mask, dtype=torch.bool)
    valid_mask = torch.tensor(valid_mask)
    test_mask = torch.tensor(test_mask)

    data = Data(x=features, edge_index=train_edge_index, y=label)
    data.test_edge_index = test_edge_index
    data.train_mask = train_mask
    data.val_mask = valid_mask
    data.test_mask = test_mask
    data.num_classes = num_classes
    return data
