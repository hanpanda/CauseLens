import os
from tqdm import tqdm
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
import random
from dgl.dataloading import GraphDataLoader

from models import loss_func_customgae, GraphAE
from utils import get_cur_time, euclidean
from run import load_dataset
from rca import compute_rca_metrics, compute_distance_statistics_for_windows, compute_anomaly_score
from log import Logger

logger = Logger(__name__)


def train_graphae(train_dataset, model):
    train_loader = GraphDataLoader(dataset=train_dataset, batch_size=args.batch_size, drop_last=False, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_loss = 0
        loop = tqdm(enumerate(train_loader), total=len(train_loader))

        for batch, (graph, _) in loop:
            optimizer.zero_grad()
            node_feat_dict = graph.ndata['feat']

            predict_node_feat_dict = model(graph, node_feat_dict, args.device)
            loss = loss_func_customgae(
                g=graph, X=node_feat_dict, X_hat=predict_node_feat_dict, recon_ntypes=['api', 'pod']
            )

            l = torch.mean(loss)
            train_loss += l.item()
            l.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        logger.info('epoch: {}. train loss: {}.'.format(epoch, train_loss))

    model_filepath = os.path.join(args.store_dir, 'graphae_model.pth')
    torch.save(model.state_dict(), model_filepath)
    logger.info('Model filepath: {}'.format(model_filepath))


def compute_statistics(train_dataset, model, nan_nodes):
    train_loader = GraphDataLoader(dataset=train_dataset, batch_size=args.batch_size, drop_last=False, shuffle=True)
    dist = defaultdict(list)
    dfs = []
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    timestamps = []

    for batch, (graph, label) in loop:
        node_feat_dict = graph.ndata['feat']
        predict_node_feat_dict = model(graph, node_feat_dict, args.device)

        for ntype in node_feat_dict.keys():
            dist[ntype].append(
                euclidean(node_feat_dict[ntype],
                          predict_node_feat_dict[ntype]).view(-1, graph.batch_size).cpu().detach().numpy()
            )
        timestamps += label['timestamp']
        # print(type(label['timestamp']), len(label['timestamp']))

    for ntype in dist.keys():
        dist_ntype = np.concatenate(dist[ntype], axis=-1)
        raw_dist_df = pd.DataFrame(data=dist_ntype.swapaxes(0, -1))
        raw_dist_df[nan_nodes[ntype].numpy()] = np.nan
        raw_dist_df['timestamp'] = timestamps
        raw_dist_df.to_csv(os.path.join(args.run_dir, 'raw_euclidean_dist_{}_.csv'.format(ntype)), index=False)
        mean = np.mean(dist_ntype, axis=-1)
        std = np.std(dist_ntype, axis=-1)
        df = pd.DataFrame({'mean': mean, 'std': std})
        df['ntype'] = ntype
        dfs.append(df)
    pd.concat(dfs).to_csv(os.path.join(args.run_dir, 'dist_stats.csv'), index_label='id')


def test_graphae(test_dataset, model, groundtruths, nan_nodes):
    loop = tqdm(enumerate(test_dataset), total=len(test_dataset))
    results = []

    for i, (graph, label) in loop:
        level = None
        if 'level' in label:
            level = label['level']

        logger.info(
            '--------------Timestamp: {}. Failure: {}. Level: {}. cmdb_id: {}.--------------'.format(
                label['timestamp'], label['failure_type'], level, label['cmdb_id']
            )
        )

        node_feat_dict = graph.ndata['feat']
        predict_node_feat_dict = model(graph, node_feat_dict, args.device)
        score_dict = score_nodes(node_feat_dict, predict_node_feat_dict, label['timestamp'], args.window)

        result = {
            'timestamp': label['timestamp'],
            'failure_type': label['failure_type'],
            'level': level,
            'cmdb_id': label['cmdb_id']
        }
        candidates = []
        for ntype in score_dict:
            # print(nan_nodes[ntype].numpy().shape, nan_nodes[ntype].numpy().dtype, nan_nodes[ntype].numpy()[0])
            indices = np.where(nan_nodes[ntype].numpy()[i] == True)[0]
            for index in indices:
                score_dict[ntype][index] = 0
            candidates += [{
                'ntype': ntype,
                'id': id,
                'score': score,
                'name': test_dataset.dgl_id_to_name(id, ntype)
            } for id, score in enumerate(score_dict[ntype])]
        result['cand'] = sorted(candidates, key=lambda item: item['score'], reverse=True)
        results.append(result)

        for i in range(10):
            logger.info(result['cand'][i])

    for failure_type in args.failure_types:
        logger.info('-----------------{}-----------------'.format(failure_type))
        compute_rca_metrics(results, groundtruths, failure_type)
    compute_rca_metrics(results, groundtruths)


def score_nodes(node_feat_dict, predict_node_feat_dict, timestamp, window=0):
    if window != 0:
        score_dict = {}
        dist_stats = compute_distance_statistics_for_windows(args.run_dir, 'euclidean', '', window)

        for ntype in node_feat_dict.keys():
            score, _, _, _ = compute_anomaly_score(
                ntype, node_feat_dict[ntype].detach(), predict_node_feat_dict[ntype].detach(), timestamp, dist_stats
            )
            score_dict[ntype] = score.tolist()

    else:
        df = pd.read_csv(os.path.join(args.run_dir, 'dist_stats.csv'))
        score_dict = {}

        for ntype in node_feat_dict.keys():
            feat = node_feat_dict[ntype]
            predict_feat = predict_node_feat_dict[ntype]
            # error = torch.sum(torch.pow(feat - predict_feat, 2), 1)
            error = euclidean(feat, predict_feat).detach().numpy()

            mean = df[df['ntype'] == ntype]['mean'].to_numpy()
            std = df[df['ntype'] == ntype]['std'].to_numpy()
            score_dict[ntype] = np.abs((error - mean) / std).tolist()

    return score_dict


class args:
    bidirected = False
    mode = None
    store_dir = None
    window = 0

    @classmethod
    def basic_config_for_aiops22(cls):
        cls.mode = 'test'
        cls.store_dir = './model-store/aiops22/GraphAE/train_2024_04_09_22_43'

        # train parameters
        cls.seed = 42
        cls.device = 'cuda'
        cls.epochs = 100
        cls.lr = 1e-3
        cls.batch_size = 64
        cls.conv_type = 'GATConv'
        cls.conv_type = 'DotGATConv'
        cls.conv_type = 'GraphConv'
        cls.in_feats = 10
        cls.hidden_feats = 10
        cls.used_etypes = ['api2api', 'api2pod']

        # data attribute
        cls.dataset = 'aiops22'
        cls.data_dir = '../datasets/aiops_2022/'
        cls.cloudbeds = ['cloudbed-1']
        cls.dates = [
            '2022-03-19', '2022-03-20', '2022-03-21', '2022-03-24', '2022-03-26', '2022-03-28', '2022-03-29',
            '2022-03-30', '2022-03-31', '2022-04-01'
        ]
        # dates = ['2022-03-19']
        cls.nfeat_select = {
            'api': ['count', 'mean', 'min', 'q10', 'q20', 'q30', 'q40', 'q50', 'q60', 'q70', 'q80', 'q90', 'max'],
            'pod': [
                'container_cpu_usage_seconds', 'container_cpu_cfs_throttled_seconds', 'container_memory_usage_MB',
                'container_memory_working_set_MB', 'container_fs_io_time_seconds./dev/vda1',
                'container_fs_io_current./dev/vda1', 'container_network_receive_errors.eth0',
                'container_network_transmit_errors.eth0', 'container_network_receive_MB.eth0',
                'container_network_transmit_MB.eth0', 'container_network_receive_packets_dropped.eth0',
                'container_network_transmit_packets_dropped.eth0', 'container_threads', 'count'
            ]
        }
        cls.node_in_feats = {ntype: len(feats) for ntype, feats in cls.nfeat_select.items()}
        cls.failure_types = ['']
        cls.failure_duration = 15

        # data process
        cls.edge_reverse = True
        cls.mask = False
        cls.process_extreme = True
        cls.process_miss = 'interpolate'
        cls.log_before_scale = True
        cls.node_scale_type = 'nodewise'
        cls.edge_scale_type = 'edgewise'

    @classmethod
    def train_mode_for_aiops22(cls):
        cls.basic_config_for_aiops22()
        cls.mode = 'train'

    @classmethod
    def test_mode_for_aiops22(cls):
        cls.basic_config_for_aiops22()
        cls.mode = 'test'
        cls.device = 'cpu'
        cls.failure_types = [
            'k8s容器cpu负载', 'k8s容器网络延迟', 'k8s容器网络资源包损坏', 'k8s容器网络丢包', 'k8s容器内存负载', 'k8s容器读io负载', 'k8s容器写io负载', 'k8s容器进程中止'
        ]
        cls.failure_duration = 2
        cls.dates = ['']
        cls.process_extreme = False
        cls.model_file = os.path.join(cls.store_dir, 'graphae_model.pth')
        cls.run_dir = os.path.join(cls.store_dir, 'run-info')

    @classmethod
    def compute_mode_for_aiops22(cls):
        cls.basic_config_for_aiops22()
        cls.model_file = os.path.join(cls.store_dir, 'graphae_model.pth')
        cls.run_dir = os.path.join(cls.store_dir, 'run-info')

    @classmethod
    def create_directory(cls):
        cls.store_dir = './model-store/{}/GraphAE/{}_{}'.format(cls.dataset, cls.mode, get_cur_time())
        cls.run_dir = os.path.join(cls.store_dir, 'run-info')
        if os.path.exists(cls.run_dir) == False:
            os.makedirs(cls.run_dir)

    @classmethod
    def basic_config_for_trainticket(cls):
        # train parameters
        cls.seed = 42
        cls.device = 'cuda'
        cls.epochs = 200
        cls.lr = 1e-3
        cls.batch_size = 64
        # cls.conv_type = 'GATConv'
        # cls.conv_type = 'DotGATConv'
        cls.conv_type = 'GraphConv'
        cls.in_feats = 12
        cls.hidden_feats = 8
        cls.used_etypes = ['api2api', 'api2pod']
        cls.num_heads = 12

        # data attribute
        cls.dataset = 'TrainTicket'
        cls.data_dir = '../datasets/Nezha/TrainTicket_2024'
        cls.dates = ['2024-04-22', '2024-04-23', '2024-04-24']
        cls.nfeat_select = {
            'api': ['count', 'mean', 'q10', 'q20', 'q30', 'q40', 'q50', 'q60', 'q70', 'q80', 'q90', 'q100'],
            'pod': [
                'CpuUsage(m)', 'CpuUsageRate(%)', 'MemoryUsage(Mi)', 'MemoryUsageRate(%)', 'SyscallRead',
                'SyscallWrite', 'NetworkReceiveBytes', 'NetworkTransmitBytes', 'PodWorkload(Ops)'
            ]
        }
        cls.node_in_feats = {ntype: len(feats) for ntype, feats in cls.nfeat_select.items()}
        cls.failure_types = ['']
        cls.failure_duration = 15

        # data process
        cls.edge_reverse = False
        cls.mask = False
        cls.process_extreme = True
        cls.k_sigma = 3
        cls.process_miss = 'interpolate'
        cls.log_before_scale = True
        cls.node_scale_type = 'nodewise'
        cls.edge_scale_type = 'edgewise'

    @classmethod
    def train_mode_for_trainticket(cls):
        cls.basic_config_for_trainticket()
        cls.mode = 'train'
        # cls.dates = ['2024-04-22', '2024-04-23', '2024-04-24']
        cls.dates = ['2024-04-02', '2024-04-03', '2024-04-04']

    @classmethod
    def test_mode_for_trainticket(cls):
        cls.basic_config_for_trainticket()
        cls.mode = 'test'
        cls.device = 'cpu'
        # cls.dates = ['2024-04-25']
        cls.dates = ['2024-04-05', '2024-04-06']
        # cls.failure_types = ['code_delay', 'exception']
        cls.failure_types = ['cpu_contention', 'network_delay']
        cls.failure_duration = 3
        cls.process_extreme = False
        cls.model_file = os.path.join(cls.store_dir, 'graphae_model.pth')
        cls.run_dir = os.path.join(cls.store_dir, 'run-info')

    @classmethod
    def compute_mode_for_trainticket(cls):
        cls.mode = 'compute'
        cls.basic_config_for_trainticket()
        # cls.dates = ['2024-04-22', '2024-04-23', '2024-04-24', '2024-04-25']
        cls.dates = ['2024-04-02', '2024-04-03', '2024-04-04', '2024-04-05', '2024-04-06']
        cls.model_file = os.path.join(cls.store_dir, 'graphae_model.pth')
        cls.run_dir = os.path.join(cls.store_dir, 'run-info')


def main():
    args.mode = 'test'
    args.window = 60 * 5
    args.bidirected = False
    args.dataset = 'TrainTicket'
    args.store_dir = './model-store/TrainTicket/GraphAE/train_2024_05_07_20_23/'
    logger.info('Mode: {}'.format(args.mode))

    if args.dataset == 'aiops22':
        if args.mode == 'train':
            args.train_mode_for_aiops22()
            args.create_directory()
        elif args.mode == 'compute':
            args.compute_mode_for_aiops22()
        elif args.mode == 'test':
            args.test_mode_for_aiops22()

    elif args.dataset == 'TrainTicket':
        if args.mode == 'train':
            args.train_mode_for_trainticket()
            args.create_directory()
        elif args.mode == 'compute':
            args.compute_mode_for_trainticket()
        elif args.mode == 'test':
            args.test_mode_for_trainticket()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset, _, _, groundtruths, nan_nodes = load_dataset(args=args, dates=args.dates, run_dir=args.run_dir)

    model = GraphAE(
        conv_type=args.conv_type,
        node_in_feats=args.node_in_feats,
        in_feats=args.in_feats,
        hidden_feats=args.hidden_feats,
        num_enc_layers=1,
        num_dec_layers=1,
        num_heads=args.num_heads,
        bidirected=args.bidirected
    )
    model.to(args.device)

    for argname, argvalue in vars(args).items():
        if not argname.startswith('__') and not callable(argvalue):
            logger.info('{}: {}'.format(argname, argvalue))

    if args.mode == 'train':
        model.train()
        train_graphae(dataset, model)
    elif args.mode == 'compute':
        model.load_state_dict(torch.load(args.model_file))
        model.eval()
        compute_statistics(dataset, model, nan_nodes)
    elif args.mode == 'test':
        model.load_state_dict(torch.load(args.model_file))
        model.eval()
        test_graphae(dataset, model, groundtruths, nan_nodes)


try:
    main()
except Exception as e:
    os.remove(logger.logname)
