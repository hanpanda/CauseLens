import pickle
import os
import json
import random
import argparse
from tqdm import tqdm
from copy import deepcopy
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split
from dgl.dataloading import GraphDataLoader
from sklearn.neighbors import KernelDensity

from dataset import myDGLDataset
from dataset_aiops_2022 import DatasetAIOPS2022
from dataset_trainticket_2024 import DatasetTrainTicket2024
from models import CustomGAE, loss_func_customgae
from utils import (
    stats_to_json, dict_to_json, load_node_scalers, load_edge_scalers, save_node_scalers, save_edge_scalers, euclidean,
    mahalanobis, tsne_visualization, get_cur_time
)
from mask import mask_operation, load_mask_feat, save_mask_feat
from rca import compute_anomaly_score_for_all_samples, counterfactual_rca, unify_results, compute_rca_metrics
from ad import cal_metrics_for_ad, get_y_true
from log import Logger

logger = Logger(__name__)


def load_dataset(args: argparse.Namespace, dates: list = None, max_samples: int = 1e9) -> myDGLDataset:
    if args.dataset == 'aiops_2022':
        dataset_class = DatasetAIOPS2022
    elif args.dataset == 'trainticket_2024':
        dataset_class = DatasetTrainTicket2024

    dataset = dataset_class(
        data_dir=args.data_dir,
        dates=dates,
        node_feature_selector=args.nfeat_select,
        add_self_loop=args.add_self_loop,
        edge_reverse=args.edge_reverse,
        max_samples=max_samples,
        failure_types=args.failure_types,
        failure_duration=args.failure_duration,
        is_mask=args.mask,
        process_extreme=args.process_extreme,
        process_miss=args.process_miss,
        k_sigma=args.k_sigma,
        use_split_info=args.use_split_info
    )

    logger.info('Load dataset completed.')

    return dataset


def scale_dataset(args: argparse.Namespace, dataset: myDGLDataset, runtime_dir: str = '') -> myDGLDataset:
    use_history = True
    save = False
    if args.mode == 'train':
        use_history = False
        save = True

    logger.info(f'Scaling dataset using history: {use_history}.')

    # load scalers
    node_scalers, edge_scalers = None, None
    if use_history:
        node_scalers = load_node_scalers(store_dir=runtime_dir, type=args.node_scale_type)
        edge_scalers = load_edge_scalers(store_dir=runtime_dir, type=args.edge_scale_type)

    # scale
    node_scalers, edge_scalers = dataset.scale(
        node_scalers=node_scalers,
        edge_scalers=edge_scalers,
        node_scale_type=args.node_scale_type,
        edge_scale_type=args.edge_scale_type,
        log=args.log_before_scale
    )

    if save:
        save_node_scalers(node_scalers, runtime_dir, args.node_scale_type)
        save_edge_scalers(edge_scalers, runtime_dir, args.edge_scale_type)
        logger.info('Save new scalers.')

    return dataset


def mask_dataset(args: argparse.Namespace, dataset: myDGLDataset, runtime_dir: str = '') -> myDGLDataset:
    use_history = True
    save = False
    if args.mode == 'train':
        use_history = False
        save = True

    stats_names = ['mean', 'median', 'q10', 'q20', 'q30', 'q40', 'q50', 'q60', 'q70', 'q80', 'q90']
    stats = {}

    # 加载历史特征 或 数据集统计量
    if use_history:
        for stats_name in stats_names:
            stats[stats_name] = load_mask_feat(store_dir=runtime_dir, prefix='dataset_' + stats_name)
    else:
        stats = dataset.get_feats_stats()

    # mask
    nfeats_dict = deepcopy(args.nfeat_select)
    if args.mask:
        mask_operation(dataset, stats[args.mask_feat_type])
        nfeats_dict['masked_api'] = nfeats_dict['api']
        logger.info('Mask completed.')

    # 保存mask特征
    if save:
        for stats_name in stats_names:
            if stats_name in stats.keys():
                save_mask_feat(runtime_dir, stats[stats_name], 'dataset_' + stats_name)
        logger.info('Save statistics.')

    return dataset


def load_model(args: argparse.Namespace, num_nodes_dict: Optional[Dict[str, int]] = None):
    # 加载模型
    node_in_feats = {}
    for ntype, feats in args.nfeat_select.items():
        node_in_feats[ntype] = len(feats)
    if args.mask:
        node_in_feats['masked_api'] = node_in_feats['api']
    model = CustomGAE(
        in_feats=args.proj_feats,
        hidden_feats=args.hidden_dim,
        conv_type=args.conv_type,
        num_heads=args.num_heads,
        feat_drop=args.feat_drop,
        attn_drop=args.attn_drop,
        num_enc_layers=args.num_enc_layers,
        num_dec_layers=args.num_dec_layers,
        num_etypes=args.num_etypes,
        etype_feats=args.etype_feats,
        edge_feats=args.edge_feats,
        residual=args.residual,
        node_in_feats=node_in_feats,
        decoder_type=args.decoder_type,
        use_etype_feats=args.use_etype_feats,
        use_edge_feats=args.use_edge_feats,
        num_nodes_dict=num_nodes_dict,
        embedding=args.embedding,
    )
    if args.mode != 'train':
        model.load_state_dict(torch.load(args.model_file))
        model.eval()
    model.to(args.device)
    logger.info('Load model completed.')

    return model


def train_model(args: argparse.Namespace, model_dir: str, runtime_dir: str):
    """训练异构图模型。"""

    # 加载并划分数据
    dataset = load_dataset(args, args.train_dates, 1e9)
    dataset = scale_dataset(args, dataset, runtime_dir)
    dataset = mask_dataset(args, dataset, runtime_dir)

    valid_size = int(len(dataset) * args.valid_ratio)
    train_size = len(dataset) - valid_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    train_loader = GraphDataLoader(dataset=train_dataset, batch_size=args.batch_size, drop_last=False, shuffle=True)
    if args.valid_ratio != 0:
        valid_loader = GraphDataLoader(dataset=valid_dataset, batch_size=args.batch_size, drop_last=False, shuffle=True)
    logger.info('Split dataset into training and validation.')

    # 定义模型 & 训练
    model = load_model(args, num_nodes_dict=dataset.num_nodes_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epoch):
        # 模型训练
        model.train()
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        train_loss = 0
        for batch, (graph, _) in loop:
            optimizer.zero_grad()
            feats = graph.ndata['feat']
            graph = graph.to(args.device)
            if isinstance(feats, dict):
                for ntype, feat in feats.items():
                    feats[ntype] = feat.to(args.device)
            else:
                feats = feats.to(args.device)
            X_hat = model(graph, feats)

            if args.model == 'CustomGAE':
                loss = loss_func_customgae(
                    graph, graph.ndata['feat'], X_hat, args.mask, args.loss_type, args.recon_ntypes
                )
            mean_loss = torch.mean(loss)
            mean_loss.backward()
            train_loss += mean_loss.item()
            optimizer.step()

        train_loss /= len(train_loader)

        # 验证集评估模型性能
        valid_loss = 0
        if args.valid_ratio != 0:
            model.eval()
            with torch.no_grad():
                valid_loss = 0
                for batch, (graph, _) in enumerate(valid_loader):
                    feats = graph.ndata['feat']
                    graph = graph.to(args.device)
                    if isinstance(feats, dict):
                        for ntype, feat in feats.items():
                            feats[ntype] = feat.to(args.device)
                    else:
                        feats = feats.to(args.device)

                    X_hat = model(graph, feats)
                    if args.model == 'CustomGAE':
                        loss = loss_func_customgae(
                            graph, graph.ndata['feat'], X_hat, args.mask, args.loss_type, args.recon_ntypes
                        )

                    valid_loss += loss.item()
                valid_loss /= len(valid_loader)

        logger.info('Epoch {}. Train Loss: {}. Validation Loss: {}.'.format(epoch, train_loss, valid_loss))
    logger.info('Training completed.')

    # 模型文件 & 参数配置保存
    model_filepath = os.path.join(model_dir, 'model.pth')
    torch.save(model.state_dict(), model_filepath)
    logger.info('Model filepath: {}。'.format(model_filepath))

    trainargs_filepath = os.path.join(model_dir, f'train_args.txt')
    with open(trainargs_filepath, 'w') as f:
        print(
            'loss: {}. \ndataset: {}. \n. training samples: {}. \n'.format(
                train_loss, args.dataset, len(train_dataset)
            ),
            file=f
        )
        for var, val in vars(args).items():
            print('{}: {}.'.format(var, val), file=f)
        print(model, file=f)

    # 计算RCA需要的统计量
    # cal_stat_for_rca(model, dataset, stacked_feat, nan_nodes, args)

    return model, train_loss


def test_model_for_rca(args: argparse.Namespace, model: CustomGAE, runtime_dir: str):
    rca_dir = os.path.join(os.path.dirname(args.model_file), 'rca')
    os.makedirs(rca_dir, exist_ok=True)

    # 加载测试集，加载已有的scaler进行缩放
    test_dataset = load_dataset(args, args.test_dates, 1e9)
    test_dataset = scale_dataset(args, test_dataset, runtime_dir)
    stacked_nfeat = test_dataset.get_stacked_nfeat()
    labels = test_dataset.get_labels()
    groundtruths = test_dataset.get_groundtruths()
    nan_nodes = test_dataset.get_nan_nodes()
    test_dataset = mask_dataset(args, test_dataset, runtime_dir)

    # 加载模型
    if model is None:
        model = load_model(args, num_nodes_dict=test_dataset.num_nodes_dict)

    # 预测样本的特征
    feats_preds = model_inference(args, test_dataset, model, get_attention=False)

    device = 'cpu'
    model = model.to(device)
    for ntype in feats_preds.keys():
        feats_preds[ntype] = feats_preds[ntype].to(device)

    # 异常检测：预测节点特征和正常预测节点特征的均值之差 判断 预测节点特征 是否异常
    # score_type = 'pred_and_mean'
    # score_type = 'kde'
    # dist_type = 'euclidean'
    # dist_type = 'mahalanobis'
    propagation_ad_rsts, data_stats, dist_stats, = compute_anomaly_score_for_all_samples(
        runtime_dir, stacked_nfeat, feats_preds, labels, args.score_type, args.dist_type_for_counterfactual, device,
        args.ad_threshold, args.window, nan_nodes, args.dist_direction, args.debug
    )

    # 因果rca：反事实计算节点累积因果效应
    causal_rca_results = counterfactual_rca(
        runtime_dir, model, test_dataset, labels, stacked_nfeat, data_stats, dist_stats, propagation_ad_rsts, device,
        args.causal_threshold, 'sum', args.dist_type_for_counterfactual, args.score_type, args
    )

    # 重建误差rca：原始节点特征和预测节点特征之差 判断 原始节点特征是否异常
    dist_type = 'euclidean'
    score_type = 'train_and_pred'
    reconstruction_rca_results, _, _ = compute_anomaly_score_for_all_samples(
        runtime_dir, stacked_nfeat, feats_preds, labels, score_type, dist_type, device, 0, args.window, nan_nodes,
        'both', args.debug
    )

    # debug: check results
    dict_to_json(propagation_ad_rsts, os.path.join(rca_dir, 'propagation_ad_rsts.json'))

    # 累积因果效应 + α * 节点独立异常分数
    if args.alpha_sensity_expr:
        alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    else:
        alphas = [args.alpha]

    for alpha in alphas:
        logger.info('-----------------RCA parameters: alpha: {}.-----------------'.format(alpha))

        rca_record = defaultdict(dict)

        dgl_id_to_name = None
        if hasattr(test_dataset, 'dgl_id_to_name'):
            dgl_id_to_name = test_dataset.dgl_id_to_name

        unified_rsts = unify_results(reconstruction_rca_results, causal_rca_results, alpha, dgl_id_to_name)

        for failure_type in args.failure_types:
            metrics = compute_rca_metrics(unified_rsts, groundtruths, failure_type)
            rca_record['metric'][failure_type] = metrics

        metrics = compute_rca_metrics(unified_rsts, groundtruths)
        rca_record['metric']['all'] = metrics
        rca_record['detail'] = unified_rsts
        dict_to_json(rca_record, os.path.join(rca_dir, 'rca_alpha_{:.1f}.json').format(alpha))


def compute_statistics_for_rca(args: Optional[argparse.Namespace], runtime_dir: str = ''):
    # 加载数据
    dataset = load_dataset(args, args.train_dates, 1e9)
    dataset = scale_dataset(args, dataset, runtime_dir)
    stacked_nfeat = dataset.get_stacked_nfeat()
    labels = dataset.get_labels()
    nan_nodes = dataset.get_nan_nodes()
    dataset = mask_dataset(args, dataset, runtime_dir)

    # timestamps
    timestamps = []
    for label in labels:
        timestamps.append(label['timestamp'])

    # 加载模型
    model = load_model(args, num_nodes_dict=dataset.num_nodes_dict)

    # 推理
    feats_preds = model_inference(args, dataset, model)
    for ntype in feats_preds.keys():
        feats_preds[ntype] = feats_preds[ntype].to('cpu')
    ntypes = ['api', 'pod']

    def compute_data_statistics(feats, runtime_dir, suffix):
        # save feature statistics: mean, cov
        data_stats = {'mean': {}, 'cov_inv': {}}
        data_stats_file = os.path.join(runtime_dir, 'data_stats_{}.json'.format(suffix))
        for ntype in ntypes:
            # (nnodes, nfeats)
            feats[ntype][nan_nodes[ntype]] = 0
            feats_mean = torch.sum(feats[ntype], dim=0) / (nan_nodes[ntype] != True).sum(dim=0).view(-1, 1)
            data_stats['mean'][ntype] = feats_mean

            # (nnodes, nfeats, nsamples)
            swap_tensor = feats[ntype].permute(1, 2, 0)
            # (nfeats, nnodes)
            nan_nodes_permuted = nan_nodes[ntype].permute(1, 0)
            cov_invs = []
            for i, t in enumerate(swap_tensor):
                t = t[:, torch.nonzero(nan_nodes_permuted[i] == False)].squeeze()
                cov = torch.cov(t)
                cov_inv = torch.pinverse(cov)
                cov_invs.append(cov_inv)
            # (nnodes, nfeats, nfeats)
            data_stats['cov_inv'][ntype] = torch.stack(cov_invs)

        stats_to_json(data_stats, data_stats_file)
        logger.info('Save to:{}'.format(data_stats_file))

        return data_stats

    def compute_dist_statistics(feats, feats_sub, cov_inv, runtime_dir, suffix, dist_type):
        # save dist statistics: mean, std
        dist_stats_file = os.path.join(runtime_dir, '{}_dist_{}.csv'.format(dist_type, suffix))
        dfs = []

        for ntype in ntypes:
            # (nsamples, nnodes)
            dists = torch.zeros((feats[ntype].shape[0], feats[ntype].shape[1]))

            if feats[ntype].dim() != feats_sub[ntype].dim():
                # (nnodes, nsamples, nfeats)
                swap_tensor = torch.swapaxes(feats[ntype], 0, 1)
                for i, t in enumerate(swap_tensor):
                    # t: (nsamples, nfeats)
                    if dist_type == 'mahalanobis':
                        dist = []
                        for sample_feat in t:
                            dist.append(
                                mahalanobis(sample_feat, feats_sub[ntype][i], cov_inv[ntype][i],
                                            args.dist_direction).item()
                            )
                        dist = torch.tensor(dist)
                    elif dist_type == 'euclidean':
                        dist = euclidean(t, feats_sub[ntype][i], args.dist_direction)
                    dists[:, i] = dist
            else:
                if dist_type == 'mahalanobis':
                    raise Exception('未实现的逻辑')
                else:
                    dists = euclidean(feats[ntype], feats_sub[ntype], args.dist_direction)

            dists = pd.DataFrame(dists.cpu().numpy())
            dists[nan_nodes[ntype].numpy()] = np.nan
            mean = dists.mean()
            std = dists.std()
            df = pd.DataFrame({'mean_per_node': mean, 'std_per_node': std, 'ntype': ntype})
            dfs.append(df)

            filepath = os.path.join(runtime_dir, f'raw_{dist_type}_dist_{ntype}_{suffix}.csv')
            dists['timestamp'] = timestamps
            dists.to_csv(filepath, index=False)
            logger.info('Save to:{}'.format(filepath))

        rst = pd.concat(dfs, axis=0)
        rst.to_csv(dist_stats_file, index_label='id')
        logger.info('Save to:{}'.format(dist_stats_file))

    # 计算每个节点重构误差的mean、std
    compute_dist_statistics(stacked_nfeat, feats_preds, None, runtime_dir, 'train_and_pred', 'euclidean')

    # 计算节点预测特征的统计量
    data_stats = compute_data_statistics(feats_preds, runtime_dir, 'pred_and_mean')

    # 计算每个节点预测特征与其平均特征欧、马氏距离的mean、std
    for dist_type in ['euclidean', 'mahalanobis']:
        compute_dist_statistics(
            feats_preds, data_stats['mean'], data_stats['cov_inv'], runtime_dir, 'pred_and_mean', dist_type
        )

    # 核密度估计
    # kde(feats_preds, timestamps, nan_nodes)


def kde(feat_dict, timestamps, nan_nodes, runtime_dir):
    bandwidth = 'silverman'

    for ntype, feat in feat_dict.items():
        # (num_samples, num_nodes, num_feats)
        feat = feat.numpy()
        estimators = []
        score_dict = {}

        for i in tqdm(range(feat.shape[1])):
            node_feat = feat[:, i, :]
            estimator = KernelDensity(bandwidth=bandwidth, kernel='gaussian', metric='euclidean')
            estimator.fit(node_feat)
            estimators.append(estimator)
            score_dict[i] = estimator.score_samples(node_feat)

        with open(os.path.join(runtime_dir, 'kde_estimator_{}.pkl'.format(ntype)), 'wb') as f:
            pickle.dump(estimators, file=f)

        df = pd.DataFrame(data=score_dict)
        df[nan_nodes[ntype].numpy()] = np.nan
        df['timestamp'] = timestamps
        df.to_csv(os.path.join(runtime_dir, 'kde_score_{}.csv'.format(ntype)), index=False)

        logger.info('KDE estimation for {} completed. bandwidth: {}.'.format(ntype, bandwidth))


def model_inference(args: argparse.Namespace, dataset: myDGLDataset, model: CustomGAE, get_attention: bool = False):
    dataloader = GraphDataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False)

    ntypes = [ntype for ntype in dataset.graphs[0].ntypes]
    etypes = [etype for etype in dataset.graphs[0].canonical_etypes]
    feats_preds = {ntype: [] for ntype in ntypes}
    attns = {etype: [] for etype in etypes}

    with torch.no_grad():
        for (g, label) in dataloader:
            g = g.to(args.device)
            feats = g.ndata['feat']
            if get_attention:
                feats_pred, attn = model(g, feats, get_attention)
                for etype in etypes:
                    attns[etype].append(attn[etype].view(g.batch_size, -1, attn[etype].shape[1]))
            else:
                feats_pred = model(g, feats)
            for ntype in ntypes:
                feats_preds[ntype].append(feats_pred[ntype].view(g.batch_size, -1, feats_pred[ntype].shape[1]))

    for ntype in ntypes:
        feats_preds[ntype] = torch.concat(feats_preds[ntype], dim=0)
    if get_attention:
        for etype in etypes:
            attns[etype] = torch.concat(attns[etype], dim=0)

    # if args.mask, then predict features are in masked_api, then replace api with masked_api
    if args.mask:
        feats_preds['api'] = feats_preds['masked_api']
        del feats_preds['masked_api']

    logger.info('Model prediction completed.')

    if get_attention:
        return feats_preds, attns
    else:
        # nnodes: (nsamples, nnodes, nfeats)
        return feats_preds
