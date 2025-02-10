import pickle
import os
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
from typing import List, Dict, Callable, Optional, Union

import pandas as pd
import numpy as np
import networkx as nx
import torch as th
from dgl import DGLGraph

from mask import load_mask_feat, replace_node_feats
from utils import euclidean, mahalanobis, load_stats
from log import Logger

logger = Logger(__name__)


def get_anomaly_subgraph(g: DGLGraph, entry_apis: Optional[List[int]] = None):
    """Get Networkx subgraph of apis and their parent apis and pods."""
    # if apis is None, include all apis
    if entry_apis is None:
        entry_apis = g.nodes('api').tolist()

    # map dgl id to networkx id
    nodes = {ntype: {} for ntype in g.ntypes}
    nx_g = nx.DiGraph()
    for v in entry_apis:
        v_ = nx_g.number_of_nodes()
        nx_g.add_node(v_, id=v, ntype='api')
        nodes['api'][v] = v_

    for v in entry_apis:
        v_ = nodes['api'][v]
        # (api, to, api), (pod, to, api)
        for etype in g.canonical_etypes:
            if etype[2] != 'api':
                continue
            if etype[1] == 'self':
                continue
            us = g.predecessors(v, etype).tolist()
            for u in us:
                if u not in nodes[etype[0]]:
                    u_ = nx_g.number_of_nodes()
                    nx_g.add_node(u_, id=u, ntype=etype[0])
                    nodes[etype[0]][u] = u_
                else:
                    u_ = nodes[etype[0]][u]
                nx_g.add_edge(u_, v_)

    return nx_g


def find_propagation_path_from_u(nx_g: nx.DiGraph, u, dgl_id_to_name: Optional[Callable] = None):
    """Calculate nodes' cumulative causal effect. dfs."""

    if 'path' in nx_g.nodes[u]:
        return nx_g.nodes[u]['path_length'], nx_g.nodes[u]['path']

    u_path_length = 0
    u_id = nx_g.nodes[u]['id']
    u_ntype = nx_g.nodes[u]['ntype']

    if dgl_id_to_name is not None:
        u_name = dgl_id_to_name(u_id, u_ntype)
    else:
        u_name = u_id

    u_path = 'api: {}'.format(u_name)

    for v in nx_g.successors(u):
        v_path_length, v_path = find_propagation_path_from_u(nx_g, v, dgl_id_to_name)

        # identify there is real causal relationship between u and v
        if nx_g.edges[u, v]['effect'] > 0:
            u_v_path_length = v_path_length + nx_g.edges[u, v]['effect']
        else:
            u_v_path_length = 0

        if u_v_path_length > u_path_length:
            u_path_length = u_v_path_length
            # u_path = [(nx_g.nodes[u]['ntype'], nx_g.nodes[u]['id'])] + v_path
            u_path = '{}: {} --{:.2f}--> {}'.format(u_ntype, u_name, nx_g.edges[u, v]['effect'], v_path)

    nx_g.nodes[u]['path_length'] = u_path_length
    nx_g.nodes[u]['path'] = u_path

    return u_path_length, u_path


def sum_causal_effect_along_path(nx_g: nx.DiGraph, dgl_id_to_name: Optional[Callable] = None):
    for u in nx_g.nodes:
        # root node
        if nx_g.in_degree(u) == 0:
            find_propagation_path_from_u(nx_g, u, dgl_id_to_name)

    # add (ntype, id, score) to result
    candidates = []
    for u in nx_g.nodes:
        candidates.append({
            'ntype': nx_g.nodes[u]['ntype'],
            'id': nx_g.nodes[u]['id'],
            'score': nx_g.nodes[u]['path_length'],
            'path': nx_g.nodes[u]['path']
        })

    candidates.sort(key=lambda x: x['score'], reverse=True)

    return candidates


def max_causal_effect_among_neighbors(nx_g: nx.DiGraph):
    candidates = []
    for u in nx_g.nodes:
        effect = 0
        for v, e_data in nx_g.adj[u].items():
            effect = max(effect, e_data['effect'])

        candidates.append({'ntype': nx_g.nodes[u]['ntype'], 'id': nx_g.nodes[u]['id'], 'score': effect})

    candidates.sort(key=lambda x: x['score'], reverse=True)

    return candidates


def counterfactual_rca(
    run_dir,
    model,
    dataset,
    labels,
    stacked_nfeat,
    data_stats,
    dist_stats,
    ad_rsts,
    device,
    causal_threshold=0.3,
    postprocess='sum',
    dist_type='euclidean',
    score_type='',
    args=None
):

    logger.info('Calculating causal effect...')

    # load estimators
    if score_type == 'kde':
        ntypes = ['api', 'pod']
        estimators = {}
        for ntype in ntypes:
            with open(os.path.join(run_dir, 'kde_estimator_{}.pkl'.format(ntype)), 'rb') as f:
                estimators[ntype] = pickle.load(f)
        logger.debug('Load KDE estimators. {}. {}.'.format(len(estimators['api']), len(estimators['pod'])))

    # copy anomaly detection results
    rsts = [{k: v for k, v in ad_rst.items() if k != 'cand'} for ad_rst in ad_rsts]
    logger.debug('Copy information.')

    # get mask feat or normal feat
    mask_feats = load_mask_feat(run_dir, prefix=f'dataset_{args.mask_feat_type}')
    mask_feats = {ntype: feat.to(device) for ntype, feat in mask_feats.items()}
    if args.mask:
        mask_feats['masked_api'] = mask_feats['api']
    logger.debug('Load mask features for intervening in counterfactual.')

    # dgl id to name
    dgl_id_to_name = None
    if hasattr(dataset, 'dgl_id_to_name'):
        dgl_id_to_name = dataset.dgl_id_to_name

    # compute causal effect througth counterfactual
    cnt = 0
    for i, label in enumerate(tqdm(labels)):
        g = dataset.graphs[i].to(device)
        # debug
        causal_weight_records = []

        # get anomaly predicted nodes and build a networkx subgraph
        try:
            anomalous_nodes, scores = zip(*ad_rsts[cnt]['cand']['api'])
        except Exception as e:
            rsts[cnt]['cand'] = []
            cnt += 1
            continue
        feats = {ntype: stacked_nfeat[ntype][i] for ntype in stacked_nfeat.keys()}
        node2score = {v: scores[i] for i, v in enumerate(anomalous_nodes)}
        nx_g = get_anomaly_subgraph(g, anomalous_nodes)

        # traverse the subgraph and calculate the causal effect of each edge
        for u_, v_ in nx_g.edges:
            u_ntype = nx_g.nodes[u_]['ntype']
            v_ntype = nx_g.nodes[v_]['ntype']
            u = nx_g.nodes[u_]['id']
            v = nx_g.nodes[v_]['id']

            # intervene: set feats of u to normal & mask v
            if args.mask:
                ids = {u_ntype: [u], 'masked_' + v_ntype: [v]}
            else:
                ids = {u_ntype: [u]}
            feats_counter = replace_node_feats(feats, mask_feats, ids)

            # counterfactual inference
            feats_pred = model(g, feats_counter)
            if args.mask:
                feats_pred['api'] = feats_pred['masked_api']
                del feats_pred['masked_api']

            # causal effect of (u, v)
            feat_1 = feats_pred[v_ntype][v].detach()
            if score_type == 'kde':
                estimator = [estimators[v_ntype][v]]
                score_counter = compute_anomaly_score_by_kde(
                    v_ntype, feat_1.reshape((1, -1)), estimator, label['timestamp'], dist_stats
                )[0][v]
            else:
                feat_2 = data_stats['mean'][v_ntype][v].detach()
                score_counter = compute_anomaly_score(
                    v_ntype, feat_1, feat_2, label['timestamp'], dist_stats, dist_type,
                    data_stats['cov_inv'][v_ntype][v], args.dist_direction
                )[0][v]
            score = node2score[v]

            # identify real causal effect
            score_diff = score - score_counter
            if score_diff < score * causal_threshold:
                nx_g.edges[u_, v_]['effect'] = 0
            else:
                nx_g.edges[u_, v_]['effect'] = score_diff

            if args.debug and dgl_id_to_name is not None:
                causal_weight_records.append({
                    'u': u,
                    'u_name': dgl_id_to_name(u, u_ntype),
                    'v': v,
                    'v_name': dgl_id_to_name(v, v_ntype),
                    'score_diff': score_diff,
                    'score': score,
                    'score_counter': score_counter
                })

        # postprocess: adjust causal effect score
        if postprocess == 'sum':
            candidates = sum_causal_effect_along_path(nx_g, dgl_id_to_name)
        elif postprocess == 'max':
            candidates = max_causal_effect_among_neighbors(nx_g)

        rsts[cnt]['cand'] = candidates
        cnt += 1

        # debug
        if args.debug:
            debug_dir = get_debug_dir(run_dir, label)
            df = pd.DataFrame.from_records(causal_weight_records)
            df.to_csv(os.path.join(debug_dir, 'causal_weight.csv'), index=False)

    return rsts


# ----------------------------------------------------------------------------------------------------------------------


def unify_results(
    individual_ad_rsts, causal_rsts, individual_coef, dgl_id_to_name, num_cands=10, agg=False, agg_duration=2
):
    """Unify results of causal effect score and individual anomaly score."""
    if individual_coef == 1:
        unified_rsts = deepcopy(individual_ad_rsts)
        for i, unified_rst in enumerate(unified_rsts):
            cands = []
            for ntype in unified_rst['cand'].keys():
                for item in unified_rst['cand'][ntype]:
                    cands.append({'ntype': ntype, 'id': item[0], 'score': item[1]})
            unified_rst['cand'] = cands
    else:
        unified_rsts = deepcopy(causal_rsts)
        for i, individual_rst in enumerate(individual_ad_rsts):
            unified_rst = unified_rsts[i]
            for ntype, array in individual_rst['cand'].items():
                for id, score in array:
                    find = False
                    for cand in unified_rst['cand']:
                        if cand['ntype'] == ntype and cand['id'] == id:
                            cand['score'] = (1 - individual_coef) * cand['score'] + individual_coef * score
                            find = True
                            break
                    if find == False:
                        unified_rst['cand'].append({'ntype': ntype, 'id': id, 'score': individual_coef * score})

    # drop candidate whose score is 0
    for rst in unified_rsts:
        new_cands = []
        for i, cand in enumerate(rst['cand']):
            if cand['score'] != 0:
                new_cands.append(cand)
        rst['cand'] = new_cands

    # aggregate_results
    if agg:
        unified_rsts = aggregate_results(unified_rsts, duration=agg_duration)

    # sort candidates
    for rst in unified_rsts:
        rst['cand'].sort(key=lambda x: x['score'], reverse=True)

    # reserve first num_cands candidates
    if num_cands != -1:
        for rst in unified_rsts:
            rst['cand'] = rst['cand'][:num_cands]

    # set name
    if dgl_id_to_name is not None:
        for rst in unified_rsts:
            for cand in rst['cand']:
                cand['name'] = dgl_id_to_name(dgl_id=cand['id'], ntype=cand['ntype'])

    return unified_rsts


def aggregate_results(results, duration=2):
    results = sorted(results, key=lambda x: x["timestamp"])
    results_out = []
    i, j = 0, 0

    while i < len(results):
        time_i = results[i]["timestamp"]
        candidates = []
        candidates.extend(results[i]["cand"])

        for j in range(i + 1, i + duration + 1):
            if j == len(results):
                break
            time_j = results[j]["timestamp"]
            if pd.Timestamp(time_i) + pd.Timedelta(minutes=j - i) != pd.Timestamp(time_j):
                break
            candidates.extend(results[j]["cand"])

        cand_dict = defaultdict(list)
        for candidate in candidates:
            cand_dict[(candidate["ntype"], candidate["id"])].append(candidate)

        candidates = []
        for (ntype, id), candidates_cur in cand_dict.items():
            paths = []
            for cand in candidates_cur:
                if "path" in cand:
                    paths.append(cand["path"])
            candidates.append({
                "ntype": ntype,
                "id": id,
                "score": np.mean([cand["score"] for cand in candidates_cur]),
                "path": max(paths, key=len)
            })
        results[i]["cand"] = candidates
        results[i]["index"] = i
        results_out.append(results[i])

        i = j

    return results_out


def compute_rca_metrics(rsts_all, groundtruths_all, failure_type=None, agg=False):
    # filter by failure_type
    if failure_type is None:
        rsts = rsts_all
        groundtruths = groundtruths_all
    else:
        rsts = []
        groundtruths = []
        for i, rst in enumerate(rsts_all):
            if rst['failure_type'] == failure_type:
                rsts.append(rst)
                groundtruths.append(groundtruths_all[i])

    # aggregate groundtruths
    if agg:
        groundtruths = []
        for rst in rsts:
            groundtruths.append(groundtruths_all[rst['index']])

    # calculate top-acc & average rank
    max_rank = 5
    top_acc = {i: 0 for i in range(1, max_rank + 1)}
    avg_rank = 0
    skipped = 0

    for idx, rst in enumerate(rsts):
        groundtruth = groundtruths[idx]
        is_find = False

        if 'cand' not in rst.keys():
            skipped += 1
            continue

        for i, candidate in enumerate(rst['cand']):
            if (candidate['ntype'], candidate['id']) in groundtruth:
                avg_rank += i + 1
                for j in range(i + 1, max_rank + 1):
                    top_acc[j] += 1
                is_find = True
                break

        if is_find == False:
            avg_rank += 10

    for i in top_acc.keys():
        top_acc[i] /= len(rsts)
    avg_rank /= len(rsts)

    # print results
    logger.info(f'-----------------RCA metrics: {failure_type}-----------------')
    logger.info('n: {}. skipped: {}.'.format(len(rsts), skipped))
    logger.info('avg_rank: {}. '.format(avg_rank))
    for i, acc in top_acc.items():
        logger.info('top {} acc: {}. '.format(i, acc))

    metrics = {'n': len(rsts), 'skipped': skipped}
    for i, acc in top_acc.items():
        metrics['top-{}'.format(i)] = acc
    metrics[f'avg@{max_rank}'] = np.mean(list(top_acc.values()))
    metrics['avg_rank'] = avg_rank

    avg_at_max_rank = metrics[f'avg@{max_rank}']
    logger.info(f'avg@{max_rank}: {avg_at_max_rank}')

    return metrics


# ----------------------------------------------------------------------------------------------------------------------


def compute_anomaly_score(
    ntype,
    feat_1,
    feat_2,
    timestamp,
    dist_stats,
    dist_type='euclidean',
    cov_inv=None,
    direction='both',
    output_func=None
):
    # (num_nodes)
    if dist_type == 'euclidean':
        error = euclidean(feat_1, feat_2, direction).numpy()
    elif dist_type == 'mahalanobis':
        error = mahalanobis(feat_1, feat_2, cov_inv, direction).numpy()

    timestamp = pd.Timestamp(timestamp, unit='s')
    index = dist_stats[ntype]['mean'].index
    timestamp = index[index <= timestamp][-1]

    mean = dist_stats[ntype]['mean'].loc[timestamp].to_numpy()
    std = dist_stats[ntype]['std'].loc[timestamp].to_numpy()

    # test: only consider error larger than mean
    diff = error - mean
    diff[diff < 0] = 0
    score = np.abs(diff) / std

    if output_func is not None:
        score = output_func(th.Tensor(score)).numpy()

    return score, mean, std, error


def compute_anomaly_score_by_kde(ntype, feat, estimators, timestamp, score_statistics):
    feat_numpy = feat.numpy()
    raw_score = []
    for j, estimator in enumerate(estimators):
        raw_score.append(estimator.score_samples(feat_numpy[j].reshape(1, -1)))
    raw_score = np.concatenate(raw_score)

    timestamp = pd.Timestamp(timestamp, unit='s')
    index = score_statistics[ntype]['mean'].index
    timestamp = index[index <= timestamp][-1]

    mean = score_statistics[ntype]['mean'].loc[timestamp].to_numpy()
    std = score_statistics[ntype]['std'].loc[timestamp].to_numpy()

    # test: only consider score smaller than mean
    diff = raw_score - mean
    diff[diff > 0] = 0
    score = np.abs(diff) / std

    return score, mean, std, raw_score


def compute_distance_statistics_for_windows(run_dir, dist_type='euclidean', score_type='', window=60 * 2):
    dist_stats = {}

    for ntype in ['api', 'pod']:
        dist_df = pd.read_csv(os.path.join(run_dir, f'raw_{dist_type}_dist_{ntype}_{score_type}.csv'))
        timestamps = dist_df['timestamp']
        dist_df = dist_df.drop(columns='timestamp')
        roll = dist_df.rolling(window=window, min_periods=1)
        mean_df = roll.mean()[window - 1:].interpolate(limit_direction='both')
        std_df = roll.std()[window - 1:].interpolate(limit_direction='both') + 1e-5
        mean_df.index = pd.to_datetime(timestamps[window - 1:])
        std_df.index = pd.to_datetime(timestamps[window - 1:])
        dist_stats[ntype] = {'mean': mean_df, 'std': std_df}

    return dist_stats


def compute_distance_statistics_for_windows_by_kde(run_dir, window=60 * 2):
    dist_stats = {}

    for ntype in ['api', 'pod']:
        dist_df = pd.read_csv(os.path.join(run_dir, f'kde_score_{ntype}.csv'))
        timestamps = dist_df['timestamp']
        dist_df = dist_df.drop(columns='timestamp')
        roll = dist_df.rolling(window=window, min_periods=1)
        mean_df = roll.mean()[window - 1:].interpolate(limit_direction='both')
        std_df = roll.std()[window - 1:].interpolate(limit_direction='both') + 1e-5
        mean_df.index = pd.to_datetime(timestamps[window - 1:])
        std_df.index = pd.to_datetime(timestamps[window - 1:])
        dist_stats[ntype] = {'mean': mean_df, 'std': std_df}

    return dist_stats


def compute_anomaly_score_for_all_samples(
    run_dir,
    stacked_nfeat,
    feats_preds,
    labels,
    score_type,
    dist_type,
    device,
    threshold=2,
    window=60 * 2,
    nan_nodes=None,
    direction='both',
    debug=False
):
    """"""
    logger.info('Anomaly detection for graph nodes...')

    # def my_output_func(x: th.Tensor) -> th.Tensor:
    #     return th.sigmoid(x - threshold)

    # 样本特征、预测特征的平均特征、协方差矩阵
    data_stats = None
    if score_type == 'pred_and_mean':
        data_stats = load_stats(os.path.join(run_dir, 'data_stats_{}.json'.format(score_type)), device)

    # 分数 = 绝对值((距离 - 均值) / 标准差)
    ntypes = ['api', 'pod']
    scores = defaultdict(list)
    means = defaultdict(list)
    stds = defaultdict(list)
    dists = defaultdict(list)

    if score_type == 'kde':
        dist_stats = compute_distance_statistics_for_windows_by_kde(run_dir=run_dir, window=window)
    else:
        dist_stats = compute_distance_statistics_for_windows(
            run_dir=run_dir, dist_type=dist_type, score_type=score_type, window=window
        )

    for ntype in ntypes:
        if score_type == 'kde':
            with open(os.path.join(run_dir, 'kde_estimator_{}.pkl'.format(ntype)), 'rb') as f:
                estimators = pickle.load(f)

        for i, label in enumerate(tqdm(labels)):

            if score_type == 'train_and_pred':
                feat_1 = stacked_nfeat[ntype][i]
                feat_2 = feats_preds[ntype][i]
                score, mean, std, dist = compute_anomaly_score(ntype, feat_1, feat_2, label['timestamp'], dist_stats)
                # debug
                if debug:
                    print_feat(
                        run_dir, label, score, mean, std, dist, ntype, '1', {
                            'feat_pred': feat_1,
                            'feat_mean': feat_2,
                            'feat': stacked_nfeat[ntype][i]
                        }
                    )

            elif score_type == 'pred_and_mean':
                feat_1 = feats_preds[ntype][i]
                feat_2 = data_stats['mean'][ntype]
                score, mean, std, dist = compute_anomaly_score(
                    ntype, feat_1, feat_2, label['timestamp'], dist_stats, dist_type, data_stats['cov_inv'][ntype],
                    direction
                )
                # debug
                if debug:
                    print_feat(
                        run_dir, label, score, mean, std, dist, ntype, '2', {
                            'feat_pred': feat_1,
                            'feat_mean': feat_2,
                            'feat': stacked_nfeat[ntype][i]
                        }
                    )

            elif score_type == 'kde':
                feat = feats_preds[ntype][i]
                score, mean, std, dist = compute_anomaly_score_by_kde(
                    ntype, feat, estimators, label['timestamp'], dist_stats
                )

            scores[ntype].append(score)
            means[ntype].append(mean)
            stds[ntype].append(std)
            dists[ntype].append(dist)

        # 修正分数: 输入特征为0的节点非异常
        scores[ntype] = np.stack(scores[ntype])
        scores[ntype][nan_nodes[ntype]] = 0

    # 阈值threshold过滤 & 排序
    rsts = []
    for i, label in enumerate(tqdm(labels)):
        cand = {}
        for ntype in ntypes:
            indices = np.where(scores[ntype][i] > threshold)[0]
            cand_list = list(zip(indices.tolist(), scores[ntype][i][indices].tolist()))
            cand[ntype] = sorted(cand_list, key=lambda x: x[1], reverse=True)

        # aiops22: timestamp, cloudbed, level, cmdb_id, failure_type
        # trainticket: timestamp, cmdb_id, failure_type
        rst = deepcopy(label)
        rst['cand'] = cand
        rsts.append(rst)

    return rsts, data_stats, dist_stats


from datetime import datetime


def get_debug_dir(runtime_dir, label):
    failure_type, timestamp, cmdb_id = label['failure_type'], label['timestamp'], label['cmdb_id']
    time_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

    debug_dir = os.path.join(runtime_dir, '..', 'debug', f'{failure_type}_{time_str}_{cmdb_id}')
    os.makedirs(debug_dir, exist_ok=True)

    return debug_dir


def print_feat(
    runtime_dir: str, label: Dict, score: np.ndarray, mean: np.ndarray, std: np.ndarray, dist: np.ndarray, ntype: str,
    suffix: str, feat_dict: Dict[str, th.Tensor]
):
    debug_dir = get_debug_dir(runtime_dir, label)

    train_args_file_path = os.path.join(runtime_dir, '..', 'train_args.txt')
    with open(train_args_file_path, 'r') as file:
        for line in file:
            if line.startswith('data_dir: '):
                data_dir = line.split(' ')[-1].rstrip()[:-1]

    node_df = pd.read_csv(os.path.join(data_dir, 'graph_nodes.csv'))
    node_df = node_df[node_df['node_type'] == ntype]

    dfs = []
    for feat_name, feat in feat_dict.items():
        df = pd.DataFrame(feat.numpy()).reset_index()
        df['feat_name'] = feat_name
        df['score'] = score
        df['mean'] = mean
        df['std'] = std
        df['dist'] = dist
        df = pd.merge(left=node_df, right=df, left_on='node_id', right_on='index')
        df = df.drop(columns=['index'])
        dfs.append(df)
    df = pd.concat(dfs).sort_values(by=['node_id', 'feat_name'])
    df.to_csv(os.path.join(debug_dir, f'{ntype}_{suffix}.csv'), index=False)
