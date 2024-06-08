import torch
import pandas as pd
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from dgl.dataloading import GraphDataLoader
from dataset_aiops22 import HetGraphDataset_AIOPS22
from dataset_trainticket import HetGraphDataset_TrainTicket
from models import *
from sklearn.neighbors import KernelDensity
import os
import json
import random

from copy import deepcopy
from utils import *
from mask import *
from rca import *
from log import logger
from tqdm import tqdm


seed = 42
th.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def load_dataset(args, dates=None, max_samples=1e9, run_dir=''):
    # load dataset according to different mode
    use_history=True
    save = False
    
    if args.mode == 'train':
        use_history = False        
        save = True
        
    if args.dataset == 'aiops22':
        dataset = HetGraphDataset_AIOPS22(
            data_dir=args.data_dir,
            dates=dates,
            cloudbeds=args.cloudbeds,
            node_feature_selector=args.nfeat_select,
            edge_reverse=args.edge_reverse, 
            used_etypes=args.used_etypes,
            failure_types=args.failure_types,
            failure_duration=args.failure_duration,
            max_samples=max_samples,
            is_mask=args.mask,
            process_extreme=args.process_extreme,
            process_miss=args.process_miss
        )
    
    elif args.dataset == 'TrainTicket':
        dataset = HetGraphDataset_TrainTicket(
            data_dir=args.data_dir,
            dates=dates,
            node_feature_selector=args.nfeat_select,
            max_samples=max_samples,
            failure_types=args.failure_types,
            failure_duration=args.failure_duration,
            is_mask=args.mask,
            process_extreme=args.process_extreme,
            process_miss=args.process_miss,
            k_sigma=args.k_sigma
        )

    stats_names = ['mean', 'median', 'q10', 'q20', 'q30', 'q40', 'q50', 'q60', 'q70', 'q80', 'q90']
    stats = {}
    node_scalers, edge_scalers = None, None
    
    # load scalers
    if use_history:
        run_dir = os.path.join(*args.model_file.split('/')[:-1], 'run-info')
        node_scalers = load_node_scalers(store_dir=run_dir, type=args.node_scale_type)
        edge_scalers = load_edge_scalers(store_dir=run_dir, type=args.edge_scale_type)
        for stats_name in stats_names:
            stats[stats_name] = load_mask_feat(store_dir=run_dir, prefix='dataset_' + stats_name)
            
    # debug
    # feats_no_scale = dataset.get_stacked_nfeat()
    # feats_statistics = dataset.get_feats_stats()
    # logger.debug('{:<10}: {}.\n {:<10}: {}.\n'.format('mean',   feats_statistics['mean']['api'][124].numpy(), 
    #                                                   'std',    feats_statistics['std']['api'][124].numpy()))
    # raise Exception
                    
    # scale
    node_scalers, edge_scalers = dataset.scale(node_scalers=node_scalers,
                                               edge_scalers=edge_scalers,
                                               node_scale_type=args.node_scale_type, 
                                               edge_scale_type=args.edge_scale_type,
                                               log=args.log_before_scale)
    
    # get feats, labels, groundtruths, nan_nodes
    feats_all = dataset.get_stacked_nfeat()
    nan_nodes = dataset.get_nan_nodes()
    labels = dataset.get_labels()
    groundtruths = dataset.get_groundtruths()
    if use_history == False:
        stats = dataset.get_feats_stats()
    nfeats_dict = deepcopy(args.nfeat_select)
    
    # mask
    if args.mask: 
        mask_operation(dataset, stats[args.mask_feat_type])
        nfeats_dict['masked_api'] = nfeats_dict['api']
        logger.info('Mask completed.')
        
    # save scalers & dataset statistics
    if save:
        save_node_scalers(node_scalers, run_dir, args.node_scale_type)
        save_edge_scalers(edge_scalers, run_dir, args.edge_scale_type)
        logger.info('Save new scalers.')
        
        for stats_name in stats_names:
            if stats_name in stats.keys():
                save_mask_feat(run_dir, stats[stats_name], 'dataset_' + stats_name)
        logger.info('Save statistics.')
        
    # debug
    # examples = {'2024-04-06 04:55:00': [124, 119, 216]}
    
    # for i, label in enumerate(labels):
    #     timestamp = label['timestamp']
    #     if timestamp in examples.keys():
    #         for api in examples[timestamp]:
    #             logger.debug('{:<10}: {}.\n {:<10}: {}.\n'.format(
    #                 'feat_no_scale', feats_no_scale['api'][i][api].numpy(), 'feat_scale', feats_all['api'][i][api].numpy()))
            
    logger.info('Load dataset completed.')
            
    return dataset, feats_all, labels, groundtruths, nan_nodes


def load_model(args, num_nodes_dict=None):
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
    

def train_model_het(args):
    """训练异构图模型。"""
    
    # 加载并划分数据
    dataset, feats_all, lbls, _, nan_nodes = load_dataset(args=args, dates=args.train_dates, max_samples=1e9, run_dir=run_dir)
    # debug nan_nodes
    # with open('debug/nan_nodes.txt', 'w') as f:
    #     for ntype, znodes in nan_nodes.items():
    #         znodes = znodes.numpy()
    #         for i, znodes_per_sample in enumerate(znodes):
    #             indices = np.where(znodes_per_sample == True)[0]
    #             if indices.shape[0] != 0:
    #                 print('timestamp: {}. ntype: {}. znodes: {}.'.format(lbls[i]['timestamp'], ntype, indices), file=f)
    
    valid_size = int(len(dataset) * args.valid_ratio)
    train_size = len(dataset) - valid_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    train_loader = GraphDataLoader(dataset=train_dataset, 
                                   batch_size=args.batch_size,
                                   drop_last=False,
                                   shuffle=True)
    if args.valid_ratio != 0:
        valid_loader = GraphDataLoader(dataset=valid_dataset,
                                       batch_size=args.batch_size,
                                       drop_last=False,
                                       shuffle=True)
    logger.info('Split dataset into training and validation.')
    
    # 定义模型 & 训练    
    model = load_model(args, num_nodes_dict=dataset.num_nodes_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    for epoch in range(args.epoch):
        # 模型训练
        model.train()
        loop = tqdm(enumerate(train_loader), total = len(train_loader))
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
                loss = loss_func_customgae(graph, graph.ndata['feat'], X_hat, args.mask, args.loss_type, args.recon_ntypes)
            l = torch.mean(loss)
            l.backward()
            train_loss += l.item()
            optimizer.step()
            
            # tensorboard
            if batch + 1 == len(train_loader): 
                writer.add_scalar('Loss', l.item(), epoch)
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
                            graph, graph.ndata['feat'], X_hat, args.mask, args.loss_type, args.recon_ntypes)
                        
                    valid_loss += loss.item()
                valid_loss /= len(valid_loader)
                writer.add_scalar('Valid Loss', valid_loss, epoch)
                
        logger.info('Epoch {}. Train Loss: {}. Validation Loss: {}.'.format(epoch, train_loss, valid_loss))
    logger.info('Training completed.')
            
    # 模型文件 & 参数配置保存
    model_filepath = os.path.join(store_dir, 'model.pth')
    torch.save(model.state_dict(), model_filepath)
    logger.info('Model filepath: {}。'.format(model_filepath))
    
    train_args_dir = os.path.join(base_dir, f'train_args')
    os.makedirs(train_args_dir, exist_ok=True)
    trainargs_filepath = os.path.join(train_args_dir, f'train_args_{cur_time}.txt')
    
    with open(trainargs_filepath, 'w') as f:
        print('seed: {}. loss: {}. \ndataset: {}. \n. training samples: {}. \n'.format(seed, train_loss, args.dataset, len(train_dataset)), file=f)
        for var, val in vars(args).items():
            print('{}: {}.'.format(var, val), file=f)
        print(model, file=f)

    # 计算RCA需要的统计量
    # cal_stat_for_rca(model, dataset, feats_all, nan_nodes, args)
    
    return model, train_loss
            

def test_model_for_rca_het(model, args):
    # 加载测试集，加载已有的scaler进行缩放
    test_dataset, feats_all, labels, groundtruths, nan_nodes = load_dataset(args=args, dates=args.test_dates)

    # 加载模型
    if model is None:
        model = load_model(args, num_nodes_dict=test_dataset.num_nodes_dict)
        
    # 预测样本的特征
    feats_preds = model_predict(test_dataset, model, args, get_attention=False)
    
    # # debug: 查看预测输出和原始输入
    # print('debug...')
    # for etype in attns.keys():
    #     attns[etype] = attns[etype].to('cpu')
    
    # # indices = [0, 1, 2, 3, 4]
    # print('{}. {}. {}.'.format(len(test_dataset.labels), len(feats_preds['api']), len(feats_all['api'])))
    # for i, lbl in enumerate(test_dataset.labels):
    #     if lbl['failure_type'] != 'cpu_contention':
    #         continue
    #     columns = args.nfeat_select['api']
    #     timestamp = test_dataset.labels[i]['timestamp']
    #     tensor_to_csv(feats_preds['api'][i], f'{timestamp}_predict_api', 'debug', columns)
    #     tensor_to_csv(feats_all['api'][i], f'{timestamp}_input_api', 'debug', columns)
        
    #     for etype in [('api', 'to', 'api'), ('pod', 'to', 'api')]:
    #         efeat = test_dataset.graphs[i].edata['feat'][etype]
    #         edges = test_dataset.graphs[i].edges(etype=etype)
    #         etensor = th.concat((edges[0].unsqueeze(1), edges[1].unsqueeze(1), efeat, attns[etype][i]), dim=1)
    #         tensor_to_csv(etensor, f'{timestamp}_input_efeat_attn_{etype}', 'debug', 
    #                       ['src', 'dst', 'count'] + [f'attn_{i}' for i in range(12)])
    
    # window = 60 * 5
    device = 'cpu'
    model = model.to(device)
    for ntype in feats_preds.keys():
        feats_preds[ntype] = feats_preds[ntype].to(device)
        
    # 异常检测：预测节点特征和正常预测节点特征的均值之差 判断 预测节点特征 是否异常
    # score_type = 'pred_and_mean'
    # score_type = 'kde'
    dist_type = 'euclidean'
    propagation_ad_rsts, data_stats, dist_stats, = compute_anomaly_score_for_all_samples(run_dir, 
                                                                                         feats_all, 
                                                                                         feats_preds, 
                                                                                         labels, 
                                                                                         args.score_type, 
                                                                                         dist_type, 
                                                                                         device, 
                                                                                         args.ad_threshold, 
                                                                                         args.window, 
                                                                                         nan_nodes)  

    # 因果：反事实计算节点累积因果效应
    causal_rsts = causal_effect(run_dir, 
                                model, 
                                test_dataset, 
                                labels, 
                                feats_all, 
                                data_stats, 
                                dist_stats, 
                                propagation_ad_rsts, 
                                device, 
                                args.causal_threshold, 
                                'sum', 
                                dist_type, 
                                args.score_type, 
                                args)
    
    # 异常检测：原始节点特征和预测节点特征之差 判断 原始节点特征是否异常
    dist_type = 'euclidean'
    score_type = 'train_and_pred'
    individual_ad_rsts, _, _ = compute_anomaly_score_for_all_samples(run_dir, 
                                                                     feats_all, 
                                                                     feats_preds, 
                                                                     labels, 
                                                                     score_type, 
                                                                     dist_type, 
                                                                     device, 
                                                                     0, 
                                                                     args.window, 
                                                                     nan_nodes)  
    
    # debug: check results
    dict_to_json(propagation_ad_rsts, os.path.join(store_dir, 'propagation_ad_rsts.json'))

    # 累积因果效应 + 节点独立异常分数
    individual_coefs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    for individual_coef in individual_coefs:
        logger.info('-----------------RCA parameters: individual_coef: {}.-----------------'.format(individual_coef))
        
        dgl_id_to_name = None
        if hasattr(test_dataset, 'dgl_id_to_name'):
            dgl_id_to_name = test_dataset.dgl_id_to_name
        
        unified_rsts = unify_results(individual_ad_rsts, causal_rsts, individual_coef, dgl_id_to_name)
        dict_to_json(unified_rsts, os.path.join(store_dir, 'rca_result_{:.1f}.json').format(individual_coef))
        
        for failure_type in args.failure_types:
            metrics = compute_rca_metrics(unified_rsts, groundtruths, failure_type)
            dict_to_json(metrics, os.path.join(store_dir, 'rca_metrics_{:.1f}_{}.json').format(individual_coef, failure_type))
            
        metrics = compute_rca_metrics(unified_rsts, groundtruths)
        dict_to_json(metrics, os.path.join(store_dir, 'rca_metrics_{:.1f}_all.json').format(individual_coef))


def compute_statistics_for_rca(model, dataset, feats_all, nan_nodes=None, args=None):
    # 加载数据
    if dataset is None:
        dataset, feats_all, labels, _, nan_nodes = load_dataset(args=args, dates=args.train_dates)
        
    # timestamps
    timestamps = []
    for label in labels:
        timestamps.append(label['timestamp'])
    
    # 加载模型
    if model is None:
        model = load_model(args, num_nodes_dict=dataset.num_nodes_dict)
    
    # 推理
    feats_preds = model_predict(dataset, model, args)
    for ntype in feats_preds.keys():
        feats_preds[ntype] = feats_preds[ntype].to('cpu')
    ntypes = ['api', 'pod']
    
    def compute_data_stats(feats, run_dir, suffix):
        # save feature statistics: mean, cov
        data_stats = {'mean': {}, 'cov': {}}
        data_stats_file = os.path.join(run_dir, 'data_stats_{}.json'.format(suffix)) 
        for ntype in ntypes:
            # (nnodes, nfeats)
            feats[ntype][nan_nodes[ntype]] = 0
            feats_mean = th.sum(feats[ntype], dim=0) / (nan_nodes[ntype] != True).sum(dim=0).view(-1, 1)
            data_stats['mean'][ntype] = feats_mean
            
            # (nnodes, nfeats, nsamples)
            swap_tensor = feats[ntype].permute(1, 2, 0)
            covs = []
            for i, t in enumerate(swap_tensor):
                t = t[:, th.nonzero(nan_nodes[ntype][i] == False)].squeeze()
                cov = th.cov(t)
                covs.append(cov)
            data_stats['cov'][ntype] = th.stack(covs)
            
        stats_to_json(data_stats, data_stats_file)
        logger.info('Save to:{}'.format(data_stats_file))
        
        return data_stats

    def compute_dist_stats(feats, feats_sub, covs, run_dir, suffix, type):
        # save dist statistics: mean, std
        dist_stats_file = os.path.join(run_dir, '{}_dist_{}.csv'.format(type, suffix))
        dfs = []
        
        for ntype in ntypes:
            # (nsamples, nnodes)
            dists = th.zeros((feats[ntype].shape[0], feats[ntype].shape[1]))

            if feats[ntype].dim() != feats_sub[ntype].dim():
                # (nnodes, nsamples, nfeats)
                swap_tensor = th.swapaxes(feats[ntype], 0, 1)
                for i, t in enumerate(swap_tensor):
                    # t: (nsamples, nfeats)
                    if type == 'mahalanobis':
                        dist = mahalanobis(t, feats_sub[ntype][i], covs[ntype][i])
                    elif type == 'euclidean':
                        dist = euclidean(t, feats_sub[ntype][i])
                    dists[:, i] = dist
            else:
                dists = euclidean(feats[ntype], feats_sub[ntype])
                
            dists = pd.DataFrame(dists.cpu().numpy())
            dists[nan_nodes[ntype].numpy()] = np.nan
            mean = dists.mean()
            std = dists.std()
            df = pd.DataFrame({'mean_per_node': mean, 'std_per_node': std, 'ntype': ntype})
            dfs.append(df)
            
            filepath = os.path.join(run_dir, f'raw_{type}_dist_{ntype}_{suffix}.csv')
            dists['timestamp'] = timestamps
            dists.to_csv(filepath, index=False)  
            logger.info('Save to:{}'.format(filepath)) 
            
        rst = pd.concat(dfs, axis=0)
        rst.to_csv(dist_stats_file, index_label='id')   
        logger.info('Save to:{}'.format(dist_stats_file)) 
     
    # 计算每个节点重构误差的mean、std
    compute_dist_stats(feats_all, feats_preds, None, run_dir, 'train_and_pred', 'euclidean')
            
    # 计算每个节点预测特征与其平均特征欧式距离的mean、std
    data_stats = compute_data_stats(feats_preds, run_dir, 'pred_and_mean')
    for dist_type in ['euclidean']:
        compute_dist_stats(feats_preds, data_stats['mean'], data_stats['cov'], run_dir, 'pred_and_mean', dist_type)
    
    # 核密度估计
    # kde(feats_preds, timestamps, nan_nodes)
    
    # debug: print records
    records = {
        # '2024-04-25 11:17:00': [(270, 24), (264, 24), (258, 24)],
        # '2024-04-25 11:18:00': [(270, 24), (264, 24), (258, 24)], 
        # '2024-04-25 11:19:00': [(270, 24), (264, 24), (258, 24)], 

        # ts-auth network
        # '2024-04-06 04:41:00': [(91, 9), (80, 9)],
        # '2024-04-06 04:42:00': [(91, 9), (80, 9)],
        # '2024-04-06 04:43:00': [(91, 9), (80, 9)],
        # '2024-04-06 04:50:00': [(91, 9), (80, 9)],
        # '2024-04-06 04:51:00': [(91, 9), (80, 9)],
        # '2024-04-06 04:52:00': [(91, 9), (80, 9)],
        
        # exception
        '2024-04-24 16:28:00': [(15, 1), (11, 1)],
        '2024-04-24 17:28:00': [(15, 1), (11, 1)],
        '2024-04-24 18:28:00': [(15, 1), (11, 1)],
        '2024-04-24 19:28:00': [(15, 1), (11, 1)],
        '2024-04-25 00:00:00': [(15, 1), (11, 1)],
        '2024-04-25 14:00:00': [(15, 1), (11, 1)],
        '2024-04-25 15:00:00': [(15, 1), (11, 1)],
        '2024-04-25 17:43:00': [(15, 1), (11, 1)],
    }
    
    for timestamp, record in records.items():
        logger.debug('---------------------timestamp: {}---------------------'.format(timestamp))
        
        for (api, pod) in record:
            index = timestamps.index(timestamp)
            api_name = dataset.dgl_id_to_name(api, 'api')
            api_feat = feats_all['api'][index][api].numpy()
            pod_feat = feats_all['pod'][index][pod].numpy()
            api_feat_pred = feats_preds['api'][index][api].numpy()
            api_feat_pred_mean = data_stats['mean']['api'][api].numpy()
            dist = euclidean(th.tensor(api_feat_pred), th.tensor(api_feat_pred_mean)).numpy()
            
            logger.debug(
                '{:<10}: {}.\n {:<10}: {}.\n {:<10}: {}.\n {:<10}: {}.\n {:<10}: {}.\n {:<10}: {}.\n {:<10}: {}.\n'
                .format('api', api, 
                        'api_name', api_name,
                        'pod_feat', pod_feat,
                        'feat', api_feat, 
                        'feat_pred', api_feat_pred, 
                        'mean', api_feat_pred_mean, 
                        'dist', dist))


def kde(feat_dict, timestamps, nan_nodes):
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
        
        with open(os.path.join(run_dir, 'kde_estimator_{}.pkl'.format(ntype)), 'wb') as f:
            pickle.dump(estimators, file=f)
            
        df = pd.DataFrame(data=score_dict)
        df[nan_nodes[ntype].numpy()] = np.nan
        df['timestamp'] = timestamps
        df.to_csv(os.path.join(run_dir, 'kde_score_{}.csv'.format(ntype)), index=False)
        
        logger.info('KDE estimation for {} completed. bandwidth: {}.'.format(ntype, bandwidth))
        
      
def test_model_for_ad(model, args):
    # 加载测试集，加载已有的scaler进行缩放
    test_dataset, feats_all, lbls, _, nan_nodes = load_dataset(args=args, dates=args.test_dates)
    num_samples = len(test_dataset)
    logger.info('Number of samples for anomaly detection: {}'.format(num_samples))
    
    # 加载模型
    if model is None:
        model = load_model(args, num_nodes_dict=test_dataset.num_nodes_dict)
    
    # 预测样本的特征
    feats_preds = model_predict(test_dataset, model, args)   
    device = 'cpu'
    model = model.to(device)
    for ntype in feats_preds.keys():
        feats_preds[ntype] = feats_preds[ntype].to(device)
    
    # 判断异常
    # 参数设置
    # window = 24 * 60
    # threshold_1 = 0
    # threshold_2 = 8
    # weight_score = True
    
    window = args.window
    threshold_1 = args.threshold_1
    threshold_2 = args.threshold_2
    weight_score = args.weight_score
    
    # 计算图中所有节点(api, pod)预测值和真实值的误差
    score_type = 'train_and_pred'
    dist_type = 'euclidean'
    individual_rsts, _, _ = compute_anomaly_score_for_all_samples(run_dir=run_dir, 
                                                        feats_all=feats_all, 
                                                        feats_preds=feats_preds, 
                                                        labels=lbls, 
                                                        score_type=score_type, 
                                                        dist_type=dist_type, 
                                                        device=device, 
                                                        threshold=threshold_1, 
                                                        window=window, 
                                                        nan_nodes=nan_nodes)  
    
    # 读取pod到api的调用时延
    def read_pod2api_duration(args, origin_id_to_dgl_id=None):
        dfs = []
        feat_dir = os.path.join(args.data_dir, 'graph', args.cloudbeds[0], 'feats')
        for date in os.listdir(feat_dir):
            if date not in args.test_dates:
                continue
            
            edge_dir = os.path.join(feat_dir, date, 'edge')
            for file in os.listdir(edge_dir):
                timestamp = file.split(sep='_')[-1].split(sep='.')[0]
                df = pd.read_csv(os.path.join(edge_dir, file))
                df['timestamp'] = pd.Timestamp(timestamp)
                dfs.append(df)
        
        result = pd.concat(dfs)
        # convert id
        if origin_id_to_dgl_id is not None:
            result['src'] = result['src'].apply(origin_id_to_dgl_id)
            result['dst'] = result['dst'].apply(origin_id_to_dgl_id)
        
        return result 
    
    def read_api_duration(args, origin_id_to_dgl_id=None):
        dfs = []
        feat_dir = os.path.join(args.data_dir, 'graph', args.cloudbeds[0], 'feats')
        for date in os.listdir(feat_dir):
            if date not in args.test_dates:
                continue
            
            api_dir = os.path.join(feat_dir, date, 'api')
            for file in os.listdir(api_dir):
                timestamp = file.split(sep='_')[-1].split(sep='.')[0]
                df = pd.read_csv(os.path.join(api_dir, file))
                df['timestamp'] = pd.Timestamp(timestamp)
                dfs.append(df)
        
        result = pd.concat(dfs)
        # convert id
        if origin_id_to_dgl_id is not None:
            result['node_id'] = result['node_id'].apply(origin_id_to_dgl_id)
        
        return result 
        
    # 为各个pod节点打分并判断是否为异常
    logger.info('Scoring each pod...')
    g = test_dataset.graphs[0]
    num_pods = test_dataset.num_nodes_dict['pod']
    y_pred = np.zeros((num_samples, num_pods))
    api_cnt = np.zeros((num_samples, num_pods))
    
    # read pod2api duration df
    if weight_score:
        duration_columns = ['mean', 'q10', 'q20', 'q30', 'q40', 'q50', 'q60', 'q70', 'q80', 'q90']
        
        pod2api_df = read_pod2api_duration(args, test_dataset.origin_id_to_dgl_id)
        pod2api_df['src'] = pod2api_df['src'].astype(int)
        pod2api_df = pod2api_df.fillna(0).set_index(['timestamp', 'src', 'dst']).sort_index()[duration_columns]
        
        api_df = read_api_duration(args, test_dataset.origin_id_to_dgl_id)
        api_df = api_df.fillna(0).set_index(['timestamp', 'node_id']).sort_index()[duration_columns]
            
    for i, result in enumerate(tqdm(individual_rsts)):
        for (api, score) in result['cand']['api']:
            if nan_nodes['api'][i][api] == False:
                pods = g.predecessors(api, etype=('pod', 'to', 'api')).tolist()
                
                # calculate pearson correlation as weight of edge from pod to api
                if weight_score:
                    # locate (pod, api, timestamp)
                    timestamp = lbls[i]['timestamp']
                    pod_df = pod2api_df.loc[(timestamp, api)]
                    api_series = api_df.loc[(timestamp, api)].squeeze().reset_index(drop=True)
                    
                    weight = {}
                    for pod in pods:
                        api_cnt[i][pod] += 1
                        try:
                            pod_series = pod_df.loc[pod].squeeze().reset_index(drop=True)
                            weight[pod] = abs(api_series.corr(pod_series))
                            if np.isnan(weight[pod]):
                                weight[pod] = 0
                        except:
                            weight[pod] = 0
                                                                    
                    for pod in pods:
                        y_pred[i][pod] += score * weight[pod] 
                else:
                    api_cnt[i][pods] += 1
                    y_pred[i][pods] += score
                
        for pod in range(num_pods):
            if api_cnt[i][pod] != 0:
                y_pred[i][pod] = y_pred[i][pod] / api_cnt[i][pod]
                
        for (pod, score) in result['cand']['pod']:
            y_pred[i][pod] += score

        for pod in range(num_pods):
            if y_pred[i][pod] > threshold_2:
                y_pred[i][pod] = 1
            else:
                y_pred[i][pod] = 0
            
    # 获取groundtruth y_true
    y_true = get_y_true(labels=lbls, num_pods=num_pods, name_to_dgl_id=test_dataset.name_to_dgl_id)
    
    # 计算AD指标: precision, recall, f1 score
    cal_metrics_for_ad(y_true=y_true, y_pred=y_pred, store_dir=store_dir, args=args, dgl_id_to_name=test_dataset.dgl_id_to_name)

    # debug: visualize
    # while True:
    #     pod_name = input('pod name: ')
    #     if pod_name == 'all':
    #         nodes_df = test_dataset.nodes_df
    #         pod_names = nodes_df[nodes_df['node_type'] == 'pod']['node_name'].tolist()
    #     else:
    #         pod_names = [pod_name]
                
    #     for pod_name in pod_names:
    #         pod_id = test_dataset.name_to_dgl_id(pod_name)
    #         metrics = ['container_cpu_usage_seconds', 'container_memory_usage_MB', 'container_fs_io_time_seconds./dev/vda1', 
    #                 'container_threads', 'container_network_transmit_MB.eth0', 'container_network_transmit_packets_dropped.eth0']
    #         fig, axs = plt.subplots(len(metrics) + 2, figsize=(100, 25))
    #         for i, metric in enumerate(metrics):
    #             metric_id = args.nfeat_select['pod'].index(metric)
    #             metric_arr = feats_all['pod'][:, pod_id, metric_id].numpy()
    #             axs[i].plot(metric_arr)
    #             axs[i].set_title('{}'.format(metric))
                
    #         axs[-1].plot(y_pred[:, pod_id], color='red')
    #         axs[-1].set_title('y_pred')
    #         axs[-2].plot(y_true[:, pod_id], color='lightgreen')
    #         axs[-2].set_title('y_true')
            
    #         xticks = np.arange(0, len(metric_arr), 60)
    #         xticklabels = [lbls[j]['timestamp'] for j in xticks]
    #         for i in range(len(metrics) + 2):
    #             axs[i].set_xticks(xticks)
    #             axs[i].set_xticklabels(xticklabels, rotation=45)
            
    #         plt.tight_layout()
    #         plt.savefig(os.path.join(store_dir, 'pod_{}_{}.png'.format(pod_id, pod_name)))
    

def model_predict(dataset, model, args, get_attention=False):
    dataloader = GraphDataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False)
        
    # if args.mask, then predict features are in masked_api
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
        feats_preds[ntype] = th.concat(feats_preds[ntype], dim=0)       
    if get_attention:
        for etype in etypes:
            attns[etype] = th.concat(attns[etype], dim=0)

    # replace api with masked_api
    if args.mask:
        feats_preds['api'] = feats_preds['masked_api']
        del feats_preds['masked_api']
        
    logger.info('Model prediction completed.')
            
    if get_attention:
        return feats_preds, attns        
    else:
        # nnodes: (nsamples, nnodes, nfeats)
        return feats_preds


def visualize(args):
    dataset, feats_all, _, _, nan_nodes = load_dataset(args=args, dates=args.train_dates)
    model = load_model(args, num_nodes_dict=dataset.num_nodes_dict)
    feats_preds = model_predict(dataset, model, args)
    ntypes = ['api', 'pod']

    for ntype in ntypes:
        array = feats_preds[ntype].to('cpu').numpy()
        feats_preds[ntype] = array.reshape((-1, array.shape[-1]))
        
        array = feats_all[ntype].numpy()
        feats_all[ntype] = array.reshape((-1, array.shape[-1]))
    
    # extract labels
    node_file = '../datasets/AIOps-2022/graph/cloudbed-1/graph_nodes.csv'
    node_df = pd.read_csv(node_file)
    labels = {}
    
    for ntype in ntypes:
        df = node_df[node_df['node_type'] == ntype].reset_index()['node_name']
        df = df.str.split('-').str[0].str.rstrip('2')
        labels[ntype] = df.to_dict() 
    
    # tsne plot
    num_samples = 400
    
    for ntype in ntypes:
        if ntype == 'api':
            tag = 'Operations'
        else:
            tag = 'Entities'
        tsne_visualization(X=feats_preds[ntype][:len(labels[ntype]) * num_samples],
                           labels=labels[ntype],
                           title=f't-SNE Visualization of Reconstructed Features of {tag} in Services',
                           path=f'images/feat_pred_{ntype}.png')
        tsne_visualization(X=feats_all[ntype][:len(labels[ntype]) * num_samples],
                           labels=labels[ntype],
                           title=f't-SNE Visualization of Features of {tag} in Services',
                           path=f'images/feat_{ntype}.png')
     

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='', help='模型名称。')
    parser.add_argument('--model_file',default='',type=str,help='模型文件路径。测试时使用。')
    parser.add_argument('--mode', type=str, default='train', help='运行模式。train, rca。')
    
    # 常规训练超参数
    parser.add_argument('--epoch', type=int, default=80, help='training epoch')
    parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')
    parser.add_argument('--valid_ratio', default=0.2, type=float, help='验证集占训练集比例。')
    
    # 数据集参数: 使用的时间段、故障类型、特征、预处理方式。
    parser.add_argument('--dataset', default='aiops22', type=str, help='数据集名称。')
    parser.add_argument('--data_dir', default='', type=str, help='数据集所在目录。')
    parser.add_argument('--failure_types', nargs='+', help='所使用数据的故障类型。正常数据则为""。')
    parser.add_argument('--failure_duration', type=int, help='故障持续时间。单位：分钟。')
    parser.add_argument('--train_dates', nargs='+', help='用于训练的数据的日期。')
    parser.add_argument('--test_dates', nargs='+', help='用于测试的数据的日期。')
    parser.add_argument('--cloudbeds', nargs='*', type=str, default=['cloudbed-1'], help='cloudbeds。')
    parser.add_argument('--used_etypes', nargs='+', help='所使用边的类型。')
    parser.add_argument('--nfeat_select', type=str,help='不同类型节点特征选择。json字符串。')
    parser.add_argument('--node_scale_type', type=str, default='nodewise', help='数据集的节点特征缩放方式。nodewise/global。')
    parser.add_argument('--edge_scale_type', type=str, default='edgewise', help='数据集的边特征缩放方式。edgewise/global。')
    parser.add_argument('--edge_reverse', action='store_true', help='是否需要将边反向。')
    parser.add_argument('--log_before_scale', action='store_true', help='是否使用取对数。')
    parser.add_argument('--process_miss', default='interpolate', type=str, help='使用什么方式填补缺失值。interpolate/zero。')
    parser.add_argument('--process_extreme', action='store_true', help='是否处理数据中的极端值。')
    parser.add_argument('--k_sigma', type=float, default=3, help='k值。用于检测极端值。')

    # CustomGAE模型参数
    parser.add_argument('--conv_type', type=str, default='DotGAT', help='点积注意力或加法注意力。DotGAT/myGAT。')
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--feat_drop', type=float, default=0)
    parser.add_argument('--attn_drop', type=float, default=0)
    parser.add_argument('--num_enc_layers', type=int, default=1)
    parser.add_argument('--num_dec_layers', type=int, default=1)
    parser.add_argument('--decoder_type', type=str, help='decoder类型。mlp or gnn。', default='mlp')
    parser.add_argument('--num_etypes', type=int, default=6)
    parser.add_argument('--edge_feats', type=int, default=1, help='边特征数。')
    parser.add_argument('--etype_feats', type=int, default=5, help='不同类型边嵌入的维度数。')
    parser.add_argument('--residual', action='store_true', help='是否使用残差连接。')
    parser.add_argument('--hidden_dim', type=int, default=10, help='隐藏层维数。')
    parser.add_argument('--proj_feats', type=int, help='投影的特征空间维度。')
    parser.add_argument('--mask', action='store_true', help='是否对于图节点使用mask。')
    parser.add_argument('--mask_feat_type', type=str, help='mask节点特征。zero, median, mean。', default='mean')
    parser.add_argument('--loss_type', type=str, help='loss函数类型。', default='mse')
    parser.add_argument('--use_edge_feats', action='store_true', help='是否使用边上的特征。')
    parser.add_argument('--use_etype_feats', action='store_true', help='是否使用边的类型嵌入。')
    parser.add_argument('--embedding', action='store_true', help='是否为节点设置embedding以计算注意力分数。')
    parser.add_argument('--recon_ntypes', nargs='+', help='训练时重建的节点类型。api, pod。')
    
    # rca 参数
    parser.add_argument('--window', type=int, default=5 * 60, help='异常分数规范化窗口。')
    parser.add_argument('--score_type', type=str, default='pred_and_mean', help='异常分数类型。')
    parser.add_argument('--causal_threshold', type=float, default=0.3, help='阈值: 异常分数降低比例。')
    parser.add_argument('--ad_threshold', type=float, default=1.0, help='阈值: 异常分数。')
    
    # ad 参数
    parser.add_argument('--threshold_1', type=int, default=0)
    parser.add_argument('--threshold_2', type=int, default=8)
    parser.add_argument('--weight_score', action='store_true')
    
    args = parser.parse_args()

    logger.info('mode: {}'.format(args.mode))
    
    # 将包含字典的字符串解析为字典对象
    try:
        args.nfeat_select = json.loads(args.nfeat_select)
    except json.JSONDecodeError as e:
        logger.error(f"无法解析输入的字符串为字典: {e}")
        exit(1)
        
    # 记录参数
    for var, value in vars(args).items():
        logger.info('{}: {}.'.format(var, value))
        
    # 存储目录
    global cur_time, base_dir, store_dir, run_dir
    cur_time = get_cur_time()
    base_dir = f'./model-store/{args.dataset}/{args.model}/'
    if args.mode == 'train':
        store_dir = os.path.join(base_dir, f'train_{cur_time}')
        run_dir = os.path.join(store_dir, 'run-info')
    else:
        store_dir = os.path.join(base_dir, f'{args.mode}_{cur_time}')
        model_dir = args.model_file[:-9]
        run_dir = os.path.join(model_dir, 'run-info')
    os.makedirs(store_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    # tensorboard
    writer = SummaryWriter(comment='_{}_{}'.format(args.dataset, args.model))
    final_loss = -1
    model = None
    if args.mode == 'train':
        logger.info('Training...')
        model, final_loss = train_model_het(args)
    elif args.mode == 'rca':
        logger.info('RCA Testing...')
        test_model_for_rca_het(model, args)
    elif args.mode == 'rca_prepare':
        compute_statistics_for_rca(None, None, None, None, args)
    elif args.mode == 'ad':
        logger.info('AD Testing...')
        test_model_for_ad(model, args)
    else:
        visualize(args)

    writer.flush()
    writer.close()