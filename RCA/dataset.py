import torch as th
import numpy as np
from copy import deepcopy as dp
from dgl.data import DGLDataset
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from utils import *
from mask import *
from log import logger
from collections import defaultdict


class myDGLDataset(DGLDataset):
    def __init__(
        self,
        dataset_name=None,
        process_miss='interpolate',
        process_extreme=False,
        k=3,
    ):
        # data process
        self.process_miss = process_miss
        self.process_extreme = process_extreme
        self.k = k  # k-sigma to process extreme values
        self.scaler_instance = MinMaxScaler(feature_range=(0, 1))
        # data container
        self.graphs = []
        self.labels = []
        self.groundtruths = []
        
        super().__init__(name=dataset_name)
        
    def process(self):
        pass
        
    def scale(self, 
              node_scalers=None, 
              edge_scalers=None, 
              attr='feat', 
              node_scale_type='nodewise',  
              edge_scale_type='edgewise', 
              log=True):
        
        # 先全部取对数
        if log:
            bias = 1
            for g in self.graphs:
                for ntype in g.ntypes:
                    g.ndata[attr] = {ntype: th.log(g.ndata[attr][ntype] + bias)}
            
        self.node_scale(
            node_scalers=node_scalers, 
            attr='feat', 
            node_scale_type=node_scale_type
        )        
        self.edge_scale(
            edge_scalers=edge_scalers,
            attr='feat',
            edge_scale_type=edge_scale_type
        )
        
        return self.node_scalers, self.edge_scalers
    
    def node_scale(self, node_scalers=None, attr='feat', node_scale_type='nodewise'):
        
        feats = defaultdict(list)
        self.node_scalers = defaultdict(list)
        
        if node_scale_type == 'nodewise':
            for g in self.graphs:
                for ntype, feat in g.ndata[attr].items():
                    feats[ntype].append(feat.numpy())
                    
            for ntype in self.graphs[0].ntypes:
                # stk: (n_nodes, n_samples, n_feats)
                stk = np.stack(feats[ntype], axis=1)
                for i, slice in enumerate(stk):
                    if node_scalers is None:
                        scaler = dp(self.scaler_instance)
                        stk[i] = scaler.fit_transform(slice)
                        self.node_scalers[ntype].append(scaler)
                    else:
                        scaler = node_scalers[ntype][i]
                        stk[i] = scaler.transform(slice)
                        self.node_scalers[ntype].append(scaler)
                
                stk = stk.swapaxes(0, 1)
                for i, g in enumerate(self.graphs):
                    g.ndata[attr] = {ntype: th.tensor(stk[i])} 
                    
        elif node_scale_type == 'global':
            for g in self.graphs:
                for ntype, feat in g.ndata[attr].items():
                    feats[ntype].append(feat.numpy())
                    
            if node_scalers is None:
                for ntype, feat in feats.items():
                    feat = np.concatenate(feat)
                    scaler = dp(self.scaler_instance)
                    scaler = scaler.fit(feat)                
                    self.node_scalers[ntype] = scaler
            else:
                self.node_scalers = node_scalers
                    
            for g in self.graphs:
                scaled_feats = {}
                for ntype, feat in g.ndata[attr].items():
                    scaled_feats[ntype] = th.tensor(
                        self.node_scalers[ntype].transform(feat.numpy())
                    )
                g.ndata[attr] = scaled_feats
    
    def edge_scale(self, edge_scalers=None, attr='feat', edge_scale_type='edgewise'):
        
        feats = {etype: [] for etype in self.graphs[0].canonical_etypes}
        self.edge_scalers = {etype: [] for etype in self.graphs[0].canonical_etypes}
        
        for g in self.graphs:
            for etype, feat in g.edata[attr].items():
                feats[etype].append(feat.numpy())
        
        if edge_scale_type == 'edgewise':
            for etype in self.graphs[0].canonical_etypes:
                # stk: (n_nodes, n_samples, n_feats)
                stk = np.stack(feats[etype], axis=1)
                for i, slice in enumerate(stk):
                    if edge_scalers is None or etype not in edge_scalers.keys():
                        scaler = dp(self.scaler_instance)
                        stk[i] = scaler.fit_transform(slice)
                    else:
                        scaler = edge_scalers[etype][i]
                        stk[i] = scaler.transform(slice)
                    self.edge_scalers[etype].append(scaler)
                    
                stk = stk.swapaxes(0, 1)
                for i, g in enumerate(self.graphs):
                    g.edata[attr] = {etype: th.tensor(stk[i])} 
                    
        elif edge_scale_type == 'global':
            if edge_scalers is None:
                for etype, feat in feats.items():
                    feat = np.concatenate(feat)
                    scaler = dp(self.scaler_instance)
                    scaler.fit(feat)
                    self.edge_scalers[etype] = scaler
            else:
                self.edge_scalers = edge_scalers
                
            for g in self.graphs:
                scaled_feats = {}
                for etype, feat in g.edata[attr].items():
                    scaled_feats[etype] = th.tensor(
                        self.edge_scalers[etype].transform(feat.numpy())
                    )
                g.edata[attr] = scaled_feats
        
    def inverse_scale(self, feats, type='nodewise', log=True):
        """将特征反向缩放回原来的尺度。
        Args:
            feats (dict): 单个样本的特征。
        Returns:
            feats_inv (dict): 反缩放后的特征。
        """
        feats_inv = {}
        
        if type == 'nodewise':
            for ntype, feat in feats.items():
                feat = feat.cpu()
                feats_inv[ntype] = th.tensor([])
                for i in range(feat.shape[0]):
                    inv_t = th.tensor(
                        self.node_scalers[ntype][i].inverse_transform(
                            th.unsqueeze(feat[i], 0).detach().numpy()
                        )
                    )
                    feats_inv[ntype] = th.cat((feats_inv[ntype], inv_t))
        
        elif type == 'global':
            for ntype, feat in feats.items():
                feat = feat.cpu().detach().numpy()
                feats_inv[ntype] = th.from_numpy(                    
                    self.node_scalers[ntype].inverse_transform(feat)
                )
        
        # e指数
        if log:
            bias = 1
            for ntype, feat in feats_inv.items():
                feats_inv[ntype] = th.exp(feat) - bias
        
        return feats_inv
        
    def get_feats_stats(self):
        if hasattr(self, 'stats') == False:
            feats = defaultdict(list)
            
            for graph in self.graphs:
                for ntype in graph.ntypes:
                    # (num_nodes, num_feats)
                    feat = graph.ndata['feat'][ntype]
                    feats[ntype].append(feat)
                    
            rst = defaultdict(dict)
            
            for ntype, feat in feats.items():
                # (num_samples, num_nodes, num_feats)
                stk = th.stack(feat)
                rst['mean'].update({ntype: th.mean(stk, dim=0)})
                rst['std'].update({ntype: th.std(stk, dim=0)})
                rst['median'].update({ntype: th.median(stk, dim=0)[0]})
                stk = th.swapaxes(stk, 0, 1)
                covs = []
                for slice in stk:
                    covs.append(th.cov(slice))
                rst['cov'].update({ntype: th.stack(covs)}) 
            self.stats = rst
            
        return self.stats

    def get_labels(self):
        return self.labels
    
    def get_groundtruths(self):
        return self.groundtruths

    def get_stacked_nfeat(self):
        feats = {}
        for ntype in self.graphs[0].ntypes:
            feats[ntype] = []
            
            for g in self.graphs:
                feats[ntype].append(g.ndata['feat'][ntype])
            
            # (num_samples, num_nodes, num_feats)
            feats[ntype] = th.stack(feats[ntype], dim=0)
        
        return feats
    
    def get_stacked_efeat(self):
        feats = {}
        for etype in self.graphs[0].canonical_etypes:
            feats[etype] = []
            
            for g in self.graphs:
                feats[etype].append(g.edata['feat'][etype])
                
            # (num_samples, num_edges, num_feats)
            feats[etype] = th.stack(feats[etype], dim=0)
        
        return feats

    def get_nan_nodes(self):
        if hasattr(self, 'nan_nodes'):
            return self.nan_nodes
        
        self.nan_nodes = {}
        for ntype in self.graphs[0].ntypes:
            nans = []
            
            for g in self.graphs:
                # (num_nodes, num_feats)
                data = g.ndata['feat'][ntype]
                nan = th.where(th.isnan(data).all(dim=1), th.tensor(True), th.tensor(False))
                g.ndata['nan'] = {ntype: nan.reshape(-1, 1)}
                nans.append(nan)
            # (num_samples, num_nodes)
            self.nan_nodes[ntype] = th.stack(nans, dim=0)
        
        return self.nan_nodes
    
    def get_nan_edges(self):
        if hasattr(self, 'nan_edges'):
            return self.nan_edges
        
        self.nan_edges = {}
        for etype in self.graphs[0].canonical_etypes:
            nans = []
            
            for g in self.graphs:
                # （num_edges, num_feats)
                data = g.edata['feat'][etype]
                nan = th.where(th.isnan(data).all(dim=1), th.tensor(True), th.tensor(False))
                g.edata['nan'] = {etype: nan.reshape(-1, 1)}
                nans.append(nan)
            self.nan_edges[etype] = th.stack(nans, dim=0)
        
        return self.nan_edges
                
    def process_missing_values(self):
        kind = self.process_miss
        logger.info('Processing missing values using {}...'.format(kind))
        def interp_numpy(arr):
            isnan = np.isnan(arr)
            if np.all(isnan):
                return np.nan_to_num(arr, nan=0)
            arr[isnan] = np.interp(np.flatnonzero(isnan), np.flatnonzero(~isnan), arr[~isnan])
            
            return arr
        
        # padding ndata 
        logger.info('Padding ndata...')
        feats = self.get_stacked_nfeat()

        for ntype, data in feats.items():
            data = data.numpy()
            if kind == 'interpolate':
                for i in range(data.shape[1]):
                    for j in range(data.shape[2]):
                        data[:, i, j] = interp_numpy(data[:, i, j])                
            elif kind == 'zero':
                data = np.nan_to_num(data, nan=0)            
            for i, feat in enumerate(data):
                self.graphs[i].ndata['feat'] = {ntype: th.tensor(feat).float()}
                
        # padding edata
        logger.info('Padding edata...')
        feats = self.get_stacked_efeat()
                        
        for etype, data in feats.items():
            data = data.numpy()
            if kind == 'interpolate':
                for i in range(data.shape[1]):
                    for j in range(data.shape[2]):
                        data[:, i, j] = interp_numpy(data[:, i, j])                
            elif kind == 'zero':
                data = np.nan_to_num(data, nan=0)                
            for i, feat in enumerate(data):
                self.graphs[i].edata['feat'] = {etype: th.tensor(feat).float()}

    def process_extreme_values(self):
        logger.info('Processing extreme values using nan...')
        k = self.k
        
        def detect_and_replace(data, k):
            # (num_samples, num_nodes, num_feats)
            data = data.numpy()
            mean = np.nanmean(data, axis=0)
            std = np.nanstd(data, axis=0)
            
            for i in range(data.shape[1]):
                for j in range(data.shape[2]):
                    indices = data[:, i, j] > mean[i, j] + std[i, j] * k
                    data[indices, i, j] = np.nan
                    
                    indices = data[:, i, j] < mean[i, j] - std[i, j] * k
                    data[indices, i, j] = np.nan
            
            return data
        
        logger.info('Padding ndata...')
        feats = self.get_stacked_nfeat()

        for ntype, data in feats.items():
            data = detect_and_replace(data, k)
                        
            for i, feat in enumerate(data):
                self.graphs[i].ndata['feat'] = {ntype: th.tensor(feat).float()}   

        logger.info('Padding edata...')
        feats = self.get_stacked_efeat()
        
        for etype, data in feats.items():
            data = detect_and_replace(data, k)          
            
            for i, feat in enumerate(data):
                self.graphs[i].edata['feat'] = {etype: th.tensor(feat).float()}   
                
    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)
    