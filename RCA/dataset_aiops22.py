import torch as th
import numpy as np
import pandas as pd
import os
import dgl
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from dataset import myDGLDataset
from utils import *
from mask import *
from log import logger
from tqdm import tqdm


class HetGraphDataset_AIOPS22(myDGLDataset):

    def __init__(
        self,
        data_dir='../datasets/AIOps-2022/',
        cloudbeds=[],
        dates=[],
        node_feature_selector={},
        edge_feature_selector={},
        edge_reverse=False,
        add_self_loop=False,
        used_etypes=('api2api', 'api2pod', 'pod2node'),
        timestamps=None,     # debug
        max_samples=1e9,
        failure_types = [],
        failure_duration=5,
        is_mask=False,
        process_miss='interpolate',
        process_extreme=False,
    ):
        # data info
        self.data_dir = data_dir
        self.dates = dates
        self.cloudbeds = cloudbeds
        self.max_samples = max_samples
        # graph structure
        self.ntypes = list(node_feature_selector.keys())
        self.node_feature_selector = node_feature_selector
        self.edge_feature_selector = edge_feature_selector
        self.edge_reverse = edge_reverse
        self.add_self_loop = add_self_loop
        self.use_etypes = used_etypes
        # failure
        self.failures = {}
        self.failure_types = failure_types
        self.failure_duration = failure_duration
        # mask
        self.is_mask = is_mask
        
        super().__init__(dataset_name='aiops22',
                         process_extreme=process_extreme,
                         process_miss=process_miss)
        
    def origin_id_to_dgl_id(self, origin_id):
        for l, r in self.origin_id_range.values():
            if origin_id >= l and origin_id <= r:
                return origin_id - l
        
    def origin_id_to_ntype(self, origin_id):
        for ntype, (l, r) in self.origin_id_range.items():
            if origin_id >= l and origin_id <= r:
                return ntype
        
    def dgl_id_to_origin_id(self, dgl_id, ntype):
        return dgl_id + self.origin_id_range[ntype][0]
    
    def dgl_id_to_name(self, dgl_id, ntype):
        origin_id = self.dgl_id_to_origin_id(dgl_id, ntype)
        node_name = self.nodes_df[self.nodes_df['node_id'] == origin_id]['node_name'].values[0]
        return node_name
       
    def name_to_dgl_id(self, name):
        origin_id = self.nodes_df[self.nodes_df['node_name'] == name]['node_id'].values[0]
        dgl_id = self.origin_id_to_dgl_id(origin_id)
        return dgl_id
        
    def read_edges(self, filepath, num_nodes_dict):
        # 读取边
        edges_df = pd.read_csv(filepath)
        edges_df = edges_df.sort_values(by=['src', 'dst'])
        edges_df['src'] = edges_df['src'].apply(self.origin_id_to_dgl_id)
        edges_df['dst'] = edges_df['dst'].apply(self.origin_id_to_dgl_id)
        data_dict = {}
        
        if 'api2api' in self.use_etypes:
            edges = edges_df[edges_df['edge_type'] == 'api2api']
            data_dict[('api', 'to', 'api')] = (
                th.tensor(edges['src'].to_numpy()), 
                th.tensor(edges['dst'].to_numpy()) 
            )
        
        if 'api2pod' in self.use_etypes:
            edges = edges_df[edges_df['edge_type'] == 'api2pod']
            data_dict[('api', 'to', 'pod')] = (
                th.tensor(edges['src'].to_numpy()), 
                th.tensor(edges['dst'].to_numpy()) 
            )
        
        if 'pod2node' in self.use_etypes:
            edges = edges_df[edges_df['edge_type'] == 'pod2node']
            data_dict[('pod', 'to', 'node')] = (
                th.tensor(edges['src'].to_numpy()), 
                th.tensor(edges['dst'].to_numpy()) 
            )
        
        # 边反向
        if self.edge_reverse:
            new_data_dict = {}
            for etype, value in data_dict.items():
                new_data_dict[(etype[2], etype[1], etype[0])] = (
                    value[1], value[0]
                )
            data_dict = new_data_dict
        
        if self.add_self_loop:
        # 节点全部添加自环
            for ntype, num_nodes in num_nodes_dict.items():
                nodes = th.arange(num_nodes)
                data_dict[(ntype, 'self', ntype)] = (nodes, nodes)                    
        else:
        # 节点入度为0则添加自环
            # 入度不为0的所有节点集合
            nodes_dict = {
                ntype: set() for ntype in num_nodes_dict.keys()
            }
            for etype, value in data_dict.items():
                nodes_dict[etype[2]] |= set(value[1].tolist())
            for ntype, num_nodes in num_nodes_dict.items():
                nodes = th.arange(num_nodes)
                mask = th.isin(nodes, th.tensor(list(nodes_dict[ntype])))
                nodes = nodes[~mask]
                if nodes.shape[0] != 0:
                    data_dict[(ntype, 'self', ntype)] = (nodes, nodes)
                    
        return data_dict

    def read_edge_feats(self, filepath, edge_dict):
        # 读取边特征
        edge_feats = {}
        edge_feats_df = pd.read_csv(filepath)
        
        if self.edge_reverse:
            edge_feats_df = edge_feats_df.rename(
                columns={'src': 'dst', 'dst': 'src'}
            )
                                        
        edge_feats_df['src_type'] = edge_feats_df['src'].apply(
            self.origin_id_to_ntype
        )
        edge_feats_df['dst_type'] = edge_feats_df['dst'].apply(
            self.origin_id_to_ntype
        )
        edge_feats_df['src'] = edge_feats_df['src'].apply(
            self.origin_id_to_dgl_id
        )
        edge_feats_df['dst'] = edge_feats_df['dst'].apply(
            self.origin_id_to_dgl_id
        )
        
        edge_feats_df = edge_feats_df.set_index(
            ['src_type', 'dst_type', 'src', 'dst']
        )
                        
        for etype, (src, dst) in edge_dict.items():
            src = src.tolist()
            dst = dst.tolist()
            src_type = [etype[0] for _ in range(len(src))]
            dst_type = [etype[2] for _ in range(len(dst))]
            idx = pd.MultiIndex.from_arrays(
                [src_type, dst_type, src, dst], 
                names=['src_type', 'dst_type', 'src', 'dst']
            )
            df = edge_feats_df.reindex(index=idx)
            edge_feats[etype] = th.from_numpy(df['count'].to_numpy()).unsqueeze(dim=1).float()
            
        return edge_feats
            
    def read_node_feats(self, api_filepath, pod_filepath, node_filepath):
        path = {'api': api_filepath, 'pod': pod_filepath, 'node': node_filepath}
        feats = {}
        for ntype in self.ntypes:
            df = pd.read_csv(path[ntype])
            # df = df.fillna(0)
            feats[ntype] = th.tensor(df[self.node_feature_selector[ntype]].values).float()
            
        return feats
                    
    def read_labels(self, cloudbed):
        failures = {}
        
        delta = pd.Timedelta(minutes=1)     
        for date in self.dates:
            timestamp = pd.to_datetime(date)
            loop = [timestamp + delta * i for i in range(60 * 24)]
            for timestamp in loop:
                failures[str(timestamp)] = None
                
        for date in self.dates:
            failures_file = self.data_dir + 'groundtruth_csv/groundtruth-k8s-{}-{}.csv' \
                .format(cloudbed[-1], date)
            if os.path.exists(failures_file):
                failures_df = pd.read_csv(failures_file)
                # 向上取整到分钟
                failures_df = failures_df.set_index('timestamp')
                for _ in range(self.failure_duration):
                    failures_df.index = pd.to_datetime(failures_df.index).floor('T')
                    failures_df.index += pd.Timedelta(minutes=1)
                    failures_df.index = failures_df.index.astype(str)
                    failures_dict = failures_df.to_dict(orient='index')
                    failures.update(failures_dict)
        
        # {<cloudbed>: {<timestamp>: {<failure_infos>}}}
        self.failures[cloudbed] = failures
        
    def get_label(self, target, cloudbed):
        target = str(target)
        label = {
            'level': '',
            'cmdb_id': '',
            'failure_type': '',
            'cloudbed': cloudbed,
            'timestamp': target,
        }
        groundtruth = ''
                    
        if self.failures[cloudbed][target] != None:
            groundtruth = self.failures[cloudbed][target]['groundtruth']
            for key in label.keys():
                if label[key] == '':
                    try:
                        label[key] = self.failures[cloudbed][target][key]
                    except:
                        pass
                    
        return label, groundtruth
    
    def label_to_groundtruth(self, cloudbed, nodes_df):
        for failure in self.failures[cloudbed].values():
            if failure == None:
                continue
            # root cause pods & apis
            groundtruth = []
            
            # add pod
            if failure['level'] == 'service':
                service = failure['cmdb_id']
                pods = nodes_df[(nodes_df['node_type'] == 'pod') &
                                (nodes_df['node_name'].str.startswith(service))]['node_id'].tolist()
            elif failure['level'] == 'pod':
                service = failure['cmdb_id'].split('-')[0].rstrip('2')
                pods = nodes_df[nodes_df['node_name'] == failure['cmdb_id']]['node_id'].tolist()
            elif failure['level'] == 'node':
                failure['groundtruth'] = groundtruth
                continue
            groundtruth += [('pod', self.origin_id_to_dgl_id(pod)) for pod in pods]
            
            # add api
            apis = nodes_df[(nodes_df['node_type'] == 'api') & \
                            (nodes_df['node_name'].str.startswith(service))]['node_id'].tolist()

            if failure['failure_type'] in ['k8s容器网络延迟', 'k8s容器网络资源包损坏', 'k8s容器网络丢包']:
                extra = nodes_df[(nodes_df['node_type'] == 'api') & \
                                 (~nodes_df['node_name'].str.startswith(service)) & \
                                 (nodes_df['node_name'].str.contains(service, case=False))]['node_id'].tolist()
                apis += extra
            groundtruth += [('api', self.origin_id_to_dgl_id(api)) for api in apis]
            
            failure['groundtruth'] = groundtruth
            
    def process(self):
        self.origin_id_range = {}
        for cloudbed in self.cloudbeds:
            graph_dir = self.data_dir + 'graph/{}/'.format(cloudbed)
            # 读取标签
            self.read_labels(cloudbed)
            # 读取节点
            self.nodes_df = pd.read_csv(graph_dir + 'graph_nodes.csv')
            num_nodes_dict = self.nodes_df['node_type'].value_counts().to_dict()
            self.num_nodes_dict = num_nodes_dict
            if self.origin_id_range == {}:
                self.origin_id_range['api'] = (0, num_nodes_dict['api'] - 1)
                self.origin_id_range['pod'] = (self.origin_id_range['api'][1] + 1, self.origin_id_range['api'][1] +  num_nodes_dict['pod'])
                self.origin_id_range['node'] = (self.origin_id_range['pod'][1] + 1, self.origin_id_range['pod'][1] +  num_nodes_dict['node'])
            # 删除不使用的节点类型
            for ntype in list(self.num_nodes_dict.keys()):
                if ntype not in self.node_feature_selector.keys():
                    del num_nodes_dict[ntype]
            # 读取边
            edge_dict = self.read_edges(graph_dir + 'edges.csv', num_nodes_dict)
            # 生成groundtruth
            self.label_to_groundtruth(cloudbed, self.nodes_df)
                        
            # 读取各个日期数据
            for date in self.dates:
                timestamp = pd.to_datetime(date)
                delta = pd.Timedelta(minutes=1)     
                loop = tqdm([timestamp + delta * i for i in range(60 * 24)])
                loop.set_description(f'cloudbed: {cloudbed}. date: {date}.')
                for timestamp in loop:
                    # 获取标签
                    label, groundtruth = self.get_label(timestamp, cloudbed)
                    # 过滤掉部分类型的故障
                    if label['failure_type'] not in self.failure_types:
                        continue
                    if groundtruth != []:
                        self.groundtruths.append(groundtruth)
                    self.labels.append(label)
                    try:       
                        # 读取节点特征
                        node_feats = self.read_node_feats(
                            api_filepath = graph_dir + \
                                'feats/{}/api/api_feats_{}.csv'.format(date, timestamp),
                            pod_filepath = graph_dir + \
                                'feats/{}/pod/pod_feats_{}.csv'.format(date, timestamp),
                            node_filepath = graph_dir + \
                                'feats/{}/node/node_feats_{}.csv'.format(date, timestamp),
                        )
                        # 读取边特征
                        edge_feats = self.read_edge_feats(
                            filepath = graph_dir + 'feats/{}/edge/edge_feats_{}.csv'
                            .format(date, timestamp),
                            edge_dict=edge_dict
                        )
                    except Exception as e:
                        logger.error('Timestamp: {}. Exception! {}'.format(timestamp, e))
                        self.labels = self.labels[:-1]
                        if groundtruth != []:
                            self.groundtruths = self.groundtruths[:-1]
                        continue
                        
                    # mask准备
                    if self.is_mask:
                        mask_prepare(edge_dict, num_nodes_dict, node_feats, edge_feats)
                        
                    # 构造异构图
                    g = dgl.heterograph(data_dict=edge_dict, num_nodes_dict=num_nodes_dict)
                    g.ndata['feat'] = node_feats
                    g.edata['feat'] = edge_feats
                    self.graphs.append(g)
                                    
                    if len(self.graphs) >= self.max_samples:
                        break
        
        # process extreme and missing values
        self.get_nan_nodes()
        self.get_nan_edges()
        
        if self.process_extreme:
            self.process_extreme_values()
        
        self.process_missing_values()
        
        logger.info(f'Number of graphs: {len(self.graphs)}')
