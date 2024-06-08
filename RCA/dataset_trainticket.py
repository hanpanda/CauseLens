import torch as th
import numpy as np
import pandas as pd
import os
import dgl
import json
from sklearn.preprocessing import MinMaxScaler
from dataset import myDGLDataset
from utils import *
from mask import *
from log import logger
from tqdm import tqdm


class HetGraphDataset_TrainTicket(myDGLDataset):

    def __init__(
        self,
        data_dir='../datasets/Nezha/trainticket',
        dates=[],
        node_feature_selector={},
        edge_feature_selector={},
        edge_reverse=False,
        add_self_loop=False,
        max_samples=1e9,
        failure_types = [],
        failure_duration=3,
        is_mask=False,
        process_miss='interpolate',
        process_extreme=False,
        k_sigma=3,
    ):        
        self.data_dir = data_dir
        self.dates = dates
        self.node_feature_selector = node_feature_selector
        self.edge_feature_selector = edge_feature_selector
        self.max_samples = max_samples
        self.failure_types = failure_types
        self.failure_duration = failure_duration
        self.is_mask = is_mask
        self.dtype = th.float32
        
        super().__init__(dataset_name='TrainTicket',
                         process_extreme=process_extreme,
                         process_miss=process_miss,
                         k=k_sigma)
                
    def read_labels(self):
        dfs = []
        for dirname in sorted(os.listdir(self.data_dir)):
            if dirname.endswith('.py') or dirname.endswith('.log'):
                continue
            
            # find filepath
            label_filepath = None
            dirpath = os.path.join(self.data_dir, dirname)
            for filename in os.listdir(dirpath):
                if filename.endswith('fault_list.json'):
                    label_filepath = os.path.join(dirpath, filename)
            
            if label_filepath is not None:
                # dict to dataframe
                records = []
                with open(label_filepath, 'r') as json_file:
                    for record in list(json.load(json_file).values()):
                        records += record
                df = pd.DataFrame().from_records(records)
                dfs.append(df)
        
        if dfs != []:
            labels_df = pd.concat(dfs)
            labels_df = labels_df.fillna('').astype(str)
            labels_df = labels_df.drop(columns=['inject_time'])
            
            labels_df['inject_timestamp'] = pd.to_datetime(labels_df['inject_timestamp'], unit='s').dt.ceil('1min')
            dfs = []
            for i in range(1, self.failure_duration):
                df = labels_df.copy()
                df['inject_timestamp'] += pd.Timedelta(minutes=i)
                dfs.append(df)
            
            # todo
            # def custom_ceil(ts, interval):
            #     ceil_ts = ts.ceil(freq='{}T'.format(interval))
            #     delta = ceil_ts - ts
            #     if delta > pd.Timedelta(minutes=interval) / 2:
            #         ceil_ts += pd.Timedelta(minutes=interval)
            #     return ceil_ts
            
            # labels_df['inject_timestamp'] = pd.to_datetime(labels_df['inject_timestamp'], unit='s')
            # labels_df['inject_timestamp'].apply(custom_ceil, interval=sample_interval)
            
            self.labels_df = pd.concat(dfs).sort_values(by='inject_timestamp')
            self.labels_df = self.labels_df.rename(columns={'inject_timestamp': 'timestamp',
                                                            'inject_pod': 'cmdb_id',
                                                            'inject_type': 'failure_type'})
            print(self.labels_df.head())
        else:
            self.labels_df = None        

    def get_label(self, timestamp, node_df, edge_df):
        if self.labels_df is None:
            label = {'timestamp': str(timestamp), 'cmdb_id': '', 'failure_type': '', 'inject_method': ''}
            groundtruth = []
            
            return label, groundtruth
        
        labels = self.labels_df[self.labels_df['timestamp'] == timestamp].to_dict('records')
        if labels == []:
            label = {'timestamp': str(timestamp), 'cmdb_id': '', 'failure_type': '', 'inject_method': ''}
            groundtruth = []
        else:
            label = labels[0]
            label['timestamp'] = str(label['timestamp'])
            df = node_df[node_df['node_name'] == label['cmdb_id']]
            service = df['service'].drop_duplicates().tolist()[0]
            
            if label['failure_type'] == 'network_delay':
                server_apis = node_df[(node_df['service'] == service) & 
                                      (node_df['node_name'].str.contains('HTTP'))]['node_id'].tolist()
                client_apis = edge_df[(edge_df['edge_type'] == 'api2api') & 
                                      (edge_df['node_id_src'].isin(server_apis))]['node_id_dst'].tolist()
                apis = server_apis + client_apis
            elif label['failure_type'] == 'code_delay' :
            # or label['failure_type'] == 'exception':
                apis = node_df[(node_df['service'] == service) & (node_df['node_name'].str.contains(label['inject_method']))]['node_id'].tolist()
            else:
                apis = node_df[node_df['service'] == service]['node_id'].tolist()
                
            apis = [('api', i) for i in apis]
            pods = df['node_id'].tolist()
            pods = [('pod', i) for i in pods]
            groundtruth = pods + apis
        
        return label, groundtruth
    
    def get_node_feats(self, timestamp, node_df, api_feats_df, pod_feats_df):      
        # get node feats and process(fill nan)  
        api_feats_df = pd.merge(left=node_df[node_df['node_type'] == 'api'],
                                right=api_feats_df[api_feats_df['timestamp'] == timestamp],
                                on='node_id', 
                                how='left',
                                validate='1:1')[self.node_feature_selector['api']]
        # api_feats_df = api_feats_df.fillna(0)
        pod_feats_df = pd.merge(left=node_df[node_df['node_type'] == 'pod'],
                                right=pod_feats_df[pod_feats_df['timestamp'] == timestamp],
                                on='node_id',
                                how='left',
                                validate='1:1')[self.node_feature_selector['pod']]
        # pod_feats_df = pod_feats_df.fillna(0)
        node_feats = {
            'api': th.from_numpy(api_feats_df.to_numpy()).to(dtype=self.dtype),
            'pod': th.from_numpy(pod_feats_df.to_numpy()).to(dtype=self.dtype)
        }
        
        return node_feats
    
    def get_edge_feats(self, timestamp, edge_df, edge_feats_df, edge_dict):
        edge_feats = {}
        for etype in edge_dict.keys():
            if etype[1] == 'self':
                num_self_edges = edge_dict[etype][0].shape[0]
                edge_feats[etype] = th.ones((num_self_edges, 1))
            else:
                # debug: how=left have all edges
                etype_str = etype[0] + '2' + etype[2]
                etype_edge_df = edge_df[edge_df['edge_type'] == etype_str]
                df = edge_feats_df[(edge_feats_df['edge_type'] == etype_str) & (edge_feats_df['timestamp'] == timestamp)]
                df = pd.merge(left=etype_edge_df, right=df, on=['node_id_src', 'node_id_dst'], how='left')['count']
                # df = df.fillna(0)
                edge_feats[etype] = th.from_numpy(df.to_numpy()).unsqueeze(dim=1).to(dtype=self.dtype)
        
        return edge_feats
    
    def process(self):
        self.graphs = []
        self.labels = []
        self.groundtruths = []
        self.read_labels()
        
        for dirname in sorted(os.listdir(self.data_dir)):
            if dirname not in self.dates:
                continue
            dirname = os.path.join(self.data_dir, dirname, 'graph')
            graph_nodes_filepath = os.path.join(dirname, 'graph_nodes.csv')
            graph_edges_filepath = os.path.join(dirname, 'graph_edges.csv')
            api_feats_filepath = os.path.join(dirname, 'api_feats.csv')
            pod_feats_filepath = os.path.join(dirname, 'pod_feats.csv')
            edge_feats_filepath = os.path.join(dirname, 'edge_feat.csv')
            
            # read nodes
            node_df = pd.read_csv(graph_nodes_filepath)
            num_nodes_dict = {
                'api': len(node_df[node_df['node_type'] == 'api']),
                'pod': len(node_df[node_df['node_type'] == 'pod'])
            }
            # debug: 
            self.node_df = node_df
            self.num_nodes_dict = num_nodes_dict
            # read edges
            edge_df = pd.read_csv(graph_edges_filepath)
            edge_dict = {}
            etypes = [('api', 'to', 'api'), ('pod', 'to', 'api')]
            for etype in etypes:
                etype_str = etype[0] + '2' + etype[2]
                df = edge_df[edge_df['edge_type'] == etype_str]
                edge_dict[etype] = (th.from_numpy(df['node_id_src'].to_numpy()),
                                    th.from_numpy(df['node_id_dst'].to_numpy()))
                
            # add self loop for nodes whose indegree are 0
            nonzero_indegree_nodes = {
                ntype: set() for ntype in num_nodes_dict.keys()
            }
            for etype, value in edge_dict.items():
                nonzero_indegree_nodes[etype[2]] |= set(value[1].tolist())
            for ntype, num_nodes in num_nodes_dict.items():
                nodes = th.arange(num_nodes)
                mask_for_nonzero = th.isin(nodes, th.tensor(list(nonzero_indegree_nodes[ntype])))
                nan_nodes = nodes[~mask_for_nonzero]
                if nan_nodes.shape[0] != 0:
                    edge_dict[(ntype, 'self', ntype)] = (nan_nodes, nan_nodes)
            
            # read feats
            api_feats_df = pd.read_csv(api_feats_filepath).drop_duplicates(subset=['node_id', 'timestamp'])
            pod_feats_df = pd.read_csv(pod_feats_filepath).drop_duplicates(subset=['node_id', 'timestamp'])
            edge_feats_df = pd.read_csv(edge_feats_filepath).drop_duplicates(subset=['node_id_src', 'node_id_dst', 'timestamp'])
            timestamps = api_feats_df['timestamp'].drop_duplicates()
            
            # build graphs
            for timestamp in tqdm(timestamps):
                # get label & filter
                label, groundtruth = self.get_label(timestamp, node_df, edge_df)
                if label['failure_type'] not in self.failure_types:
                    continue
                
                # get feats
                node_feats = self.get_node_feats(timestamp, node_df, api_feats_df, pod_feats_df)
                edge_feats = self.get_edge_feats(timestamp, edge_df, edge_feats_df, edge_dict)
                
                # mask prepare
                if self.is_mask:
                    mask_prepare(edge_dict, num_nodes_dict, node_feats, edge_feats)
                
                # build graph
                g = dgl.heterograph(data_dict=edge_dict, num_nodes_dict=num_nodes_dict)
                g.ndata['feat'] = node_feats
                g.edata['feat'] = edge_feats
                self.graphs.append(g)
                self.labels.append(label)
                if groundtruth != []:
                    self.groundtruths.append(groundtruth)
                if len(self.graphs) >= self.max_samples:
                    break
        
        # process extreme and missing values
        self.get_nan_nodes()
        self.get_nan_edges()
        
        if self.process_extreme:
            self.process_extreme_values()
        
        self.process_missing_values()
                
        logger.info(f'Number of graphs: {len(self.graphs)}')
                
    def dgl_id_to_name(self, dgl_id, ntype):
        return self.node_df[(self.node_df['node_type'] == ntype) & (self.node_df['node_id'] == dgl_id)]['node_name'].values[0]
    