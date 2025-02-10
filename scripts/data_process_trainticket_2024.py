import sys
import os
import json
import argparse
from datetime import datetime
from collections import defaultdict
from typing import Optional, Dict, List

import pandas as pd
from tqdm import tqdm
from line_profiler import profile

sys.path.append('..')
from utils.log import Logger

logger = Logger(__file__)

BASE_DIR = '/root/lqh/multimodal-RCA/datasets/TrainTicket_2024'
TARGET_DIR = '/root/lqh/multimodal-RCA/datasets/TrainTicket_2024/graph'

total_trace_df: Optional[pd.DataFrame] = None
span2api: Optional[Dict[str, str]] = None


def groundtruth_to_csv():
    logger.info('groundtruth_csv生成...')
    groundtruth_dir = os.path.join(BASE_DIR, 'groundtruth_csv')
    os.makedirs(groundtruth_dir, exist_ok=True)

    records_per_date = defaultdict(list)

    for file in os.listdir(BASE_DIR):
        if '-fault_list.json' not in file:
            continue
        with open(os.path.join(BASE_DIR, file), 'r') as fp:
            tmp = json.load(fp)
        for fault in tmp:
            record = {
                'timestamp': fault['inject_timestamp'],
                'level': 'pod',
                'cmdb_id': fault['inject_pod'],
                'failure_type': fault['inject_type'],
                'method': fault['inject_method']
            }
            date = datetime.fromtimestamp(record['timestamp']).strftime('%Y-%m-%d')
            records_per_date[date].append(record)

    for date, records in records_per_date.items():
        records = sorted(records, key=lambda r: r['timestamp'])
        pd.DataFrame.from_records(records).to_csv(
            os.path.join(groundtruth_dir, f'groundtruth-k8s-1-{date}.csv'), index=False
        )


def round_to_nearest_minute(timestamp):
    dt = pd.Timestamp(timestamp, unit='s')
    # 检查秒数是否大于30
    if dt.second > 30:
        # 如果秒数大于30，向上取整到最近的一分钟
        rounded_dt = dt.ceil('min')
    else:
        # 如果秒数小于等于30，向下取整到最近的一分钟
        rounded_dt = dt.floor('min')
    return int(rounded_dt.timestamp())


def pod_to_service(pod: str):
    return '-'.join(pod.split('-')[:-2])


def read_traces():
    global total_trace_df
    if total_trace_df is not None:
        return total_trace_df
    trace_dir = os.path.join(DATA_DIR, 'trace')
    dfs = []
    for trace_file in tqdm(sorted(os.listdir(trace_dir)), desc='read_traces: '):
        df = pd.read_csv(os.path.join(trace_dir, trace_file))
        dfs.append(df)
    total_trace_df = pd.concat(dfs)
    total_trace_df = total_trace_df.drop_duplicates(subset=['TraceID', 'SpanID'])

    return total_trace_df


def read_graph_nodes():
    node_df = pd.read_csv(os.path.join(TARGET_DIR, 'graph_nodes.csv'))
    return node_df


def span_to_api():
    global span2api
    if span2api is not None:
        return span2api
    span2api = {}

    sample_trace_df = read_traces()

    for trace_id, trace_df in tqdm(sample_trace_df.groupby(by='TraceID'), desc='span_to_api: '):
        trace_dict = trace_df.set_index('SpanID').to_dict(orient='index')

        for span_id, span in trace_dict.items():
            service = pod_to_service(span['PodName'])
            op = span['OperationName']
            if op.startswith('HTTP'):
                method = op
                route = trace_df[trace_df['ParentID'] == span_id]['OperationName'].tolist()
                if len(route) > 0:
                    route = route[0]
                else:
                    continue
                api = f'{service}-{method} {route}'
            elif op.startswith('/api'):
                method = trace_dict[span['ParentID']]['OperationName']
                route = op
                api = f'{service}-{method} {route}'
            else:
                api = f'{service}-{op}'
            span2api[span_id] = api

    return span2api


def count_graph_nodes(nrows: int = 10**6):
    node_file_path = os.path.join(TARGET_DIR, 'graph_nodes.csv')
    if os.path.exists(node_file_path):
        logger.info('节点文件已经存在。')
        return

    sample_trace_df = read_traces().head(nrows)
    api_set = set()
    pod_set = set()

    for trace_id, trace_df in sample_trace_df.groupby('TraceID'):
        trace_dict = trace_df.set_index('SpanID').to_dict(orient='index')

        for span_id, span in trace_dict.items():
            service = pod_to_service(span['PodName'])
            op = span['OperationName']
            if op.startswith('HTTP'):
                method = op
                route = trace_df[trace_df['ParentID'] == span_id]['OperationName'].tolist()
                if len(route) > 0:
                    route = route[0]
                else:
                    continue
                api = f'{service}-{method} {route}'
            elif op.startswith('/api'):
                method = trace_dict[span['ParentID']]['OperationName']
                route = op
                api = f'{service}-{method} {route}'
            else:
                api = f'{service}-{op}'
            api_set.add(api)
            pod_set.add(span['PodName'])

    records = []
    records.extend([{
        'node_id': id,
        'node_name': node_name,
        'node_type': 'api'
    } for id, node_name in enumerate(sorted(list(api_set)))])
    records.extend([{
        'node_id': id,
        'node_name': node_name,
        'node_type': 'pod'
    } for id, node_name in enumerate(sorted(list(pod_set)))])
    node_df = pd.DataFrame.from_records(records)
    node_df.to_csv(node_file_path, index=False)
    logger.info('节点信息已经保存到{}。'.format(node_file_path))


def count_graph_edges(nrows: int = 10**6):
    edge_file_path = os.path.join(TARGET_DIR, 'graph_edges.csv')
    if os.path.exists(edge_file_path):
        logger.info('边文件已经存在。')
        return

    logger.info('统计边...')
    node_df = read_graph_nodes()
    api2id = {}
    pod2id = {}
    for _, row in node_df.iterrows():
        if row['node_type'] == 'api':
            api2id[row['node_name']] = row['node_id']
        elif row['node_type'] == 'pod':
            pod2id[row['node_name']] = row['node_id']

    span2api = span_to_api()

    record_set = set()
    sample_trace_df = read_traces().head(nrows)
    for _, span in sample_trace_df.iterrows():
        api = span2api[span['SpanID']]
        record_set.add((pod2id[span['PodName']], api2id[api], 'pod2api'))
        if span['ParentID'] != 'root':
            parent_api = span2api[span['ParentID']]
            record_set.add((api2id[api], api2id[parent_api], 'api2api'))

    edge_df = pd.DataFrame.from_records(list(record_set), columns=['src', 'dst', 'edge_type'])
    edge_df.to_csv(edge_file_path, index=False)


def extract_api_feats_from_traces():
    logger.info('trace提取api节点特征...')
    sample_trace_df = read_traces()
    sample_trace_df['Duration'] = sample_trace_df['Duration'].astype(float)
    sample_trace_df['Duration'] = sample_trace_df['Duration'].interpolate(limit_direction='both')
    span2api = span_to_api()
    sample_trace_df['api'] = [
        span2api[span_id] if span_id in span2api else 'None' for span_id in sample_trace_df['SpanID'].tolist()
    ]
    sample_trace_df = sample_trace_df[sample_trace_df['api'] != 'None']
    sample_trace_df['minute'] = sample_trace_df['EndTimeUnixNano'] // 10**9 // 60 * 60 + 60

    node_df = read_graph_nodes()
    node_df = node_df[node_df['node_type'] == 'api']

    for timestamp, trace_df in tqdm(sample_trace_df.groupby(by='minute'), desc='extract_api_feats_from_traces: '):
        duration_funcs = []
        duration_funcs.extend(['mean', 'std', 'median', 'min', 'max'])
        duration_funcs.extend([(lambda x, j=i: x.quantile(j * 0.05)) for i in range(1, 20, 1)])
        agg_dict = {'api': 'count', 'Duration': duration_funcs}
        # print(trace_df.groupby(by='api').head())
        # exit(-1)
        trace_df = trace_df[['api', 'Duration']]
        # print(trace_df.dtypes)
        # exit(-1)
        feat_df = trace_df.groupby(by='api').agg(agg_dict).reset_index()
        feat_df.columns = [
            'api', 'count', 'mean', 'std', 'median', 'min', 'max', 'q5', 'q10', 'q15', 'q20', 'q25', 'q30', 'q35',
            'q40', 'q45', 'q50', 'q55', 'q60', 'q65', 'q70', 'q75', 'q80', 'q85', 'q90', 'q95'
        ]
        feat_df = pd.merge(left=node_df, right=feat_df, left_on='node_name', right_on='api', how='left')
        feat_df = feat_df.drop(columns=['node_name'])

        dt = datetime.fromtimestamp(timestamp)
        time = dt.strftime('%Y-%m-%d %H:%M:%S')
        date = time.split(' ')[0]
        feat_dir = os.path.join(TARGET_DIR, 'feats', date, 'api')
        os.makedirs(feat_dir, exist_ok=True)
        feat_df.to_csv(os.path.join(feat_dir, f'api_feats_{time}.csv'), index=False)


def extract_pod_feats_from_metrics():
    logger.info('metric提取pod节点特征...')
    metric_dir = os.path.join(DATA_DIR, 'metric')
    dfs = []
    for metric_file in tqdm(sorted(os.listdir(metric_dir)), desc='read metrics: '):
        if not metric_file.startswith('ts-'):
            continue
        df = pd.read_csv(os.path.join(metric_dir, metric_file))
        dfs.append(df)
    df = pd.concat(dfs)
    df = df.dropna(subset=['TimeStamp'])

    node_df = read_graph_nodes()
    node_df = node_df[node_df['node_type'] == 'pod']

    df['TimeStamp'] = df['TimeStamp'].apply(lambda x: round_to_nearest_minute(int(x)))
    for timestamp, metric_df in tqdm(df.groupby(by='TimeStamp'), desc='extract_pod_feats_from_metrics: '):
        feat_df = metric_df.drop(columns=['Time', 'TimeStamp']).rename(columns={'PodName': 'pod'})
        feat_df = feat_df.sort_values(by='pod')
        feat_df = feat_df.reset_index(drop=True)
        feat_df = pd.merge(left=node_df, right=feat_df, left_on='node_name', right_on='pod', how='left')
        feat_df = feat_df.drop(columns=['node_name'])

        dt = datetime.fromtimestamp(timestamp)
        time = dt.strftime('%Y-%m-%d %H:%M:%S')
        date = time.split(' ')[0]
        feat_dir = os.path.join(TARGET_DIR, 'feats', date, 'pod')
        os.makedirs(feat_dir, exist_ok=True)
        feat_df.to_csv(os.path.join(feat_dir, f'pod_feats_{time}.csv'), index=False)


def extract_edge_feats_from_traces():
    logger.info('trace提取边特征...')
    node_df = read_graph_nodes()
    name2id = node_df.set_index('node_name')['node_id'].to_dict()

    sample_trace_df = read_traces()
    span2api = span_to_api()
    sample_trace_df['minute'] = sample_trace_df['EndTimeUnixNano'] // 10**9 // 60 * 60 + 60

    for timestamp, trace_df in tqdm(sample_trace_df.groupby(by='minute'), desc='extract_edge_feats_from_traces: '):
        edges = []
        for _, row in trace_df.iterrows():
            if row['ParentID'] not in span2api or row['SpanID'] not in span2api:
                continue
            parent_api = span2api[row['ParentID']]
            api = span2api[row['SpanID']]
            pod = row['PodName']
            duration = row['Duration']
            if api not in name2id or parent_api not in name2id or pod not in name2id:
                continue
            edges.append({
                'api': name2id[api],
                'parent_api': name2id[parent_api],
                'pod': name2id[pod],
                'duration': duration
            })
        edges = pd.DataFrame(edges)
        if edges.empty:
            continue

        columns = ['dst', 'src', 'count', 'mean']
        dfs = []

        grouped = edges.groupby(by=['parent_api', 'api'])
        df1 = grouped.size().reset_index(name='count')
        df2 = grouped.agg({'duration': ['mean']})
        feat_df = pd.merge(left=df1, right=df2, on=['parent_api', 'api'])
        feat_df.columns = columns
        feat_df['etype'] = 'api2api'
        dfs.append(feat_df)

        grouped = edges.groupby(by=['api', 'pod'])
        df1 = grouped.size().reset_index(name='count')
        df2 = grouped.agg({'duration': ['mean']})
        feat_df = pd.merge(left=df1, right=df2, on=['api', 'pod'])
        feat_df.columns = columns
        feat_df['etype'] = 'pod2api'
        dfs.append(feat_df)

        dt = datetime.fromtimestamp(timestamp)
        time = dt.strftime('%Y-%m-%d %H:%M:%S')
        date = time.split(' ')[0]
        feat_dir = os.path.join(TARGET_DIR, 'feats', date, 'edge')
        os.makedirs(feat_dir, exist_ok=True)
        pd.concat(dfs).to_csv(os.path.join(feat_dir, f'edge_feats_{time}.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default='2024-04-25')
    parser.add_argument('--target_dir', type=str, default='/root/lqh/multimodal-RCA/datasets/TrainTicket_2024/graph_1')
    args = parser.parse_args()

    DATA_DIR = os.path.join(BASE_DIR, f'{args.date}')
    TARGET_DIR = args.target_dir
    os.makedirs(TARGET_DIR, exist_ok=True)

    groundtruth_to_csv()
    count_graph_nodes()
    count_graph_edges()
    extract_api_feats_from_traces()
    extract_pod_feats_from_metrics()
    extract_edge_feats_from_traces()
