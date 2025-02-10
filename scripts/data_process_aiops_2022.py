import sys
import os
import argparse
import datetime
import json
from collections import namedtuple

import pandas as pd
from tqdm import tqdm
from line_profiler import profile

sys.path.append('..')
from utils.log import Logger

logger = Logger(__file__)
traces_df = None
span2api = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default='2022-03-20')
    parser.add_argument('--cloudbed', type=str, default='cloudbed-1')
    args = parser.parse_args()

    # 处理某一天、某个cloudbed的数据；每1分钟的数据作为1个样本；
    cloudbed = args.cloudbed
    date = args.date
    sample_interval = '1min'
    time_offset = pd.Timedelta(hours=8)

    data_dir = '/root/lqh/multimodal-RCA/datasets/aiops_2022/raw/{}/{}/'.format(date, cloudbed)  # 原始数据目录
    container_metrics_dir = data_dir + 'metric/container/'  # 原始container指标数据目录
    node_metrics_dir = data_dir + 'metric/node/'  # 原始node指标数据目录
    traces_dir = data_dir + 'trace/all/'  # 原始trace数据目录
    logs_dir = data_dir + 'log/all/'  # 原始log数据目录

    graph_dir = '/root/lqh/multimodal-RCA/datasets/aiops_2022/graph/'  # 处理后的图数据目录
    api_feats_dir = graph_dir + 'feats/{}/api/'.format(date)  # api节点特征目录
    pod_feats_dir = graph_dir + 'feats/{}/pod/'.format(date)  # pod节点特征目录
    node_feats_dir = graph_dir + 'feats/{}/node/'.format(date)  # node节点特征目录
    edge_feats_dir = graph_dir + 'feats/{}/edge/'.format(date)  # edge特征目录

    dirs = [graph_dir, api_feats_dir, pod_feats_dir, node_feats_dir, edge_feats_dir]
    for dir in dirs:
        if os.path.exists(dir) == False:
            os.makedirs(dir)
    logger.info('创建文件夹完成。')


def groundtruth_to_csv():
    logger.info('groundtruth_csv生成...')
    json_dir = os.path.join(data_dir, '..', '..')
    for file in os.listdir(json_dir):
        if not file.endswith('.json'):
            continue
        date = '-'.join(file.split('.')[0].split('-')[3:])
        print(date)

        json_file_path = os.path.join(json_dir, file)
        csv_file_path = os.path.join(json_dir, '..', 'groundtruth_csv', file.split('.')[0] + '.csv')

        # 读取 JSON 文件
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        # 将 JSON 数据转换为 DataFrame
        df = pd.DataFrame(data)

        # 将 timestamp 列转换为整数类型
        df['timestamp'] = df['timestamp'].astype(int)

        # 检查 timestamp 列的值是否小于 1648000000
        df['timestamp'] = df['timestamp'].apply(
            lambda x: x + int(datetime.strptime(date, '%Y-%m-%d').timestamp()) if x < 1648000000 else x
        )

        df = df.sort_values(by='timestamp')

        # 将 DataFrame 转换为 CSV 文件
        df.to_csv(csv_file_path, index=False)


def read_traces(nrows=None):
    global traces_df
    if traces_df is None:
        logger.info('读取traces...')
        traces_df = pd.read_csv(traces_dir + 'trace_jaeger-span.csv', nrows=nrows)
        traces_df['service'] = traces_df['cmdb_id'].str.split('-', expand=True)[0].str.rstrip('2')
        traces_df['api'] = traces_df['service'] + '-' + traces_df['operation_name']
        traces_df['timestamp'] = pd.to_datetime(traces_df['timestamp'], unit='ms') + time_offset
        traces_df = traces_df.set_index('timestamp')

    return traces_df


def span_to_api():
    global span2api
    logger.info('span to api...')
    if span2api is None:
        span2api = traces_df.set_index('span_id')['api'].to_dict()

    return span2api


def read_graph_nodes(ntype=None):
    graph_nodes_df = pd.read_csv(graph_dir + 'graph_nodes.csv')
    if ntype is not None:
        graph_nodes_df = graph_nodes_df[graph_nodes_df['node_type'] == ntype]
        graph_nodes_df = graph_nodes_df.rename(columns={'node_name': ntype})

    return graph_nodes_df


################################################################################
#       统计图的所有节点，api，pod，node                                        #
################################################################################


def count_graph_nodes(nrows: int = 100000):
    node_file = graph_dir + 'graph_nodes.csv'
    if os.path.exists(node_file):
        logger.info('节点文件已经存在。')
        return

    logger.info('统计节点...')
    sample_trace_df = read_traces().head(nrows)

    dfs = []
    for ntype in ['api', 'pod', 'node']:
        if ntype == 'node':
            df = pd.read_csv(node_metrics_dir + os.listdir(node_metrics_dir)[0])['cmdb_id']
        elif ntype == 'pod':
            df = sample_trace_df['cmdb_id']
        else:
            df = sample_trace_df[ntype]

        df = df.value_counts().reset_index()['index'].to_frame()
        df = df.rename(columns={'index': 'node_name'})
        df = df.sort_values(by=['node_name'], ignore_index=True)
        df = df.reset_index()
        df = df.rename(columns={'index': 'node_id'})
        df['node_type'] = ntype
        dfs.append(df)

    graph_nodes_df = pd.concat(dfs, ignore_index=True)
    graph_nodes_df.to_csv(node_file, index=False)
    logger.info('节点信息已经保存到{}。'.format(node_file))


################################################################################
#       统计图的所有边：version2                                                #
#       对于AIOPS22数据集: api图固定，api到pod图固定，pod到node固定              #
#       对于没有调用的api仍然有连接，可在数据处理阶段设置为0                     #
################################################################################


@profile
def count_graph_edges(nrows: int = 10000):
    edge_file = graph_dir + 'graph_edges.csv'
    if os.path.exists(edge_file):
        logger.info('边文件已经存在。')
        return

    logger.info('统计边...')
    graph_nodes_df = read_graph_nodes()
    name2id = graph_nodes_df.set_index('node_name')['node_id'].to_dict()

    sample_trace_df = read_traces().head(nrows)
    span2api = span_to_api()

    metrics_df = pd.read_csv(container_metrics_dir + 'kpi_container_cpu_system_seconds.csv')
    metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'], unit='s') + time_offset
    metrics_df = metrics_df.set_index('timestamp')

    Edge = namedtuple('Edge', ['src', 'dst', 'edge_type'])
    edges = set()
    for _, row in tqdm(sample_trace_df.iterrows()):
        # 添加 pod to api
        callee = name2id[span2api[row['span_id']]]
        pod = name2id[row['cmdb_id']]
        edges.add(Edge(pod, callee, 'pod2api'))
        # 添加 api to api
        try:
            if pd.isna(row['parent_span']) == False:
                caller = name2id[span2api[row['parent_span']]]
                edges.add(Edge(callee, caller, 'api2api'))
        except Exception as e:
            logger.info(e)
            continue

    # 添加 node to pod
    pod_node = metrics_df['cmdb_id'].drop_duplicates().str.split('.')
    for node, pod in pod_node:
        pod = name2id[pod]
        node = name2id[node]
        edges.add(Edge(node, pod, 'node2pod'))

    graph_edges_df = pd.DataFrame(edges)
    graph_edges_df.to_csv(edge_file, index=False)
    logger.info('边信息已经保存到{}。'.format(edge_file))


################################################################################
#       从trace文件中聚合出api节点特征：1min的粒度                              #
#           mean, std, min, median, q25, q75, max (duration)                   #
#           success_rate, count                                                #
#       不被调用的api没有特征                                                   #
################################################################################


@profile
def extract_api_feats_from_traces():
    logger.info('trace提取api节点特征...')
    sample_trace_df = read_traces()
    graph_nodes_df = read_graph_nodes(ntype='api')

    time_groups = sample_trace_df.resample(sample_interval)
    for timestamp, group in tqdm(time_groups):
        timestamp = timestamp.floor(sample_interval)
        # 根据api进行分组并聚合
        api_groups = group.groupby(by=['api'])
        funcs = [(lambda x, j=i: x.quantile(j * 0.05)) for i in range(1, 20, 1)]
        agg_dict = {
            'api':
                'count',
            'duration': ['mean', 'std', 'median', 'min', 'max'] + funcs,
            'status_code':
                lambda x: len(x[(x != 0) & (x != 200) & (x != 'ok') & (x != 'oK') & (x != 'Ok') &
                                (x != 'OK')]) / len(x)
        }
        api_feats_df = api_groups.agg(agg_dict).reset_index()
        # 为计算列创建适当的列名
        api_feats_df.columns = [
            'api', 'count', 'mean', 'std', 'median', 'min', 'max', 'q5', 'q10', 'q15', 'q20', 'q25', 'q30', 'q35',
            'q40', 'q45', 'q50', 'q55', 'q60', 'q65', 'q70', 'q75', 'q80', 'q85', 'q90', 'q95', 'success_rate'
        ]
        api_feats_df = pd.merge(left=graph_nodes_df, right=api_feats_df, how='left', on='api')
        api_feats_df.to_csv(api_feats_dir + 'api_feats_{}.csv'.format(timestamp), index=False)


# ###############################################################################
#       从metric文件中聚合出pod节点的特征：1min的粒度                            #
#       包括所有的container指标无论时间段内是否有api调用                         #
# ###############################################################################


@profile
def extract_pod_feats_from_metrics():
    logger.info('metric提取pod节点特征...')
    graph_nodes_df = read_graph_nodes(ntype='pod')

    metrics_df = None
    for file in sorted(os.listdir(container_metrics_dir)):
        # logger.info(file)
        df = pd.read_csv(container_metrics_dir + file)
        for metric_name, metric_df in df.groupby(by=['kpi_name']):
            metric_df = metric_df.rename(columns={'value': metric_name})
            metric_df = metric_df.drop(columns=['kpi_name'])
            metric_df = metric_df.drop_duplicates(subset=['timestamp', 'cmdb_id'])

            if metrics_df is None:
                metrics_df = metric_df
            else:
                # 根据左边的df的列：timestamp, cmdb_id 进行merge。
                metrics_df = pd.merge(left=metrics_df, right=metric_df, how='left', on=['timestamp', 'cmdb_id'])

    metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'], unit='s') + time_offset
    metrics_df = metrics_df.set_index('timestamp')
    time_groups = metrics_df.resample(sample_interval)

    for timestamp, group in tqdm(time_groups):
        file = pod_feats_dir + 'pod_feats_{}.csv'.format(timestamp)
        try:
            pod_feats_df = group
            pod_feats_df[['node', 'pod']] = pod_feats_df['cmdb_id'].str.split('.', expand=True)
            pod_feats_df = pd.merge(left=graph_nodes_df, right=pod_feats_df, how='left', on='pod')
            pod_feats_df.to_csv(file, index=False)
        except Exception as e:
            logger.info(e)
            continue


################################################################################
#       从metric文件中聚合出node(主机）节点的特征：1min的粒度                     #
#       包括所有的system指标                                                    #
################################################################################


def extract_node_feats_from_metrics():
    logger.info('metric提取node节点特征...')
    graph_nodes_df = read_graph_nodes(ntype='node')

    for file in os.listdir(node_metrics_dir):
        node_metrics_df = pd.read_csv(node_metrics_dir + file)
        # 将dataframe变形
        node_metrics_df = node_metrics_df.drop_duplicates(subset=['timestamp', 'cmdb_id', 'kpi_name'])
        node_metrics_df = node_metrics_df.pivot(index=['timestamp', 'cmdb_id'], columns='kpi_name', values='value')
        node_metrics_df = node_metrics_df.reset_index()
        node_metrics_df = node_metrics_df.rename(columns={'cmdb_id': 'node'})
        # 时间处理
        node_metrics_df['timestamp'] = \
            pd.to_datetime(node_metrics_df['timestamp'], unit='s') + time_offset
        node_metrics_df = node_metrics_df.set_index('timestamp')
        # 按照1分钟分组
        time_groups = node_metrics_df.resample(sample_interval)
        for timestamp, node_feats_df in time_groups:
            # logger.info(timestamp)
            node_feats_df = pd.merge(left=graph_nodes_df, right=node_feats_df, how='left', on='node')
            node_feats_df.to_csv(node_feats_dir + 'node_feats_{}.csv'.format(timestamp), index=False)


################################################################################
#       从trace文件中聚合出pod节点特征：1min粒度                                #
#                   count                                                      #
################################################################################


@profile
def extract_pod_feats_from_traces():
    logger.info('trace提取pod节点特征...')
    traces_df = read_traces()

    time_groups = traces_df.resample(sample_interval)
    funcs = [(lambda x, j=i: x.quantile(j * 0.1)) for i in range(1, 10, 1)]
    for timestamp, group in tqdm(time_groups):
        timestamp = timestamp.floor(sample_interval)
        # logger.info(timestamp)
        try:
            df = group.groupby(by='cmdb_id').agg({
                'cmdb_id': 'count',
                'duration': ['mean', 'std', 'median', 'min', 'max'] + funcs
            })
            # debug
            df.columns = [
                'count', 'mean', 'std', 'median', 'min', 'max', 'q10', 'q20', 'q30', 'q40', 'q50', 'q60', 'q70', 'q80',
                'q90'
            ]
            df.index.name = 'pod'

            pod_feats_file = 'pod_feats_{}.csv'.format(timestamp)
            pod_feats_df = pd.read_csv(pod_feats_dir + pod_feats_file)
            pod_feats_df = pd.merge(left=pod_feats_df, right=df, how='left', on='pod')

            pod_feats_df.to_csv(pod_feats_dir + pod_feats_file, index=False)
        except Exception as e:
            logger.info(e)
            continue


################################################################################
#       从trace文件中聚合出api->api, api->pod边的特征                            #
#                   count                                                      #
################################################################################


@profile
def extract_edge_feats_from_traces():
    logger.info('trace提取边特征...')
    graph_nodes_df = read_graph_nodes()
    name2id = graph_nodes_df.set_index('node_name')['node_id'].to_dict()

    traces_df = read_traces()
    span2api = span_to_api()

    time_groups = traces_df.resample(sample_interval)
    for timestamp, group_df in tqdm(time_groups):
        # extract features of each sample interval
        timestamp = timestamp.floor(sample_interval)
        edges = []
        for _, row in group_df.iterrows():
            current_span_api = span2api[row['span_id']]
            if current_span_api not in name2id:
                continue
            api = name2id[current_span_api]
            pod = name2id[row['cmdb_id']]
            duration = row['duration']
            try:
                parent_api = name2id[span2api[row['parent_span']]]
            except Exception as e:
                parent_api = None
            edges.append({'api': api, 'parent_api': parent_api, 'pod': pod, 'duration': duration})

        edges = pd.DataFrame(edges)
        # columns = ['src', 'dst', 'count', 'mean', 'q10', 'q20', 'q30', 'q40', 'q50', 'q60', 'q70', 'q80', 'q90']
        # quantiles = [(lambda x, j=i: x.quantile(j * 0.1)) for i in range(1, 10, 1)]
        columns = ['dst', 'src', 'count', 'mean']
        quantiles = []

        api2api_grouped = edges.dropna(subset=['parent_api']).groupby(by=['parent_api', 'api'])
        api2api_edges = api2api_grouped.size().reset_index(name='count')
        api2api_duration = api2api_grouped['duration'].agg(['mean'] + quantiles).reset_index()
        api2api_feats = pd.merge(left=api2api_edges, right=api2api_duration, on=['parent_api', 'api'])
        api2api_feats.columns = columns
        api2api_feats['etype'] = 'api2api'

        api2pod_grouped = edges.groupby(by=['api', 'pod'])
        api2pod_edges = api2pod_grouped.size().reset_index(name='count')
        api2pod_duration = api2pod_grouped['duration'].agg(['mean'] + quantiles).reset_index()
        api2pod_feats = pd.merge(left=api2pod_edges, right=api2pod_duration, on=['api', 'pod'])
        api2pod_feats.columns = columns
        api2pod_feats['etype'] = 'pod2api'

        pd.concat([api2api_feats, api2pod_feats]).to_csv(
            edge_feats_dir + 'edge_feats_{}.csv'.format(timestamp), index=False
        )


################################################################################
#                   聚合pod网络特征到api节点                                     #
################################################################################
def agg_api_network_feats():
    network_metrics = [
        'container_network_receive_errors.eth0', 'container_network_transmit_errors.eth0',
        'container_network_receive_MB.eth0', 'container_network_transmit_MB.eth0',
        'container_network_receive_packets.eth0', 'container_network_transmit_packets.eth0',
        'container_network_receive_packets_dropped.eth0', 'container_network_transmit_packets_dropped.eth0'
    ]
    interval = pd.Timedelta(sample_interval)
    timestamps = [pd.Timestamp(date) + i * interval for i in range(24 * 60 * 60 // interval.seconds)]
    for timestamp in tqdm(timestamps):
        try:
            api_df = pd.read_csv(api_feats_dir + f'api_feats_{timestamp}.csv')
            pod_df = pd.read_csv(pod_feats_dir + f'pod_feats_{timestamp}.csv')

            for col in network_metrics:
                if col in api_df.columns:
                    # api_df = api_df.drop(col, axis=1)
                    continue

            api_df[['service', '_']] = api_df['api'].str.split('-', 1, expand=True)
            pod_df[['service', '_']] = pod_df['pod'].str.split('-', 1, expand=True)
            pod_df['service'] = pod_df['service'].str.rstrip('2')
            pod_df = pod_df.groupby(by='service')[network_metrics].sum().reset_index()

            api_df = pd.merge(api_df, pod_df, on='service')
            api_df = api_df.drop(columns=['service', '_'])
            api_df.to_csv(api_feats_dir + f'api_feats_{timestamp}.csv', index=False)
            # api_df.to_csv('tmp.csv')
        except Exception as e:
            logger.info(f'{cloudbed}: {timestamp}. Exception: {e}.')


################################################################################
#                   按照次序执行脚本提取特征                                     #
################################################################################


def preprocess():
    groundtruth_to_csv()
    count_graph_nodes()
    count_graph_edges()
    extract_api_feats_from_traces()
    extract_pod_feats_from_metrics()
    # extract_node_feats_from_metrics()
    extract_edge_feats_from_traces()
    extract_pod_feats_from_traces()
    # agg_api_network_feats()


if __name__ == '__main__':
    preprocess()
