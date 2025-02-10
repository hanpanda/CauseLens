import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from os import path
from tqdm import tqdm
from collections import defaultdict


def read_data_for_methods(dataset='TT'):
    df = pd.read_excel(io='实验记录.xlsx', sheet_name='对比实验', usecols='A:J')
    methods = ['[NAME]', 'CIRCA', 'MicroRCA', 'MicroScope', 'GATAE', 'GCNAE', 'DotGATAE']
    data_dict = {}

    for method in methods:
        print('---------Method: {}---------'.format(method))
        method_df = df[(df['Data'] == dataset) & (df['Method'] == method) & (df['Directed'] == 0)]
        if method_df.empty:
            continue
        method_df = method_df.reset_index(drop=True)
        method_df = method_df.drop(columns=['Data', 'Method', 'Directed'])
        method_df = method_df.rename(
            columns={
                'Top@1': 'HR@1',
                'Top@2': 'HR@2',
                'Top@3': 'HR@3',
                'Top@4': 'HR@4',
                'Top@5': 'HR@5',
            }
        )
        method_df = method_df[1:]
        print(method_df)
        # method_df['Label'], _ = method_df['Label'].str.split('(')
        method_df = method_df.set_index('Label')
        data_dict[method] = method_df.to_dict()
    print(data_dict)

    return data_dict


# read json and compute avg@k
def compute_metrics(result_dir, fault_types, K):
    num_fault_dict = defaultdict(int)
    avg_k_dict = defaultdict(dict)
    top_k_dict = defaultdict(lambda: defaultdict(dict))

    for i in range(11):
        for fault_type in fault_types:
            coefficient = '{:.1f}'.format(i / 10)
            file_path = path.join(result_dir, 'rca_metrics_{:.1f}_{}.json'.format(i / 10, fault_type))
            if path.exists(file_path) == False:
                # print('False. {}.'.format(file_path))
                continue
            with open(file_path, 'r') as f:
                metric_dict = json.load(f)

            avg_k = 0
            for j in range(1, K + 1):
                avg_k += metric_dict['top-{}'.format(j)]
            avg_k /= K
            avg_k_dict[fault_type][coefficient] = avg_k
            print('fault_type: {}. coefficient: {:.1f}. avg@k: {}.'.format(fault_type, i / 10, avg_k))

            for j in range(1, K + 1):
                top_k_dict[j][fault_type][coefficient] = metric_dict['top-{}'.format(j)]

            num_fault_dict[fault_type] = metric_dict['n']

    return avg_k_dict, top_k_dict, num_fault_dict


# parameter sensity visualize (top@1 & avg@k)
def plot_params_and_metrics(
    result_dir,
    metric_data,
    metric_names='',
    fault_types=[],
    param_name='alpha',
    figsize=None,
    xlabel=r'$\alpha$',
    ylabel=''
):
    if os.path.exists(result_dir) == False:
        os.makedirs(result_dir, exist_ok=True)

    matplotlib.rcParams['font.family'] = 'Times New Roman'
    matplotlib.rcParams['font.size'] = 11
    matplotlib.rcParams['font.weight'] = 'bold'

    if figsize is not None:
        plt.figure(figsize=figsize)
    else:
        plt.figure(figsize=(15, 3))

    if isinstance(metric_names, str):
        metric_names = [metric_names]

    for i, metric_name in enumerate(metric_names):
        metric_dict = metric_data[metric_name]

        for j, fault_type in enumerate(fault_types, start=1):
            data = metric_dict[fault_type]
            x_values = list(data.keys())
            y_values = list(data.values())

            if param_name == 'alpha':
                plt.subplot(1, len(metric_dict), j)
                # plt.subplot(int(i/2)+1, i%2+1, i)
                plt.plot(x_values, y_values, marker='o')
                plt.title(f'Fault Type: {fault_type}')
                plt.xlabel(xlabel)
                plt.ylabel(metric_name)
                plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5)

            elif param_name.startswith('tau'):
                color = plt.cm.tab10(i)
                plt.plot(x_values, y_values, marker='o', markersize=5, label=metric_name, color=color)

    if param_name.startswith('tau'):
        plt.xlabel(xlabel)
        plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5)
        plt.legend(bbox_to_anchor=(0.5, 1.2), loc='upper center', ncol=4)

    plt.tight_layout()
    file_path = path.join(result_dir, '{}_{}_{}.pdf'.format(param_name, metric_names, fault_types))
    plt.savefig(file_path)
    print('save to: {}'.format(file_path))


def plot_params_and_metrics_on_one_graph(result_dir, metric_dict, metric_name=''):
    plt.figure(figsize=(10, 6))

    for i, (fault_type, data) in enumerate(metric_dict.items(), start=1):
        coefficients = list(data.keys())
        avg_k_values = list(data.values())

        # 选择颜色
        color = plt.cm.tab10(i)  # 使用tab10调色板
        plt.plot(coefficients, avg_k_values, marker='o', label=fault_type, color=color)

    plt.legend()
    plt.title('{} vs Coefficient for Different Fault Types'.format(metric_name))
    plt.xlabel('Coefficient')
    plt.ylabel(metric_name)

    file_path = path.join(result_dir, '{}_one_graph.png'.format(metric_name))
    plt.savefig(file_path)
    print('save to: {}'.format(file_path))


def plot_radar(result_dir, data_dict, coefficient, metric_name='', fault_types=[]):
    # 计算角度
    angles = np.linspace(0, 2 * np.pi, len(fault_types), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # 绘制雷达图
    for method, metric_dict in data_dict.items():
        data = []
        for fault_type in fault_types:
            value = metric_dict[metric_name][fault_type]
            if isinstance(value, dict):
                value = value[coefficient]
            data.append(value)
        data += data[:1]
        ax.plot(angles, data, label=f'{method}')
        ax.fill(angles, data, alpha=0.25)

    ax.set_rlim(0, 1)
    # ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(fault_types)
    ax.legend(loc='upper right')
    ax.set_title('{}'.format(metric_name))

    file_path = path.join(result_dir, '{}_radar.png'.format(metric_name))
    plt.savefig(file_path)
    print('save to: {}'.format(file_path))


def compute_overall_metrics(top_k_dict, avg_k_dict, num_fault_dict, fault_types, K):
    num_samples = 0

    for fault_type in fault_types:
        num_samples += num_fault_dict[fault_type]

    for coefficient in range(11):
        coefficient = '{:.1f}'.format(coefficient / 10)
        avg_k = 0
        for i, k in enumerate(top_k_dict.keys()):
            hit_samples = 0
            for fault_type in fault_types:
                hit_samples += num_fault_dict[fault_type] * top_k_dict[k][fault_type][coefficient]
            top_k_dict[k]['all'][coefficient] = hit_samples / num_samples
            avg_k += top_k_dict[k]['all'][coefficient]

            if i + 1 == K:
                break
        avg_k /= K
        avg_k_dict['all'][coefficient] = avg_k

    # print top@K, avg@K
    for coefficient in range(11):
        coefficient = '{:.1f}'.format(coefficient / 10)
        print('-------------------coefficient: {}-------------------'.format(coefficient))

        for key in top_k_dict.keys():
            print('{}: {}'.format(key, top_k_dict[key]['all'][coefficient]))
        print('Avg@{}: {}'.format(K, avg_k_dict['all'][coefficient]))


def get_all_metrics(times, fault_types, K):
    # {failure_type: {cofficient: avg_k}}
    avg_k_dict = defaultdict(dict)
    # {HR@k: {failure_type: {cofficient: top_k}}}
    top_k_dict = defaultdict(lambda: defaultdict(dict))
    num_fault_dict = defaultdict(int)

    for time in times:
        result_dir = 'model-store/TrainTicket/CustomGAE/rca_{}'.format(time)
        avg_k_dict_tmp, top_k_dict_tmp, num_fault_dict_tmp = compute_metrics(result_dir, fault_types, K)
        avg_k_dict.update(avg_k_dict_tmp)
        num_fault_dict.update(num_fault_dict_tmp)
        for k in top_k_dict_tmp.keys():
            top_k_dict[f'HR@{k}'].update(top_k_dict_tmp[k])

    fault_types = [fault_type for fault_type in fault_types if fault_type != 'all']
    compute_overall_metrics(top_k_dict, avg_k_dict, num_fault_dict, fault_types, K)

    return top_k_dict, avg_k_dict


def main():
    dataset = 'tt'
    K = 5
    fault_types = ['cpu_contention', 'network_delay', 'code_delay', 'exception', 'all']

    # Experiment 1: alpha
    def alpha_experiment():
        # pred_and_mean

        # tau_1: 2.0 tau_2: 0.3
        times = ['2024_04_29_10_48', '2024_04_17_14_52']
        times = ['2024_06_03_16_45', '2024_06_03_17_11']  # no process extreme; 1440;                     **
        times = ['2024_06_03_16_45', '2024_06_03_19_40']  # no process extreme; 1440; method;
        times = ['2024_06_03_20_42', '2024_06_03_20_08']  # no process extreme; 1440; method; +q95,q99;
        times = ['', '2024_06_03_20_19']  # no process extreme; 1200; method; +q95,q99;
        times = ['2024_06_03_20_42', '2024_06_03_20_36']  # no process extreme; 1440; +q95,q99;           *
        # times = ['2024_06_03_20_52', '2024_06_03_20_56']    # no process extreme; 1600; +q95,q99;
        # times = ['2024_06_03_21_03']    # no process extreme; 1440; +q95,q99,-q10~q50;

        # tau_1: 3.0 tau_2: 0.6
        times = ['2024_06_04_23_18', '2024_06_04_21_40']

        # all
        image_dir = 'images/alpha_tau1_3.0_tau2_0.6_{}_{}'.format(times[0], times[1])

        # code_delay & exception
        # times = ['2024_06_03_20_36']
        # image_dir = 'images/alpha_tt_{}'.format(times[0])
        # fault_types = ['code_delay', 'exception']

        # cpu_contention & network_delay
        # times = ['2024_06_03_16_45']
        # image_dir = 'images/alpha_tt_{}'.format(times[0])
        # fault_types = ['cpu_contention', 'network_delay']

        top_k_dict, avg_k_dict = get_all_metrics(times, fault_types, K)
        metric_data = top_k_dict
        metric_data['Avg@5'] = avg_k_dict

        plot_params_and_metrics(image_dir, metric_data, metric_names='Avg@5', fault_types=fault_types)
        plot_params_and_metrics(image_dir, metric_data, metric_names='HR@1', fault_types=fault_types)
        plot_params_and_metrics(image_dir, metric_data, metric_names='HR@2', fault_types=fault_types)
        plot_params_and_metrics(image_dir, metric_data, metric_names='HR@3', fault_types=fault_types)

    def tau_experiment():
        # tau_1: anomaly threshold
        # tau_2 的两种取值差不多
        # set tau_2 = 0.3
        tau_dict = {
            '0.0': ['2024_06_04_21_10', '2024_06_04_21_19'],
            '1.0': ['2024_06_04_20_59', '2024_06_04_21_17'],  # 0.1
            '2.0': ['2024_06_03_20_42', '2024_06_03_20_36'],  # 0.2
            '3.0': ['2024_06_04_21_04', '2024_06_04_21_25'],  # 0.3
            '4.0': ['2024_06_04_21_06', '2024_06_04_21_26'],  # 0.4
            '5.0': ['2024_06_04_21_07', '2024_06_04_21_27'],  # 0.5
            '6.0': ['2024_06_04_21_08', '2024_06_04_21_28'],  # 0.6
            # '7.0': ['', ''],        # 0.7
            # '8.0': ['', ''],        # 0.8
            # '9.0': ['', ''],        # 0.9
            # '10.0': ['', ''],        # 1.0
        }
        # set tau_2 = 0.6
        tau_dict = {
            '0.0': ['2024_06_04_22_50', '2024_06_04_21_43'],
            '1.0': ['2024_06_04_22_44', '2024_06_04_21_42'],  # 0.1
            '2.0': ['2024_06_04_22_48', '2024_06_04_22_41'],  # 0.2
            '3.0': ['2024_06_04_23_18', '2024_06_04_21_40'],  # 0.3
            '4.0': ['2024_06_04_23_20', '2024_06_04_21_39'],  # 0.4
            '5.0': ['2024_06_04_23_22', '2024_06_04_21_38'],  # 0.5
            '6.0': ['2024_06_04_23_24', '2024_06_04_21_37'],  # 0.6
            # '7.0': ['', ''],        # 0.7
            # '8.0': ['', ''],        # 0.8
            # '9.0': ['', ''],        # 0.9
            # '10.0': ['', ''],        # 1.0
        }
        param_name = 'tau_1'
        best_alpha = '0.8'
        image_dir = 'images/tau1_alpha_{}_tau2_{}/'.format(best_alpha, '0.6')
        xlabel = r'$\tau_1$'

        # tau_2: causal threshold
        param_name = 'tau_2'
        # # set tau_1 = 2.0
        # tau_dict = {
        #     '0.0': ['2024_06_04_17_07', '2024_06_04_17_06'],        # 0.0
        #     '0.1': ['2024_06_04_17_11', '2024_06_04_16_15'],        # 0.1
        #     '0.2': ['2024_06_04_17_31', '2024_06_04_16_18'],        # 0.2
        #     '0.3': ['2024_06_03_20_42', '2024_06_03_20_36'],        # 0.3
        #     '0.4': ['2024_06_04_17_34', '2024_06_04_16_22'],        # 0.4
        #     '0.5': ['2024_06_04_17_36', '2024_06_04_16_24'],        # 0.5
        #     '0.6': ['2024_06_04_17_38', '2024_06_04_16_26'],        # 0.6
        #     '0.7': ['2024_06_04_17_41', '2024_06_04_16_27'],        # 0.7
        #     '0.8': ['2024_06_04_17_42', '2024_06_04_16_29'],        # 0.8
        #     '0.9': ['2024_06_04_17_45', '2024_06_04_16_30'],        # 0.9
        #     '1.0': ['2024_06_04_17_47', '2024_06_04_16_33'],        # 1.0
        # }
        # image_dir = 'images/tau2_alpha_{}_tau1_2.0/'.format(best_alpha)
        # set tau_1 = 3.0
        tau_dict = {
            '0.0': ['2024_06_05_15_43', '2024_06_05_16_08'],
            '0.1': ['2024_06_05_15_45', '2024_06_05_16_07'],
            '0.2': ['2024_06_05_15_46', '2024_06_05_16_06'],
            '0.3': ['2024_06_05_15_48', '2024_06_05_16_04'],
            '0.4': ['2024_06_05_15_49', '2024_06_05_16_03'],
            '0.5': ['2024_06_05_15_50', '2024_06_05_16_01'],
            '0.6': ['2024_06_04_23_18', '2024_06_04_21_40'],
            '0.7': ['2024_06_05_15_51', '2024_06_05_16_00'],
            '0.8': ['2024_06_05_15_53', '2024_06_05_15_59'],
            '0.9': ['2024_06_05_15_54', '2024_06_05_15_58'],
            '1.0': ['2024_06_05_15_55', '2024_06_05_15_57'],
        }
        image_dir = 'images/tau2_alpha_{}_tau1_3.0/'.format(best_alpha)
        best_alpha = '0.8'
        xlabel = r'$\tau_2$'

        # {HR@k: {failure_type: {cofficient: top_k}}}
        metric_data = defaultdict(lambda: defaultdict(dict))

        for tau, times in tau_dict.items():
            cur_top_k_dict, cur_avg_k_dict = get_all_metrics(times, fault_types, K)

            for fault_type in fault_types:
                for HR in cur_top_k_dict.keys():
                    metric_data[HR][fault_type][tau] = cur_top_k_dict[HR][fault_type][best_alpha]
                metric_data[f'Avg@{K}'][fault_type][tau] = cur_avg_k_dict[fault_type][best_alpha]

        plot_params_and_metrics(
            image_dir,
            metric_data,
            metric_names=['HR@1', 'HR@2', 'HR@3', 'Avg@5'],
            # metric_names=['HR@1'],
            fault_types=['all'],
            param_name=param_name,
            figsize=(5.5, 3),
            xlabel=xlabel
        )

    alpha_experiment()
    # tau_experiment()

    # data_dict = read_data_for_methods(dataset=dataset)
    # data_dict['[NAME]'] = {'Avg@5': avg_k_dict}
    # data_dict['[NAME]'].update(top_k_dict)

    # plot_radar(result_dir='images', data_dict=data_dict, coefficient='0.9', metric_name='HR@1', fault_types=fault_types)
    # plot_radar(result_dir='images', data_dict=data_dict, coefficient='0.9', metric_name='Avg@5', fault_types=fault_types)
    # plot_params_and_metrics(image_dir, avg_k_dict, metric_name='avg@5', param_name=param_name)
    # plot_params_and_metrics(image_dir, top_k_dict['HR@1'], metric_name='HR@1', param_name=param_name)
    # plot_params_and_metrics(image_dir, top_k_dict['HR@2'], metric_name='HR@2', param_name=param_name)
    # plot_params_and_metrics(image_dir, top_k_dict['HR@3'], metric_name='HR@3', param_name=param_name)


def compute_avg():
    weights = [
        48,
        69,
        72,
        45,
    ]
    # array = [
    #     0.563, 0.708, 0.708, 0.750, 0.813,
    #     0.493, 0.522, 0.580, 0.594, 0.609,
    #     0.292, 0.556, 0.569, 0.625, 0.694,
    #     0.444, 0.489, 0.533, 0.556, 0.578,
    # ]
    array = [
        0.792,
        0.896,
        0.896,
        0.896,
        0.896,
        0.464,
        0.638,
        0.739,
        0.754,
        0.768,
        0.556,
        0.694,
        0.847,
        0.903,
        0.917,
        0.600,
        0.733,
        0.778,
        0.822,
        0.844,
    ]
    array = np.array(array).reshape((4, 5))
    avgs = []
    print(array)
    for i in range(5):
        avg = np.average(a=array[:, i], weights=weights)
        avgs.append(avg)
    print(avgs)
    print(np.mean(avgs))


def compute_time_duration():
    t1 = input('t1: ')
    t2 = input('t2: ')
    t1 = pd.Timestamp(t1)
    t2 = pd.Timestamp(t2)
    if t1 < t2:
        s = (t2 - t1).total_seconds()
    else:
        s = (t1 - t2).total_seconds()
    print('sec: {}'.format(s))
    return s


def filter_rca_results():
    path = 'model-store/aiops22/CustomGAE/rca_2024_05_17_23_24/rca_result_1.0.json'
    with open(path, 'r') as f:
        results = json.load(f)

    fault = 'k8s容器进程中止'
    filter_results = []
    for result in results:
        if result['failure_type'] != fault:
            continue
        if result['cmdb_id'] not in result['cand'][0]['name']:
            filter_results.append(result)

    print('count: {}'.format(len(filter_results)))
    filter_results = sorted(filter_results, key=lambda x: x['timestamp'])
    with open('filter_results.json', 'w') as f:
        json.dump(filter_results, f, indent=4)


def overhead_analysis():

    def read_traces_tt():
        base_dir = '../datasets/Nezha/TrainTicket_2024/2024-04-25'
        trace_dir = os.path.join(base_dir, 'trace')

        # read files
        print('reading traces...')
        trace_dfs = []
        for file in tqdm(sorted(os.listdir(trace_dir))):
            filepath = path.join(trace_dir, file)
            trace_dfs.append(pd.read_csv(filepath))
        trace_df = pd.concat(trace_dfs)
        trace_df = trace_df.drop_duplicates()

        return trace_df

    def read_traces_aiops(nrows=None):
        print('读取traces...')
        traces_dir = '../datasets/aiops_2022/dataset/2022-03-19/cloudbed-1/trace/all/'
        traces_df = pd.read_csv(traces_dir + 'trace_jaeger-span.csv', nrows=nrows)
        return traces_df

    dataset = 'aiops'
    if dataset == 'tt':
        trace_df = read_traces_tt()
    elif dataset == 'aiops':
        trace_df = read_traces_aiops()
    time_data_process = compute_time_duration()

    num_spans = len(trace_df)
    num_spans_per_minute = num_spans / (60 * 24)

    print(
        'num_spans: {}.\n'
        'num_spans_per_minute: {}.\n'
        'data process time: {} s. {} s.'.format(
            num_spans, num_spans_per_minute, time_data_process, time_data_process / (60 * 24)
        )
    )


# compute_time_duration()
# filter_rca_results()
# overhead_analysis()
main()
