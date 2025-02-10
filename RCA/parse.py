import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='', help='模型名称。')
    parser.add_argument('--model_file', default='', type=str, help='模型文件路径。测试时使用。')
    parser.add_argument('--mode', type=str, default='train', help='运行模式。train/rca_prepare/rca。')

    # 常规训练超参数
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--epoch', type=int, default=100, help='training epoch')
    parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')
    parser.add_argument('--valid_ratio', default=0.2, type=float, help='验证集占训练集比例。')

    # 数据集参数: 使用的时间段、故障类型、特征、预处理方式。
    parser.add_argument('--dataset', default='aiops_2022', type=str, help='数据集名称。')
    parser.add_argument('--data_dir', default='../datasets/aiops_2022/graph/', type=str, help='数据集所在目录。')
    parser.add_argument('--failure_types', nargs='+', help='所使用数据的故障类型。正常数据则为""。')
    parser.add_argument('--failure_duration', type=int, help='故障持续时间。单位：分钟。')
    parser.add_argument('--train_dates', nargs='+', help='用于训练的数据的日期。')
    parser.add_argument('--test_dates', nargs='+', help='用于测试的数据的日期。')
    parser.add_argument('--used_etypes', nargs='+', help='所使用边的类型。')
    parser.add_argument('--nfeat_select', type=str, help='不同类型节点特征选择。json字符串。')
    parser.add_argument('--node_scale_type', type=str, default='nodewise', help='数据集的节点特征缩放方式。nodewise/global。')
    parser.add_argument('--edge_scale_type', type=str, default='edgewise', help='数据集的边特征缩放方式。edgewise/global。')
    parser.add_argument('--add_self_loop', action='store_true', help='是否添加自环边。')
    parser.add_argument('--edge_reverse', action='store_true', help='是否需要将边反向。')
    parser.add_argument('--log_before_scale', action='store_true', help='是否使用取对数。')
    parser.add_argument('--process_miss', default='interpolate', type=str, help='使用什么方式填补缺失值。interpolate/zero。')
    parser.add_argument('--process_extreme', action='store_true', help='是否处理数据中的极端值。')
    parser.add_argument('--k_sigma', type=float, default=3, help='k值。用于检测极端值。')
    parser.add_argument('--use_split_info', action='store_true', help='是否使用自定义分割数据集的文件。')

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
    parser.add_argument('--proj_feats', type=int, default=10, help='投影的特征空间维度。')
    parser.add_argument('--mask', action='store_true', help='是否对于图节点使用mask。')
    parser.add_argument('--mask_feat_type', type=str, help='mask节点特征。zero, median, mean。', default='mean')
    parser.add_argument('--loss_type', type=str, help='loss函数类型。', default='mse')
    parser.add_argument('--use_edge_feats', action='store_true', help='是否使用边上的特征。')
    parser.add_argument('--use_etype_feats', action='store_true', help='是否使用边的类型嵌入。')
    parser.add_argument('--embedding', action='store_true', help='是否为节点设置embedding以计算注意力分数。')
    parser.add_argument('--recon_ntypes', nargs='+', help='训练时重建的节点类型。api, pod。')

    # rca 参数
    parser.add_argument('--window', type=int, default=12 * 60, help='异常分数规范化窗口。')
    parser.add_argument('--score_type', type=str, default='pred_and_mean', help='异常分数类型。')
    parser.add_argument('--causal_threshold', type=float, default=0.6, help='阈值: 异常分数降低比例。')
    parser.add_argument('--ad_threshold', type=float, default=3.0, help='阈值: 异常分数。')
    parser.add_argument(
        '--dist_type_for_counterfactual', type=str, default='euclidean', help='欧式(euclidean)、马氏距离(mahalanobis)。'
    )
    parser.add_argument('--dist_direction', type=str, default='gt', help='both/gt/lt')
    parser.add_argument('--alpha', type=float, default=0.8, help='控制重建误差rca分数和反事实rca分数的占比。')

    # 废弃：ad 参数
    parser.add_argument('--threshold_1', type=int, default=0)
    parser.add_argument('--threshold_2', type=int, default=8)
    parser.add_argument('--weight_score', action='store_true')

    # debug & 实验
    parser.add_argument('--debug', action='store_true', help='是否打印debug文件。')
    parser.add_argument('--alpha_sensity_expr', action='store_true', help='是否进行α敏感性实验。')

    args = parser.parse_args()

    return args
