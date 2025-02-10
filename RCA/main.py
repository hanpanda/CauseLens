import json
import os

from run import train_model, test_model_for_rca, compute_statistics_for_rca
from log import Logger
from utils import get_cur_time, set_random_seed
from parse import parse_args

RUNTIME_DIR_NAME = 'runtime'
logger = Logger(__name__)

if __name__ == '__main__':
    args = parse_args()
    logger.info('mode: {}'.format(args.mode))

    # 设置随机种子
    set_random_seed(args.seed)

    # 将包含字典的字符串解析为字典对象
    try:
        args.nfeat_select = json.loads(args.nfeat_select)
    except json.JSONDecodeError as e:
        logger.error(f"无法解析输入的字符串为字典: {e}")
        exit(1)

    # 记录参数
    for var, value in vars(args).items():
        logger.info('{}: {}.'.format(var, value))

    # 存储目录创建
    cur_time = get_cur_time()

    base_dir = f'./models/{args.dataset}/{args.model}/'
    model_dir = os.path.dirname(args.model_file)
    runtime_dir = os.path.join(model_dir, RUNTIME_DIR_NAME)

    if args.mode == 'train':
        model_dir = os.path.join(base_dir, f'model_{cur_time}')
        runtime_dir = os.path.join(model_dir, RUNTIME_DIR_NAME)

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(runtime_dir, exist_ok=True)

    # 根据不同模式运行
    model = None
    if args.mode == 'train':
        logger.info('Training...')
        train_model(args, model_dir, runtime_dir)
    elif args.mode == 'rca':
        logger.info('RCA Testing...')
        test_model_for_rca(args, model, runtime_dir)
    elif args.mode == 'rca_prepare':
        compute_statistics_for_rca(args, runtime_dir=runtime_dir)
