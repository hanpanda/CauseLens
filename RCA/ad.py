import os

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from log import Logger

logger = Logger(__name__)


def get_y_true(labels, num_pods, name_to_dgl_id=None):
    # 将标签转为y_true (num_points, num_pods)
    y_true = np.zeros((len(labels), num_pods))
    for i, lbl in enumerate(labels):
        if lbl['failure_type'] == '':
            continue
        if lbl['level'] == 'service':
            pods = [name_to_dgl_id(lbl['cmdb_id'] + suffix) for suffix in ['-0', '-1', '-2', '2-0']]
        else:
            pods = [name_to_dgl_id(lbl['cmdb_id'])]
        y_true[i][pods] = 1

    return y_true


def cal_metrics_for_ad(y_true, y_pred, store_dir, args, dgl_id_to_name=None):
    logger.info('-----------------AD metrics-----------------')
    # 结果调整
    for i in range(y_true.shape[0]):
        for j in range(y_true.shape[1]):
            if y_true[i][j] == 1 and y_pred[i][j] == 1:
                # backward
                k = i
                while k >= 0 and y_true[k][j] == 1:
                    y_pred[k][j] = 1
                    k -= 1
                # forward
                k = i
                while k < y_true.shape[0] and y_true[k][j] == 1:
                    y_pred[k][j] = 1
                    k += 1

    # 计算precision、recall、f1
    df = []
    tn_sum, fp_sum, fn_sum, tp_sum = 0, 0, 0, 0
    y_true = y_true.swapaxes(0, 1)
    y_pred = y_pred.swapaxes(0, 1)
    for i in range(len(y_true)):
        # tn, fp, fn, tp
        tn, fp, fn, tp = confusion_matrix(y_true[i], y_pred[i], labels=[0, 1]).ravel()
        tn_sum += tn
        fp_sum += fp
        fn_sum += fn
        tp_sum += tp

        pr = precision_score(y_true[i], y_pred[i], zero_division=0, average='binary')
        rc = recall_score(y_true[i], y_pred[i], zero_division=0, average='binary')
        f1 = f1_score(y_true[i], y_pred[i], zero_division=0, average='binary')

        num_samples = len(y_true[i])
        num_anomalies = np.sum(y_true[i])
        name = dgl_id_to_name(i, 'pod')

        logger.info(
            'node_name: {}. num_anomalies: {}. num_samples: {}. pr: {:.3f}. rc: {:.3f}. f1: {:.3f}.'.format(
                name, num_anomalies, num_samples, pr, rc, f1
            )
        )

        df.append({
            'id': i,
            'node_name': name,
            'pr': pr,
            'rc': rc,
            'f1': f1,
            'num_anomalies': num_anomalies,
            'num_samples': num_samples,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp
        })

    pr = tp_sum / (tp_sum + fp_sum)
    rc = tp_sum / (tp_sum + fn_sum)
    f1 = 2 * pr * rc / (pr + rc)
    logger.info('summary:\n pr: {:.3f}. rc: {:.3f}. f1: {:.3f}.'.format(pr, rc, f1))

    filepath = os.path.join(
        store_dir,
        '{}_{}_{}_ad_rst.csv'.format(args.test_dates[0], args.test_dates[-1],
                                     str(args.model_file).split('/')[-2])
    )
    df = pd.DataFrame(df).to_csv(filepath, index=False)
    logger.info('Save AD metrics to: \n{}'.format(filepath))
