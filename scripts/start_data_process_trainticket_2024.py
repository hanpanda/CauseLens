import os

dates = ['2024-04-04', '2024-04-05', '2024-04-06']
target_dir = '/root/lqh/multimodal-RCA/datasets/TrainTicket_2024/graph_1'

for date in dates:
    cmd = 'nohup python data_process_trainticket_2024.py --date {} --target_dir {} &'.format(date, target_dir)
    print(cmd)
    os.system(cmd)

dates = ['2024-04-22', '2024-04-23', '2024-04-24', '2024-04-25']
target_dir = '/root/lqh/multimodal-RCA/datasets/TrainTicket_2024/graph_2'

for date in dates:
    cmd = 'nohup python data_process_trainticket_2024.py --date {} --target_dir {} &'.format(date, target_dir)
    print(cmd)
    os.system(cmd)
