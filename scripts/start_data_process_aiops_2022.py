import os

dates = [
    # '2022-03-20',
    '2022-03-21',
    '2022-03-22',
    '2022-03-24',
    '2022-03-26',
    '2022-03-28',
    '2022-03-29',
    '2022-03-30',
    '2022-03-31',
    '2022-04-01',
    '2022-04-02'
]

cloudbeds = ['cloudbed-1']

for cloudbed in cloudbeds:
    for date in dates:
        cmd = 'nohup python data_process_aiops22.py --date {} --cloudbed {} &'.format(date, cloudbed)
        print(cmd)
        os.system(cmd)
