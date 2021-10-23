import pandas as pd
import os
import re
import json
import pandas as pd
import math

datasets = ['VLCS','PACS','OfficeHome']
env_list = [0,1,2,3]

if not os.path.exists('collection'):
    os.mkdir('collection')
if not os.path.exists('renamed'):
    os.mkdir('renamed')

def floatequ(a,b):
    return math.fabs(a-b) < 1e-8

for dataset in datasets:
    for env in env_list:
        # collect features
        dir_list = ['{}_CORAL_test_env{}'.format(dataset, env), '{}_ERM_test_env{}'.format(dataset, env),
                    '{}_GroupDRO_test_env{}'.format(dataset, env),
                    '{}_IRM_test_env{}'.format(dataset, env),
                    '{}_Mixup_test_env{}'.format(dataset, env)
            ]
        
        output_file = 'collection/data_collection_{}_env{}.csv'.format(dataset, env)
        dt = None
        for dir0 in dir_list:
            if not os.path.isdir(dir0) or 'env' not in dir0:
                continue
            file_list = sorted(os.listdir(dir0))
            for sub_name in file_list:
                if not os.path.isdir(os.path.join(dir0, sub_name)):
                    continue
                sub_file_list = os.listdir(os.path.join(dir0, sub_name))
                try:
                    fname = 'result.csv'
                    res = pd.read_csv(os.path.join(dir0, sub_name, fname), index_col=False)
                    if dt is None:
                        dt = res
                    else:
                        dt = pd.concat([dt, res], axis=0)
                        #print('count')
                    print('collect {}'.format(sub_name))
                except FileNotFoundError as e:
                    print(e)
                    print(dir0, sub_name)
            if dt is not None:
                dt.to_csv(output_file, index=False)

        # rename columns
        output_file = 'renamed/collection_{}_test{}_feanum.csv'.format(dataset, env)
        read_file = 'collection/data_collection_{}_env{}.csv'.format(dataset, env) 
        if not os.path.exists(read_file):
            continue
        dt = None
        read_file_frame = pd.read_csv(read_file)
        names = read_file_frame['name']
        lr_list = []
        step_list = []
        pn_list = []
        sd_list = []
        method_list = []
        pn_name_list = ["groupdro_eta", "irm_lambda", "mixup_alpha", "vrex_lambda", 'mmd_gamma', 'no_pn']
        index_list = []

        idx = -1
        for name in names:
            idx += 1
            tp = re.search(r'\{[^\}]+\}', name)
            dp = name[tp.span()[0]:tp.span()[1]]
            dp = dp.replace('=', ':').replace('*', '.').replace(r'_"', r',"')
            dt = json.loads(dp)
            try:
                assert 'lr' in dt
            except Exception as e:
                print(e)

            tmp_lr = dt['lr']
            if not (floatequ(tmp_lr, 1e-4) or floatequ(tmp_lr, 5e-5)):
                continue
            for pn in pn_name_list:
                if pn == 'no_pn':
                    tmp_pn = 0
                if pn in dt:
                    tmp_pn = dt[pn]
                    break

            tp = re.search(r'step_[0-9]*', name)
            tmp_step = int(name[tp.span()[0] + 5:tp.span()[1]])
            if tmp_step not in [2500,5000]:
                continue

            tp = re.search(r'_[0-9]*_', name)
            tmp_sd = int(name[tp.span()[0] + 1:tp.span()[1] - 1])
            if tmp_sd not in [0,1,2,3,4]:
                continue

            tp = re.search(r'\A[A-Za-z]*_', name)
            dp = name[tp.span()[0]: tp.span()[1] - 1]
            method_list.append(dp)
            lr_list.append(tmp_lr)
            pn_list.append(tmp_pn)
            step_list.append(tmp_step)
            sd_list.append(tmp_sd)

            index_list.append(idx)


        read_file_frame2 = read_file_frame.iloc[index_list]

        read_file_frame2['algorithm'] = method_list
        read_file_frame2['step'] = step_list
        read_file_frame2['lr'] = lr_list
        read_file_frame2['penalty'] = pn_list

        read_file_frame2['seed'] = sd_list

        first_list = ['algorithm', 'step', 'lr', 'penalty', 'seed']
        last_list = ['name']

        column_list = list(read_file_frame2)
        for _ in first_list:
            column_list.remove(_)
        for _ in last_list:
            column_list.remove(_)

        column_list = first_list + column_list + last_list
        read_file_frame2 = read_file_frame2[column_list]

        thr_list = []
        for thr in range(9):
            thr_list.append(round(thr * 0.05, 2))
        cls = {}
        for thr in thr_list:
            cls['test_dis_{}'.format(thr)] = ''
            cls['train_info_{}'.format(thr)] = ''
            cls['train_dis_{}'.format(thr)] = thr
        read_file_frame2 = read_file_frame2.rename(columns=cls)
        read_file_frame2.index = [_ for _ in range(len(index_list))]
        read_file_frame2.to_csv(output_file)

        #print(cls)
        #print(read_file_frame2.columns)
