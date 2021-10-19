import argparse
import collections
import json
import os
import random
import sys
import time
import uuid
import copy

import numpy as np

import itertools
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
matplotlib.use('AGG')



def to_str(lis):
    s = ""
    for w in lis:
        s = s + str(w).ljust(10," ") + ", "
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    args = parser.parse_args()
    lr_list = ['0*0001','0*0003','0*0005']
    file_dir = ['OfficeHome_ERM_test_env1','OfficeHome_GroupDRO_test_env1','OfficeHome_IRM_test_env1',
                'OfficeHome_VREx_test_env1','OfficeHome_Mixup_test_env1']

    file_list ={'OfficeHome_ERM_test_env1': ['ERM_{"lr"=0*0005}_3','ERM_{"lr"=0*0003}_3','ERM_{"lr"=0*0001}_2'],
                'OfficeHome_GroupDRO_test_env1':['GroupDRO_{{"lr"={}_"groupdro_eta"={}}}_0'.format(lr,eta)
                                      for lr,eta in itertools.product(lr_list,['0*01','0*1','1','10'])],
                'OfficeHome_IRM_test_env1':['IRM_{{"lr"={}_"irm_penalty_anneal_iters"=1000_"irm_lambda"={}}}_0'.format(lr,pn)
                                for lr,pn in itertools.product(lr_list,['1','10','100','1000'])],
                'OfficeHome_VREx_test_env1':['VREx_{{"lr"={}_"vrex_anneal_iter"=1000_"vrex_lambda"={}}}_0'.format(lr,pn)
                                for lr,pn in itertools.product(lr_list,['1','10','100','1000'])],
                'OfficeHome_Mixup_test_env1':['Mixup_{{"lr"={}_"mixup_alpha"={}}}_0'.format(lr,alpha)
                                   for lr,alpha in itertools.product(lr_list,['0*1','0*2','0*3'])]
                }

    num_classes = 65
    env_list = ['env{}'.format(i) for i in range(4)]
    step_list_all = {'OfficeHome_ERM_test_env1':[500*(i+1) for i in range(10)],
                 'OfficeHome_GroupDRO_test_env1':[500*(i+1) for i in range(10)],
                 'OfficeHome_Mixup_test_env1':[500*(i+1) for i in range(10)],
                 'OfficeHome_IRM_test_env1':[1200 + 400*i for i in range(10)],
                 'OfficeHome_VREx_test_env1':[1200 + 400*i for i in range(10)]}
    before = True
    use_csv = False
    #print(step_list_all)
    if not use_csv:
        plotx = np.array([])
        ploty = np.array([])
        plotc = np.array([])
        str0 = ''
        for i in range(2048):
            str0 += ','
        str0 += '\n'
        with open('logs/expansion{}.csv'.format('_before' if before else ''), 'w+') as f:
            f.write(str0)
        for dir0 in file_dir:
            step_list = step_list_all[dir0]
            print(dir0,step_list)
            for file_name in file_list[dir0]:

                for step in step_list:
                    if before:
                        npy_file_name = 'logs/{}/{}/extracted_{}before_new_L1__mean_save.npy'.format(dir0, file_name,
                                                                                                     step)
                    else:
                        npy_file_name = 'logs/{}/{}/extracted_{}new_L1__mean_save.npy'.format(dir0, file_name, step)
                    try:
                        compute_result = np.load(npy_file_name, allow_pickle=True).item()
                    except Exception as e:
                        print(e)
                        continue
                    # train_results, label_num*feature_num
                    train_distance = compute_result['train_results'].max(axis=0).reshape(-1)
                    test_distance = compute_result['test_results'].max(axis=0).reshape(-1)
                    train_info = compute_result['train_info'].reshape(-1)

                    plotx = np.concatenate([plotx, train_distance])
                    ploty = np.concatenate([ploty, test_distance])
                    plotc = np.concatenate([plotc, train_info])
                    l1 = npy_file_name + ','
                    l1 += to_str(train_distance)
                    l1 += '\n'
                    l1 += npy_file_name + ','
                    l1 += to_str(test_distance)
                    l1 += '\n'
                    l1 += npy_file_name + ','
                    l1 += to_str(train_info)
                    l1 += '\n'
                    with open('logs/expansion{}.csv'.format('_before' if before else '').format(dir0), 'a+') as f:
                        f.write(l1)
        fig = plt.figure()
        plt.title('all', fontsize=20)
        plt.scatter(plotx, ploty, s=4., c=plotc)
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.savefig('logs/expansion{}.jpg'.format('_before' if before else ''))
        plotx = np.array([])
        ploty = np.array([])
        plotc = np.array([])

    else:
        plotx = np.array([])
        ploty = np.array([])
        plotc = np.array([])
        df = pd.read_csv('expansion.csv')
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        print(df)
        assert df.shape[0]%3 == 0,print(df.shape[0])
        for i in range(len(df)//3):
            plotx = np.concatenate([plotx, np.array(df.iloc[3*i][1:],dtype=np.float)])
            ploty = np.concatenate([ploty, np.array(df.iloc[3*i+1][1:],dtype=np.float)])
            plotc = np.concatenate([plotc, np.array(df.iloc[3*i+2][1:],dtype=np.float)])
        fig = plt.figure()
        plt.title(dir, fontsize=20)
        plt.scatter(plotx, ploty, s=4., c=plotc)
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.savefig('expansion_all.jpg')