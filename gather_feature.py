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
import PIL
import torch
import torchvision
import torch.utils.data

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

import matplotlib.pyplot as plt

def to_str(lis):
    s = ""
    for w in lis:
        s = s + str(w).ljust(10," ") + ", "
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    args = parser.parse_args()

    file_list = ['ERM_{"lr"=0*0005}_3','ERM_{"lr"=0*0003}_3','ERM_{"lr"=0*0001}_2']
    file_dir = []

    num_classes = 65
    env_list = ['env{}'.format(i) for i in range(4)]
    step_list = [500*(i+1) for i in range(10)]
    before = False
    'extracted_500new_L1__mean_save.npy'
    plotx = np.array([])
    ploty = np.array([])
    plotc = np.array([])
    for dir in file_dir:
        for file_name in file_list:
            for step in step_list:
                if before:
                    npy_file_name = 'logs/{}/{}/extracted_{}before_new_L1__mean_save.npy'.format(dir,file_name,step)
                else:
                    npy_file_name = 'logs/{}/{}/extracted_{}new_L1__mean_save.npy'.format(dir,file_name,step)
                compute_result = np.load(npy_file_name, allow_pickle=True).item()
                # train_results, label_num*feature_num
                train_distance = compute_result['train_results'].max(axis=0).reshape(-1)
                test_distance = compute_result['test_results'].max(axis=0).reshape(-1)
                train_info = compute_result['train_info'].reshape(-1)

                plotx = np.concatenate([plotx, train_distance])
                ploty = np.concatenate([ploty, test_distance])
                plotc = np.concatenate([plotc, train_info])
                l1 = to_str(train_distance)
                l1 += '\n'
                l1 += to_str(test_distance)
                l1 += '\n'
                with open('expansion.csv', 'a+') as f:
                    f.write(l1)
            fig = plt.figure()
            plt.scatter(plotx, ploty, c=plotc)
            plt.savefig('logs/{}/expansion.jpg'.format(dir))