# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Run sweeps
"""

import argparse
import copy
import getpass
import hashlib
import json
import os
import random
import shutil
import time
import uuid

import numpy as np
import torch


from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed import command_launchers

import tqdm
import shlex
import itertools

class Job:
    NOT_LAUNCHED = 'Not launched'
    INCOMPLETE = 'Incomplete'
    DONE = 'Done'

    def __init__(self, train_args):
        self.output_dir = train_args['output_dir']
        self.train_args = copy.deepcopy(train_args)
        self.extract = self.output_dir + '/' + train_args['extract_feature']
        command = ['python', 'main.py']
        for k, v in sorted(self.train_args.items()):
            if v == '':
                command.append(f'--{k} ')
                continue
            if isinstance(v, list):
                v = ' '.join([str(v_) for v_ in v])
            elif isinstance(v, str):
                v = shlex.quote(v)
            command.append(f'--{k} {v}')
        self.command_str = ' '.join(command)

        if os.path.exists(os.path.join(self.extract, 'done')):
            self.state = Job.DONE
        elif os.path.exists(self.extract):
            self.state = Job.INCOMPLETE
        else:
            self.state = Job.NOT_LAUNCHED

    def __str__(self):
        job_info = (self.train_args['dataset'],
            self.train_args['algorithm'],
            self.train_args['test_envs'],self.train_args['extract_feature'])
        return '{}: {} {}'.format(
            self.state,
            self.extract,
            job_info)

    @staticmethod
    def launch(jobs, launcher_fn,available_list=[0,1,2,3]):
        print('Launching...')
        jobs = jobs.copy()
        #np.random.shuffle(jobs)
        print('Making job directories:')
        for job in tqdm.tqdm(jobs, leave=False):
            os.makedirs(job.output_dir, exist_ok=True)
        commands = [job.command_str for job in jobs]
        launcher_fn(commands,available_list)
        print(f'Launched {len(jobs)} jobs!')

    @staticmethod
    def delete(jobs):
        print('Deleting...')
        for job in jobs:
            if os.path.isdir(job.extract):
                shutil.rmtree(job.extract)
        print(f'Deleted {len(jobs)} jobs!')

def all_test_env_combinations(n):
    """
    For a dataset with n >= 3 envs, return all combinations of 1 and 2 test
    envs.
    """
    assert(n >= 3)
    for i in range(n):
        yield [i]
        for j in range(i+1, n):
            yield [i, j]

def make_args_list(n_trials, dataset_names, algorithms, n_hparams_from, n_hparams, steps,
    data_dir, task, holdout_fraction, single_test_envs, hparams):
    args_list = []
    for trial_seed in range(n_trials):
        for dataset in dataset_names:
            for algorithm in algorithms:
                if single_test_envs:
                    all_test_envs = [
                        [i] for i in range(datasets.num_environments(dataset))]
                else:
                    all_test_envs = all_test_env_combinations(
                        datasets.num_environments(dataset))
                for test_envs in all_test_envs:
                    for hparams_seed in range(n_hparams_from, n_hparams):
                        train_args = {}
                        train_args['dataset'] = dataset
                        train_args['algorithm'] = algorithm
                        train_args['test_envs'] = test_envs
                        train_args['holdout_fraction'] = holdout_fraction
                        train_args['hparams_seed'] = hparams_seed
                        train_args['data_dir'] = data_dir
                        train_args['task'] = task
                        train_args['trial_seed'] = trial_seed
                        train_args['seed'] = misc.seed_hash(dataset,
                            algorithm, test_envs, hparams_seed, trial_seed)
                        if steps is not None:
                            train_args['steps'] = steps
                        if hparams is not None:
                            train_args['hparams'] = hparams
                        args_list.append(train_args)
    return args_list

def ask_for_confirmation():
    response = input('Are you sure? (y/n) ')
    if not response.lower().strip()[:1] == "y":
        print('Nevermind!')
        exit(0)

DATASETS = [d for d in datasets.DATASETS if "Debug" not in d]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a sweep')
    parser.add_argument('command', choices=['launch', 'delete_incomplete','just_view'])
    parser.add_argument('--datasets', nargs='+', type=str, default=DATASETS)
    parser.add_argument('--algorithms', nargs='+', type=str, default=algorithms.ALGORITHMS)
    parser.add_argument('--task', type=str, default="domain_generalization")
    parser.add_argument('--n_hparams_from', type=int, default=0)
    parser.add_argument('--n_hparams', type=int, default=20)
    #parser.add_argument('--output_dir', type=str, required=True)
    #parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_trials', type=int, default=3)
    parser.add_argument('--command_launcher', type=str, required=True)
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--hparams', type=str, default=None)
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--single_test_envs', action='store_true')
    parser.add_argument('--skip_confirmation', action='store_true')
    parser.add_argument('--train_algorithm',type=str, default='')
    args = parser.parse_args()
    debug = False

    available_list = [1,2,3]
    algorithm_dict = {}
    # test_env0
    # ERM Mixup in 171
    # other on ctl

    # test_env1

    # mixup 1e-4, 3e-4,working on 171,
    # VREx 1e-4, working on 171
    # ERM,mixup 5e-5 working on 175
    # other on 173

    # test_env2
    # all in 173
    #

    # test_env3
    # ERM, Mixup,in ctl4
    # GroupDRO,IRM ?  in ctl3
    # VREx  in 173

    #
    # algorithm_dict['ERM'] = {'times': 5, 'hparam': {'lr': [1e-4, 3e-4, 5e-4]},
    #                                                     'start_step':0}
    # algorithm_dict['Mixup'] = {'times': 5, 'hparam': {'lr': [1e-4, 5e-5],
    #                                                   'mixup_alpha': [0.1, 0.2]},
    #                            'start_step': 0,'freq':2500}
    # algorithm_dict['GroupDRO'] = {'times':5,'hparam':{'lr':[1e-4,5e-5],
    #                                                   'groupdro_eta':[0.01,0.1]},
    #                               'start_step':0,'freq':2500}
    # algorithm_dict['IRM'] = {'times':5,
    #                           'hparam':{'lr':[1e-4],
    #                                     'irm_penalty_anneal_iters':[1000],'irm_lambda':[1,10]},
    #                            'start_step':1000,'freq':2500}
    # algorithm_dict['VREx'] = {'times':5,
    #                           'hparam':{'lr':[1e-4,3e-4,5e-4],
    #                                     'vrex_anneal_iter':[1000],'vrex_lambda':[1,10,100,1000]},
    #                            'start_step':1000}

    algorithm_dict['CORAL'] = {'times':5,
                               'hparam':{'lr':[1e-4,5e-5],
                                         'mmd_gamma':[0.01,0.1,1,10]},
                               'start_step':0,'freq':500}
    #
    # algorithm_dict['ERM'] = {'times': 5, 'hparam': {'lr': [1e-4,5e-5]},
    #                                                     'start_step':0,'freq':2500}

    if args.train_algorithm:
        algorithm_dict_tmp = {}
        args.train_algorithm = args.train_algorithm.split(',')
        for algorithm in args.train_algorithm:
            algorithm_dict_tmp[algorithm] = copy.deepcopy(algorithm_dict[algorithm])
        algorithm_dict = algorithm_dict_tmp

    val_test = False
    if val_test:
        algorithm_dict = {}
        algorithm_dict['ERM'] = {'times': 1, 'hparam': {'lr': [1e-4, 3e-4, 5e-4]},
                                                'start_step':0,}

    args_list = []
    dataset_list = ['ColoredMNIST']

    for data_set in dataset_list:
        if data_set == 'OfficeHome':
            test_env_list = [0]
        elif data_set == 'VLCS':
            test_env_list = [0,1,2,3]
        elif data_set == 'PACS':
            test_env_list = [0,1,2,3]
        elif data_set == 'ColoredMNIST':
            test_env_list = [2]
        for test_env in test_env_list:
            for alg in algorithm_dict:
                hparams = {}
                train_args = {}
                train_args['algorithm'] = alg
                if os.path.exists('domainbed/{}'.format(data_set)):
                    train_args['data_dir'] = 'domainbed'
                else:
                    train_args['data_dir'] = 'domainbed/datasets'
                train_args['algorithm'] = alg
                train_args['dataset'] = data_set
                train_args['test_envs'] = test_env
                train_args['steps'] = 1001 if data_set == 'ColoredMNIST' else 5001
                train_args['start_step'] = algorithm_dict[alg]['start_step']
                train_args['output_dir'] = 'logs/{}_{}_test_env{}'.format(data_set,alg, test_env) if not val_test else \
                    'logs/val_test_{}_test_env{}'.format(alg, test_env)
                if val_test:
                    train_args['val'] = 'out'
                train_args['output_result_file'] = 'result.csv'
                if 'freq' in algorithm_dict[alg]:
                    train_args['checkpoint_freq'] = algorithm_dict[alg]['freq']
                else:
                    train_args['checkpoint_freq'] = (train_args['steps'] - train_args['start_step'] - 1) // 10
                train_args['save_feature_every_checkpoint'] = ''
                train_args['skip_model_save'] = ''
                train_args['follow_plot'] = ''
                param_iter = itertools.product(*list(algorithm_dict[alg]['hparam'].values()))
                para_title = algorithm_dict[alg]['hparam'].keys()
                for para_comb in param_iter:
                    # exp_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                    for i, key in enumerate(para_title):
                        hparams[key] = para_comb[i]
                    raw_hparams = json.dumps(hparams)
                    hparam_str = raw_hparams.replace('.', '*').replace(':', '=').replace(' ', '').replace(',', '_')
                    full_times = algorithm_dict[alg]['times']
                    train_args['hparams'] = raw_hparams
                    for times in range(full_times):
                        file_name = "%s_%s_%s" % (alg, hparam_str, times) if not debug else \
                            "debug_%s_%s_%s" % (alg, hparam_str, times)
                        train_args['extract_feature'] = file_name
                        train_args['trial_seed'] = random.randint(100000, 999999)
                        args_list.append(copy.deepcopy(train_args))


    jobs = [Job(train_args) for train_args in args_list]

    for job in jobs:
        print(job)
    print("{} jobs: {} done, {} incomplete, {} not launched.".format(
        len(jobs),
        len([j for j in jobs if j.state == Job.DONE]),
        len([j for j in jobs if j.state == Job.INCOMPLETE]),
        len([j for j in jobs if j.state == Job.NOT_LAUNCHED]))
    )

    if args.command == 'launch':
        to_launch = [j for j in jobs if j.state == Job.NOT_LAUNCHED]
        print(f'About to launch {len(to_launch)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        launcher_fn = command_launchers.REGISTRY[args.command_launcher]
        Job.launch(to_launch, launcher_fn,available_list)

    elif args.command == 'delete_incomplete':
        to_delete = [j for j in jobs if j.state == Job.INCOMPLETE]
        print(f'About to delete {len(to_delete)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        Job.delete(to_delete)
    elif args.command == 'just_view':
        pass