# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

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
from domainbed.feature_checker import feature_extractor_for_pipline,feature_extractor_for_train
import domainbed.matrix_opt_for_train as moft
from logs.matrix_optimizer_new import opt_kde


def torch_to_numpy(d):
    return {
        key: d[key].cpu().numpy()
        for key in d.keys()
        if d[key] is not None
    }

def to_str(lis):
    s = ""
    for w in lis:
        s = s + str(w).ljust(10," ") + ", "
    return s


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--task', type=str, default="domain_generalization",
                        help='domain_generalization | domain_adaptation')
    parser.add_argument('--hparams', type=str,
                        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
                        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
                        help='Trial number (used for seeding split_dataset and '
                             'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
                        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=1000,
                        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output/")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0)
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_feature_every_checkpoint', action='store_true')
    parser.add_argument('--extract_feature', type=str, default=None)  # 是否extract每个特征的分布
    parser.add_argument('--output_result_file', type=str, default=None)  # 是否extract每个特征的分布
    parser.add_argument('--follow_plot', action='store_true')
    parser.add_argument('--start_step',type=int,default=0)
    parser.add_argument('--val',type = str, default='in')


    args = parser.parse_args()
    threshold_list = [round(0.05 * i, 2) for i in range(9)]
    title_flag = True
    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None
    if args.extract_feature is not None:
        args.output_dir = args.output_dir + args.extract_feature if args.output_dir[
                                                                        -1] == "/" else args.output_dir + "/" + args.extract_feature

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(
            args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
                                                  misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
                                               args.test_envs, hparams)
    else:
        raise NotImplementedError

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []

        out, in_ = misc.split_dataset(env,
                                      int(len(env) * args.holdout_fraction),
                                      misc.seed_hash(args.trial_seed, env_i))

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                                          int(len(in_) * args.uda_holdout_fraction),
                                          misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]

    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)
        if i in args.test_envs]

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=256,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]
    eval_weights = [None for _, weights in (
            in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
                         for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
                          for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
                          for i in range(len(uda_splits))]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
                                len(dataset) - len(args.test_envs), hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env) / hparams['batch_size']
                           for env, _ in in_splits])

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ


    def save_checkpoint(filename):
        import copy
        cpa = copy.deepcopy(algorithm)
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": cpa.cpu().state_dict()
        }
        del cpa
        torch.save(save_dict, os.path.join(args.output_dir, filename))


    last_results_keys = None

    for step in range(start_step, n_steps):
        step_start_time = time.time()
        minibatches_device = [(x.to(device), y.to(device))
                              for x, y in next(train_minibatches_iterator)]
        # print(len(minibatches_device))
        if args.task == "domain_adaptation":
            uda_device = [x.to(device)
                          for x, _ in next(uda_minibatches_iterator)]
        else:
            uda_device = None

        if args.algorithm == 'CutERM' and step==hparams['cut_step']and step not in [0,n_steps-1]:
            if hparams['cut_percent'] > 1e-8:
                datas = feature_extractor_for_train(algorithm, zip(
                    eval_loader_names, eval_loaders), device, dataset.num_classes)
                env_list = ['env{}'.format(i) for i in range(len(dataset))]
                train_env = copy.deepcopy(env_list)
                for ev in args.test_envs:
                    train_env.remove('env{}'.format(ev))
                feature_num = 512 if hparams['resnet18'] else 2048
                opt_for_train = moft.opt_kde(env_list, train_env, dataset.num_classes,
                                        feature_num, datas, percent=hparams['cut_percent'],
                                        sample_size=1000, device=device)
                opt_for_train.pca()
                trans_matrix = opt_for_train.forward(cal_info=True, set_mask=True)
            else:
                feature_num = 512 if hparams['resnet18'] else 2048
                trans_matrix = torch.eye(feature_num,device=device)
            algorithm.update_classifer(trans_matrix)


        step_vals = algorithm.update(minibatches_device, uda_device)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)
        if ((step % checkpoint_freq == 0) or (step == n_steps - 1)) and (step > args.start_step):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for name, loader, weights in evals:
                start = time.time()
                loaderlen = len(loader)
                acc = misc.accuracy(algorithm, loader, weights, device)
                # print("eavl " + name + " with loader len " + str(loaderlen) + " use time " + str(round(time.time()-start,3)))
                results[name + '_acc'] = acc

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                           colwidth=12)
            if args.dataset == 'ColoredMNIST':
                store_list = ['env0_in_acc','env0_out_acc',	'env1_in_acc',
                          'env1_out_acc','env2_in_acc',
                          'env2_out_acc']
            else:
                store_list = ['env0_in_acc','env0_out_acc',	'env1_in_acc',
                          'env1_out_acc','env2_in_acc',
                          'env2_out_acc','env3_in_acc','env3_out_acc']

            if args.output_result_file is not None:
                # print('enter this step')
                assert args.extract_feature is not None
                if title_flag and not os.path.exists(args.output_dir +'/'+ args.output_result_file):
                    title = "name"
                    for key in store_list:
                        title += ',' + key
                    for thr in threshold_list:
                        title += ',train_dis_{},test_dis_{},train_info_{},feature_num_{}'.format(thr,thr,thr,thr)
                    title += '\n'
                    with open(args.output_dir +'/'+ args.output_result_file, 'a+') as f:
                        f.write(title)
                    print('csv file created')
                    title = "name"
                    for key in store_list:
                        title += ',' + key
                    for thr in threshold_list:
                        title += ',train_dis_{},test_dis_{},train_info_{},feature_num_{}'.format(thr, thr, thr,thr)
                    title += '\n'
                    with open(args.output_dir + '/' + 'before_'+args.output_result_file, 'a+') as f:
                        f.write(title)
                    print('before csv file created')
                title_flag = False

                with open(args.output_dir +'/'+ args.output_result_file, 'a+') as f:
                    res = args.extract_feature + '_step_{},'.format(step) + \
                          str([results[key] for key in store_list])[1:-1] + ','
                    if not args.follow_plot:
                        res += '\n'
                    f.write(res)
                with open(args.output_dir +'/'+ 'before_'+args.output_result_file, 'a+') as f:
                    res = args.extract_feature + '_step_{},'.format(step) + \
                          str([results[key] for key in store_list])[1:-1] + ','
                    if not args.follow_plot:
                        res += '\n'
                    f.write(res)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_feature_every_checkpoint:
                if args.extract_feature is not None:
                    #print('________start feature extract_________')
                    if args.output_dir[-1] == '/':
                        marker = args.output_dir + "extracted_{}".format(step)
                    else:
                        marker = args.output_dir + "/" + "extracted_{}".format(step)
                    datas = feature_extractor_for_pipline(algorithm, zip(
                        eval_loader_names, eval_loaders), device, dataset.num_classes, marker,val=args.val)
                    env_list = ['env{}'.format(i) for i in range(len(dataset))]
                    train_env = copy.deepcopy(env_list)
                    for ev in args.test_envs:
                        train_env.remove('env{}'.format(ev))
                    if args.dataset == 'ColoredMNIST':
                        feature_num = 128
                    else:
                        feature_num = 512 if hparams['resnet18'] else 2048
                    opt_for_pipline = opt_kde(env_list, train_env, dataset.num_classes,
                                            feature_num, datas, sample_size=10000, device=device)
                    compute_result = torch_to_numpy(
                        opt_for_pipline.forward(cal_info=True, use_mean_info=True))
                    compute_result['eig_value'] = opt_for_pipline.eig_val()

                    mmstr = '_mean'
                    new_for_save = np.array(compute_result)
                    np.save(marker + "before_new_L1_" + mmstr + "_save.npy", new_for_save)
                    del new_for_save
                    train_distance = compute_result['train_results'].max(axis=0)
                    test_distance = compute_result['test_results'].max(axis=0)
                    info = compute_result['train_info']
                    print("———————— before info filter ————————")
                    print("train_dis:", train_distance)
                    print("test_dis:", test_distance)
                    print("info:", info)
                    line = ''
                    for thr in threshold_list:
                        select_index = [i for i in range(len(info)) if info[i] >= thr]
                        # print(select_index)
                        if len(select_index) == 0:
                            train_mean = float('nan')
                            test_mean = float('nan')
                            info_mean = float('nan')
                            line += to_str([train_mean, test_mean, info_mean,0])
                        else:
                            train_mean = train_distance[select_index].mean()
                            test_mean = test_distance[select_index].mean()
                            info_mean = info[select_index].mean()
                            line += to_str([train_mean, test_mean, info_mean,len(select_index)])
                    line += '\n'
                    del compute_result

                    if args.output_result_file is not None:
                        with open(args.output_dir + '/' + 'before_' + args.output_result_file, 'a+') as f:
                            f.write(line)

                    for _ in range(4000):
                        opt_for_pipline.backward(backward_method='mean',lr=1.0)
                    compute_result = torch_to_numpy(
                        opt_for_pipline.forward(cal_info=True, use_mean_info=True))
                    compute_result['eig_value'] = opt_for_pipline.eig_val()

                    mmstr = '_mean'
                    new_for_save = np.array(compute_result)
                    np.save(marker + "new_L1_" + mmstr + "_save.npy", new_for_save)
                    del new_for_save

                    train_distance = compute_result['train_results'].max(axis=0)
                    test_distance = compute_result['test_results'].max(axis=0)
                    info = compute_result['train_info']
                    print("———————— info filter ————————")
                    print("train_dis:",train_distance)
                    print("test_dis:",test_distance)
                    print("info:",info)
                    line = ''
                    for thr in threshold_list:
                        select_index = [i for i in range(len(info)) if info[i] >= thr]
                        #print(select_index)
                        if len(select_index) == 0:
                            train_mean = float('nan')
                            test_mean = float('nan')
                            info_mean = float('nan')
                            line += to_str([train_mean, test_mean, info_mean,0])
                        else:
                            train_mean = train_distance[select_index].mean()
                            test_mean = test_distance[select_index].mean()
                            info_mean = info[select_index].mean()
                            line += to_str([train_mean, test_mean, info_mean,len(select_index)])
                    line += '\n'
                    if args.output_result_file is not None:
                        with open(args.output_dir +'/'+ args.output_result_file, 'a+') as f:
                            f.write(line)
                    del opt_for_pipline

                if not args.skip_model_save:
                    save_checkpoint(f'model_step{step}.pkl')
    if not args.skip_model_save:
        save_checkpoint('model.pkl')

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')

def donothing():
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass