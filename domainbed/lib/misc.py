# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Things that don't belong anywhere else
"""

import hashlib
import json
import os
import sys
from shutil import copyfile
from collections import OrderedDict
from numbers import Number
import operator

import numpy as np
import torch
import tqdm
from collections import Counter

def make_weights_for_balanced_classes(dataset):
    counts = Counter()
    classes = []
    for _, y in dataset:
        y = int(y)
        counts[y] += 1
        classes.append(y)

    n_classes = len(counts)

    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights

def pdb():
    sys.stdout = sys.__stdout__
    import pdb
    print("Launching PDB, enter 'n' to step to parent function.")
    pdb.set_trace()

def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)

def print_separator():
    print("="*80)

def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)

class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]
    def __len__(self):
        return len(self.keys)

def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert(n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)

def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs

def accuracy(network, loader, weights, device):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            #total += len(x)
            x = x.to(device,non_blocking=True)
            y = y.to(device,non_blocking=True)
            p = network.predict(x)
            #assert weights is None, "An extended mode which is not support!"
            #'''
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset:
                                        weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device,non_blocking=True)
            
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() *
                            batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() *
                            batch_weights).sum().item()
            total += batch_weights.sum().item()
            '''
            if p.size(1) == 1:
                correct += p.gt(0).eq(y).float().sum()
            else:
                correct += p.argmax(1).eq(y).float().sum()
            '''
            
    network.train()

    return correct / total
    #return correct.item() / total


def get_feature(network, loader, device, num_classes, f_star, perturbation=False, return_raw=False):   # loader为一个环境的
    # here, f_star is a vector representing the initial point of each feature
    # perturbation indicates whether to add a sign(nabla) term before calculating the norm
    # return_raw indicates whether to return the direct difference

    raw_difference = [[] for i in range(num_classes)]
    FSSD = [[]for i in range(num_classes)]
    label_set = [torch.tensor(label).to(device)
                 for label in range(num_classes)]

    network.eval()
    set_zero = True if f_star is None else False
    with torch.no_grad():

        network.to(device)
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            feature = network.feature(x)    # get feature

            for label in range(num_classes):
                
                #label_difference = (label_set[label] == y).reshape((-1,1)) * difference
                select = (label_set[label] == y)
                label_difference = feature[select, :]
                if not set_zero:
                    label_difference = label_difference - f_star[label].reshape((1,-1))
                # if perturbation:
                #    sign_nabla = torch.sign(torch.autograd.grad(difference,x)[0])  # a matrix, #feature * dim(x)
                #    modified_x = sign_nabla + torch.mm(torch.ones_like(f_star),x)   # \tilde x
                FSSD[label].append(torch.norm(input=label_difference, p=1, dim=1))
                if return_raw:
                    raw_difference[label].append(label_difference)
    network.train()
    final_fssd = [torch.cat(FSSD[i], dim=0) for i in range(num_classes)]
    if not return_raw:
        return final_fssd
    return final_fssd, [torch.cat(raw_difference[i], dim=0) for i in range(num_classes)]


class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

class ParamDict(OrderedDict):
    """Code adapted from https://github.com/Alok/rl_implementations/tree/master/reptile.
    A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)
