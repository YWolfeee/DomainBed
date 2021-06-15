# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import copy
import numpy as np

from domainbed import networks
from domainbed.lib.misc import random_pairs_of_minibatches
from domainbed.matrix_opt_for_train import opt_kde

ALGORITHMS = [
    'ERM',
    'RegularERM',
    'AblationRERM',
    'IRM',
    'GroupDRO',
    'Mixup',
    'MLDG',
    'CORAL',
    'RegularCORAL'
    'MMD',
    'DANN',
    'CDANN',
    'MTL',
    'SagNet',
    'ARM',
    'VREx',
    'RSC',
    'SD',
    'MyRSC'
]

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)

    # 提取特征
    def feature(self, x):
        return self.featurizer(x)

class CutERM(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CutERM, self).__init__(input_shape, num_classes, num_domains,
                                         hparams)
        self.label_num = num_classes
        self.feature_num = self.featurizer.n_outputs
        self.env_num = num_domains
        self.update_cal = 0

    def update(self, minibatches, unlabeled=None):
        self.update_cal += 1 # only difference with ERM
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}
    def update_classifer(self,cut_param:torch.Tensor):
        cut_param.detach_()
        cut_param.requires_grad = False
        self.classifier.weight.data = torch.matmul(self.classifier.weight.data,cut_param)


class RegularERM(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(RegularERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.use_erm = False
        if 'use_erm' in hparams:
            self.use_erm = hparams['use_erm']
        self.label_num = num_classes
        self.feature_num = self.featurizer.n_outputs
        self.env_num = num_domains
        self.gamma = 0
        self.mean_method = 'full_decay'
        print('mean_method:',self.mean_method)
        if 'mean_method' in hparams:
            mth = hparams['mean_method']
            assert mth in ['decay_per_step','decay_per_data','full_decay'], 'make sure mean_method right'
            self.mean_method = mth

        if 'RERM_gamma' in hparams:
            self.gamma = hparams['RERM_gamma']
        self.lam = 0.0
        if 'RERM_lam' in hparams:
            self.lam = hparams['RERM_lam']
        if self.lam < 1e-10:
            self.use_erm = True
        if not self.use_erm:
            self.register_buffer('mean_table',
                                 torch.zeros([self.label_num, self.env_num, self.feature_num]))
            self.register_buffer('visit_table',
                                 torch.zeros([self.label_num, self.env_num]))
            # visit_table will be used in a different way in full_decay method


    def update(self, minibatches, unlabeled=None):
        if self.use_erm:
            return super().update(minibatches)

        # calculate mean
        # only the newest data will cause gradient
        if self.mean_method == 'decay_per_data':
            self.mean_table.detach_()
            for env_id, (x, y) in enumerate(minibatches):
                feats = self.feature(x)

                # sweep all label
                for i in range(self.label_num):
                    fetch_ids = torch.where(y == i)[0]
                    if len(fetch_ids):
                        # if there is label i, average them
                        tmp = torch.mean(feats[fetch_ids, :], dim=0)
                        if self.visit_table[i, env_id] > 0.01:
                            # if it's not the fist time, do decay
                            self.mean_table[i, env_id, :] = (1 - self.gamma) * self.mean_table[i, env_id, :] \
                                                            + self.gamma * tmp
                        else:
                            # otherwise, no decay, set visit to 1
                            self.mean_table[i, env_id, :] = tmp
                            self.visit_table[i, env_id] = 1.0
                # else:
                #    self.mean_table[y,env_id,:] += self.gamma*feats

            loss1 = self.lam * torch.mean(torch.var(self.mean_table, dim=1))
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            loss2 = F.cross_entropy(self.predict(all_x), all_y)
            self.optimizer.zero_grad()
            loss = loss1 + loss2
            loss.backward()
            self.optimizer.step()

            return {'loss': loss.item(),
                    'loss_acc': loss2.item(),
                    'loss_mean': loss1.item(),
                    }
        if self.mean_method == 'full_decay':
            x0,_ = minibatches[0]
            visit_table = x0.new_zeros([self.label_num,self.env_num])
            mean_table = x0.new_zeros([self.label_num,self.env_num,self.feature_num])
            for env_id, (x, y) in enumerate(minibatches):
                feats = self.feature(x)
                for i in range(self.label_num):
                    fetch_ids = torch.where(y == i)[0]
                    if len(fetch_ids):
                        # if there is label i, average them
                        visit_table[i,env_id] = 1.0
                        mean_table[i,env_id,:] = torch.mean(feats[fetch_ids, :], dim=0)
            mean_loss = x0.new_zeros([1])
            nzlabel = 0.0
            for i in range(self.label_num):
                # TODO: vars sum or difference sum ?
                fetch_envs = torch.where(visit_table[i,:]>0.01)[0]
                if len(fetch_envs)>1:
                    mean_loss += torch.mean(torch.var(mean_table[i,fetch_envs,:],dim=1))
                    nzlabel += 1.0
            mean_loss = self.lam * mean_loss / nzlabel

            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            loss2 = F.cross_entropy(self.predict(all_x), all_y)
            loss = mean_loss + loss2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return {'loss': loss.item(),
                    'loss_acc': loss2.item(),
                    'loss_mean': mean_loss.item(),
                    }

    def feature_predict(self,x):
        return self.classifier(x)


class AblationRERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(AblationRERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        # TODO: find those para in dataset or args
        self.label_num = 75
        self.feature_num = self.featurizer.n_outputs
        env_num = 3
        self.layer_norm = nn.LayerNorm([self.feature_num])
        self.featurizer = nn.Sequential(self.featurizer,self.layer_norm)
        self.network = nn.Sequential(self.featurizer,self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)

    # 提取特征
    def feature(self, x):
        return self.featurizer(x)


class ARM(ERM):
    """ Adaptive Risk Minimization (ARM) """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        original_input_shape = input_shape
        input_shape = (1 + original_input_shape[0],) + original_input_shape[1:]
        super(ARM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.context_net = networks.ContextNet(original_input_shape)
        self.support_size = hparams['batch_size']

    def predict(self, x):
        batch_size, c, h, w = x.shape
        if batch_size % self.support_size == 0:
            meta_batch_size = batch_size // self.support_size
            support_size = self.support_size
        else:
            meta_batch_size, support_size = 1, batch_size
        context = self.context_net(x)
        context = context.reshape((meta_batch_size, support_size, 1, h, w))
        context = context.mean(dim=1)
        context = torch.repeat_interleave(context, repeats=support_size, dim=0)
        x = torch.cat([x, context], dim=1)
        return self.network(x)


class AbstractDANN(Algorithm):
    """Domain-Adversarial Neural Networks (abstract class)"""

    def __init__(self, input_shape, num_classes, num_domains,
                 hparams, conditional, class_balance):

        super(AbstractDANN, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.register_buffer('update_count', torch.tensor([0]))
        self.conditional = conditional
        self.class_balance = class_balance

        # Algorithms
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.discriminator = networks.MLP(self.featurizer.n_outputs,
            num_domains, self.hparams)
        self.class_embeddings = nn.Embedding(num_classes,
            self.featurizer.n_outputs)

        # Optimizers
        self.disc_opt = torch.optim.Adam(
            (list(self.discriminator.parameters()) +
                list(self.class_embeddings.parameters())),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams['weight_decay_d'],
            betas=(self.hparams['beta1'], 0.9))

        self.gen_opt = torch.optim.Adam(
            (list(self.featurizer.parameters()) +
                list(self.classifier.parameters())),
            lr=self.hparams["lr_g"],
            weight_decay=self.hparams['weight_decay_g'],
            betas=(self.hparams['beta1'], 0.9))

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        self.update_count += 1
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_z = self.featurizer(all_x)
        if self.conditional:
            disc_input = all_z + self.class_embeddings(all_y)
        else:
            disc_input = all_z
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat([
            torch.full((x.shape[0], ), i, dtype=torch.int64, device=device)
            for i, (x, y) in enumerate(minibatches)
        ])

        if self.class_balance:
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1. / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction='none')
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        disc_softmax = F.softmax(disc_out, dim=1)
        input_grad = autograd.grad(disc_softmax[:, disc_labels].sum(),
            [disc_input], create_graph=True)[0]
        grad_penalty = (input_grad**2).sum(dim=1).mean(dim=0)
        disc_loss += self.hparams['grad_penalty'] * grad_penalty

        d_steps_per_g = self.hparams['d_steps_per_g_step']
        if (self.update_count.item() % (1+d_steps_per_g) < d_steps_per_g):

            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {'disc_loss': disc_loss.item()}
        else:
            all_preds = self.classifier(all_z)
            classifier_loss = F.cross_entropy(all_preds, all_y)
            gen_loss = (classifier_loss +
                        (self.hparams['lambda'] * -disc_loss))
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            return {'gen_loss': gen_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))

class DANN(AbstractDANN):
    """Unconditional DANN"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DANN, self).__init__(input_shape, num_classes, num_domains,
            hparams, conditional=False, class_balance=False)


class CDANN(AbstractDANN):
    """Conditional DANN"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CDANN, self).__init__(input_shape, num_classes, num_domains,
            hparams, conditional=True, class_balance=True)


class IRM(ERM):
    """Invariant Risk Minimization"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IRM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        penalty_weight = (self.hparams['irm_lambda'] if self.update_count
                          >= self.hparams['irm_penalty_anneal_iters'] else
                          1.0)
        nll = 0.
        penalty = 0.

        all_x = torch.cat([x for x,y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)

        if self.update_count == self.hparams['irm_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
            'penalty': penalty.item()}


class VREx(ERM):
    """V-REx algorithm from http://arxiv.org/abs/2003.00688"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(VREx, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, unlabeled=None):
        if self.update_count >= self.hparams["vrex_penalty_anneal_iters"]:
            penalty_weight = self.hparams["vrex_lambda"]
        else:
            penalty_weight = 1.0

        nll = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        losses = torch.zeros(len(minibatches))
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll = F.cross_entropy(logits, y)
            losses[i] = nll

        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()
        loss = mean + penalty_weight * penalty

        if self.update_count == self.hparams['vrex_penalty_anneal_iters']:
            # Reset Adam (like IRM), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
                'penalty': penalty.item()}


class Mixup(ERM):
    """
    Mixup of minibatches from different domains
    https://arxiv.org/pdf/2001.00677.pdf
    https://arxiv.org/pdf/1912.01805.pdf
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Mixup, self).__init__(input_shape, num_classes, num_domains,
                                    hparams)

    def update(self, minibatches, unlabeled=None):
        objective = 0

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            lam = np.random.beta(self.hparams["mixup_alpha"],
                                 self.hparams["mixup_alpha"])

            x = lam * xi + (1 - lam) * xj
            predictions = self.predict(x)

            objective += lam * F.cross_entropy(predictions, yi)
            objective += (1 - lam) * F.cross_entropy(predictions, yj)

        objective /= len(minibatches)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item()}

class RegularMixup(Mixup):
    class Mixup(ERM):
        """
        Mixup of minibatches from different domains
        https://arxiv.org/pdf/2001.00677.pdf
        https://arxiv.org/pdf/1912.01805.pdf
        """

        def __init__(self, input_shape, num_classes, num_domains, hparams):
            super(Mixup, self).__init__(input_shape, num_classes, num_domains,
                                        hparams)

        def update(self, minibatches, unlabeled=None):
            objective = 0

            for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
                lam = np.random.beta(self.hparams["mixup_alpha"],
                                     self.hparams["mixup_alpha"])

                x = lam * xi + (1 - lam) * xj
                predictions = self.predict(x)

                objective += lam * F.cross_entropy(predictions, yi)
                objective += (1 - lam) * F.cross_entropy(predictions, yj)

            objective /= len(minibatches)

            self.optimizer.zero_grad()
            objective.backward()
            self.optimizer.step()

            return {'loss': objective.item()}


class GroupDRO(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(GroupDRO, self).__init__(input_shape, num_classes, num_domains,
                                        hparams)
        self.register_buffer("q", torch.Tensor())

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        for m in range(len(minibatches)):
            x, y = minibatches[m]
            losses[m] = F.cross_entropy(self.predict(x), y)
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

        self.q /= self.q.sum()

        loss = torch.dot(losses, self.q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class MLDG(ERM):
    """
    Model-Agnostic Meta-Learning
    Algorithm 1 / Equation (3) from: https://arxiv.org/pdf/1710.03463.pdf
    Related: https://arxiv.org/pdf/1703.03400.pdf
    Related: https://arxiv.org/pdf/1910.13580.pdf
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MLDG, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)

    def update(self, minibatches, unlabeled=None):
        """
        Terms being computed:
            * Li = Loss(xi, yi, params)
            * Gi = Grad(Li, params)

            * Lj = Loss(xj, yj, Optimizer(params, grad(Li, params)))
            * Gj = Grad(Lj, params)

            * params = Optimizer(params, Grad(Li + beta * Lj, params))
            *        = Optimizer(params, Gi + beta * Gj)

        That is, when calling .step(), we want grads to be Gi + beta * Gj

        For computational efficiency, we do not compute second derivatives.
        """
        num_mb = len(minibatches)
        objective = 0

        self.optimizer.zero_grad()
        for p in self.network.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            # fine tune clone-network on task "i"
            inner_net = copy.deepcopy(self.network)

            inner_opt = torch.optim.Adam(
                inner_net.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )

            inner_obj = F.cross_entropy(inner_net(xi), yi)

            inner_opt.zero_grad()
            inner_obj.backward()
            inner_opt.step()

            # The network has now accumulated gradients Gi
            # The clone-network has now parameters P - lr * Gi
            for p_tgt, p_src in zip(self.network.parameters(),
                                    inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_(p_src.grad.data / num_mb)

            # `objective` is populated for reporting purposes
            objective += inner_obj.item()

            # this computes Gj on the clone-network
            loss_inner_j = F.cross_entropy(inner_net(xj), yj)
            grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(),
                allow_unused=True)

            # `objective` is populated for reporting purposes
            objective += (self.hparams['mldg_beta'] * loss_inner_j).item()

            for p, g_j in zip(self.network.parameters(), grad_inner_j):
                if g_j is not None:
                    p.grad.data.add_(
                        self.hparams['mldg_beta'] * g_j.data / num_mb)

            # The network has now accumulated gradients Gi + beta * Gj
            # Repeat for all train-test splits, do .step()

        objective /= len(minibatches)

        self.optimizer.step()

        return {'loss': objective}

    # This commented "update" method back-propagates through the gradients of
    # the inner update, as suggested in the original MAML paper.  However, this
    # is twice as expensive as the uncommented "update" method, which does not
    # compute second-order derivatives, implementing the First-Order MAML
    # method (FOMAML) described in the original MAML paper.

    # def update(self, minibatches, unlabeled=None):
    #     objective = 0
    #     beta = self.hparams["beta"]
    #     inner_iterations = self.hparams["inner_iterations"]

    #     self.optimizer.zero_grad()

    #     with higher.innerloop_ctx(self.network, self.optimizer,
    #         copy_initial_weights=False) as (inner_network, inner_optimizer):

    #         for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
    #             for inner_iteration in range(inner_iterations):
    #                 li = F.cross_entropy(inner_network(xi), yi)
    #                 inner_optimizer.step(li)
    #
    #             objective += F.cross_entropy(self.network(xi), yi)
    #             objective += beta * F.cross_entropy(inner_network(xj), yj)

    #         objective /= len(minibatches)
    #         objective.backward()
    #
    #     self.optimizer.step()
    #
    #     return objective


class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractMMD, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, minibatches, unlabeled=None):
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)

        self.optimizer.zero_grad()
        (objective + (self.hparams['mmd_gamma']*penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'loss': objective.item(), 'penalty': penalty}


class MMD(AbstractMMD):
    """
    MMD using Gaussian kernel
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MMD, self).__init__(input_shape, num_classes,
                                          num_domains, hparams, gaussian=True)


class CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CORAL, self).__init__(input_shape, num_classes,
                                         num_domains, hparams, gaussian=False)

class RegularCORAL(CORAL):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(RegularCORAL, self).__init__(input_shape, num_classes,
                                         num_domains, hparams,)
        self.use_erm = False
        if 'use_erm' in hparams:
            self.use_erm = hparams['use_erm']
        self.label_num = num_classes
        self.feature_num = self.featurizer.n_outputs
        self.env_num = num_domains
        self.lam = 0.0
        if 'regular_lam' in hparams:
            self.lam = hparams['regular_lam']
        if self.lam < 1e-10:
            self.use_erm = True
        if not self.use_erm:
            self.register_buffer('mean_table',
                                 torch.zeros([self.label_num, self.env_num, self.feature_num]))
            self.register_buffer('visit_table',
                                 torch.zeros([self.label_num, self.env_num]))
            # visit_table will be used in a different way in full_decay method

    def update(self, minibatches, unlabeled=None):
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)

        x0, _ = minibatches[0]
        visit_table = x0.new_zeros([self.label_num, self.env_num])
        mean_table = x0.new_zeros([self.label_num, self.env_num, self.feature_num])
        for env_id, feats in enumerate(features):
            for i in range(self.label_num):
                fetch_ids = torch.where(targets[env_id] == i)[0]
                if len(fetch_ids):
                    visit_table[i, env_id] = 1.0
                    mean_table[i, env_id, :] = torch.mean(feats[fetch_ids, :], dim=0)
        mean_loss = x0.new_zeros([1])
        nzlabel = 0.0
        for i in range(self.label_num):
            # TODO: vars sum or difference sum ?
            fetch_envs = torch.where(visit_table[i, :] > 0.01)[0]
            if len(fetch_envs) > 1:
                mean_loss += torch.mean(torch.var(mean_table[i, fetch_envs, :], dim=1))
                nzlabel += 1.0
        mean_loss = self.lam * mean_loss / nzlabel

        self.optimizer.zero_grad()
        (objective + (self.hparams['mmd_gamma']*penalty) + mean_loss).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'loss': objective.item(), 'penalty': penalty,'loss_mean':mean_loss.item()}



class MTL(Algorithm):
    """
    A neural network version of
    Domain Generalization by Marginal Transfer Learning
    (https://arxiv.org/abs/1711.07910)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MTL, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs * 2,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) +\
            list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        self.register_buffer('embeddings',
                             torch.zeros(num_domains,
                                         self.featurizer.n_outputs))

        self.ema = self.hparams['mtl_ema']

    def update(self, minibatches, unlabeled=None):
        loss = 0
        for env, (x, y) in enumerate(minibatches):
            loss += F.cross_entropy(self.predict(x, env), y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def update_embeddings_(self, features, env=None):
        return_embedding = features.mean(0)

        if env is not None:
            return_embedding = self.ema * return_embedding +\
                               (1 - self.ema) * self.embeddings[env]

            self.embeddings[env] = return_embedding.clone().detach()

        return return_embedding.view(1, -1).repeat(len(features), 1)

    def predict(self, x, env=None):
        features = self.featurizer(x)
        embedding = self.update_embeddings_(features, env).normal_()
        return self.classifier(torch.cat((features, embedding), 1))

class SagNet(Algorithm):
    """
    Style Agnostic Network
    Algorithm 1 from: https://arxiv.org/abs/1910.11645
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SagNet, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        # featurizer network
        self.network_f = networks.Featurizer(input_shape, self.hparams)
        # content network
        self.network_c = networks.Classifier(
            self.network_f.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        # style network
        self.network_s = networks.Classifier(
            self.network_f.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        # # This commented block of code implements something closer to the
        # # original paper, but is specific to ResNet and puts in disadvantage
        # # the other algorithms.
        # resnet_c = networks.Featurizer(input_shape, self.hparams)
        # resnet_s = networks.Featurizer(input_shape, self.hparams)
        # # featurizer network
        # self.network_f = torch.nn.Sequential(
        #         resnet_c.network.conv1,
        #         resnet_c.network.bn1,
        #         resnet_c.network.relu,
        #         resnet_c.network.maxpool,
        #         resnet_c.network.layer1,
        #         resnet_c.network.layer2,
        #         resnet_c.network.layer3)
        # # content network
        # self.network_c = torch.nn.Sequential(
        #         resnet_c.network.layer4,
        #         resnet_c.network.avgpool,
        #         networks.Flatten(),
        #         resnet_c.network.fc)
        # # style network
        # self.network_s = torch.nn.Sequential(
        #         resnet_s.network.layer4,
        #         resnet_s.network.avgpool,
        #         networks.Flatten(),
        #         resnet_s.network.fc)

        def opt(p):
            return torch.optim.Adam(p, lr=hparams["lr"],
                    weight_decay=hparams["weight_decay"])

        self.optimizer_f = opt(self.network_f.parameters())
        self.optimizer_c = opt(self.network_c.parameters())
        self.optimizer_s = opt(self.network_s.parameters())
        self.weight_adv = hparams["sag_w_adv"]

    def forward_c(self, x):
        # learning content network on randomized style
        return self.network_c(self.randomize(self.network_f(x), "style"))

    def forward_s(self, x):
        # learning style network on randomized content
        return self.network_s(self.randomize(self.network_f(x), "content"))

    def randomize(self, x, what="style", eps=1e-5):
        device = "cuda" if x.is_cuda else "cpu"
        sizes = x.size()
        alpha = torch.rand(sizes[0], 1).to(device)

        if len(sizes) == 4:
            x = x.view(sizes[0], sizes[1], -1)
            alpha = alpha.unsqueeze(-1)

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + eps).sqrt()

        idx_swap = torch.randperm(sizes[0])
        if what == "style":
            mean = alpha * mean + (1 - alpha) * mean[idx_swap]
            var = alpha * var + (1 - alpha) * var[idx_swap]
        else:
            x = x[idx_swap].detach()

        x = x * (var + eps).sqrt() + mean
        return x.view(*sizes)

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        # learn content
        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()
        loss_c = F.cross_entropy(self.forward_c(all_x), all_y)
        loss_c.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        # learn style
        self.optimizer_s.zero_grad()
        loss_s = F.cross_entropy(self.forward_s(all_x), all_y)
        loss_s.backward()
        self.optimizer_s.step()

        # learn adversary
        self.optimizer_f.zero_grad()
        loss_adv = -F.log_softmax(self.forward_s(all_x), dim=1).mean(1).mean()
        loss_adv = loss_adv * self.weight_adv
        loss_adv.backward()
        self.optimizer_f.step()

        return {'loss_c': loss_c.item(), 'loss_s': loss_s.item(),
                'loss_adv': loss_adv.item()}

    def predict(self, x):
        return self.network_c(self.network_f(x))


class RSC(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(RSC, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.network
        self.drop_f = (1 - hparams['rsc_f_drop_factor']) * 100
        self.drop_b = (1 - hparams['rsc_b_drop_factor']) * 100
        self.num_classes = num_classes

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        # inputs
        all_x = torch.cat([x for x, y in minibatches])
        # labels
        all_y = torch.cat([y for _, y in minibatches])
        # one-hot labels
        all_o = torch.nn.functional.one_hot(all_y, self.num_classes)
        # features
        all_f = self.featurizer(all_x)
        # predictions
        all_p = self.classifier(all_f)

        # Equation (1): compute gradients with respect to representation
        all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]

        # Equation (2): compute top-gradient-percentile mask
        percentiles = np.percentile(all_g.cpu(), self.drop_f, axis=1)
        percentiles = torch.Tensor(percentiles)
        percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
        mask_f = all_g.lt(percentiles.to(device)).float()

        # Equation (3): mute top-gradient-percentile activations
        all_f_muted = all_f * mask_f

        # Equation (4): compute muted predictions
        all_p_muted = self.classifier(all_f_muted)

        # Section 3.3: Batch Percentage
        all_s = F.softmax(all_p, dim=1)
        all_s_muted = F.softmax(all_p_muted, dim=1)
        changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
        percentile = np.percentile(changes.detach().cpu(), self.drop_b)
        mask_b = changes.lt(percentile).float().view(-1, 1)
        mask = torch.logical_or(mask_f, mask_b).float()

        # Equations (3) and (4) again, this time mutting over examples
        all_p_muted_again = self.classifier(all_f * mask)

        # Equation (5): update
        loss = F.cross_entropy(all_p_muted_again, all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class SD(ERM):
    """
    Gradient Starvation: A Learning Proclivity in Neural Networks
    Equation 25 from [https://arxiv.org/pdf/2011.09468.pdf]
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SD, self).__init__(input_shape, num_classes, num_domains,
                                        hparams)
        self.sd_reg = hparams["sd_reg"]

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        all_p = self.predict(all_x)

        loss = F.cross_entropy(all_p, all_y)
        penalty = (all_p ** 2).mean()
        objective = loss + self.sd_reg * penalty

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': loss.item(), 'penalty': penalty.item()}

from torch.autograd import Variable
import random,math
class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class MyRSC(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MyRSC, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.avgpool = copy.deepcopy(self.featurizer.network.avgpool)
        self.feature_num = self.featurizer.n_outputs
        del self.featurizer.network.avgpool
        self.featurizer.network.avgpool = Identity()
        self.step = 0
        self.pecent = 0

    def update(self, minibatches, unlabeled=None):
        self.step += 1
        all_x = torch.cat([x for x, y in minibatches])
        # labels
        all_y = torch.cat([y for _, y in minibatches])

        x = self.featurizer.network(all_x)
        x = x.view(x.size(0),self.feature_num,-1)
        feature_map_size = math.floor(math.sqrt(x.size(2)))
        x = x.view(x.size(0),self.feature_num,feature_map_size,feature_map_size)

        interval = 1000
        if self.step % interval == 0:
            self.pecent = 3.0 / 10 + (self.step / interval) * 2.0 / 10

        self.eval()
        x_new = x.clone().detach()
        x_new = Variable(x_new.data, requires_grad=True)
        #print('x_new:',x_new.size())
        x_new_view = self.avgpool(x_new)
        x_new_view = x_new_view.view(x_new_view.size(0), -1)
        #print('size:',x_new_view.size())
        output = self.classifier(x_new_view)
        class_num = output.shape[1]
        index = all_y
        num_rois = x_new.shape[0]
        num_channel = x_new.shape[1]
        H = x_new.shape[2]
        HW = x_new.shape[2] * x_new.shape[3]
        one_hot = torch.zeros((1), dtype=torch.float32).cuda()
        one_hot = Variable(one_hot, requires_grad=False)
        sp_i = torch.ones([2, num_rois]).long()
        sp_i[0, :] = torch.arange(num_rois)
        sp_i[1, :] = index
        sp_v = torch.ones([num_rois])
        one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v, torch.Size([num_rois, class_num])).to_dense().cuda()
        one_hot_sparse = Variable(one_hot_sparse, requires_grad=False)
        one_hot = torch.sum(output * one_hot_sparse)
        self.zero_grad()
        one_hot.backward()
        grads_val = x_new.grad.clone().detach()
        grad_channel_mean = torch.mean(grads_val.view(num_rois, num_channel, -1), dim=2)
        channel_mean = grad_channel_mean
        grad_channel_mean = grad_channel_mean.view(num_rois, num_channel, 1, 1)
        spatial_mean = torch.sum(x_new * grad_channel_mean, 1)
        spatial_mean = spatial_mean.view(num_rois, HW)
        self.zero_grad()

        choose_one = random.randint(0, 9)
        if choose_one <= 4:
            # ---------------------------- spatial -----------------------
            spatial_drop_num = math.ceil(HW * 1 / 3.0)
            th18_mask_value = torch.sort(spatial_mean, dim=1, descending=True)[0][:, spatial_drop_num]
            th18_mask_value = th18_mask_value.view(num_rois, 1).expand(num_rois, 49)
            mask_all_cuda = torch.where(spatial_mean > th18_mask_value, torch.zeros(spatial_mean.shape).cuda(),
                                        torch.ones(spatial_mean.shape).cuda())
            mask_all = mask_all_cuda.reshape(num_rois, H, H).view(num_rois, 1, H, H)
        else:
            # -------------------------- channel ----------------------------
            vector_thresh_percent = math.ceil(num_channel * 1 / 3.2)
            vector_thresh_value = torch.sort(channel_mean, dim=1, descending=True)[0][:, vector_thresh_percent]
            vector_thresh_value = vector_thresh_value.view(num_rois, 1).expand(num_rois, num_channel)
            vector = torch.where(channel_mean > vector_thresh_value,
                                 torch.zeros(channel_mean.shape).cuda(),
                                 torch.ones(channel_mean.shape).cuda())
            mask_all = vector.view(num_rois, num_channel, 1, 1)

        # ----------------------------------- batch ----------------------------------------
        cls_prob_before = F.softmax(output, dim=1)
        x_new_view_after = x_new * mask_all
        x_new_view_after = self.avgpool(x_new_view_after)
        x_new_view_after = x_new_view_after.view(x_new_view_after.size(0), -1)
        x_new_view_after = self.classifier(x_new_view_after)
        cls_prob_after = F.softmax(x_new_view_after, dim=1)

        sp_i = torch.ones([2, num_rois]).long()
        sp_i[0, :] = torch.arange(num_rois)
        sp_i[1, :] = index
        sp_v = torch.ones([num_rois])
        one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v, torch.Size([num_rois, class_num])).to_dense().cuda()
        before_vector = torch.sum(one_hot_sparse * cls_prob_before, dim=1)
        after_vector = torch.sum(one_hot_sparse * cls_prob_after, dim=1)
        change_vector = before_vector - after_vector - 0.0001
        change_vector = torch.where(change_vector > 0, change_vector, torch.zeros(change_vector.shape).cuda())
        th_fg_value = torch.sort(change_vector, dim=0, descending=True)[0][
            int(round(float(num_rois) * self.pecent))]
        drop_index_fg = change_vector.gt(th_fg_value).long()
        ignore_index_fg = 1 - drop_index_fg
        not_01_ignore_index_fg = ignore_index_fg.nonzero()[:, 0]
        mask_all[not_01_ignore_index_fg.long(), :] = 1

        self.train()
        mask_all = Variable(mask_all, requires_grad=True)
        x = x * mask_all

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #print(x.size())
        pred = self.classifier(x)
        loss = F.cross_entropy(pred, all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}

    def predict(self, x):
        x = self.featurizer.network(x)
        x = x.view(x.size(0), self.feature_num, -1)
        feature_map_size = math.floor(math.sqrt(x.size(2)))
        x = x.view(x.size(0), self.feature_num, feature_map_size, feature_map_size)
        x = self.avgpool(x)
        #print(x.size())
        x = x.view(x.size(0), -1)
        return self.classifier(x)