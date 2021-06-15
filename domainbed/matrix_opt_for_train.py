import torch as torch
import torch.nn as nn
import numpy as np
import time
import argparse


def shape_to_matrix(feature_num, env_list, label_num, max_data, data_len, data, device='cuda'):
    env_num = len(env_list)
    matrix = torch.zeros([env_num, label_num, max_data,
                       feature_num], device=device)
    #print('env_list',env_list)
    for env in range(env_num):
        for label in range(label_num):
            matrix[env][label][0:data_len[env, label]
            ] = data[label][env_list[env]]
            #print('data_len:',data_len[env, label])
            #print('data:',data[label][env_list[env]])
        #print('______env______:',env)
    #print(matrix)
    return matrix


class opt_kde(torch.nn.Module):
    def __init__(self, env_list, train_env, num_classes, feature_num,data,percent=0.5,
                 sample_size=1000,device='cuda'):
        self.sample_size = sample_size
        self.device = device
        self.envs = env_list
        self.train_env = train_env
        self.envs_num = len(self.envs)
        self.mask = None
        self.percent = percent

        # 准备初始化数据
        data_len = np.zeros(
            (len(env_list), num_classes), dtype=np.int32)
        for i in range(len(env_list)):
            for j in range(num_classes):
                data_len[i][j] = len(data[j][env_list[i]])
        #print('data:',data)
        matrix = shape_to_matrix(feature_num=feature_num, env_list=env_list, label_num=num_classes,
                                 max_data=int(
                                     max([max(w) for w in data_len])), data_len=data_len, data=data,
                                 device=device)

        # 确认参数匹配
        self.feature_num = matrix.shape[3]
        assert self.feature_num == feature_num, "Error when loading feature"
        self.label_num = matrix.shape[1]
        assert self.label_num == num_classes, "Error when dealing with labels"
        self.max_sample = matrix.shape[2]
        assert matrix.shape[0] == len(
            env_list), "length of envs in data does match provided envs"

        self.matrix = matrix
        #print('matrix', self.matrix)

        self.data_len = torch.tensor(data_len, dtype=torch.float32)
        self.data_mask = torch.ones(
            (self.envs_num, self.label_num, self.max_sample), dtype=torch.int32).to(self.device)
        for env in range(self.envs_num):
            for label in range(self.label_num):
                self.data_mask[env, label, data_len[env, label]:] -= 1
        self.len_unsqueeze = self.data_len.unsqueeze(2).to(self.device)

        self.bandwidth = 1.06 * \
                         self.max_sample ** (-1. / (1 + 4)) * \
                         torch.std(matrix, dim=2).mean().clone().detach()
        self.offset = torch.exp(-0.5 / (self.bandwidth ** 2)).to(self.device)
        # self.sample_size = int(sample_size * (torch.max(matrix) - torch.min(matrix)).cpu().item())

        self.batch_len = 1
        self.batch_size = (self.sample_size +
                           self.batch_len - 1) // self.batch_len

        self.params = torch.eye(
            self.feature_num, requires_grad=True).to(device)

    def normalize(self):  # do normalization in params
        self.params = self.params / torch.sqrt(torch.sum(self.params ** 2, dim=0, keepdim=True)).detach().clamp_min_(
            1e-3)

    def forward(self, cal_info=False, verbose=False,set_mask=False):
        # matmul matrix params, s.t. check the results in this linear combination
        matrix = torch.matmul(self.matrix, self.params).detach().unsqueeze(dim=-1)
        left, right = torch.min(matrix).cpu(
        ).item(), torch.max(matrix).cpu().item()

        if verbose:
            print("sample message: from %.4f to %.4f, size is %d" %
                  (left, right, self.sample_size))
        delta = (right - left) / self.sample_size
        x_gird = torch.linspace(left, right, self.sample_size).to(self.device)
        divisor = np.sqrt(2 * np.pi) * self.bandwidth
        store_dis = torch.zeros(
            (self.envs_num * self.envs_num, self.label_num, self.feature_num)).to(self.device)
        if cal_info:
            store_info = torch.zeros((
                self.label_num * self.label_num, self.envs_num, self.feature_num
            )).to(self.device)
        reduce_zeros = torch.tensor(
            self.max_sample, dtype=torch.float32).to(self.device)

        index = 0
        train_index = []
        for envi in range(self.envs_num):
            for envj in range(self.envs_num):
                if self.envs[envi] in self.train_env and self.envs[envj] in self.train_env:
                    train_index.append(index)
                index += 1

        timing = 1000 // self.batch_len
        for batch in range(self.batch_size):
            if batch % timing == 0:
                start = time.time()
            points = x_gird[batch *
                            self.batch_len:min((batch + 1) * self.batch_len, self.sample_size)].reshape((1, -1))
            reducer = (torch.sum(torch.pow(self.offset, (matrix - points) ** 2), dim=2) -
                       ((reduce_zeros - self.len_unsqueeze) *
                        torch.pow(self.offset, points ** 2)).unsqueeze(dim=2)
                       ) / self.len_unsqueeze.unsqueeze(dim=3)

            dis_expand = reducer.expand(
                (self.envs_num, self.envs_num, self.label_num, self.feature_num, reducer.shape[-1]))
            store_dis += torch.sum(torch.abs(dis_expand - dis_expand.permute(1, 0, 2, 3, 4)), dim=-1).reshape(
                (-1, self.label_num, self.feature_num)) / divisor
            #print(store_dis)
            if cal_info:
                info_expand = reducer.permute(1, 0, 2, 3).expand(
                    (self.label_num, self.label_num, self.envs_num, self.feature_num, reducer.shape[-1]))
                store_info += torch.sum(torch.abs(info_expand - info_expand.permute(1, 0, 2, 3, 4)), dim=-1).reshape(
                    (-1, self.envs_num, self.feature_num)) / divisor

            if batch % timing == timing - 1 and verbose:
                print("epoch %d, avg time: %f" %
                      ((batch + 1) * self.batch_len, (time.time() - start) / timing / self.batch_len))
                # print("pure cal:" + str(cal_time / timing/self.batch_len))

        test_results = (store_dis * delta / 2).max(dim=0)[0]
        train_results = (store_dis[train_index] * delta / 2).max(dim=0)[0]
        if verbose:
            print("finish forward once.")

        if set_mask:
            feature_dis = train_results.max(dim=0)[0].view(-1)
            self.mask = torch.topk(feature_dis,int(self.percent*len(feature_dis)),largest=True)[1]
            # find smallest channel, set param to 0
            print('mask len:',len(self.mask))
            if len(self.mask) == 0:
                return torch.eye(self.params.size(0),device=self.device)
            save_param = self.params.detach().clone()
            save_param[:,self.mask] = 0
            inverse_param = torch.inverse(self.params)
            inverse_param[self.mask,:]=0
            res = torch.matmul(save_param,inverse_param)
            print('diff from identity:',torch.norm(res - torch.eye(self.feature_num,device=self.device)))
            return res

        if cal_info:
            # should consider min env s.t. this to feature is exhibit, and select the biggest label pair
            # train_info = (store_info * delta / 2).max(dim=0)[0]
            # return a (1, feature_num) dimension
            train_info = (store_info * delta /
                          2).min(dim=1)[0].max(dim=0)[0].reshape((1, -1))
            return {
                "train_results": train_results,
                "test_results": test_results,
                "train_info": train_info,
                "train_dis": torch.mean(train_results.max(dim=0)[0]),
                "test_dis": torch.mean(test_results.max(dim=0)[0])
            }
        return {
            "train_results": train_results,
            "test_results": test_results,
            "train_info": None,
            "train_dis": torch.mean(train_results.max(dim=0)[0]),
            "test_dis": torch.mean(test_results.max(dim=0)[0])
        }

    @torch.no_grad()
    def pca(self):
        mean_value = torch.mean(self.matrix @ self.params, dim=2) * (
                self.max_sample / self.len_unsqueeze)
        # mean_valuse is of shape (env,label,feature)
        x = mean_value.unsqueeze(1)
        y = mean_value.unsqueeze(0)
        feat = (x - y).view(-1, self.feature_num)
        feat1 = feat.unsqueeze(2)
        feat2 = feat.unsqueeze(1)
        mat = torch.mean(feat1 * feat2, dim=0)
        eig = torch.eig(mat, eigenvectors=True)
        print('min eig of data:{}'.format(torch.min(eig[0]).item()))
        lam = torch.diag(torch.sqrt(eig[0]))
        self.params = lam * eig[1]

    def backward(self, backward_method='mean', lr=1):
        if backward_method == 'L1':
            # 这里考虑对env取完max之后对label做mean，这样子可以增加数据量
            # argmax: label x feature → train_index上的index
            # 表示的是对这个params的分量，这个label，是哪两个环境参与了max dis的计算
            print("L1 backward is not ready, please use mean method to backward")
            exit()
            cluster_index = torch.gather(torch.from_numpy(
                np.array(train_index, dtype=np.longlong)).to(self.device), 0, argmax.view(-1)).reshape((-1, 1))
            index = torch.cat([cluster_index // 4, cluster_index % 4], dim=1).reshape((-1, 2, 1))
            # index is (label*feature)*2(represent 2 env taken by this pair)

            update_matrix = self.matrix.permute(1, 3, 0, 2).reshape((
                -1, self.envs_num, self.max_sample)).gather(dim=1,
                                                            index=index.expand(index.shape[0], index.shape[1],
                                                                               self.max_sample)).reshape((
                self.label_num, self.feature_num, 2, self.max_sample
            ))

            argmax = store_dis[train_index].reshape(
                (-1, self.feature_num)).max(dim=0)[1].reshape((-1, 1))
            # TODO: add appropriate index
            index = torch.cat([argmax % self.label_num, ])
            index = [(i, argmax[i] % self.label_num, [train_index[w] // 4, train_index[w] % 4])
                     for i, w in enumerate((argmax // self.label_num))]
            update_matrix = matrix.squeeze(-1).permute(3, 1, 0, 2)[index]
            print(update_matrix.shape)
        elif backward_method == 'mean':
            mean_value = torch.mean(self.matrix @ self.params, dim=2) * (
                    self.max_sample / self.len_unsqueeze)
            train_env_index = [w for w in range(self.envs_num) if self.envs[w] in self.train_env]
            variance = torch.var(mean_value[train_env_index], dim=0)
            grad = torch.autograd.grad(variance.mean(), self.params)
            self.params -= lr * grad[0]
            self.normalize()

    def eig_val(self):  # return sorted eig value, to check whether degenerate
        eigs = torch.eig(self.params)
        return np.sort(eignormalizes[0].detach().cpu().numpy()[:, 0])


class opt_mmd(torch.nn.Module):
    def __init__(self, matrix, data_len, sample_size, env_list, device='cuda'):
        print("This method is not prepared. Please use opt_kde instead.")
        exit()
        self.device = device

        self.feature_num = matrix.shape[3]
        self.label_num = matrix.shape[1]
        self.max_sample = matrix.shape[2]
        assert matrix.shape[0] == len(
            env_list), "length of envs in data does match provided envs"
        self.sample_size = [torch.tensor(
            10 ** (gamma)).to(self.device) for gamma in range(-3, 4)]

        self.matrix = matrix  # Label x Env x Data_num x feature_num
        self.data_len = torch.tensor(
            data_len, dtype=torch.float32).to(self.device)
        self.envs = env_list
        self.envs_num = len(self.envs)

        self.data_mask = torch.ones(
            (self.envs_num, self.label_num, self.max_sample)).to(self.device)
        for env in range(self.envs_num):
            for label in range(self.label_num):
                self.data_mask[env, label, data_len[env, label]:] -= 1.0

        self.params = torch.eye(
            self.feature_num).to(device)

        self.global_MMD = True

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

    def mmd(self, x, y):  # calculate the value of each two point in x_list

        Kxx = self.gaussian_kernel(x, x).mean()
        Kyy = self.gaussian_kernel(y, y).mean()
        Kxy = self.gaussian_kernel(x, y).mean()
        return Kxx + Kyy - 2 * Kxy


    def forward(self):
        # Global MMD = \mean_{i,j} \sum_{g} \mean{env, env'}
        matrix = self.matrix @ self.params
        Kenv = torch.zeros((self.envs_num, 1)).to(self.device)
        for env in range(self.envs_num):  # 先计算自己和自己的
            x_norm = (matrix[env].pow(2).sum(
                dim=-1, keepdim=True)) @ self.data_mask[env].unsqueeze(-1).transpose(-2, -1)
            res = -2 * \
                  matrix[env] @ matrix[env].transpose(-2, -1) + \
                  x_norm.transpose(-2, -1) + x_norm
            res.clamp_min_(1e-30)
            for g in self.sample_size:  # MMD中的kernel
                Kenv[env] += torch.mean(torch.exp(res.mul(-g)).mean(dim=[-1, -2]).add_(-1) * torch.pow(
                    self.max_sample / self.data_len[env], 2) + 1)
        print(Kenv)