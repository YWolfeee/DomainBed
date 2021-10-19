import torch as torch
import torch.nn as nn
import numpy as np
import time
import argparse


def shape_to_matrix(feature_num, env_list, label_num, max_data, data_len, data, device='cuda',input_tensor=False):
    env_num = len(env_list)
    if input_tensor:
        matrix = torch.zeros((env_num,label_num,max_data,feature_num),device=device)
    else:
        matrix = np.zeros((env_num, label_num, max_data,
                       feature_num), dtype=np.float32)
    for env in range(env_num):
        for label in range(label_num):
            matrix[env][label][0:data_len[env, label]
            ] = data[label][env_list[env]]
    if input_tensor:
        return matrix
    else:
        return torch.from_numpy(matrix).to(device)


def torch_to_numpy(d):
    return {
        key: d[key].cpu().numpy()
        for key in d.keys()
        if d[key] is not None
    }


class opt_kde(torch.nn.Module):
    def __init__(self, env_list, train_env, num_classes, feature_num, data,sample_size,device):
        self.sample_size = sample_size
        self.device = device
        self.envs = env_list
        self.train_env = train_env
        self.envs_num = len(self.envs)
        self.train_env_index = [i for i in range(self.envs_num) if env_list[i] in self.train_env]
        input_tensor = False
        if isinstance(data[0][env_list[0]],torch.Tensor):
            input_tensor = True

        # 准备初始化数据
        data_len = np.zeros(
            (len(env_list), num_classes), dtype=np.int32)
        for i in range(len(env_list)):
            for j in range(num_classes):
                data_len[i][j] = data[j][env_list[i]].shape[0]
        matrix = shape_to_matrix(feature_num=feature_num, env_list=env_list, label_num=num_classes,
                                 max_data=int(
                                     max([max(w) for w in data_len])), data_len=data_len, data=data,
                                 device=device,input_tensor=input_tensor)

        # 确认参数匹配
        self.feature_num = matrix.shape[3]
        assert self.feature_num == feature_num, "Error when loading feature"
        self.label_num = matrix.shape[1]
        assert self.label_num == num_classes, "Error when dealing with labels"
        self.max_sample = matrix.shape[2]
        assert matrix.shape[0] == len(
            env_list), "length of envs in data does match provided envs"

        std = torch.std(matrix, dim=2).mean().clone().detach()
        self.matrix = matrix / std

        self.data_len = torch.tensor(data_len, dtype=torch.float32)
        self.data_mask = torch.ones(
            (self.envs_num, self.label_num, self.max_sample), dtype=torch.int32).to(self.device)
        for env in range(self.envs_num):
            for label in range(self.label_num):
                self.data_mask[env, label, data_len[env, label]:] -= 1
        self.len_unsqueeze = self.data_len.unsqueeze(2).to(self.device)

        self.bandwidth = 1.06 * \
                         self.max_sample ** (-1. / (1 + 4)) * \
                         torch.std(self.matrix, dim=2).mean().clone().detach()

        self.offset = torch.exp(-0.5 / (self.bandwidth ** 2)).to(self.device)
        # self.sample_size = int(sample_size * (torch.max(matrix) - torch.min(matrix)).cpu().item())

        self.batch_len = 10
        self.batch_size = (self.sample_size +
                           self.batch_len - 1) // self.batch_len

        self.params = torch.eye(
            self.feature_num, requires_grad=True).to(device)

    def normalize(self):  # do normalization in params
        self.params = self.params / torch.sqrt(torch.sum(self.params ** 2, dim=0, keepdim=True)).detach().clamp_min_(
            1e-3)

    def forward(self, cal_info=False, verbose=False,use_mean_info=True):
        # matmul matrix params, s.t. check the results in this linear combination
        matrix = torch.matmul(self.matrix, self.params).unsqueeze(dim=-1).detach()
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
        test_index = []
        for envi in range(self.envs_num):
            for envj in range(self.envs_num):
                if self.envs[envi] in self.train_env and self.envs[envj] in self.train_env and envi < envj:
                    train_index.append(index)
                if envi < envj:
                    test_index.append(index)
                index += 1

        index = 0
        info_index = []
        for labeli in range(self.label_num):
            for labelj in range(self.label_num):
                if labeli < labelj:
                    info_index.append(index)
                index += 1

        timing = 1000 // self.batch_len
        for batch in range(self.batch_size):
            if batch % timing == 0:
                start = time.time()
            points = x_gird[batch *
                            self.batch_len:min((batch + 1) * self.batch_len, self.sample_size)].reshape((1, -1))
            #print(
                #'over:',torch.sum(torch.pow(self.offset, (matrix - points) ** 2), dim=2) -
                       #((reduce_zeros - self.len_unsqueeze) *
                       # torch.pow(self.offset, points ** 2)).unsqueeze(dim=2))

            reducer = (torch.sum(torch.pow(self.offset, (matrix - points) ** 2), dim=2) -
                       ((reduce_zeros - self.len_unsqueeze) *
                        torch.pow(self.offset, points ** 2)).unsqueeze(dim=2)
                       ) / self.len_unsqueeze.unsqueeze(dim=3)
            #print('dw:',self.len_unsqueeze.unsqueeze(dim=3))
            #print('reducer:',reducer)

            dis_expand = reducer.expand(
                (self.envs_num, self.envs_num, self.label_num, self.feature_num, reducer.shape[-1]))

            store_dis += torch.sum(torch.abs(dis_expand - dis_expand.permute(1, 0, 2, 3, 4)), dim=-1).reshape(
                    (-1, self.label_num, self.feature_num)) / divisor

            if cal_info:
                info_expand = reducer.permute(1, 0, 2, 3).expand(
                    (self.label_num, self.label_num, self.envs_num, self.feature_num, reducer.shape[-1]))
                store_info += torch.sum(torch.abs(info_expand - info_expand.permute(1, 0, 2, 3, 4)), dim=-1).reshape(
                    (-1, self.envs_num, self.feature_num)) / divisor

            if batch % timing == timing - 1 and verbose:
                print("epoch %d, avg time: %f" %
                      ((batch + 1) * self.batch_len, (time.time() - start) / timing / self.batch_len))
                # print("pure cal:" + str(cal_time / timing/self.batch_len))

        test_results = (store_dis[test_index] * delta / 2).max(dim=0)[0]
        train_results = (store_dis[train_index] * delta / 2).max(dim=0)[0]
        if verbose:
            print("finish forward once.")

        if cal_info:
            # should consider min env s.t. this to feature is exhibit, and select the biggest label pair
            # train_info = (store_info * delta / 2).max(dim=0)[0]
            # return a (1, feature_num) dimension
            if use_mean_info:
                train_info_raw = (store_info[info_index][:, self.train_env_index, :] * delta /
                                  2).min(dim=1)[0].mean(dim=0).reshape((-1))
            else:
                train_info_raw = (store_info[info_index][:, self.train_env_index, :] * delta /
                                  2).min(dim=1)[0].max(dim=0).reshape((-1))
            return {
                "train_results": train_results,
                "test_results": test_results,
                "train_info": train_info_raw,
                "train_dis": torch.mean(train_results.max(dim=0)[0]),
                "test_dis": torch.mean(test_results.max(dim=0)[0]),
                "info_mean": train_info_raw.mean()
            }
        return {
            "train_results": train_results,
            "test_results": test_results,
            "train_info": None,
            "train_dis": torch.mean(train_results.max(dim=0)[0]),
            "test_dis": torch.mean(test_results.max(dim=0)[0]),
            "info_mean": None,
        }

    # def backward(self, backward_method='mean', lr=1):
    #     if backward_method == 'L1':
    #         # 这里考虑对env取完max之后对label做mean，这样子可以增加数据量
    #         # argmax: label x feature → train_index上的index
    #         # 表示的是对这个params的分量，这个label，是哪两个环境参与了max dis的计算
    #         results = torch_to_numpy(self.forward(whether_backward=True, lr=lr))
    #         print("Before training, Train dis is %.4f, test dis is %.4f" %
    #               (results['train_dis'], results['test_dis']))

    #         '''
    #         cluster_index = torch.gather(torch.from_numpy(
    #             np.array(train_index,dtype=np.longlong)).to(self.device), 0, argmax.view(-1)).reshape((-1, 1))
    #         index = torch.cat([cluster_index // 4, cluster_index % 4], dim=1).reshape((-1,2,1))
    #         # index is (label*feature)*2(represent 2 env taken by this pair)
    #         update_matrix = self.matrix.permute(1, 3, 0, 2).reshape((
    #                 -1, self.envs_num,self.max_sample)).gather(dim=1, 
    #                 index=index.expand(index.shape[0],index.shape[1],self.max_sample)).reshape((
    #                     self.label_num,self.feature_num,2,self.max_sample
    #                 ))
    #         argmax = store_dis[train_index].reshape(
    #             (-1, self.feature_num)).max(dim=0)[1].reshape((-1, 1))
    #         # TODO: add appropriate index
    #         index = torch.cat([argmax % self.label_num, ])
    #         index = [(i, argmax[i] % self.label_num, [train_index[w]//4, train_index[w] % 4])
    #                 for i, w in enumerate((argmax // self.label_num))]
    #         update_matrix = matrix.squeeze(-1).permute(3, 1, 0, 2)[index]
    #         print(update_matrix.shape)
    #         '''

    #     elif backward_method == 'mean':
    #         mean_value = torch.mean(self.matrix @ self.params, dim=2) * (
    #                 self.max_sample / self.len_unsqueeze)
    #         train_env_index = [w for w in range(self.envs_num) if self.envs[w] in self.train_env]
    #         variance = torch.var(mean_value[train_env_index], dim=0)
    #         grad = torch.autograd.grad(variance.mean(), self.params)
    #         self.params -= lr * grad[0]
    #         self.normalize()

    def eig_val(self):  # return sorted eig value, to check whether degenerate
        eigs = torch.eig(self.params)
        return np.sort(eigs[0].detach().cpu().numpy()[:, 0])
