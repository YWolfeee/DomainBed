from scipy.stats import ks_2samp
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch as torch
import torch.nn as nn
from gpu_kde import Dis_Calculation
import time
import argparse

parser = argparse.ArgumentParser(description='Domain generalization')
parser.add_argument('--dir', type=str, required=True)
parser.add_argument('--sample_size',type=int,default=100000)
parser.add_argument('--show', action="store_true")
parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
args = parser.parse_args()


has_extract_infor = False
save_some_image = False
info_threshold = 0.0
var_threshold = 0.4
#marker_lis = ["029_0.15_rex","029_0.15_pen1","559_0.15_pen1"]
marker_lis = args.dir.split(",")
if len(marker_lis) > 1:
    assert save_some_image == False

# 设定环境

env_list = ['env0', 'env1', 'env2', 'env3']
test_env = ['env1']
#env_list = ['env0','env1','env2']
#test_env = ['env2']
train_env = [env for env in env_list if env not in test_env]
# 设定label
num_classes = 65


# 定义相似函数

def distribution_distance(data, method="ks_distance", range_lis=None):
    if method == "ks_distance":
        #assert len(data) == 2, "KS distance is available only for two distribution!"
        raw_list = {}
        for indexi in range(len(range_lis)):
            for indexj in range(indexi + 1, len(range_lis)):
                raw_list[str(range_lis[indexi])+","+str(range_lis[indexj])
                         ] = ks_2samp(data[indexi], data[indexj]).statistic
        return max([value for value in raw_list.values()]), raw_list
        # return ks_2samp(data[0], data[1]).statistic
    elif method == "L1":
        return Dis_Calculation(data=data, method='L1', sample_cplx=10000, device='cuda', range_lis=range_lis)
    elif method == "mean":
        raw_list = {}
        meaner = [np.average(w) for w in data]
        for indexi in range(len(range_lis)):
            for indexj in range(indexi + 1, len(range_lis)):
                raw_list[str(range_lis[indexi])+","+str(range_lis[indexj])
                         ] = abs(meaner[indexi] - meaner[indexj])
        return max([value for value in raw_list.values()]), raw_list
    else:
        raise AssertionError


def shape_to_matrix(feature_num, env_num, label_num, max_data, data_len, data, device='cuda'):
    matrix = np.zeros((env_num, label_num, max_data,
                       feature_num), dtype=np.float32)
    for env in range(env_num):
        for label in range(label_num):
            matrix[env][label][0:data_len[env, label]
                               ] = data[label][env_list[env]]
    return torch.from_numpy(matrix).to(device)


class opt_kde(torch.nn.Module):
    def __init__(self, matrix, data_len, sample_size, device='cuda'):
        self.feature_num = matrix.shape[3]
        self.label_num = matrix.shape[1]
        self.max_sample = matrix.shape[2]
        assert matrix.shape[0] == len(
            env_list), "length of envs in data does match provided envs"
        self.bandwidth = self.max_sample ** (-1./(1+4))
        self.offset = torch.exp(
            torch.tensor(-0.5 / (self.bandwidth**2))).to(device)
        self.sample_size = sample_size
        self.device = device

        self.batch_len = 4
        self.batch_size = (self.sample_size +
                           self.batch_len - 1) // self.batch_len

        self.matrix = matrix
        self.data_len = torch.tensor(data_len, dtype=torch.float32)
        self.envs = env_list
        self.envs_num = len(self.envs)

        self.params = torch.eye(
            self.feature_num).to(device)

    def forward(self, cal_info=False):
        matrix = torch.matmul(self.matrix, self.params).unsqueeze(dim=-1)
        left, right = torch.min(matrix).cpu(
        ).item(), torch.max(matrix).cpu().item()
        print("sample message: from %.4f to %.4f, size is %d" %
              (left, right, self.sample_size))
        delta = (right - left) / self.sample_size
        x_gird = torch.linspace(left, right, self.sample_size).to(self.device)
        divisor = torch.tensor(
            np.sqrt(2*np.pi) * self.bandwidth, dtype=torch.float32).to(self.device)
        store_dis = torch.zeros(
            (self.envs_num * self.envs_num, self.label_num, self.feature_num)).to(self.device)
        if cal_info:
            store_info = torch.zeros((
                self.label_num * self.label_num, self.envs_num, self.feature_num
            )).to(self.device)
        reduce_zeros = torch.tensor(
            self.max_sample, dtype=torch.float32).to(self.device)
        len_unsqueeze = self.data_len.unsqueeze(2).to(self.device)

        index = 0
        train_index = []
        for envi in range(self.envs_num):
            for envj in range(self.envs_num):
                if self.envs[envi] in train_env and self.envs[envj] in train_env:
                    train_index.append(index)
                index += 1

        timing = 1000 // self.batch_len
        for batch in range(self.batch_size):
            if batch % timing == 0:
                start = time.time()
            points = x_gird[batch *
                            self.batch_len:min((batch+1)*self.batch_len, self.sample_size)].reshape((1, -1))
            reducer = (torch.sum(torch.pow(self.offset, (matrix - points)**2), dim=2) -
                       ((reduce_zeros - len_unsqueeze) *
                        torch.pow(self.offset, points**2)).unsqueeze(dim=2)
                       ) / len_unsqueeze.unsqueeze(dim=3)

            dis_expand = reducer.expand(
                (self.envs_num, self.envs_num, self.label_num, self.feature_num, reducer.shape[-1]))
            store_dis += torch.sum(torch.abs(dis_expand - dis_expand.permute(1, 0, 2, 3, 4)), dim=-1).reshape(
                (-1, self.label_num, self.feature_num)) / divisor

            if cal_info:
                info_expand = reducer.permute(1, 0, 2, 3).expand(
                    (self.label_num, self.label_num, self.envs_num, self.feature_num, reducer.shape[-1]))
                store_info += torch.sum(torch.abs(info_expand - info_expand.permute(1, 0, 2, 3, 4)), dim=-1).reshape(
                    (-1, self.envs_num, self.feature_num)) / divisor

            #index = 0
            # for envi in range(self.envs_num):
            #    for envj in range(envi+1, self.envs_num):
            #        store_dis[index, :, :] += torch.abs(
            #            (reducer[envi] - reducer[envj])).sum(dim=-1) / divisor
            #        index += 1
            #index = 0
            # if cal_info:
            #    for labeli in range(self.label_num):
            #        for labelj in range(labeli + 1, self.label_num):
            #            store_info[index,] += torch.abs(
            #                (reducer[:,labeli,:] - reducer[:,labelj,:]) / divisor
            #            )
            if batch % timing == timing - 1:
                print("epoch %d, avg time: %f" %
                      ((batch+1)*self.batch_len, (time.time()-start)/timing/self.batch_len))
                #print("pure cal:" + str(cal_time / timing/self.batch_len))

        test_results = (store_dis*delta/2).max(dim=0)[0]
        train_results = (store_dis[train_index] * delta/2).max(dim=0)[0]
        print("finish forward once.")
        if cal_info:
            train_info = (store_info * delta / 2).max(dim=0)[0]
            return {
                "train_results": train_results,
                "test_results": test_results,
                "train_info": train_info
            }
        return {
            "train_results": train_results,
            "test_results": test_results,
            "train_info": None
        }


# Plot the s function
plt.figure()
for marker in marker_lis:
    print("start extracting [" + marker + "]")
    #marker = "029_0.15_rex"
    method = "L1"

    # 处理文件
    npy_file_path = "./"+marker + "/"
    file_list = [file_name for file_name in os.listdir(
        npy_file_path) if ".npy" in file_name]

    def get_file_name(name, label):
        for file in file_list:
            if name in file and "label"+str(label) in file:
                return file
        raise EnvironmentError

    def has_test_env(s):
        for env in test_env:
            if env in s:
                return True
        return False

    # data = label x env x samples x feature
    data = [{
        name: np.load(npy_file_path + get_file_name(name, label)) for name in env_list
    } for label in range(num_classes)]

    feature_num = data[0][env_list[0]].shape[-1]

    # 提取特征信息
    if not has_extract_infor:
        feature = []
        print("———————— Starting feature kde for %d features ————————" %
              feature_num)
        # get_feature_num
        data_length = np.zeros((len(env_list), num_classes), dtype=np.int32)
        for i in range(len(env_list)):
            for j in range(num_classes):
                data_length[i][j] = data[j][env_list[i]].shape[0]
        big_matrix = shape_to_matrix(feature_num=feature_num, env_num=len(env_list), label_num=num_classes, max_data=int(
            max([max(w) for w in data_length])), data_len=data_length, data=data, device=args.device)

        optimizer = opt_kde(big_matrix, data_length,args.sample_size)

        steps = 0
        debug_new_method = True
        compute_result = optimizer.forward(cal_info=debug_new_method)
        compute_result = {
            key:compute_result[key].cpu().numpy()
            for key in compute_result.keys()
        }
        train_distance = compute_result['train_results']
        test_distance = compute_result['test_results']
        if debug_new_method:
            train_info = compute_result['train_info']
            plt.scatter(np.max(train_distance, axis=0), np.max(test_distance, axis=0),
                        c=np.min(train_info, axis=0), s=[1 + 100 * w for w in np.average(train_info,axis=0)])
        else:
            plt.scatter(np.max(train_distance, axis=0),
                        np.max(test_distance, axis=0))
        plt.xlim(-0.005, 1.005)
        plt.ylim(-0.005, 1.005)
        plt.savefig(npy_file_path+"new_"+method+"_"+marker)
        new_for_save = np.array(compute_result)
        np.save(npy_file_path+"new_"+method+"_"+marker+"_save.npy", new_for_save)
        exit()

        for index in range(feature_num):
            # 对每一个特征
            feature.append({})  # dict
            start = time.time()
            feature[-1]['information_all'] = {}
            feature[-1]['information_test'] = {}

            # 计算distinguishable, 每组label对(i,j)在环境env上的difference，并返回最小的
            for i in range(num_classes):
                for j in range(i+1, num_classes):
                    # label (i,j)
                    feature[-1]['information_all'][str(i) + ',' + str(j)] = 1.0
                    feature[-1]['information_test'][str(i) +
                                                    ',' + str(j)] = 1.0

            for env in env_list:   # 对所有集合计算
                _, distance = distribution_distance([data[i][env][:, index] for i in range(
                    num_classes)], method=method, range_lis=list(range(num_classes)))

                if env in train_env:    # 只对training计算information_max
                    for key in feature[-1]['information_all']:
                        feature[-1]['information_all'][key] = min(
                            feature[-1]['information_all'][key], distance[key])
                else:
                    for key in feature[-1]['information_test']:
                        feature[-1]['information_test'][key] = min(
                            feature[-1]['information_test'][key], distance[key])

            # 计算这个特征的overall可区分性
            feature[-1]['max_info'] = max([value for value in feature[-1]
                                           ['information_all'].values()])

            # 下面计算在train和all上的distance
            feature[-1]['invariance_all'] = []
            feature[-1]['train_var'], feature[-1]["all_var"] = [], []
            for label in range(num_classes):
                # feature[-1]['invariance_all'].append({})
                # feature[-1]['train_var'].append(0.0)
                # feature[-1]['all_var'].append(0.0)

                max_dis, distance = distribution_distance(
                    data=[data[label][w][:, index] for w in env_list], method=method, range_lis=env_list)
                feature[-1]['invariance_all'].append(distance)
                feature[-1]['all_var'].append(max_dis)
                feature[-1]['train_var'].append(
                    max([v for k, v in distance.items() if not has_test_env(k)]))

                '''
                for i in range(len(env_list)):
                    for j in range(i+1, len(env_list)):
                        envi, envj = env_list[i], env_list[j]
                        
                        # 计算在label上环境envi,envj之间的distance
                        distance = distribution_distance(
                            data[label][envi][:, index], data[label][envj][:, index])
                        feature[-1]['invariance_all'][label][envi +
                                                            ","+envj] = distance
                        feature[-1]['all_var'][-1] = max(feature[-1]
                                                        ['all_var'][-1], distance)
                        if envi in train_env and envj in train_env:
                            feature[-1]['train_var'][-1] = max(
                                feature[-1]['train_var'][-1], distance)
                '''

            print("feature " + str(index) + " use time: " +
                  str(round(time.time() - start, 2)))

        for_save = np.array(feature)
        np.save(npy_file_path+marker+"feature_information_all.npy", for_save)
    else:   # 直接load feature
        feature = np.load(file=npy_file_path + marker +
                          "feature_information_all.npy", allow_pickle=True)

    def get_row(lis):
        return ",".join([str(w) for w in lis])+"\n"

    writer = open(npy_file_path+marker+"key_message_all.txt", mode='w')
    header = ["index", 'max_info', 'test_max']
    for label in range(num_classes):
        header.append("l"+str(label)+"_train_var")
        header.append("l"+str(label)+"_all_var")
    writer.write(get_row(header))
    for index in range(len(feature)):
        line = [index]
        line.append(feature[index]['max_info'])
        line.append(np.max(feature[index]['information_test']))
        for label in range(num_classes):
            line.append(feature[index]['train_var'][label])
            line.append(feature[index]['all_var'][label])
        writer.write(get_row(line))
    writer.close()

    lab_n_env = []
    for label in range(num_classes):
        # if label > 5 :
        #    continue
        for env in env_list:
            lab_n_env.append((label, env))

    filter_feature = [w for w in feature if w['max_info'] > info_threshold]
    training_variance = [
        np.arctan(max(w['train_var'])) * 2 / np.pi for w in filter_feature]
    all_variance = [np.arctan(max(w['all_var'])) *
                    2 / np.pi for w in filter_feature]
    max_info = [w['max_info'] for w in filter_feature]
    info = [np.mean(list(w['information_all'].values()))
            for w in filter_feature]
    test_info = [np.mean(list(w['information_test'].values()))
                 for w in filter_feature]

    invariant_rate = [(b-a)/a for (a, b)
                      in zip(training_variance, all_variance)]
    info_rate = [(b-a)/a for (a, b) in zip(info, test_info)]

    # plt.scatter(invariant_rate,info_rate, c = training_variance, s = [
    #    1 + 100 * w for w in info
    # ])
    plt.scatter(training_variance, all_variance, s=[
                1 + 100 * w for w in info], c=test_info)

#plt.vlines(var_threshold, 0, 1, colors="r", linestyles="dashed")
plt.title("S Function with info_threshold[%.3f] and var_threshold[%.3f]" % (
    info_threshold, var_threshold))
plt.xlim(left=-0.005)
plt.ylim(bottom=-0.05)
if args.show:
    plt.show()
else:
    plt.savefig(npy_file_path+marker+method+".png")
feature_num = [index for index in range(len(feature)) if (feature[index]['max_info']
                                                          > info_threshold and max(feature[index]['train_var']) < var_threshold)]
if save_some_image:
    #feature_num = [84, 56, 54, 23, 83]
    #feature_num = [0,1,2,3,4]
    # while(True):
    #    temp = input()
    #    if temp == "c":
    #        break
    #    feature_num.append(int(temp))
    if not os.path.exists(npy_file_path+"feature_image/"):
        os.mkdir(npy_file_path+"feature_image/")
    print("———— strat creating images ————")
    for num in feature_num:
        plt.figure(figsize=(18, 10))
        # plt.figure()
        color_set = ['black', 'grey', 'coral', 'red', 'peru', 'darkorange', 'gold', 'yellow', 'lawngreen',
                     'green', 'turquoise', 'aqua', 'dodgerblue', 'royalblue', 'blueviolet', 'm', 'crimson']
        # for w in range(2):
        # label_lis = max(feature[num]
        #                            ['information_all'],key = feature[num]['information_all'].get).split(",")
        label_index = np.argmax(feature[num]['train_var'])
        for i, env in enumerate(env_list):
            feature_set = data[label_index][env][:, num]
            color = 5 * i
            plt.hist(feature_set, bins='auto', density=True, color=color_set[color],
                     alpha=1, label="label"+str(label_index)+"_" + env, histtype='step', linewidth=5)
        # plt.savefig("feature_imgae/feature"+str(num)+"_label"+str(label)+"_" +
        #                env+".png")
        plt.legend(fontsize=15)
        plt.title('Feature'+str(num)+",disting:"+str(round(feature[num]['max_info'], 3))+"\n"
                  + "mean_var_in_train:" +
                  str(round(max(feature[num]['train_var']), 4))
                  + ";mean_var_in_all:"+str(round(max(feature[num]['all_var']), 4)), fontsize=20)
        plt.savefig(npy_file_path+"feature_image/feature" +
                    str(num)+".png", dpi=100)
        plt.close()
    print("———— end creating images ————")
