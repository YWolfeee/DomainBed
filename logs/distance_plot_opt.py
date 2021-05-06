from scipy.stats import ks_2samp
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import torch as torch
import torch.nn as nn
from gpu_kde import Dis_Calculation
import time
import argparse
from matrix_optimizer import opt_kde
# matplotlib.use('AGG')
parser = argparse.ArgumentParser(description='Domain generalization')
parser.add_argument('--dir', type=str, required=True,
                    help="Input dir lsit you want to extract feature and train matrix, splited by `,`")
parser.add_argument('--sample_size', type=int, default=10000, help="sample size when calculate the integral")
parser.add_argument('--show', action="store_true", help= "to show the figure or save it")
parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
parser.add_argument('--steps', type=int, default=500, help= 'combinational matrix training steps')
parser.add_argument('--check_freq', type=int, default=100)
parser.add_argument('--lr', type=float, default=1.0)
parser.add_argument('--no_extracted', action="store_true", help="whether extract the filges, or directly read")
parser.add_argument('--per_feature_image', action="store_true", help="whether plot each feature, not supported yet")
parser.add_argument('--info_threshold', type=float, default=0.0, help="feature with information lower than this will not be plotted")
parser.add_argument('--test_index', type=str, default="1", help="index of test environment")
parser.add_argument('--num_classes', type=int, help="number of classes")
parser.add_argument('--add_info_plot', type=bool, default=True, help="whether to add informative as color")
parser.add_argument('--plot_method', type=str, default='L1', help="using what method to calculate the distance of two distributions")
parser.add_argument('--backward_method', type=str, default='mean',help="using what metrix to train the matrix")
parser.add_argument('--concentrate_image',type=str,default=None,help="input a dir, where you want all images to be saved here")
args = parser.parse_args()

marker_lis = args.dir.split(",")




# 设定环境 需要手动修改
env_list = ['env0', 'env1', 'env2', 'env3']
test_env = [env_list[int(i)] for i in args.test_index]
# env_list = ['env0','env1','env2']
# test_env = ['env2']
train_env = [env for env in env_list if env not in test_env]
# 设定label
num_classes = 65 if args.num_classes is None else args.num_classes
print('---------args----------')
for k in list(vars(args).keys()):
    print('%s: %s' % (k, vars(args)[k]))

print("")

print('########:    This program will deal with:', str(marker_lis))
print("########:    Environment list:" + str(env_list))
print("")

# Plot the s function

# 定义相似函数

def distribution_distance(data, method="ks_distance", range_lis=None):
    if method == "ks_distance":
        # assert len(data) == 2, "KS distance is available only for two distribution!"
        raw_list = {}
        for indexi in range(len(range_lis)):
            for indexj in range(indexi + 1, len(range_lis)):
                raw_list[str(range_lis[indexi]) + "," + str(range_lis[indexj])
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
                raw_list[str(range_lis[indexi]) + "," + str(range_lis[indexj])
                         ] = abs(meaner[indexi] - meaner[indexj])
        return max([value for value in raw_list.values()]), raw_list
    else:
        raise AssertionError


def train_process(optimier, steps=100, check_freq=50):
    for step in range(steps):
        if (step % check_freq == 0) or (step == steps-1):
            compute_result = optimizer.forward(cal_info=False)
            compute_result = {
                key: compute_result[key].cpu().numpy()
                for key in compute_result.keys()
                if compute_result[key] is not None
            }
            train_dis = compute_result['train_dis']
            test_dis = compute_result['test_dis']
            print("In step %d / %d, train dis is %.4f, test dis is %.4f" %
                  (step, steps, train_dis, test_dis))

        optimizer.backward(backward_method=args.backward_method, lr=args.lr)


def plot_once(results, npy_file_path, marker, steps):
    # 画图部分
    plt.figure(figsize=(36, 20))
    train_distance = results['train_results']
    test_distance = results['test_results']
    if args.add_info_plot:
        train_info = results['train_info']
        plt.scatter(np.max(train_distance, axis=0), np.max(test_distance, axis=0),
                    c=train_info.reshape((-1)), s=[1 + 400 * w for w in np.average(train_info, axis=0)])
    else:
        plt.scatter(np.max(train_distance, axis=0),
                    np.max(test_distance, axis=0))
    plt.xlim(-0.005, 1.005)
    plt.ylim(-0.005, 1.005)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.xlabel("Invarnance in training envs",fontsize=20)
    plt.ylabel("Invarnance in all envs",fontsize=20)
    variance_str = str(int(
        1000 * results['train_dis'])).ljust(4, '0') + "+" + str(int(1000 * results['test_dis'])).ljust(4, '0')
    print("ready to show/save features with averagely [ train dis %.4f, test dis %.4f, avg info %.4f ]" %(
        results['train_dis'],results['test_dis'],np.average(results['train_info'])))
    if args.show:
        plt.show()
    else:
        fig_name = marker + "_drag_" + args.plot_method + "_steps" + \
            str(steps) + "_lr" + str(args.lr) + "_" + variance_str + '.png'
        plt.savefig(npy_file_path + fig_name)
        print("[saved image " + fig_name + "] in " + npy_file_path)

    plt.close()


def torch_to_numpy(d):
    return {
        key: d[key].cpu().numpy()
        for key in d.keys()
        if d[key] is not None
    }


if __name__ == "__main__":
    if args.concentrate_image is not None:
        if not os.path.exists(args.concentrate_image):
            os.mkdir(args.concentrate_image)
    for marker in marker_lis:
        print("———————— start extracting [" + marker + "] ————————")

        # 处理文件
        npy_file_path = "./" + marker + \
            "/" if not marker[-1] == "/" else "./" + marker
        file_list = [file_name for file_name in os.listdir(
            npy_file_path) if ".npy" in file_name]

        def get_file_name(name, label):
            for file in file_list:
                if name in file and "label" + str(label) in file:
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
        if not args.no_extracted:
            print("Start feature kde for %d features" %
                  feature_num)

            # 初始化网络\训练
            optimizer = opt_kde(env_list, train_env,
                                num_classes, feature_num, args, data)

            save_place = npy_file_path
            if args.concentrate_image is not None:
                save_place = "./" + args.concentrate_image + \
            "/" if not args.concentrate_image[-1] == "/" else "./" + args.concentrate_image

            # 先画一次steps = 0的图
            compute_result = torch_to_numpy(
                optimizer.forward(cal_info=args.add_info_plot))
            plot_once(compute_result, 
                save_place,
                marker, 0)

            # 然后开始训练
            train_process(optimizer, steps=args.steps,
                          check_freq=args.check_freq)

            # 训练数据处理部分
            compute_result = torch_to_numpy(
                optimizer.forward(cal_info=args.add_info_plot))
            compute_result['eig_value'] = optimizer.eig_val()

            # 画训练好的图
            plot_once(compute_result, save_place, marker, args.steps)

            # 储存训练结果, 方便日后使用
            new_for_save = np.array(compute_result)
            np.save(npy_file_path + "new_" + args.plot_method + "_" +
                    marker + "_save.npy", new_for_save)

            old_method = False
            if old_method:
                feature = []
                for index in range(feature_num):
                    # 对每一个特征
                    feature.append({})  # dict
                    start = time.time()
                    feature[-1]['information_all'] = {}
                    feature[-1]['information_test'] = {}

                    # 计算distinguishable, 每组label对(i,j)在环境env上的difference，并返回最小的
                    for i in range(num_classes):
                        for j in range(i + 1, num_classes):
                            # label (i,j)
                            feature[-1]['information_all'][str(
                                i) + ',' + str(j)] = 1.0
                            feature[-1]['information_test'][str(i) +
                                                            ',' + str(j)] = 1.0

                    for env in env_list:  # 对所有集合计算
                        _, distance = distribution_distance([data[i][env][:, index] for i in range(
                            num_classes)], method=method, range_lis=list(range(num_classes)))

                        if env in train_env:  # 只对training计算information_max
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

                    print("feature " + str(index) + " use time: " +
                          str(round(time.time() - start, 2)))

                for_save = np.array(feature)
                np.save(npy_file_path + marker +
                        "feature_information_all.npy", for_save)

                def get_row(lis):
                    return ",".join([str(w) for w in lis]) + "\n"

                writer = open(npy_file_path + marker +
                              "key_message_all.txt", mode='w')
                header = ["index", 'max_info', 'test_max']
                for label in range(num_classes):
                    header.append("l" + str(label) + "_train_var")
                    header.append("l" + str(label) + "_all_var")
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

                filter_feature = [
                    w for w in feature if w['max_info'] > args.info_threshold]
                training_variance = [
                    np.arctan(max(w['train_var'])) * 2 / np.pi for w in filter_feature]
                all_variance = [np.arctan(max(w['all_var'])) *
                                2 / np.pi for w in filter_feature]
                max_info = [w['max_info'] for w in filter_feature]
                info = [np.mean(list(w['information_all'].values()))
                        for w in filter_feature]
                test_info = [np.mean(list(w['information_test'].values()))
                             for w in filter_feature]

                invariant_rate = [(b - a) / a for (a, b)
                                  in zip(training_variance, all_variance)]
                info_rate = [(b - a) / a for (a, b) in zip(info, test_info)]

                # plt.scatter(invariant_rate,info_rate, c = training_variance, s = [
                #    1 + 100 * w for w in info
                # ])
                plt.scatter(training_variance, all_variance, s=[
                    1 + 100 * w for w in info], c=test_info)

            print("Finish feature kde for %d features" %
                  feature_num)
            print("")

        else:  # 直接load feature
            feature = np.load(file=npy_file_path + "new_" + args.plot_method + "_" +
                              marker + "_save.npy", allow_pickle=True)

        if args.per_feature_image:
            # feature_num = [84, 56, 54, 23, 83]
            # feature_num = [0,1,2,3,4]
            # while(True):
            #    temp = input()
            #    if temp == "c":
            #        break
            #    feature_num.append(int(temp))
            if not os.path.exists(npy_file_path + "feature_image/"):
                os.mkdir(npy_file_path + "feature_image/")
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
                             alpha=1, label="label" + str(label_index) + "_" + env, histtype='step', linewidth=5)
                # plt.savefig("feature_imgae/feature"+str(num)+"_label"+str(label)+"_" +
                #                env+".png")
                plt.legend(fontsize=15)
                plt.title('Feature' + str(num) + ",disting:" + str(round(feature[num]['max_info'], 3)) + "\n"
                          + "mean_var_in_train:" +
                          str(round(max(feature[num]['train_var']), 4))
                          + ";mean_var_in_all:" + str(round(max(feature[num]['all_var']), 4)), fontsize=20)
                plt.savefig(npy_file_path + "feature_image/feature" +
                            str(num) + ".png", dpi=100)
                plt.close()
            print("———— end creating images ————")
