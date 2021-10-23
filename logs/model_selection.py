import numpy as np
import csv as csv
import matplotlib.pyplot as plt
from scipy import stats
import heapq
import os

algorithm = ['ERM','IRM','CORAL','Mixup','GroupDRO']

file_dir = "renamed"
file_name_list = [os.path.join(file_dir, w) for w in os.listdir(file_dir)]


info_threshold = [0.3]

# accuracy threshold used for train_mix selection criterion
acc_thr = 0.1

ymode = 'ood_acc'
xmode_list = ['train_acc','train_dis','train_mix']
hyper = ['algorithm']
assert hyper[0] == 'algorithm'


print(" ")
mix_accs = []
train_accs = []
for file_name in file_name_list:
    print("################ File: " + file_name +
          "; Using Algorithm: " + str(algorithm) + " ################")
    file_result = csv.reader(open(file_name, 'r'))
    
    # find test env in file name
    env_list = set(['env0', 'env1', 'env2','env3'])
    test_env = 'env' + [str(i)
                    for i in range(len(env_list)) if str(i) in file_name][0]
    train_env = env_list - set([test_env])

    data = []

    for i, w in enumerate(file_result):
        if i == 0:
            header = w
            for i in range(len(header)):
                if header[i] == '0':
                    header[i] = '0.0'
            hyper_index = [header.index(w) for w in hyper]
        else:
            if w[hyper_index[0]] in algorithm :
                data.append(w)

    def cluster(w):
        return ",".join([s + ":" + str(w[j]) for s, j in zip(hyper, hyper_index)])

    name_list = sorted(list(set([cluster(w) for w in data])))
    name_index = {
        name: [i for i in range(len(data)) if cluster(data[i]) == name]
        for name in name_list}
    for xmode in xmode_list:
        # print("———————— Xmode: " + xmode +" ————————")
        for thr in info_threshold:
            if xmode == "train_acc" and thr != info_threshold[0]:
                continue

            # calculate the avg acc anyhow
            train_acc_index = [header.index(
                    w + '_out_acc') for w in train_env]
            avg_acc = [np.average([float(w[index])
                                       for index in train_acc_index]) for w in data]

            filter_num = [float(128) for w in data]
                
            if xmode == 'train_dis' or xmode == 'train_mix':
                train_dis_index = header.index(str(thr))
                x_value = [float(w[train_dis_index]) for w in data]
            elif xmode == "train_acc":
                x_value = avg_acc  # direct use
            else:
                raise NotImplementedError
            
            if test_env + "_in_acc" in header:
                acc_index = header.index(test_env + '_in_acc')
                y_value = np.array(
                    [float(w[acc_index])*0.8 + float(w[acc_index+1]) * 0.2 for w in data])
            else:
                acc_index = header.index("test_acc")
                y_value = np.array([float(w[acc_index]) for w in data])
        


            x, y = [], []
            acc = []
            sub_name_list = []
            
            plot_number = 0
            for name in name_list:
                
                this_x = [x_value[i] for i in name_index[name]
                            if not np.isnan(x_value[i])]
                this_y = [y_value[i] for i in name_index[name]
                            if not np.isnan(x_value[i])]
                this_acc = [avg_acc[i] for i in name_index[name]
                            if not np.isnan(x_value[i])]

                x += this_x
                y += this_y
                acc += this_acc
                plot_number += len(this_x)
                sub_name_list += [name for _ in range(len(this_x))]
               
            x, y = np.array(x), np.array(y)
            acc = np.array(acc)
            if xmode == 'train_acc':
                assert acc.all() == x.all(), "the calculation of acc is not accurate"
            
            topnum = 3
            if xmode == "train_acc":
                topk = heapq.nlargest(topnum, range(len(x)), x.take)
                assert len(x) == len(sub_name_list)
                train_accs.append(np.mean(y[topk[0]])*100)
                print("%s Select; Thr: None; acc: %.2f; Name: %s" %
                      (xmode, np.mean(y[topk[0]])*100, sub_name_list[topk[0]]))
            elif xmode == "train_dis":
                mink = heapq.nsmallest(topnum, range(len(x)), x.take)
                assert len(x) == len(sub_name_list)
                print("%s Select; Thr: %.2f; acc: %.2f; Name: %s" %
                      (xmode, thr, np.mean(y[mink[0]])*100, sub_name_list[mink[0]]))
            elif xmode == "train_mix":
                assert len(x) == len(sub_name_list)

                acc_array = np.array(acc)
                max_acc = np.max(acc_array)
                acc_thr = 0.1
                remain_feat = np.where(acc_array > max_acc-acc_thr)[0]
                acc_std = np.std(acc_array[remain_feat])
                dis_std = np.std(x[remain_feat])

                print('estimate r:',acc_std/dis_std)
                selector = acc - x * acc_std/dis_std
                topk = heapq.nlargest(topnum, range(
                    len(selector)), selector.take)
                mix_accs.append(np.mean(y[topk[0]])*100)
                print("%s Select; acc_thr: %.2f; acc: %.2f; Name: %s" %
                      (xmode, acc_thr, np.mean(y[topk[0]])*100, sub_name_list[topk[0]]))
                
