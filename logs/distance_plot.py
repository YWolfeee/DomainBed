from scipy.stats import ks_2samp
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from gpu_kde import Dis_Calculation
import time
import argparse

parser = argparse.ArgumentParser(description='Domain generalization')
parser.add_argument('--dir', type=str,required=True)
parser.add_argument('--show',action="store_true")
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

env_list = ['env0', 'env1', 'env2','env3']
test_env = ['env1']
train_env = [env for env in env_list if env not in test_env]

# 设定label
num_classes = 65
# Plot the s function
plt.figure()
for marker in marker_lis:
    print("start extracting [" + marker + "]")
    #marker = "029_0.15_rex"
    method = "L1"


    # 处理文件
    npy_file_path = "./"+marker +"/"
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

    # get_feature_num
    feature_num = data[0][env_list[0]].shape[-1]
    #feature_num = 50


    # 定义相似函数


    def distribution_distance(data, method="ks_distance", range_lis = None):
        if method == "ks_distance":
            #assert len(data) == 2, "KS distance is available only for two distribution!"
            raw_list = {}
            for indexi in range(len(range_lis)):
                for indexj in range(indexi + 1, len(range_lis)):
                    raw_list[str(range_lis[indexi])+","+str(range_lis[indexj])] = ks_2samp(data[indexi],data[indexj]).statistic
            return max([value for value in raw_list.values()]), raw_list
            #return ks_2samp(data[0], data[1]).statistic
        elif method == "L1":
            return Dis_Calculation(data=data, method= 'L1',sample_cplx=10000,device = 'cuda', range_lis = range_lis)
        elif method == "mean":
            raw_list = {}
            meaner = [np.average(w) for w in data]
            for indexi in range(len(range_lis)):
                for indexj in range(indexi + 1, len(range_lis)):
                    raw_list[str(range_lis[indexi])+","+str(range_lis[indexj])] = abs(meaner[indexi] - meaner[indexj])
            return max([value for value in raw_list.values()]), raw_list
        else:
            raise AssertionError


    # 提取特征信息
    if not has_extract_infor:
        feature = []
        print("———————— Starting feature kde for %d features ————————" % feature_num)
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
                    feature[-1]['information_test'][str(i) + ',' + str(j)] = 1.0
                    
            for env in env_list:   # 对所有集合计算
                _, distance = distribution_distance([data[i][env][:,index] for i in range(num_classes)], method=method, range_lis = list(range(num_classes)))
                
                if env in train_env:    # 只对training计算information_max
                    for key in feature[-1]['information_all']:
                        feature[-1]['information_all'][key] = min(feature[-1]['information_all'][key], distance[key])
                else:
                    for key in feature[-1]['information_test']:
                        feature[-1]['information_test'][key] = min(feature[-1]['information_test'][key], distance[key])

            # 计算这个特征的overall可区分性
            feature[-1]['max_info'] = max([value for value in feature[-1]
                                        ['information_all'].values()])

            # 下面计算在train和all上的distance
            feature[-1]['invariance_all'] = []
            feature[-1]['train_var'], feature[-1]["all_var"] = [], []
            for label in range(num_classes):
                #feature[-1]['invariance_all'].append({})
                #feature[-1]['train_var'].append(0.0)
                #feature[-1]['all_var'].append(0.0)

                max_dis, distance = distribution_distance(data= [data[label][w][:,index] for w in env_list], method=method,range_lis=env_list)
                feature[-1]['invariance_all'].append(distance)
                feature[-1]['all_var'].append(max_dis)
                feature[-1]['train_var'].append(max([v for k,v in distance.items() if not has_test_env(k)]))

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
    header = ["index", 'max_info','test_max']
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
        #if label > 5 :
        #    continue
        for env in env_list:
            lab_n_env.append((label, env))


    filter_feature = [w for w in feature if w['max_info'] > info_threshold]
    training_variance = [np.arctan(max(w['train_var'])) * 2 / np.pi for w in filter_feature]
    all_variance = [np.arctan(max(w['all_var'])) *2 / np.pi for w in filter_feature]
    max_info = [w['max_info'] for w in filter_feature]
    info = [np.mean(list(w['information_all'].values())) for w in filter_feature]
    test_info = [np.mean(list(w['information_test'].values())) for w in filter_feature]

    invariant_rate = [(b-a)/a for (a,b) in zip(training_variance,all_variance)]
    info_rate = [(b-a)/a for (a,b) in zip(info,test_info)]

    #plt.scatter(invariant_rate,info_rate, c = training_variance, s = [
    #    1 + 100 * w for w in info
    #])
    plt.scatter(training_variance, all_variance, s=[
                1 + 100 * w for w in  info], c=test_info)

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
        color_set = ['black','grey','coral','red','peru','darkorange','gold','yellow','lawngreen','green','turquoise','aqua','dodgerblue','royalblue','blueviolet','m','crimson']
        #for w in range(2):
            #label_lis = max(feature[num]
            #                            ['information_all'],key = feature[num]['information_all'].get).split(",")    
        label_index = np.argmax(feature[num]['train_var'])
        for i, env in enumerate(env_list):
            feature_set = data[label_index][env][:, num]
            color = 5 * i 
            plt.hist(feature_set, bins='auto', density=True, color= color_set[color],
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
