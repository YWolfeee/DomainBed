import numpy as np
import os
from functools import cmp_to_key

def find_first(x,y):
    if x == y:
        return 0
    return [w for w in range(min(len(x),len(y))) if x[w] != y[w]][0]

def number_compare(x,y):
    if len(x) == 0:
        return -1
    if len(y) == 0:
        return 1
    tag = False
    for i in range(len(x)):
        if x[i] == "*":
            x = x[:i] + "." + x[i+1:]
            if i ==0:
                tag = True
    if tag:
        x = '0' + x
    tag = False
    for i in range(len(y)):
        if y[i] == "*":
            y = y[:i] + "." + y[i+1:]
            if i == 0:
                tag = True
    if tag:
        y = '0' + y
    a = float(x)
    b = float(y)
    if a == b:
        return -1 if len(x) > len(y) else 1
    return -1 if a < b else 1

def compare(x, y):
    sorted_list = ['ERM', 'CORAL', 'REx',"DRO", "RSC"]
    indexx = [i for i in range(len(sorted_list)) if sorted_list[i] in x][0]
    indexy = [i for i in range(len(sorted_list)) if sorted_list[i] in y][0]
    if indexx != indexy:
        return -1 if indexx < indexy else 1
    if set([x,y]) == set([
        'new_L1_CORAL_{lr=5e-05_mmd_gamma=0*1_resnet18=false}_2021-05-08-02-17-26_3_mean_save.npy',
        'new_L1_CORAL_{lr=5e-05_mmd_gamma=0*01_resnet18=false}_2021-05-10-15-13-57_1_mean_save.npy'
        ]):
        print(set([x,y]))
    penal_thre = max(x.index("}"), y.index("}"))
    if x[:penal_thre] == y[:penal_thre]:    # Seed 不同
        if x == y:
            return 0
        return -1 if x < y else 1
    else:                                   # Penal 不同
        first_dis = find_first(x,y)
        x_last, y_last = first_dis, first_dis
        while(True):
            if (x[x_last] >= '0' and x[x_last] <= '9') or (x[x_last] == "*"):
                x_last += 1
            else:
                break
        while(True):
            if (y[y_last] >= '0' and y[y_last] <= '9') or (y[y_last] == "*"):
                y_last += 1
            else:
                break
        return number_compare(x[first_dis:x_last],y[first_dis:y_last])
        if "*" in x and "*" not in y:
            return - 1
        elif "*" in y and "*" not in x:
            return 1
        elif "*" in x and "*" in y:
            return x < y
        else:
            return 1 if (x < y) else -1


def to_str(lis):
    s = ""
    for w in lis:
        s = s + str(w).ljust(10," ") + ", "
    return s


threshold_list = [round(0.05*i, 2) for i in range(9)]
npy_dir = "all_npy_list"
file_list = sorted(os.listdir(npy_dir), key=cmp_to_key(compare))
output_file = "all_feature.txt"
output = open(output_file, 'w')

for sub_name in file_list:
    if sub_name == "new_L1_CORAL_{lr=5e-05_mmd_gamma=1000_resnet18=false}_2021-05-09-16-37-00_0_mean_save.npy":
        sub_name = sub_name
    name = os.path.join(npy_dir, sub_name)
    result = np.load(name, allow_pickle=True).item()
    train_distance = result['train_results'].max(axis=0)
    test_distance = result['test_results'].max(axis=0)
    info = result['train_info']
    line = sub_name.ljust(125," ") + ", "
    print("———————— " + line + " ————————")
    for thr in threshold_list:
        select_index = [i for i in range(len(info)) if info[i] > thr]
        train_mean = train_distance[select_index].mean()
        test_mean = test_distance[select_index].mean()
        info_mean = info[select_index].mean()
        line += to_str([train_mean, test_mean, info_mean])
    output.write(line+"\n")
output.close()
