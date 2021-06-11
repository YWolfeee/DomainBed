import torch
import numpy as np
from domainbed.lib.misc import get_feature
from domainbed.lib import misc
import matplotlib.pyplot as plt
import time


def calculate_f_star(algorithm, loaders, device, test_envs, num_classes):
    # test envs 不应该加入FSSD的计算
    fss_list = [[] for i in range(num_classes)]
    for eval_name, eval_loader in loaders:
        if "out" not in eval_name:
            continue
        ruleout_test = False
        for env in test_envs:
            if str(env) in eval_name:
                ruleout_test = True
                continue
        if ruleout_test:
            continue

        fssd_score, fssd_raw = get_feature(algorithm, eval_loader,
                                           device, num_classes, None, False, True)    # return_raw
        for label in range(num_classes):
            fss_list[label].append(torch.mean(fssd_raw[label], dim=0))
    mean = [torch.zeros_like(fss_list[label][0])
            for label in range(num_classes)]
    for label in range(num_classes):
        for i in range(len(fss_list)):
            mean[label] += fss_list[label][i]
    return [mean[label]/len(fss_list[label]) for label in range(num_classes)]

def feature_extractor_for_train(algorithm, loaders, device, num_classes,val='in'):
    print("————————Calculate the feature distribution for train————————")
    print("")
    #eval_list = []

    fssd_list_raw = [{} for _ in range(num_classes)]

    for eval_name, eval_loader in loaders:
        if val in eval_name:
            fssd_score, fssd_raw = get_feature(
                algorithm, eval_loader, device, num_classes, None, False, True)
            #eval_list.append(eval_name)
            for label in range(num_classes):
                fssd_list_raw[label][eval_name[:-3]]=fssd_raw[label]
    #print(fssd_list_raw[0]['env0'])
    print("————————Finish Calculating————————")
    return fssd_list_raw


def feature_extractor_for_pipline(algorithm, loaders, device, num_classes, marker="",val='in'):
    print("————————Calculate the feature distribution————————")
    print("")
    eval_list = []
    fssd_list_raw = [[]for i in range(num_classes)]
    fssd_mean = [[]for i in range(num_classes)]
    fssd_variance = [[]for i in range(num_classes)]
    feature_mean = [[]for i in range(num_classes)]
    feature_var = [[]for i in range(num_classes)]
    return_feat = [{} for _ in range(num_classes)]
    for eval_name, eval_loader in loaders:
        if val in eval_name:
            # Debug 不设置f_star
            # fssd_score, fssd_raw = get_feature(
            #    algorithm, eval_loader, device, num_classes, f_star, False, True)
            start = time.time()
            fssd_score, fssd_raw = get_feature(
                algorithm, eval_loader, device, num_classes, None, False, True)
            for label in range(num_classes):
                return_feat[label][eval_name[:4]]=fssd_raw[label]
            eval_list.append(eval_name)
            print("Extract feature in env " + eval_name + " use time " + str(round(time.time()-start, 3)))

            for label in range(num_classes):
                fssd_list_raw[label].append(fssd_score[label])
                fssd_mean[label].append(torch.mean(fssd_list_raw[label][-1]))
                fssd_variance[label].append(
                    torch.var(fssd_list_raw[label][-1]))
                feature_mean[label].append(torch.mean(fssd_raw[label], dim=0))
                feature_var[label].append(torch.var(fssd_raw[label], dim=0))

                save_raw_feature = True
                if save_raw_feature:

                    np.save(marker + "_"+eval_name+"_label"+str(label) +
                            ".npy", fssd_raw[label].cpu().numpy())

                # 直观图片打印
                save_some_image = False
                if save_some_image:
                    feature_num = list(range(20))
                    for num in feature_num:
                        feature_set = fssd_raw[label][:, num].cpu().numpy()
                        plt.figure()
                        plt.hist(feature_set, bins='auto', density=True)
                        plt.savefig("feature_imgae/feature"+str(num)+"_label"+str(label)+"_" +
                                    eval_list[-1]+".png")
                        plt.close()
    return return_feat

def feature_extractor(algorithm, loaders, device, num_classes, marker=""):
    print("————————Calculate the feature distribution————————")
    print("")
    eval_list = []
    fssd_list_raw = [[]for i in range(num_classes)]
    fssd_mean = [[]for i in range(num_classes)]
    fssd_variance = [[]for i in range(num_classes)]
    feature_mean = [[]for i in range(num_classes)]
    feature_var = [[]for i in range(num_classes)]
    for eval_name, eval_loader in loaders:
        if "in" in eval_name:
            # Debug 不设置f_star
            # fssd_score, fssd_raw = get_feature(
            #    algorithm, eval_loader, device, num_classes, f_star, False, True)
            start = time.time()
            fssd_score, fssd_raw = get_feature(
                algorithm, eval_loader, device, num_classes, None, False, True)
            eval_list.append(eval_name)
            print("Extract feature in env " + eval_name + " use time " + str(round(time.time()-start, 3)))

            for label in range(num_classes):
                fssd_list_raw[label].append(fssd_score[label])
                fssd_mean[label].append(torch.mean(fssd_list_raw[label][-1]))
                fssd_variance[label].append(
                    torch.var(fssd_list_raw[label][-1]))
                feature_mean[label].append(torch.mean(fssd_raw[label], dim=0))
                feature_var[label].append(torch.var(fssd_raw[label], dim=0))

                save_raw_feature = True
                if save_raw_feature:

                    np.save(marker + "_"+eval_name+"_label"+str(label) +
                            ".npy", fssd_raw[label].cpu().numpy())

                # 直观图片打印
                save_some_image = False
                if save_some_image:
                    feature_num = list(range(20))
                    for num in feature_num:
                        feature_set = fssd_raw[label][:, num].cpu().numpy()
                        plt.figure()
                        plt.hist(feature_set, bins='auto', density=True)
                        plt.savefig("feature_imgae/feature"+str(num)+"_label"+str(label)+"_" +
                                    eval_list[-1]+".png")
                        plt.close()

    '''
    indent = 20
    print("Environment".ljust(indent), "mean".ljust(indent), "var".ljust(indent))
    for label in range(num_classes):
        for i in range(len(eval_list)):
            print((eval_list[i]+"&label"+str(label)).ljust(indent), str(round(fssd_mean[label][i].cpu().item(), 4)).ljust(
                indent), str(round(fssd_variance[label][i].cpu().item(), 4)).ljust(indent))
    print("")
    for label in range(num_classes):
        print("Results in difference feature of label "+str(label))
        for i in range(feature_mean[label][-1].shape[0]):
            misc.print_row([feature_mean[label][j][i].cpu().item()
                            for j in range(len(eval_list))]+[feature_var[label][j][i].cpu().item() for j in range(len(eval_list))], colwidth=10)
    '''