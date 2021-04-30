
import numpy as np
import math
import torch
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, gaussian_kde
#import latexify
import time
default_method = "L1"
# 正态分布的概率密度函数。可以理解成 x 是 mu（均值）和 sigma（标准差）的函数
np.random.seed()
default_device = 'cuda'


class my_gaussian_kde(object):
    def __init__(self, eval_points, samples, bw_method='scott'):
        self.samples = samples.reshape((-1, 1))
        self.eval_points = eval_points.reshape((-1, 1))
        self.sample_size = self.samples.shape[0]
        self.eval_size = self.eval_points.shape[0]
        self.bw_method = bw_method
        if self.bw_method == "scott":
            self.bandwidth = self.sample_size**(-1./(1+4))
        elif self.bw_method == "silverman":
            self.bandwidth = (self.sample_size * (1 + 2) / 4.)**(-1. / (1 + 4))
        self.batch_size = 256

    def calculate(self,device):
        #matrix = self.eval_points * np.ones((1,self.sample_size)) - np.ones((self.eval_size,1)) * self.samples.T
        batch_len = math.ceil(self.sample_size / self.batch_size)
        #results = np.zeros((self.eval_size,1))
        results = torch.zeros((self.eval_size, 1)).to(device)
        offset = torch.exp(
            torch.tensor(-0.5 / (self.bandwidth**2))).to(device)
        for i in range(batch_len):
            lower, upper = i * \
                self.batch_size, min((i+1)*self.batch_size, self.sample_size)
            #matrix = np.broadcast_to(self.eval_points,(self.eval_size,upper-lower)) - np.broadcast_to(self.samples[lower:upper].T,(self.eval_size,upper-lower))
            matrix = self.eval_points - self.samples[lower:upper].T

            # calculator = np.exp(-0.5 * (matrix/ self.bandwidth)**2)/)
            calculator = torch.pow(offset, matrix**2)
            results += torch.sum(calculator, dim=1).reshape((-1, 1))
        return (results / np.sqrt(2*np.pi) / (self.sample_size * self.bandwidth))


def normfun(x, mu, sigma):
    pdf = np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf


def pdf_distance(data1, data2, method, delta=None):
    data1, data2 = np.reshape(data1, (-1, 1)), np.reshape(data2, (-1, 1))
    if method == "ks_distance":
        return ks_2samp(data1, data2).statistic
    elif method == "L1":
        return np.sum(abs(data1-data2)) * delta / 2
    elif method == "L2":
        return np.sum((data1-data2)**2) * delta / 2
    elif method == "KL":
        return (data1.T * np.log2(data1 / data2) + data2.T * np.log2(data2 / data1)) * delta / 2
    elif method == "JS":
        return (data1.T * np.log2(2 * data1 / (data1+data2)) + data2.T *
                np.log2(2 * data2 / (data1+data2))) * delta / 2
    else:
        raise NotImplementedError


def encap(var, device,direction='torch'):
    if direction == 'torch':
        return torch.tensor(var).detach().to(device)
    else:
        return var.cpu().numpy()

# calculate the L1 distance of g1 and g_2


def Dis_Calculation(data, method=default_method, sam_method='average', sample_cplx=10000,whether_plot=False,device=default_device, range_lis = None):
    # Length of data might be long
    #data = [data1,data2]
    if range_lis is None:
        range_lis = lis(range(len(data)))
    left = min([min(w) for w in data])
    right = max([max(w) for w in data])
    tolerance = 0.2 # extend degree
    left, right = left - (right-left)*tolerance, right + (right-left)*tolerance
    #sample_size = int(sample_cplx * max(10,right - left))
    sample_size = int(sample_cplx * 10)
    delta = (right - left) / sample_size
    x_gird = np.linspace(left, right, sample_size)

    # Prepare pdf for each data scheme
    pdf = []
    for w in range(len(data)):        
        pdf.append(encap(my_gaussian_kde(
            encap(x_gird,device), encap(data[w],device)).calculate(device),device,'numpy'))

    raw_list = {}
    # Calculate and return the result
    for indexi in range(len(range_lis)):
        for indexj in range(indexi + 1, len(range_lis)):
            raw_list[str(range_lis[indexi])+","+str(range_lis[indexj])] = pdf_distance(
                data1=pdf[indexi], data2=pdf[indexj], method=default_method, delta=delta)
    max_result = max([value for value in raw_list.values()])

    if whether_plot:
        plt.figure()
        color = ['r', 'b']
        for w in range(2):
            plt.plot(x_gird, args[w]['pdf'], label=args[w]['type'], c=color[w])
        plt.legend()
        plt.title(default_method + "_distance: " + str(result))
        plt.show()
    return max_result, raw_list  # bounded in [0, 1]



def Gaussian_Calculation(g1, g2, method=default_method, sam_method='average',device = default_device):
    temp_lis = [g1, g2]
    args = [
        {
            'mu': temp_lis[w][0],
            'sigma':temp_lis[w][1]**0.5,
            'type': 'real' if len(temp_lis[w]) <= 2 else temp_lis[w][2],
            'kde': 'my' if len(temp_lis[w]) <= 3 else temp_lis[w][3],
        } for w in range(2)
    ]

    # Prepare global data, like sample_size and threshold
    tolerance = 3
    left = min([args[w]['mu'] - tolerance * args[w]['sigma']
                for w in range(2)])
    right = max([args[w]['mu'] + tolerance * args[w]['sigma']
                 for w in range(2)])
    sample_size = 1000 * max(int(right - left), 10)
    delta = (right - left) / sample_size
    x_gird = np.linspace(left, right, sample_size)
    #x_gird = np.sort(np.random.normal((left+right)/2, max(args[0]['sigma'],args[1]['sigma']),sample_size))

    # Prepare pdf for each data scheme
    for w in range(2):
        if args[w]['type'] == 'real':   # nead real pdf
            args[w]['pdf'] = normfun(x_gird,
                                     args[w]['mu'], args[w]['sigma'])
        else:
            data = np.sort(np.random.normal(
                args[w]['mu'], args[w]['sigma'], sample_size))
            if args[w]['kde'] == 'my':
                args[w]['pdf'] = encap(my_gaussian_kde(
                    encap(x_gird,device), encap(data,device)).calculate(device),device,'numpy')
            else:
                kde = gaussian_kde(data)
                args[w]['pdf'] = kde.evaluate(x_gird)

    # Calculate and return the result
    result = pdf_distance(
        data1=args[0]['pdf'], data2=args[1]['pdf'], method=default_method, delta=delta)

    whether_plot = True
    if whether_plot:
        plt.figure()
        color = ['r', 'b']
        for w in range(2):
            plt.plot(x_gird, args[w]['pdf'], label=args[w]['type'], c=color[w])
        plt.legend()
        plt.title(default_method + "_distance: " + str(result))
        plt.show()
    return result  # bounded in [0, 1]


#print(Gaussian_Calculation(
#    g1=[0,1,'sam'],g2=[0,1,'sam']
#))