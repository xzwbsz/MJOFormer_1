#导入相关包
import numpy as np
import torch as t
import random
import matplotlib.pyplot as plt
#import netCDF4
import datetime
import torch
from tqdm import tqdm
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.autograd as autograd
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

# 构造数据管道
class MJODataset(Dataset):
    def __init__(self, data,label):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.label = torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        return self.data[idx],self.label[idx]

class my_Function():    
    #cor函数
    def cor(Y_test,t_preds):
        #Y_test = mean_std(Y_test)
        #t_preds = mean_std(t_preds)
        cor_day = []
        a=0
        b=0
        c=0
        score_cor=0
        for i in range(0,35):
            for j in range(0,len(Y_test)):
                a+=(Y_test[j,i,:]*t_preds[j,i,:]).sum()
                b+=(Y_test[j,i]**2).sum()
                c+=(t_preds[j,i]**2).sum()
            b=np.sqrt(b)
            c=np.sqrt(c)
            cor=a/(b*c)
            cor_day.append(cor)
            a=0
            b=0
            c=0
        cor_day=np.array(cor_day)
        #print(cor_day)
        for i in range(0,len(cor_day)):
            if cor_day[i]<0.50:
                break
        score_cor=i
        return score_cor

    def rmse_new(Y_valid,preds):
        #Y_valid = mean_std(Y_valid)
        #preds = mean_std(preds)
        rmse_day = []
        a=0
        score_rmse=0
        for i in range(0,35):
            for j in range(0,len(Y_valid)):
                a+=(torch.pow((Y_valid[j,i,:]-preds[j,i,:]),2)).sum()
            rmse=a/len(Y_valid)
            rmse=np.sqrt(rmse)
            rmse_day.append(rmse)
            a=0
        rmse_day=np.array(rmse_day)
        #print(rmse_day)
        for i in range(0,len(rmse_day)):
            if rmse_day[i]>1.40:
                break
        score_rmse=i
        '''
        day_cal=np.argwhere(rmse_day<1.40)
        print(day_cal)
        if len(day_cal)==0:
            score_rmse=0
        else:
            score_rmse=day_cal.max()+1
        '''
        #score_rmse=rmse_day.max()
        return score_rmse

    def rmse_max(Y_valid,preds):
        #Y_valid = mean_std(Y_valid)
        #preds = mean_std(preds)
        rmse_day = []
        a=0
        score_rmse=0
        for i in range(0,35):
            for j in range(0,len(Y_valid)):
                a+=(torch.pow((Y_valid[j,i,:]-preds[j,i,:]),2)).sum()
            rmse=a/len(Y_valid)
            rmse=np.sqrt(rmse)
            rmse_day.append(rmse)
            a=0
        rmse_day=np.array(rmse_day)
        score_rmse=rmse_day.max()
        return score_rmse


