import torch
import random
import os
from torch import nn
import torch as t
from torch import optim
from tqdm import tqdm
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from axial_attention import AxialAttention
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from module import Attention, PreNorm, FeedForward, CNNBlock, LandOceanModule
from function import MJODataset, my_Function
from mjoformer import Transformer, ViViT, Model
import numpy as np

#设置gpu1跑
if torch.cuda.is_available():
    device=torch.device("cuda:1")
else :
    device=torch.device("cpu")

# 固定随机种子
SEED = 22

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(SEED)

 #image_size, patch_size, num_classes, num_frames   
 #image_size是图片大小，num_classes是输出大小，num_frames是时间
# 导入数据集，包括训练集和验证集
path='/WdHeDisk/users/zhangnong/MJO/20230509_test/data/'
wei = '_for7_7_35_sample.npy'
X_train = np.load(path+'X_train'+wei)
Y_train = np.load(path+'Y_train'+wei)
X_valid = np.load(path+'X_valid'+wei)
Y_valid = np.load(path+'Y_valid'+wei)

X_test = np.load(path+'X_test'+wei)
Y_test = np.load(path+'Y_test'+wei)

Y_train_1 = Y_train[:, :, 0]
Y_train_2 = Y_train[:, :, 1]

Y_valid_1 = Y_valid[:, :, 0]
Y_valid_2 = Y_valid[:, :, 1]

Y_test_1 = Y_test[:, :, 0]
Y_test_2 = Y_test[:, :, 1]

Y_train_1 = np.expand_dims(Y_train_1, axis=2)
Y_train_2 = np.expand_dims(Y_train_2, axis=2)
Y_valid_1 = np.expand_dims(Y_valid_1, axis=2)
Y_valid_2 = np.expand_dims(Y_valid_2, axis=2)

Y_test_1 = np.expand_dims(Y_test_1, axis=2)
Y_test_2 = np.expand_dims(Y_test_2, axis=2)



epochs = 100
#设置超参数，rmm1和rmm2分开  最佳batch size = 16
batch_size_1 = 16
image_size_1 = 3
path_size_1 = 3
num_classes_1 = 1
num_frames_1 = 7
criterion_1 = nn.MSELoss()
lr_1 = 1e-4
weight_deacy_1 = 0.01

batch_size_2 = 16
image_size_2 = 3
path_size_2 = 3
num_classes_2 = 1
num_frames_2 = 7
criterion_2 = nn.MSELoss()
lr_2 = 1e-4
weight_deacy_2 = 0.01


# 初始化数据
trainset_1 = MJODataset(X_train, Y_train_1)
trainloader_1 = DataLoader(trainset_1, batch_size = batch_size_1, shuffle=True)

validset_1 = MJODataset(X_valid, Y_valid_1)
validloader_1 = DataLoader(validset_1, batch_size = batch_size_1, shuffle=True)

# 初始化数据
trainset_2 = MJODataset(X_train, Y_train_2)
trainloader_2 = DataLoader(trainset_2, batch_size = batch_size_2, shuffle=True)

validset_2 = MJODataset(X_valid, Y_valid_2)
validloader_2 = DataLoader(validset_2, batch_size = batch_size_2, shuffle=True)

#

testset_1 = MJODataset(X_test, Y_test_1)
testloader_1 = DataLoader(testset_1, batch_size = 16, shuffle = False)

testset_2 = MJODataset(X_test, Y_test_2)
testloader_2 = DataLoader(testset_2, batch_size = 16, shuffle = False)


#定义训练函数
def train_function(model, trainloader, validloader, criterion, optimizer, batch_size, model_weights):
    train_losses, valid_losses = [], []
    scores_rmse = []
    best_score1 = float('-inf')
    best_score2 = float('inf')
    preds = np.zeros((len(Y_valid),35,1))
    y_valid=np.zeros((len(Y_valid),35,1))

    #模型训练
    for epoch in range(epochs):
        print('Epoch: {}/{}'.format(epoch+1, epochs))
        model.train()
        losses = 0
        for data1, labels in tqdm(trainloader):
            data1 = data1.to(device)
            labels = labels.to(device)
            model.zero_grad()

            pred = model(data1)
            pred.unsqueeze(2)
            labels.unsqueeze(2)

            #print(pred.shape)
            #print(labels.shape)

            loss = criterion(pred, labels)
            losses += loss.cpu().detach().numpy()

            loss.backward()
            optimizer.step()

        train_loss = losses / len(trainloader)
        train_losses.append(train_loss)
        print('Training Loss: {:.3f}'.format(train_loss))
    # 模型验证
        model.eval()
        losses = 0
        s=0
        ss=0
        s_rmse=0
        with torch.no_grad():
            for i, data in tqdm(enumerate(validloader)):
                data1, labels = data
                data1 = data1.to(device)
                y_valid[i*batch_size:(i+1)*batch_size] = labels.detach().cpu()
                labels = labels.to(device)
                pred = model(data1)
                pred.unsqueeze(2)
                labels.unsqueeze(2)

                loss = criterion(pred, labels)

                losses += loss.cpu().detach().numpy()
                preds[i*batch_size:(i+1)*batch_size] = pred.detach().cpu()
            valid_loss = losses / len(validloader)
            valid_losses.append(valid_loss)
            print('Validation Loss: {:.3f}'.format(valid_loss))

            preds=torch.as_tensor(preds)
            y_valid=torch.as_tensor(y_valid)


            s = my_Function.rmse_new(y_valid,preds)

            ss = my_Function.cor(y_valid,preds)
            s_rmse = my_Function.rmse_max(y_valid,preds)

            scores_rmse.append(s)
            print('------------------------------------')
            print('day_rmse: {:}'.format(s))
            print('day_cor: {:}'.format(ss))
            print('rmse_max: {:.3f}'.format(s_rmse))
            print('------------------------------------')
        
    # 保存最佳模型权重

        final_s = min(s,ss)
        if (final_s > best_score1) :
            best_score1 = final_s
            best_score2 = s_rmse
            #best_score2 = ss
            checkpoint = {'best_score': best_score1,'state_dict': model.state_dict()}
            torch.save(checkpoint, model_weights)
        if (final_s==best_score1)&(s_rmse < best_score2):
            best_score1 = final_s
            best_score2 = s_rmse
            #best_score2 = ss
            checkpoint = {'best_score': best_score1,'state_dict': model.state_dict()}
            torch.save(checkpoint, model_weights) 
        print('best_day:{:}'.format(best_score1))
        print('best_rmse:{:.3f}'.format(best_score2))

def test_function(model_1, model_2, model_1_weights, model_2_weights, testloader_1, testloader_2, batch_size = 16, Y_test = Y_test):
    # 加载最佳模型权重
    model_1.to(device)
    checkpoint = torch.load(model_1_weights)
    model_1.load_state_dict(checkpoint['state_dict'])

    model_2.to(device)
    checkpoint = torch.load(model_2_weights)
    model_2.load_state_dict(checkpoint['state_dict'])

    # 在测试集上评估模型效果
    model_1.eval()
    t_scores_rmse = []
    t_s = 0
    t_preds_1 = np.zeros((len(Y_test),35,1))
    for i, data in tqdm(enumerate(testloader_1)):
        data1, labels = data
        data1 = data1.to(device)
        labels = labels.to(device)
        t_pred = model_1(data1)
        t_preds_1[i*batch_size:(i+1)*batch_size] = t_pred.detach().cpu()
    t_preds_1 = torch.as_tensor(t_preds_1)

        # 在测试集上评估模型效果
    model_2.eval()
    t_preds_2 = np.zeros((len(Y_test),35,1))
    for i, data in tqdm(enumerate(testloader_2)):
        data1, labels = data
        data1 = data1.to(device)
        labels = labels.to(device)
        t_pred = model_2(data1)
        t_preds_2[i*batch_size:(i+1)*batch_size] = t_pred.detach().cpu()
    t_preds_2 = torch.as_tensor(t_preds_2)

    t_preds = torch.cat((t_preds_1, t_preds_2), dim = 2)

    Y_test = t.from_numpy(Y_test)


#t_s=rmse(Y_test,t_preds)
    t_s = my_Function.rmse_new(Y_test, t_preds)
    t_ss = my_Function.cor(Y_test, t_preds)
    t_scores_rmse.append(t_s)
#t_scores_cor.append(t_ss)
    print('Score_rmse: {:.3f}'.format(t_s))
    print('Score_cor: {:.3f}'.format(t_ss))




if __name__ == "__main__":


    model_1 = ViViT(image_size_1, path_size_1, num_classes_1, num_frames_1, device).to(device)
    #model_2 = ViViT(image_size_2, path_size_2, num_classes_2, num_frames_2, device).to(device)
    #model_1 = Model(input_size = 1152, output_size = 35, hidden_size = 1024, num_layers = 1, device = device).to(device)

    model_2 = Model(input_size = 1152, output_size = 35, hidden_size = 1024, num_layers = 1, device = device).to(device)

    #设置参数
    model_1_weights = '/WdHeDisk/users/zhangnong/MJO/20230509_test/rmm1.pth'
    model_2_weights = '/WdHeDisk/users/zhangnong/MJO/20230509_test/rmm2.pth'

    optimizer_1 = optim.Adam(model_1.parameters(), lr = lr_1)  # weight_decay是L2正则化参数 weight_decay = weight_deacy
    optimizer_2 = optim.Adam(model_2.parameters(), lr = lr_2) 


    #训练rmm1
    train_function(model_1, trainloader_1, validloader_1, criterion_1, optimizer_1, batch_size_1, model_1_weights)

    #train_function(model_2, trainloader_2, validloader_2, criterion_2, optimizer_2, batch_size_2, model_2_weights)


    #测试
    #test_function(model_1, model_2, model_1_weights, model_2_weights, testloader_1, testloader_2)

    
