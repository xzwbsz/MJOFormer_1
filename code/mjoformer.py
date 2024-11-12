import torch
import random
import os
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from axial_attention import AxialAttention
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from module import Attention, PreNorm, FeedForward, LandOceanModule, CNNBlock, Decoder, Encoder
from function import MJODataset, my_Function
import numpy as np


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


  
class ViViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, num_frames, device, dim = 256, depth = 4, heads = 7, pool = 'cls', in_channels = 64, dim_head = 256, dropout = 0.2,
                 emb_dropout = 0., scale_dim = 4, ):
        super().__init__()
        
        # 使用时初始化LandOceanModule，并传入输入数据的形状
        land_ocean_module = LandOceanModule(input)
        


        self.conv1 = nn.Conv2d(10, 16, kernel_size=7, stride=2, padding=3)
        self.conv2 = CNNBlock(16, 16, 3, 1, 1)
        self.conv3 = CNNBlock(16, 32, 3, 2, 1)
        self.conv4 = CNNBlock(32, 32, 3, 1, 1)
        self.conv5 = CNNBlock(32, 64, 3, 2, 1)
        self.conv6 = CNNBlock(64, 64, 3, 1, 1)

        #self.conv7 = CNNBlock(64, 128, 3, 2, 1)
        #self.conv8 = CNNBlock(128, 128, 3, 1, 1)


        
        #self.image_size = image_size
        self.avgpool = nn.AdaptiveAvgPool2d((image_size, image_size))#(16,4,69)
        
                #(1, 5, 128, 256, 256)
        self.attn = AxialAttention(dim = in_channels, dim_index = 2, heads = 8, num_dimensions = 3, )

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )


        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches + 1, dim))


        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool
        self.dim = dim
        self.device = device

        #self.decoder = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        #先定义h和c
        self.decoder = Decoder(input_size = dim, hidden_size = 256, num_layers = 1, output_size = 35)

        # embedding dimension
        # where is the embedding dimension
        # number of heads for multi-head attention
        # number of axial dimensions (images is 2, video is 3, or more)
        #attn = AxialAttention(dim = 128, odim_index = 2, heads = 8, num_dimensions = 3,)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            #nn.Linear(dim, num_classes)
            FeedForward(dim, dim, dropout=dropout),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):

        # 将输入数据X传入land_ocean_module进行处理
        x = land_ocean_module(x)

        b, t, H, W, C = x.shape
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        x = x.view(b * t, C, H, W)




        #m = nn.ReflectionPad2d((0, 0, 10, 100))

        #m(x)
        #print(x.shape)
        
        #cnn部分
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)




        #print('before:', x.shape)
        x = self.avgpool(x)

        print(x.shape)

        x = rearrange(x, '(b t) ... -> b t ...', b=b, t=t)

        x = self.attn(x)

        #print('before:', x.shape)
        #分块【batch_size， T， C， H，W】-》 b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)
        #【1，16，196，192】
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b = b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        #加位置编码【1，16，197，192】
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)


        x = rearrange(x, 'b t n d -> (b t) n d')
        #print("before space T ", x.shape)
        x = self.space_transformer(x)
        #print("after space T ", x.shape)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)

        #print("after space T ", x.shape)

        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        #print("before time T ", x.shape)
        x = self.temporal_transformer(x)

       # print(x.shape)#【64, 8, 512】
        

        h = x[:, 0, :]
        c = x[:, 0, :]

        x = x[:, 1:, :]

        h = h.squeeze()
        c = c.squeeze()

        h = h.unsqueeze(0)
        c = c.unsqueeze(0)

        #print(h.shape, c.shape, t)

       # print(x.shape)

        outputs_rmm = torch.zeros(b, t, 35).to(self.device)
        #outputs_rmm2 = torch.zeros(b, t, 35).to(self.device)

        for i in range(t):
            _input = x[:, i, :]

            #print(x.shape)

            
            output, h, c = self.decoder(_input, h, c)
            
            outputs_rmm[:, i, :] = output
        
        rmm1 = outputs_rmm[:, -1, :]
        #rmm2 = outputs_rmm2[:, -1, :]
        
        rmm1 = rmm1.squeeze()
        #rmm2 = rmm2.squeeze()
        
        rmm1 = rmm1.unsqueeze(2)
        #rmm2 = rmm2.unsqueeze(2)

        #RMM = torch.cat((rmm1, rmm2), dim = 2)
        
        return rmm1
        
        

        '''

        #print(x.shape)[8]
        #前面相当于做了encoder【1，17，192】

        #这里用informer，取【batch_size, 17, 192】

        #print(x.shape)【64, 8, 512】

        x_new = torch.randn([b, 35, self.dim]).to(self.device)
        x = torch.cat((x[:, 1:, :], x_new), dim=1)

        #x = self.decoder(x)

        #print(x.shape)[43, ]

        #取最后35天的做预测

        x = x[:, t:, :]
        #print(x.shape)






        #print(x.shape)
        #x = x.mean(dim = 2) if self.pool == 'mean' else x[:, 0]

        #print(x.shape)
        #前面相当于做了encoder

        return self.mlp_head(x)
        '''
    
#定义模型
class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, device ):
        super().__init__()
        self.input_size = input_size #1152
        self.output_size = output_size #35
        self.hidden_size = hidden_size #1024
        self.num_layers = num_layers #1
        self.device = device
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3)
        self.conv2=CNNBlock(16, 16, 3, 1, 1)
        self.conv3=CNNBlock(16, 32, 3, 2, 1)
        self.conv4=CNNBlock(32, 32, 3, 1, 1)
        self.conv5=CNNBlock(32, 64, 3, 2, 1)
        self.conv6=CNNBlock(64, 64, 3, 1, 1)
        #self.conv7=CNNBlock(64, 128, 3, 2, 1)
        #self.conv8=CNNBlock(128, 128, 3, 1, 1)

        
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))#(16,4,69)
        self.flatten = nn.Flatten()
        
        self.Encoder = Encoder(self.input_size, self.hidden_size, self.num_layers, self.device)
        self.Decoder = Decoder(self.input_size, self.hidden_size, self.num_layers, self.output_size)

    def forward(self, x):
        
        x = x[:, :, :, :, 0:3]
        batch_size, seq_len, H, W, C = x.shape
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        x = x.view(batch_size * seq_len, C, H, W)
        
        #cnn部分
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        #x = self.conv7(x)
        #x = self.conv8(x)

        
        x = self.avgpool(x)
        x = self.flatten(x)
        
        _, C_new = x.shape
        
        x = x.view(batch_size, seq_len, C_new)
        
        
        
        h, c = self.Encoder(x)
        
        outputs_rmm = torch.zeros(batch_size, seq_len, self.output_size).to(self.device)
        #outputs_rmm2 = torch.zeros(batch_size, seq_len, self.output_size).to(self.device)
        
        for t in range(seq_len):
            _input = x[:, t, :]
            
            output, h, c = self.Decoder(_input, h, c)
            
            outputs_rmm[:, t, :] = output
        
        rmm1 = outputs_rmm[:, -1, :]
        
        rmm1 = rmm1.squeeze()
        
        rmm1 = rmm1.unsqueeze(2)

        
        return rmm1

    





    