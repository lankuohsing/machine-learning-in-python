# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 20:08:32 2021

@author: lankuohsing
"""
import numpy as np
import torch.utils.data as Data
import torch
from collections import OrderedDict
from torchsummary import summary


mlp_model=torch.load( "./model/mlp.model")

# In[]
test2_data=[]
test2_labels=[]
data2=[]
labels2=[]
with open("./dataset/test2_dataset.txt",'r',encoding="UTF-8") as rf:
    for line in rf:
        split_list=line.strip().split(" ")
        x=float(split_list[0])
        y=float(split_list[1])
        label=int(split_list[2])
        test2_data.append([x,y])
        test2_labels.append([label])
criterion=torch.nn.CrossEntropyLoss()
class_num=4
test2_features=torch.tensor(test2_data,dtype=torch.float)
test2_labels=torch.tensor(test2_labels,dtype=torch.long)
one_hot_labels=torch.zeros(len(test2_labels),class_num).scatter_(1,test2_labels,1)
batch_size=64
# 将训练数据的特征和标签组合
dataset=Data.TensorDataset(test2_features,one_hot_labels)
# 随机读取小批量
test_loader=Data.DataLoader(dataset,batch_size,shuffle=True)

test_loss=0
correct=0
for data,target in test_loader:
    logits=mlp_model(data)
    test_loss+=criterion(logits,target).item()
    pred=logits.data.max(1)[1]
    correct+=pred.eq(torch.nonzero(target.data)[:,1]).sum()
test_loss/=len(test_loader.dataset)
print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{}({:.3f}%)".
      format(test_loss,correct,
             len(test_loader.dataset),
             100.*correct/len(test_loader.dataset)))
# In[]
summary(mlp_model,(1,2))