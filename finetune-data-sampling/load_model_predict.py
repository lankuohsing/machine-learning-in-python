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
train1_data=[]
train1_labels=[]
with open("./dataset/train1_dataset.txt",'r',encoding="UTF-8") as rf:
    for line in rf:
        split_list=line.strip().split(" ")
        x=float(split_list[0])
        y=float(split_list[1])
        label=int(split_list[2])
        if True:
            train1_data.append([x,y])
            train1_labels.append([label])
with open("./dataset/test2_dataset.txt",'r',encoding="UTF-8") as rf:
    for line in rf:
        split_list=line.strip().split(" ")
        x=float(split_list[0])
        y=float(split_list[1])
        label=int(split_list[2])
        test2_data.append([x,y])
        test2_labels.append([label])
class_num=4
train1_features=torch.tensor(train1_data,dtype=torch.float)
train1_labels=torch.tensor(train1_labels,dtype=torch.long)
one_hot_labels=torch.zeros(len(train1_labels),class_num).scatter_(1,train1_labels,1)
batch_size=1
# 将训练数据的特征和标签组合
train1_dataset=Data.TensorDataset(train1_features,one_hot_labels)
# 随机读取小批量
train1_loader=Data.DataLoader(train1_dataset,batch_size,shuffle=True)
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
train1_loss=0
correct=0
list_data_prob_label=[]
for data,target in train1_loader:
    logits=mlp_model(data)

    softmax=torch.nn.Softmax()
    probs=softmax(logits)
    list_data_prob_label.append((data[0],probs[0],target[0]))
    train1_loss+=criterion(logits,target).item()
    pred=logits.data.max(1)[1]
    correct+=pred.eq(torch.nonzero(target.data)[:,1]).sum()
train1_loss/=len(train1_loader.dataset)
print("\nTrain1 set: Average loss: {:.4f}, Accuracy: {}/{}({:.3f}%)".
      format(train1_loss,correct,
             len(train1_loader.dataset),
             100.*correct/len(train1_loader.dataset)))
# In[]
#summary(mlp_model,(1,2))
print(list_data_prob_label[0])