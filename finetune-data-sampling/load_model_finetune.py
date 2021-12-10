# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 22:54:30 2021

@author: lankuohsing
"""

import numpy as np
import torch.utils.data as Data
import torch
from collections import OrderedDict
from torchsummary import summary



# In[]
test2_data=[]
test2_labels=[]

train2_data=[]
train2_labels=[]
test1_data=[]
test1_labels=[]
with open("./dataset/train2_dataset.txt",'r',encoding="UTF-8") as rf:
    for line in rf:
        split_list=line.strip().split(" ")
        x=float(split_list[0])
        y=float(split_list[1])
        label=int(split_list[2])
        train2_data.append([x,y])
        train2_labels.append([label])
with open("./dataset/test2_dataset.txt",'r',encoding="UTF-8") as rf:
    for line in rf:
        split_list=line.strip().split(" ")
        x=float(split_list[0])
        y=float(split_list[1])
        label=int(split_list[2])
        test2_data.append([x,y])
        test2_labels.append([label])
with open("./dataset/test1_dataset.txt",'r',encoding="UTF-8") as rf:
    for line in rf:
        split_list=line.strip().split(" ")
        x=float(split_list[0])
        y=float(split_list[1])
        label=int(split_list[2])
        if True:
            test1_data.append([x,y])
            test1_labels.append([label])
# In[]
class_num=4
train2_features=torch.tensor(train2_data,dtype=torch.float)
train2_labels=torch.tensor(train2_labels,dtype=torch.long)
one_hot_labels=torch.zeros(len(train2_labels),class_num).scatter_(1,train2_labels,1)
batch_size=64
# 将训练数据的特征和标签组合
train2_dataset=Data.TensorDataset(train2_features,one_hot_labels)
# 随机读取小批量
train_loader=Data.DataLoader(train2_dataset,batch_size,shuffle=True)
test2_features=torch.tensor(test2_data,dtype=torch.float)
test2_labels=torch.tensor(test2_labels,dtype=torch.long)
one_hot_labels=torch.zeros(len(test2_labels),class_num).scatter_(1,test2_labels,1)
batch_size=64
# 将训练数据的特征和标签组合
test2_dataset=Data.TensorDataset(test2_features,one_hot_labels)
# 随机读取小批量
test2_loader=Data.DataLoader(test2_dataset,batch_size,shuffle=True)

test1_features=torch.tensor(test1_data,dtype=torch.float)
test1_labels=torch.tensor(test1_labels,dtype=torch.long)
one_hot_labels=torch.zeros(len(test1_labels),class_num).scatter_(1,test1_labels,1)
batch_size=64
# 将训练数据的特征和标签组合
test1_dataset=Data.TensorDataset(test1_features,one_hot_labels)
# 随机读取小批量
test1_loader=Data.DataLoader(test1_dataset,batch_size,shuffle=True)
# In[]
num_inputs=2
num_outputs=4
mlp_model=torch.load( "./model/mlp.model")
criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(mlp_model.parameters(),lr=0.03)
# In[]
epochs=80
for epoch in range(epochs):
    for batch_idx,(feature_in_on_batch,label_in_one_batch) in enumerate(train_loader):
        logits=mlp_model(feature_in_on_batch)
        loss=criterion(logits,label_in_one_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#        if batch_idx % 100==0:
#            print("Train Epoch: {} [{}/{}({:0f}%)]\tLoss: {:6f}".format(epoch,batch_idx*len(feature_in_on_batch),len(train_loader.dataset),100.*batch_idx/len(train_loader),loss.item()))

    test_loss=0
    correct=0
    for data,target in test2_loader:
        logits=mlp_model(data)
        test_loss+=criterion(logits,target).item()
        pred=logits.data.max(1)[1]
        correct+=pred.eq(torch.nonzero(target.data)[:,1]).sum()
    test_loss/=len(test2_loader.dataset)
    print("\nEpoch: {}".format(epoch))
    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{}({:.3f}%)".
          format(test_loss,correct,
                 len(test2_loader.dataset),
                 100.*correct/len(test2_loader.dataset)))

# In[]
test_loss=0
correct=0
for data,target in test2_loader:
    logits=mlp_model(data)
    test_loss+=criterion(logits,target).item()
    pred=logits.data.max(1)[1]
    correct+=pred.eq(torch.nonzero(target.data)[:,1]).sum()
test_loss/=len(test2_loader.dataset)
print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{}({:.3f}%)".
      format(test_loss,correct,
             len(test2_loader.dataset),
             100.*correct/len(test2_loader.dataset)))
# In[]
test_loss=0
correct=0
for data,target in test1_loader:
    logits=mlp_model(data)
    test_loss+=criterion(logits,target).item()
    pred=logits.data.max(1)[1]
    correct+=pred.eq(torch.nonzero(target.data)[:,1]).sum()
test_loss/=len(test1_loader.dataset)
print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{}({:.3f}%)".
      format(test_loss,correct,
             len(test1_loader.dataset),
             100.*correct/len(test1_loader.dataset)))
# In[]
summary(mlp_model,(1,2))