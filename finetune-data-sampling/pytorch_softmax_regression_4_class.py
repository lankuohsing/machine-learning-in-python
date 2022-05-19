# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 22:03:24 2021

@author: lankuohsing
"""

import numpy as np
import torch.utils.data as Data
import torch
from collections import OrderedDict
from torchsummary import summary
# In[]
data1=[]
labels1=[]
data2=[]
labels2=[]
with open("./dataset/4_class_data_2d.txt",'r',encoding="UTF-8") as rf:
    for line in rf:
        split_list=line.strip().split(" ")
        x=float(split_list[0])
        y=float(split_list[1])
        label=int(split_list[2])
        if (x-2)**2+(y-2)**2<=0.5**2:
            data2.append([x,y])
            labels2.append([label-1])
        else:
            data1.append([x,y])
            labels1.append([label-1])
# In[]
class_num=4
features=torch.tensor(data1,dtype=torch.float)
labels=torch.tensor(labels1,dtype=torch.long)
one_hot_labels=torch.zeros(len(labels),class_num).scatter_(1,labels,1)
batch_size=64
# 将训练数据的特征和标签组合
dataset=Data.TensorDataset(features,one_hot_labels)
# 随机读取小批量
train_loader=Data.DataLoader(dataset,batch_size,shuffle=True)
test_loader=train_loader
epochs=100
# In[]
num_inputs=2
num_outputs=4
class LinearNet(torch.nn.Module):
    def __init__(self,num_inputs,num_outputs):
        super(LinearNet,self).__init__()
        self.linear=torch.nn.Linear(num_inputs,num_outputs)
    def forward(self,x): # x.shape: (batch,num_input)
        y=self.linear(x.view(x.shape[0],-1))
        return y
softmax_regression=torch.nn.Sequential(
        OrderedDict([
                ("linear",torch.nn.Linear(num_inputs,num_outputs))
                ])
        )
torch.nn.init.normal_(softmax_regression.linear.weight,mean=0,std=0.01)
torch.nn.init.constant_(softmax_regression.linear.weight,val=0.01)
criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(softmax_regression.parameters(),lr=0.01)

for epoch in range(epochs):
    for batch_idx,(feature_in_on_batch,label_in_one_batch) in enumerate(train_loader):
        logits=softmax_regression(feature_in_on_batch)
        loss=criterion(logits,label_in_one_batch)
        break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#        if batch_idx % 100==0:
#            print("Train Epoch: {} [{}/{}({:0f}%)]\tLoss: {:6f}".format(epoch,batch_idx*len(feature_in_on_batch),len(train_loader.dataset),100.*batch_idx/len(train_loader),loss.item()))

    test_loss=0
    correct=0
    for data,target in test_loader:
        logits=softmax_regression(data)
        test_loss+=criterion(logits,target).item()
        pred=logits.data.max(1)[1]
        correct+=pred.eq(torch.nonzero(target.data)[:,1]).sum()
    test_loss/=len(test_loader.dataset)
    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{}({:.3f}%)".
          format(test_loss,correct,
                 len(test_loader.dataset),
                 100.*correct/len(test_loader.dataset)))
# In[]
summary(softmax_regression,(1,2))