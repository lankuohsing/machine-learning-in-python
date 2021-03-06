# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 22:35:41 2021

@author: lankuohsing
"""

import numpy as np
import torch.utils.data as Data
import torch
from collections import OrderedDict
from torchsummary import summary
# In[]
train1_data=[]
train1_labels=[]
test1_data=[]
test1_labels=[]

with open("./dataset/4_class_boundary_samples.txt",'r',encoding="UTF-8") as rf:
    for line in rf:
        split_list=line.strip().split(" ")
        x=float(split_list[0])
        y=float(split_list[1])
        label=int(split_list[2])
        if True:
            train1_data.append([x,y])
            train1_labels.append([label-1])
with open("./dataset/test_dataset.txt",'r',encoding="UTF-8") as rf:
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
train1_features=torch.tensor(train1_data,dtype=torch.float)
train1_labels=torch.tensor(train1_labels,dtype=torch.long)
one_hot_labels=torch.zeros(len(train1_labels),class_num).scatter_(1,train1_labels,1)
batch_size=64
# 将训练数据的特征和标签组合
dataset=Data.TensorDataset(train1_features,one_hot_labels)
# 随机读取小批量
train_loader=Data.DataLoader(dataset,batch_size,shuffle=True)

test1_features=torch.tensor(test1_data,dtype=torch.float)
test1_labels=torch.tensor(test1_labels,dtype=torch.long)
one_hot_labels=torch.zeros(len(test1_labels),class_num).scatter_(1,test1_labels,1)
batch_size=64
# 将训练数据的特征和标签组合
dataset=Data.TensorDataset(test1_features,one_hot_labels)
test_loader=Data.DataLoader(dataset,batch_size,shuffle=True)
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
mlp_model=torch.nn.Sequential(
    OrderedDict([
            ("linear1",torch.nn.Linear(num_inputs,num_outputs)),
            ("activation1",torch.nn.Sigmoid()) ,
            ("linear2",torch.nn.Linear(num_outputs,num_outputs)),
            ]),
        )
torch.nn.init.normal_(mlp_model.linear1.weight,mean=0,std=0.01)
torch.nn.init.constant_(mlp_model.linear1.bias,val=0.01)
torch.nn.init.normal_(mlp_model.linear2.weight,mean=0,std=0.01)
torch.nn.init.constant_(mlp_model.linear2.bias,val=0.01)
criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(mlp_model.parameters(),lr=0.03)
# In[]
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
    for data,target in test_loader:
        logits=mlp_model(data)
        test_loss+=criterion(logits,target).item()
        pred=logits.data.max(1)[1]
        correct+=pred.eq(torch.nonzero(target.data)[:,1]).sum()
    test_loss/=len(test_loader.dataset)
    print("\nEpoch: {}".format(epoch))
    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{}({:.3f}%)".
          format(test_loss,correct,
                 len(test_loader.dataset),
                 100.*correct/len(test_loader.dataset)))
# In[]
summary(mlp_model,(1,2))

# In[]
torch.save(mlp_model, "./model/mlp.model")

# In[]
state = {'model': mlp_model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
torch.save(state, "./model/mlp_dict.model")

# In[]
checkpoint = torch.load("./model/mlp_dict.model")
mlp_model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
start_epoch = checkpoint['epoch']
