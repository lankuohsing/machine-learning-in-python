# -*- coding: utf-8 -*-

import sys
sys.path.append("../")
from data_processing import dataset_split
import numpy as np
import torch.utils.data as Data
import torch
from collections import OrderedDict
# In[]
num_class_1=4
num_class_2=4

data_list=[]
label1_list=[]
label2_list=[]
label3_list=[]
one_hot_label_list=[]
with open("./dataset/multi-label-data-4-class.txt",'r',encoding="UTF-8") as rf:
    for line in rf:
        split_list=line.strip().split(" ")
        x=float(split_list[0])
        y=float(split_list[1])
        label1=int(split_list[2])
        label2=int(split_list[3])
        label3=int(label1*10+label2)
        one_hot_label=[0 for i in range(num_class_1+num_class_2)]
        one_hot_label[label1]=1
        one_hot_label[num_class_1+label2]=1
        data_list.append([x,y])
        label1_list.append(label1)
        label2_list.append(label2)
        label3_list.append(label3)
        one_hot_label_list.append(one_hot_label)
# In[]
splitter=dataset_split.MultiClassSplitter(data_list,label3_list,multi_labels=one_hot_label_list)

data_trains, label3_trains, multi_label_trains, data_tests, label3_tests, multi_label_tests=splitter.stratified_train_test( test_size=0.25, random_state=0,duplicate_minor=True)

# In[]
data_train=data_trains[0]
data_test=data_tests[0]
multi_label_train=multi_label_trains[0]
multi_label_test=multi_label_tests[0]

# In[]
def gen_torch_dataset(data,one_hot_labels):
    features=torch.tensor(data,dtype=torch.float)
    one_hot_labels=torch.tensor(one_hot_labels,dtype=torch.float)
    batch_size=64
    # 将训练数据的特征和标签组合
    dataset=Data.TensorDataset(features,one_hot_labels)
    # 随机读取小批量
    data_loader=Data.DataLoader(dataset,batch_size,shuffle=True)
    return data_loader
train_loader=gen_torch_dataset(data_train,multi_label_train)
test_loader=gen_torch_dataset(data_test,multi_label_test)
epochs=100
# In[]
num_inputs=len(data_train[0])
num_outputs=len(multi_label_train[0])

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

for epoch in range(epochs):
    for batch_idx,(feature_in_on_batch,label_in_one_batch) in enumerate(train_loader):
        logits=mlp_model(feature_in_on_batch)
        loss=criterion(logits[:,0:num_class_1],label_in_one_batch[:,0:num_class_1])+criterion(logits[:,num_class_1:num_class_1+num_class_2],label_in_one_batch[:,num_class_1:num_class_1+num_class_2])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#        if batch_idx % 100==0:
#            print("Train Epoch: {} [{}/{}({:0f}%)]\tLoss: {:6f}".format(epoch,batch_idx*len(feature_in_on_batch),len(train_loader.dataset),100.*batch_idx/len(train_loader),loss.item()))

    test_loss=0
    correct1=0
    correct2=0
    for data,target in test_loader:
        logits=mlp_model(data)
        test_loss+=criterion(logits[:,0:num_class_1],target[:,0:num_class_1]).item()+criterion(logits[:,num_class_1:num_class_1+num_class_2],target[:,num_class_1:num_class_1+num_class_2]).item()
        pred1=logits.data[:,0:num_class_1].max(1)[1]
        pred2=logits.data[:,num_class_1:num_class_1+num_class_1].max(1)[1]
        correct1+=pred1.eq(torch.nonzero(target.data[:,0:num_class_1])[:,1]).sum()
        correct2+=pred2.eq(torch.nonzero(target.data[:,num_class_1:num_class_1+num_class_2])[:,1]).sum()
    test_loss/=len(test_loader.dataset)
    print("\nTest set: Average loss: {:.4f}, Accuracy: label1: {}/{}({:.3f}%),label2: {}/{}({:.3f}%)".
          format(test_loss,correct1,
                 len(test_loader.dataset),
                 100.*correct1/len(test_loader.dataset),correct2,
                 len(test_loader.dataset),
                 100.*correct2/len(test_loader.dataset)))