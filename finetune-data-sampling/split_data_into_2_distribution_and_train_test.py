# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 22:03:40 2021

@author: lankuohsing
"""
import random
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
#        if True:
            data1.append([x,y])
            labels1.append([label-1])
# In[]
index1_list=list(range(0,len(data1)))
index2_list=list(range(0,len(data2)))
random.shuffle(index1_list)
random.shuffle(index2_list)
# In[]
train_percenage=0.8
train1_len=int(train_percenage*len(index1_list))
train2_len=int(train_percenage*len(index2_list))
train1_data=[]
train1_label=[]
train2_data=[]
train2_label=[]
test1_data=[]
test1_label=[]
test2_data=[]
test2_label=[]
for i in range(0,train1_len):
    train1_data.append(data1[index1_list[i]])
    train1_label.append(labels1[index1_list[i]])
for i in range(0,train2_len):
    train2_data.append(data2[index2_list[i]])
    train2_label.append(labels2[index2_list[i]])
for i in range(train1_len,len(index1_list)):
    test1_data.append(data1[index1_list[i]])
    test1_label.append(labels1[index1_list[i]])
for i in range(train2_len,len(index2_list)):
    test2_data.append(data2[index2_list[i]])
    test2_label.append(labels2[index2_list[i]])
# In[]
with open("./dataset/train1_dataset.txt",'w',encoding="UTF-8") as wf:
    for i, data in enumerate(train1_data):
        label=train1_label[i]
        wf.write(str(data[0])+" "+str(data[1])+" "+str(label[0])+"\n")
with open("./dataset/test1_dataset.txt",'w',encoding="UTF-8") as wf:
    for i, data in enumerate(test1_data):
        label=test1_label[i]
        wf.write(str(data[0])+" "+str(data[1])+" "+str(label[0])+"\n")
with open("./dataset/train2_dataset.txt",'w',encoding="UTF-8") as wf:
    for i, data in enumerate(train2_data):
        label=train2_label[i]
        wf.write(str(data[0])+" "+str(data[1])+" "+str(label[0])+"\n")
with open("./dataset/test2_dataset.txt",'w',encoding="UTF-8") as wf:
    for i, data in enumerate(test2_data):
        label=test2_label[i]
        wf.write(str(data[0])+" "+str(data[1])+" "+str(label[0])+"\n")