# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 12:03:49 2022

@author: lankuohsing
"""
from collections import defaultdict
import random
random.seed(1)
# In[]
"""位置0，1，2，3为第一个label，4，5，6，为第二个label"""
X_list=[]
y_list=[]
for i in range(0,100):
    X_list.append([i+1]*10)
    y_temp=[0]*7
    y_temp[i%4]=1
    y_temp[(i)%2+4]=1
    y_list.append(y_temp.copy())
i=100
X_list.append([i+1]*10)
y_temp=[0]*7
y_temp[0]=1
y_temp[-1]=1
y_list.append(y_temp.copy())

i=101
X_list.append([i+1]*10)
y_temp=[0]*7
y_temp[0]=1
y_temp[-1]=1
y_list.append(y_temp.copy())
# In[]
dict_multilabel_num=defaultdict(int)
dict_multilabel_samples=defaultdict(list)
for index,y_temp in enumerate(y_list):
    multilabel_str=""
    for j,y_j in enumerate(y_temp):
        if y_j==1:
            multilabel_str+=str(j)+"_"
    multilabel_str=multilabel_str[:-1]
    dict_multilabel_num[multilabel_str]+=1
    dict_multilabel_samples[multilabel_str].append({"X":X_list[index],"y":y_list[index]})
# In[]
test_ratio=0.2
dict_multilabel_train=defaultdict(list)
dict_multilabel_test=defaultdict(list)
train_X=[]
test_X=[]
train_y=[]
test_y=[]

for multilabel,samples in dict_multilabel_samples.items():

    print("processing label ",multilabel)
    index_list=list(range(0,len(samples)))
    random.shuffle(index_list)
    train_index_end=int(len(index_list)*(1-test_ratio))
#    test_index_end=len(index_list)
    if train_index_end<=0:
        train_index_end+=1
    if train_index_end >=len(index_list):
        assert(len(index_list)==1)
        print("warning! label",multilabel,"only has 1 sample! test set is the same as train set")
        test_index_start=train_index_end-1
    else:
        test_index_start=train_index_end
#    print("train index")
    for i in range(0,train_index_end):
        index=index_list[i]
#        print(index)
        dict_multilabel_train[multilabel].append(samples[index])
        train_X.append(samples[index]["X"])
        train_y.append(samples[index]["y"])
#    print("test index")
    for i in range(test_index_start,len(index_list)):
        index=index_list[i]
#        print(index)
        dict_multilabel_test[multilabel].append(samples[index])
        test_X.append(samples[index]["X"])
        test_y.append(samples[index]["y"])
random.seed(10)
random.shuffle(train_X)
random.seed(10)
random.shuffle(train_y)

random.seed(20)
random.shuffle(test_X)
random.seed(20)
random.shuffle(test_y)