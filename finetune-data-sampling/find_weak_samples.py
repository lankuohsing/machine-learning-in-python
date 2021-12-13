# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 21:45:50 2021

@author: lankuohsing
"""

import json
from collections import defaultdict
import random
# In[]
with open("./dataset/dict_class_samples.json",'r',encoding='UTF-8') as load_f:
    dict_class_samples_str = json.load(load_f)

# In[]
dict_class_samples=defaultdict(list)

for key,value in dict_class_samples_str.items():
    for sample_str in value:
        splist_list=sample_str.split(" ")
        x=float(splist_list[0])
        y=float(splist_list[1])
        prob_dev1=float(splist_list[2])
        prob_dev2=float(splist_list[3])
        prob_dev3=float(splist_list[4])
        prob_dev4=float(splist_list[5])
        dict_class_samples[int(key)].append((x,y,prob_dev1,prob_dev2,prob_dev3,prob_dev4))
# In[]
weak_num=20
weak_samples=[]
for key,value in dict_class_samples.items():
    labels=[0,1,2,3]
    labels.remove(key)
    for label in labels:
        value.sort(key=lambda t:t[label+2],reverse=False)
        for i in range(0,weak_num):
            weak_samples.append((value[i],key))
# In[]
with open("./dataset/weak_samples.txt",'w',encoding="UTF-8") as wf:
    for sample in weak_samples:
        wf.write(str(sample[0][0])+" "+str(sample[0][1])+" "+str(sample[1])+"\n")
# In[]
random_samples=[]
for key,value in dict_class_samples.items():
    for label in labels:
        random.shuffle(value)
        for i in range(0,weak_num):
            random_samples.append((value[i],key))
with open("./dataset/random_samples.txt",'w',encoding="UTF-8") as wf:
    for sample in random_samples:
        wf.write(str(sample[0][0])+" "+str(sample[0][1])+" "+str(sample[1])+"\n")