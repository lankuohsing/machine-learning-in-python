# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 21:13:45 2021

@author: lankuohsing
"""


import numpy as np
import matplotlib.pyplot as plt
import random
import math

#right
def GenerateDatasetInCycle(
        points_in_rectangle=None,
        point_num=1000,# number of points to generate
        start_theta=0.0,
        end_theta=2*np.pi,
        radius=0.0,
        center_x=0.0,
        center_y=0.0,
        seg_x=None,
        seg_y=None,
        label=0,
        constraint_x=0.0,
        constraint_y=0.0,
        constraint_r=0.0,
        is_constraint=False,
        point_color="black",
        gap=0.1
        ):
    theta=np.linspace(start_theta,end_theta,1000)
    x=np.cos(theta)*radius+center_x
    y=np.sin(theta)*radius+center_y
    if seg_x is not None and seg_y is not None:
        x=np.concatenate((seg_x,x))
        y=np.concatenate((seg_y,y))
    plt.plot(x,y,color="black",linewidth=2)
    dataset=[]
    for point in points_in_rectangle:
        x=point[0]
        y=point[1]
        if label==1:
            if x>=2.0 and x<=3.0-gap:
                if (x-center_x)**2+(y-center_y)**2<=(radius-gap)**2 or  y<=1:
                    dataset.append((x,y,label))
                    plt.plot(x, y, '.', color = point_color)
                    continue
            if x<=2.0:
                if y<=1.0-gap or y<=2.0 and x>=1.0 and (x-1.0)**2+(y-2.0)**2>=(radius+gap)**2:
                    dataset.append((x,y,label))
                    plt.plot(x, y, '.', color = point_color)
                    continue
        if label==2:
            if y>=1.0+gap and y<=2.0:
                if (x-center_x)**2+(y-center_y)**2<=(radius-gap)**2 or x<=1.0:
                    dataset.append((x,y,label))
                    plt.plot(x, y, '.', color = point_color)
                    continue
            if y>=2.0:
                if x<=1.0-gap or x<=2.0 and y<=3.0 and (x-2.0)**2+(y-3.0)**2>=(radius+gap)**2:
                    dataset.append((x,y,label))
                    plt.plot(x, y, '.', color = point_color)
                    continue
        if label==3:
            if x>=1.0+gap and x<=2.0:
                if y>=3.0+gap or y>=2.0 and (x-center_x)**2+(y-center_y)**2<=(radius-gap)**2:
                    dataset.append((x,y,label))
                    plt.plot(x, y, '.', color = point_color)
                    continue
            if x>=2.0:
                if y>=3.0+gap or y>=2.0 and x<=3.0 and (x-3.0)**2+(y-2.0)**2>=(radius+gap)**2:
                    dataset.append((x,y,label))
                    plt.plot(x, y, '.', color = point_color)
                    continue
        if label==4:
            if y>=2.0 and y<=3.0-gap:
                if x>=3.0+gap or x>=2.0 and (x-center_x)**2+(y-center_y)**2<=(radius-gap)**2:
                    dataset.append((x,y,label))
                    plt.plot(x, y, '.', color = point_color)
                    continue
            if y<=2.0:
                if x>=3.0+gap or x>=2.0 and y>=1.0 and (x-2.0)**2+(y-1.0)**2>=(radius+gap)**2:
                    dataset.append((x,y,label))
                    plt.plot(x, y, '.', color = point_color)
                    continue
    return dataset
if __name__=="__main__":
    pi = np.pi
    R = 1

    plt.figure(figsize=(6,6))
    plt.title("circle")
    plt.ylim(0,4)
    plt.xlim(0,4)
    point_num=10000
    points_in_rectangle=[]
    for i in range(0,point_num):
        x=random.uniform(0,4)
        y=random.uniform(0,4)
        points_in_rectangle.append((x,y))
    # In[]
    dataset_1=GenerateDatasetInCycle(points_in_rectangle=points_in_rectangle,
                                     point_num=1000,
                                     start_theta=0.0,
                                     end_theta=pi/2,
                                     radius=R,
                                     center_x=2.0,
                                     center_y=1.0,
                                     seg_x=np.linspace(3.0,3.0,1000),
                                     seg_y=np.linspace(0.0,1.0,1000),
                                     label=1,
                                     constraint_x=2.0,
                                     constraint_y=0.75,
                                     constraint_r=0.5,
                                     is_constraint=True,
                                     point_color='green')
    # In[]
    dataset_2=GenerateDatasetInCycle(points_in_rectangle=points_in_rectangle,
                                     point_num=1000,
                                     start_theta=3*pi/2,
                                     end_theta=2*pi,
                                     radius=R,
                                     center_x=1.0,
                                     center_y=2.0,
                                     seg_x=np.linspace(0.0,1.0,1000),
                                     seg_y=np.linspace(1.0,1.0,1000),
                                     label=2,
                                     constraint_x=0.75,
                                     constraint_y=2.0,
                                     constraint_r=0.5,
                                     is_constraint=True,
                                     point_color='blue')
    dataset_3=GenerateDatasetInCycle(points_in_rectangle=points_in_rectangle,
                                     point_num=1000,
                                     start_theta=pi,
                                     end_theta=3*pi/2,
                                     radius=R,
                                     center_x=2.0,
                                     center_y=3.0,
                                     seg_x=np.linspace(1.0,1.0,1000),
                                     seg_y=np.linspace(3.0,4.0,1000),
                                     label=3,
                                     constraint_x=2.0,
                                     constraint_y=3.25,
                                     constraint_r=0.5,
                                     is_constraint=True,
                                     point_color='red')
    dataset_4=GenerateDatasetInCycle(points_in_rectangle=points_in_rectangle,
                                     point_num=1000,
                                     start_theta=pi/2,
                                     end_theta=pi,
                                     radius=R,
                                     center_x=3.0,
                                     center_y=2.0,
                                     seg_x=np.linspace(3.0,4.0,1000),
                                     seg_y=np.linspace(3.0,3.0,1000),
                                     label=4,
                                     constraint_x=3.25,
                                     constraint_y=2.0,
                                     constraint_r=0.5,
                                     is_constraint=True,
                                     point_color='yellow')
    plt.legend()
    plt.grid()
    plt.savefig("./figures/4_class_data_2d.png")
    plt.show()
    # In[]
    with open("./dataset/4_class_data_2d.txt",'w',encoding="UTF-8") as wf:
        for data in dataset_1:
            wf.write(str(data[0])+" "+str(data[1])+" "+str(data[2])+"\n")
        for data in dataset_2:
            wf.write(str(data[0])+" "+str(data[1])+" "+str(data[2])+"\n")
        for data in dataset_3:
            wf.write(str(data[0])+" "+str(data[1])+" "+str(data[2])+"\n")
        for data in dataset_4:
            wf.write(str(data[0])+" "+str(data[1])+" "+str(data[2])+"\n")