# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 23:07:43 2021

@author: lankuohsing
"""

# In[]
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


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
        plt.plot(x, y, '.', color = point_color)
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
    dict_label_points=defaultdict(list)
    with open("./dataset/random_samples.txt",'r',encoding="UTF-8") as rf:
        for line in rf:
            split_list=line.strip().split(" ")
            x=float(split_list[0])
            y=float(split_list[1])
            label=int(split_list[2])
            points_in_rectangle.append((x,y))
            dict_label_points[label].append((x,y))
    # In[]
    dataset_1=GenerateDatasetInCycle(points_in_rectangle=dict_label_points[0],
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
    dataset_2=GenerateDatasetInCycle(points_in_rectangle=dict_label_points[1],
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
    dataset_3=GenerateDatasetInCycle(points_in_rectangle=dict_label_points[2],
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
    dataset_4=GenerateDatasetInCycle(points_in_rectangle=dict_label_points[3],
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
    plt.savefig("./figures/weak_points.png")
    plt.show()