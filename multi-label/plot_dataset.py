# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

# 要绘制单个点，可使用函数scatter() ，并向它传递一对 x 和 y 坐标，


# 它将在指定位置绘制一个点：
plt.scatter(2, 4, s=200,c='k') # 设置图表标题并给坐标轴加上标签
# 调用了scatter() ，并使用实参s 设置了绘制图形时使用的点的尺寸
plt.title("Square Numbers", fontsize=24)
plt.xlabel("Value", fontsize=14)
plt.ylabel("Square of Value", fontsize=14) # 设置刻度标记的大小
plt.tick_params(axis='both', which='major', labelsize=14)
plt.show()

# 使用使 scatter() 绘制一系列点
x_values = [1, 2, 3, 4, 5]
y_values = [1, 4, 9, 16, 25]
plt.scatter(x_values, y_values, s=5,c='g') # 设置图表标题并给坐标轴指定标签
plt.show()

# In[]
x_list=[]
y_list=[]
label1_list=[]
label2_list=[]
colors=['g','b','r','y']
green_list=[]
blue_list=[]
red_list=[]
yellow_list=[]

with open("dataset/4_class_data_2d.txt",'r',encoding="UTF-8") as rf:
    for line in rf:
        split_list=line.strip().split(" ")
        x=float(split_list[0])
        y=float(split_list[1])
        label1=int(split_list[2])-1
        if y<=x:
            if x+y<=4:
                label2=0
            else:
                label2=3
        else:
            if x+y<=4:
                label2=1
            else:
                label2=2
        x_list.append(x)
        y_list.append(y)
        label1_list.append(label1)
        label2_list.append(label2)
        if label2==0:
            green_list.append((x,y,label1,label2))
            continue
        if label2==1:
            blue_list.append((x,y,label1,label2))
            continue
        if label2==2:
            red_list.append((x,y,label1,label2))
            continue
        if label2==3:
            yellow_list.append((x,y,label1,label2))
            continue
# In[]
plt.scatter(x=[e[0] for e in green_list], y=[e[1] for e in green_list], s=5,c='g')
plt.scatter(x=[e[0] for e in blue_list], y=[e[1] for e in blue_list], s=5,c='b')
plt.scatter(x=[e[0] for e in red_list], y=[e[1] for e in red_list], s=5,c='r')
plt.scatter(x=[e[0] for e in yellow_list], y=[e[1] for e in yellow_list], s=5,c='y')
plt.show()

# In[]
with open("./dataset/multi-label-data-4-class.txt",'w',encoding="UTF-8") as wf:
    for data in green_list:
        line=str(data[0])+" "+str(data[1])+" "+str(data[2])+" "+str(data[3])+"\n"
        wf.write(line)
    for data in blue_list:
        line=str(data[0])+" "+str(data[1])+" "+str(data[2])+" "+str(data[3])+"\n"
        wf.write(line)
    for data in red_list:
        line=str(data[0])+" "+str(data[1])+" "+str(data[2])+" "+str(data[3])+"\n"
        wf.write(line)
    for data in yellow_list:
        line=str(data[0])+" "+str(data[1])+" "+str(data[2])+" "+str(data[3])+"\n"
        wf.write(line)