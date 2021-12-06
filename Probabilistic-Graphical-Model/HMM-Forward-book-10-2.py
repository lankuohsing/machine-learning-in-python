# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 23:15:04 2021

@author: lankuohsing
"""


"""
https://www.cnblogs.com/gongyanzh/p/12880387.html
假设从三个 袋子 {1,2,3}中 取出 4 个球 O={red,white,red,white}，模型参数λ=(A,B,π) 如下，计算序列O出现的概率
"""
# 状态 1 2 3之间的转移概率矩阵
A = [[0.5,0.2,0.3],
     [0.3,0.5,0.2],
     [0.2,0.3,0.5]]
# 初始状态概率
pi = [0.2,0.4,0.4]
# 每个袋子里，红白求的概率
# red white
B = [[0.5,0.5],
     [0.4,0.6],
     [0.7,0.3]]

# In[]
#前向算法
def hmm_forward(A,B,pi,O):
    T = len(O)# 观测序列长度
    N = len(A[0])# 状态个数
    #step1 初始化
    alpha = [[0]*T for _ in range(N)]# 每行代表不同的状态，每列代表不同的观测时刻
    for i in range(N):
        alpha[i][0] = pi[i]*B[i][O[0]]

    #step2 计算alpha(t)
    for t in range(1,T):
        for i in range(N):
            temp = 0
            for j in range(N):
                temp += alpha[j][t-1]*A[j][i]
            alpha[i][t] = temp*B[i][O[t]]

    #step3
    proba = 0
    for i in range(N):
        proba += alpha[i][-1]
    return proba,alpha

A = [[0.5,0.2,0.3],
     [0.3,0.5,0.2],
     [0.2,0.3,0.5]]
B = [[0.5,0.5],
     [0.4,0.6],
     [0.7,0.3]]
pi = [0.2,0.4,0.4]
O = [0,1,0]
proba,alpha=hmm_forward(A,B,pi,O)  #结果为 0.130218
print(proba)