# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 12:03:49 2022

@author: lankuohsing
"""
from collections import defaultdict
import random
import logging
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

# In[]

class MultiClassSplitter(object):
    def __init__(self, X_list=None, y_list=None, multi_labels=None):
        self.X_list=X_list
        self.y_list=y_list
        self.multi_labels=multi_labels
        self.dict_multilabel_num, self.dict_multilabel_samples=self.get_nums_samples_dict( X_list, y_list)


    def get_nums_samples_dict(self, X_list, y_list,multi_labels=None):
        """
        X_list:[feature1,feture2,...]
        y_list:[label1, label2, ...]
        """
        dict_multilabel_num=defaultdict(int)
        dict_multilabel_samples=defaultdict(lambda: defaultdict(list))

        for index,label in enumerate(y_list):
            dict_multilabel_num[label]+=1
            dict_multilabel_samples[label]["X"].append(X_list[index])
            dict_multilabel_samples[label]["y"].append(label)
            if multi_labels is not None:
                dict_multilabel_samples[label]["multi_label"].append(multi_labels[index])
        return dict_multilabel_num, dict_multilabel_samples

    def stratified_k_fold(self, n_splits, drop_minority=False, random_state=None):
        major_labels=[]
        minor_labels=[]
        for label, num in self.dict_multilabel_num.items():
            if num < n_splits:
                logging.warning("Label {} only has {} samples!".format(label,num))
                minor_labels.append(label)
            else:
                major_labels.append(label)
        if not drop_minority:#保留所有类别样本
            X_list=self.X_list
            y_list=self.y_list
        else:#丢弃样本数小于n_splits的类别
            for label in minor_labels:
                logging.warning("Dropping label {}!".format(label))
            X_list=[]
            y_list=[]
            for label in major_labels:
                X_list.extend(self.dict_multilabel_samples[label]["X"])
                y_list.extend(self.dict_multilabel_samples[label]["y"])
        X_folds=[]
        y_folds=[]
        skf = StratifiedKFold(n_splits=n_splits,random_state=random_state)
        for train_index, test_index in skf.split(X_list, y_list):
            X_fold_temp=[]
            y_fold_temp=[]
            for index in test_index:
                X_fold_temp.append(X_list[index])
                y_fold_temp.append(y_list[index])
            X_folds.append(X_fold_temp)
            y_folds.append(y_fold_temp)
        return X_folds, y_folds
    def stratified_k_sampling(self, n_splits, drop_minority=False, random_state=None):
        major_labels=[]
        minor_labels=[]
        for label, num in self.dict_multilabel_num.items():
            if num < n_splits:
                logging.warning("Label {} only has {} samples!".format(label,num))
                minor_labels.append(label)
            else:
                major_labels.append(label)
        if not drop_minority:#保留所有类别样本
            X_list=self.X_list
            y_list=self.y_list
        else:#丢弃样本数小于n_splits的类别
            for label in minor_labels:
                logging.warning("Dropping label {}!".format(label))
            X_list=[]
            y_list=[]
            for label in major_labels:
                X_list.extend(self.dict_multilabel_samples[label]["X"])
                y_list.extend(self.dict_multilabel_samples[label]["y"])
        X_folds=[]
        y_folds=[]
        skf = StratifiedKFold(n_splits=n_splits,random_state=random_state)
        for train_index, test_index in skf.split(X_list, y_list):
            X_fold_temp=[]
            y_fold_temp=[]
            for index in train_index:
                X_fold_temp.append(X_list[index])
                y_fold_temp.append(y_list[index])
            X_folds.append(X_fold_temp)
            y_folds.append(y_fold_temp)
        return X_folds, y_folds
    def stratified_train_test(self, n_splits=1, *, test_size=0.1, train_size=None, random_state=None):
        major_labels=[]
        minor_labels=[]
        for label, num in self.dict_multilabel_num.items():
            if num < 2:
                logging.warning("Label {} only has {} samples!".format(label,1))
                minor_labels.append(label)
            else:
                major_labels.append(label)
        for label in minor_labels:
            logging.warning("Dropping label {}!".format(label))
        X_list=[]
        y_list=[]
        multi_labels=[]
        for label in major_labels:
            X_list.extend(self.dict_multilabel_samples[label]["X"])
            y_list.extend(self.dict_multilabel_samples[label]["y"])
            if self.multi_labels is not None:
                multi_labels.extend(self.dict_multilabel_samples[label]["multi_label"])
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size,train_size=train_size, random_state=random_state)
        sss.get_n_splits(X_list, y_list)
        X_trains=[]
        y_trains=[]
        multi_label_trains=[]
        X_tests=[]
        y_tests=[]
        multi_label_tests=[]
        for train_index, test_index in sss.split(X_list, y_list):
            X_train_temp=[]
            y_train_temp=[]
            multi_label_train_temp=[]
            X_test_temp=[]
            y_test_temp=[]
            multi_label_test_temp=[]
            for index in test_index:
                X_test_temp.append(X_list[index])
                y_test_temp.append(y_list[index])
                if self.multi_labels is not None:
                    multi_label_test_temp.append(multi_labels[index])
            for index in train_index:
                X_train_temp.append(X_list[index])
                y_train_temp.append(y_list[index])
                if self.multi_labels is not None:
                    multi_label_train_temp.append(multi_labels[index])
            X_trains.append(X_train_temp)
            y_trains.append(y_train_temp)
            X_tests.append(X_test_temp)
            y_tests.append(y_test_temp)
            if self.multi_labels is not None:
                multi_label_trains.append(multi_label_train_temp)
                multi_label_tests.append(multi_label_test_temp)
        return X_trains, y_trains, multi_label_trains, X_tests, y_tests, multi_label_tests
# In[]
if __name__=="__main__":

    X=[[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]
    y=[0,0,0,0,0,0,0,0,1,1]

    splitter=MultiClassSplitter(X,y)
    X_folds, y_folds=splitter.stratified_k_sampling(3, drop_minority=False)
    print(X_folds)
    print(y_folds)
    X=[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]
    y=[0, 0, 0, 0, 1, 1]
    splitter=MultiClassSplitter(X,y)
    X_trains, y_trains, multi_label_trains, X_tests, y_tests, multi_label_tests=splitter.stratified_train_test( test_size=0.25, random_state=0)
    print(X_trains, y_trains, multi_label_trains, X_tests, y_tests, multi_label_tests)

