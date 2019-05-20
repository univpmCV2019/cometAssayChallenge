#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 12:04:02 2019

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
np.random.seed(1)

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

data_path = '' #folder dove sono contenuti i dati
img_dir = os.path.join(data_path, 'images/stdrossi_png') #folder immagini
full_labels = pd.read_excel(os.path.join(data_path,'stdrossi_png.xlsx')) #labels

#estraggo dall'excel solo le colonne di interesse
reformat = full_labels.loc[:,['CurrentFieldPath_STR50','CometRectangleLeft_I32','CometRectangleTop_I32','CometRectangleRight_I32','CometRectangleBottom_I32','CometTailIntensity_SGL',]]
reformat.columns = ['Image name','xmin','ymin','xmax','ymax','Class']
reformat.insert(1, 'Width', '768')
reformat.insert(2, 'Height', '574')

#divisione in 3 classi di intensità
conditions = [(reformat['Class'] <= 0.2), (reformat['Class'] > 0.2) & (reformat['Class'] < 0.7), (reformat['Class'] >= 0.7)]
labels = ['low', 'medium','high']
reformat['Class'] = np.select(conditions, labels)
reformat = reformat[reformat.Class != '0']

grouped = reformat.groupby('Image name')
grouped_list = [grouped.get_group(x) for x in grouped.groups]

#divido i dati in train, validation, test
train = []
other = []
val = []
test = []
i = 0
folders = sorted(listdir_nohidden(img_dir))
for item in (folders):
        folder = os.path.join(img_dir,item)
        images = []
        for imagename in sorted(listdir_nohidden(folder)):
            images.append('{0}/{1}'.format(item,imagename))

        train.append(np.random.choice(images, int(len(images)*0.6), replace=False))
        other.append(np.setdiff1d(images,train[i]))
        val.append(np.random.choice(other[i], int(len(images)*0.2), replace=False))
        test.append(np.setdiff1d(other[i], val[i]))
        i = i+1

train = np.concatenate(train, axis=0)
val = np.concatenate(val, axis=0)
test = np.concatenate(test, axis=0)

train_label = []
val_label = []
test_label = []
for i in range(0,len(grouped_list)):
    data = grouped_list[i]
    name = data.iloc[0]['Image name']
    if name in train:
        train_label.append(grouped_list[i])
    elif name in test: 
        test_label.append(grouped_list[i])
    else:
        val_label.append(grouped_list[i])
        
train_label = pd.concat(train_label, axis=0)
train_label.sort_index(axis=0, level=None, ascending=True, inplace=True, kind='quicksort', na_position='last', sort_remaining=True, by=None)
val_label = pd.concat(val_label, axis=0)
val_label.sort_index(axis=0, level=None, ascending=True, inplace=True, kind='quicksort', na_position='last', sort_remaining=True, by=None)
test_label = pd.concat(test_label, axis=0)

train_label['Class'].value_counts()

#salvo in csv
train_label.to_csv('train_labels.csv', index=None)
val_label.to_csv('val_labels.csv', index=None)
test_label.to_csv('test_labels.csv', index=None)