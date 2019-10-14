#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 16:58:12 2019

@author: mmps
"""
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

print("Full data sample")
files = ['S1_Dataset', 'S2_Dataset']
full_data = pd.DataFrame()
for f in files:
    for filename in os.listdir(f):
        if filename != 'README.txt':
            data_path = f + '/' + filename
            data=pd.read_csv(data_path, header=None)
            full_data = full_data.append(data, ignore_index=True)
            
full_data.columns = ['time','frontal','vertical','lateral','id','rssi','phase','frequency','activity']
full_data.info()

print("Apresentando o shape dos dados (dimenssoes)")
print(full_data.shape)
full_data=full_data.interpolate()

print("Visualizando o conjunto inicial (head) dos dados, ou mais claramente, os 20 primeiros registros (head(20))")
print(full_data.head(20))

print("Conhecendo os dados estatisticos dos dados carregados (describe)")
print(full_data.describe())

print("Conhecendo a distribuicao dos dados por classes (class distribution)")
print(full_data.groupby('activity').size())

print("Criando grafios de caixa da distribuicao das classes")
full_data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False, figsize  = [10, 10])
plt.show()

print("Criando histogramas dos dados por classes")
full_data.hist(figsize=[10, 10])
plt.show()

print("Criando graficos de dispersao dos dados")
colors_palette = {1: 'red', 2: 'yellow', 3: 'blue', 4: 'green'}
colors = [colors_palette[c] for c in full_data['activity']]
scatter_matrix(full_data[['time','frontal','vertical','lateral','id','rssi','phase','frequency']], c=colors, figsize  = [10, 10])
plt.show()

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
for x,c,action in [(1,'r', 'sit on bed'),(2,'g','sit on chair'),(3,'b', 'lying'),(4,'k', 'ambulating')]:
    xs = full_data.loc[full_data['activity'] == x]['frontal']
    ys = full_data.loc[full_data['activity'] == x]['vertical']
    zs = full_data.loc[full_data['activity'] == x]['lateral']
    ax.scatter(xs, ys, zs, c=c, marker='.', label=action)

ax.legend()
ax.set_xlabel('frontal')
ax.set_ylabel('vertical')
ax.set_zlabel('lateral')    
ax.view_init(None, 30)
plt.show()

fig = plt.figure(figsize=(20,10))
ax=fig.add_subplot(111)
for x,c,action in [(1,'r', 'sit on bed'),(2,'g','sit on chair'),(3,'b', 'lying'),(4,'k', 'ambulating')]:
    ts = full_data.loc[full_data['activity'] == x]['time']
    xs = full_data.loc[full_data['activity'] == x]['frontal']
    ys = full_data.loc[full_data['activity'] == x]['vertical']
    zs = full_data.loc[full_data['activity'] == x]['lateral']
    ax.scatter(ts, xs, c=c, marker='.', label=action)
    ax.scatter(ts, ys, c=c, marker='x', label=action)
    ax.scatter(ts, zs, c=c, marker='^', label=action)
ax.legend()
ax.set_xlabel('time(s)')
ax.set_ylabel('acceleration(g)')
plt.show()

pca = PCA().fit(full_data.values[0:-1, :])
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');


print("First data sample")
single_sample = pd.read_csv("S1_Dataset/d1p01M", header=None)
single_sample.columns = ['time','frontal','vertical','lateral','id','rssi','phase','frequency','activity']

print("Conhecendo a distribuicao dos dados por classes (class distribution)")
print(single_sample.groupby('activity').size())

print("Criando graficos de dispersao dos dados")
colors_palette = {1: 'red', 2: 'yellow', 3: 'blue', 4: 'green'}
colors = [colors_palette[c] for c in single_sample['activity']]
scatter_matrix(single_sample[['time','frontal','vertical','lateral','id','rssi','phase','frequency']], c=colors, figsize  = [10, 10])
plt.show()

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
for x,c,action in [(1,'r', 'sit on bed'),(2,'g','sit on chair'),(3,'b', 'lying'),(4,'k', 'ambulating')]:
    xs = single_sample.loc[single_sample['activity'] == x]['frontal']
    ys = single_sample.loc[single_sample['activity'] == x]['vertical']
    zs = single_sample.loc[single_sample['activity'] == x]['lateral']
    ax.scatter(xs, ys, zs, c=c, marker='.', label=action)

ax.legend()
ax.set_xlabel('frontal')
ax.set_ylabel('vertical')
ax.set_zlabel('lateral')    
ax.view_init(None, 30)
plt.show()

fig = plt.figure(figsize=(20,10))
ax=fig.add_subplot(111)
for x,c,action in [(1,'r', 'sit on bed'),(2,'g','sit on chair'),(3,'b', 'lying'),(4,'k', 'ambulating')]:
    ts = single_sample.loc[single_sample['activity'] == x]['time']
    xs = single_sample.loc[single_sample['activity'] == x]['frontal']
    ys = single_sample.loc[single_sample['activity'] == x]['vertical']
    zs = single_sample.loc[single_sample['activity'] == x]['lateral']
    ax.scatter(ts, xs, c=c, marker='.', label=action)
    ax.scatter(ts, ys, c=c, marker='x', label=action)
    ax.scatter(ts, zs, c=c, marker='^', label=action)
ax.legend()
ax.set_xlabel('time(s)')
ax.set_ylabel('acceleration(g)')
plt.show()
