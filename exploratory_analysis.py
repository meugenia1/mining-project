#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------
# Projeto Soluções de Mineração de Dados
# ----------------------------------------

# Base de dados: 
# Activity recognition with healthy older people using a batteryless
# wearable sensor Data Set
#
# Grupo:
# Karl Sousa (kvms) 
# Maria Eugênia (meps)
# Mateus Silva (mmps)
# Vitor Cardim (vcm3)

#%%
# *********************************
# *** Importação de bibliotecas ***
# *********************************

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score

from mpl_toolkits.mplot3d import Axes3D #corrige erro projeção 3D

#%%
# ********************************
# *** Leitura da base de dados ***
# ********************************

#%%
# Geração de listas de nomes dos arquivos por grupos (gênero e sala)
arquivosSala1 = sorted(glob.glob('S1_Dataset/d*'))
arquivosSala2 = sorted(glob.glob('S2_Dataset/d*'))

arquivosGerais = arquivosSala1 + arquivosSala2

arquivosHomens = [ arq for arq in arquivosGerais if arq.endswith('M') ]
arquivosMulheres = [ arq for arq in arquivosGerais if arq.endswith('F') ]

arquivosHomensS1 = [ arq for arq in arquivosSala1 if arq.endswith('M') ]
arquivosHomensS2 = [ arq for arq in arquivosSala2 if arq.endswith('M') ]

arquivosMulheresS1 = [ arq for arq in arquivosSala1 if arq.endswith('F') ]
arquivosMulheresS2 = [ arq for arq in arquivosSala2 if arq.endswith('F') ]

# Criação das bases de dados
dataS1 = pd.concat([pd.read_csv(f, header=None) for f in 
                arquivosSala1], ignore_index = True)

dataS2 = pd.concat([pd.read_csv(f, header=None) for f in 
                arquivosSala2], ignore_index = True)

dataS1.columns = dataS2.columns = ['tempo','frontal','vertical','lateral',
                                   'antena','rssi','fase','frequencia',
                                   'atividade']

full_data = pd.concat([dataS1, dataS2])

# *************************************
# *** Análise Exploratória de Dados ***
# *************************************

#%%
# *** Análise no conjunto dos dados ***
# Aqui será feita a AED no conjunto dos dados, ou seja, concatenando
# os diferentes ensaios de coleta de dados de diferentes pessoas

dataS1.info()
dataS2.info()

print("Apresentando as dimensões dos dados (shape)")
print("S1:")
print(dataS1.shape)

print("S2:")
print(dataS2.shape)

print("Visualizando os 10 primeiros registros (head(10))")
print("S1:")
print(dataS1.head(10))

print("S2:")
print(dataS2.head(10))

print("Conhecendo os dados estatísticos dos dados carregados (describe)")
print("S1:")
print(dataS1.describe())

print("S2:")
print(dataS2.describe())

print("Conhecendo a distribuição dos dados por classes (class distribution)")
print(dataS1.groupby('atividade').size())
print(dataS2.groupby('atividade').size())

print("Criando gráficos de caixa da distribuição das classes")
print("S1:")
dataS1.plot(kind='box', subplots=True, layout=(3,3), sharex=False, 
            sharey=False, figsize  = [10, 10])
plt.show()

print("S2:")
dataS2.plot(kind='box', subplots=True, layout=(3,3), sharex=False, 
            sharey=False, figsize  = [10, 10])
plt.show()

print("Criando histogramas dos dados por classes")
print("S1:")
dataS1.hist(figsize=[10, 10])
plt.show()

print("S2:")
dataS2.hist(figsize=[10, 10])
plt.show()

print("Criando gráficos de dispersão dos dados")
print("S1:")
colors_palette = {1: 'red', 2: 'yellow', 3: 'blue', 4: 'green'}
colors = [colors_palette[c] for c in dataS1['atividade']]
scatter_matrix(dataS1[['tempo','frontal','vertical','lateral','antena','rssi',
                       'fase','frequencia']], c=colors, figsize  = [10, 10])
plt.show()


print("S2:")
colors = [colors_palette[c] for c in dataS2['atividade']]
scatter_matrix(dataS2[['tempo','frontal','vertical','lateral','antena','rssi',
                       'fase','frequencia']], c=colors, figsize  = [10, 10])
plt.show()

# Plot 3D da classificação da classe segundo as acelerações nos 3 eixos
print("S1:")
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
for x,c,action in [(1,'r', 'sentado na cama'), (2,'g','sentado na cadeira'), 
                   (3,'b', 'deitado'), (4,'k', 'movimento')]:
    xs = dataS1.loc[dataS1['atividade'] == x]['frontal']
    ys = dataS1.loc[dataS1['atividade'] == x]['vertical']
    zs = dataS1.loc[dataS1['atividade'] == x]['lateral']
    ax.scatter(xs, ys, zs, c=c, marker='.', label=action)

ax.legend()
ax.set_xlabel('frontal')
ax.set_ylabel('vertical')
ax.set_zlabel('lateral')
ax.view_init(None, 30)
plt.show()

print("S2:")
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
for x,c,action in [(1,'r', 'sentado na cama'), (2,'g','sentado na cadeira'),
                   (3,'b', 'deitado'), (4,'k', 'em movimento')]:
    xs = dataS2.loc[dataS2['atividade'] == x]['frontal']
    ys = dataS2.loc[dataS2['atividade'] == x]['vertical']
    zs = dataS2.loc[dataS2['atividade'] == x]['lateral']
    ax.scatter(xs, ys, zs, c=c, marker='.', label=action)

#ax.legend()
#ax.set_xlabel('frontal')
#ax.set_ylabel('vertical')
#ax.set_zlabel('lateral')    
#ax.view_init(None, 30)
plt.show()

# Gráfico da aceleração x tempo, classificando segundo as classes
print("Considerando todo o conjunto de dados:")
fig = plt.figure(figsize=(20,10))
ax=fig.add_subplot(111)
for x,c,action in [(1,'r', 'sentado na cama'), (2,'g','sentado na cadeira'),
                   (3,'b', 'deitado'), (4,'k', 'em movimento')]:
    ts = full_data.loc[full_data['atividade'] == x]['tempo']
    xs = full_data.loc[full_data['atividade'] == x]['frontal']
    ys = full_data.loc[full_data['atividade'] == x]['vertical']
    zs = full_data.loc[full_data['atividade'] == x]['lateral']
    ax.scatter(ts, xs, c=c, marker='.', label=action)
    ax.scatter(ts, ys, c=c, marker='x', label=action)
    ax.scatter(ts, zs, c=c, marker='^', label=action)
ax.legend()
ax.set_xlabel('tempo (s)')
ax.set_ylabel('aceleração (g)')
plt.show()

#%%
# *** Análise em um ensaio de coleta de dados ***
# Aqui será feita a AED considerando-se somente um ensaio de coleta, 
# ou seja, a partir da sequência temporal de dados de um único arquivo

print("Primeiro ensaio de coleta (1º arquivo)")
single_sample = pd.read_csv("S1_Dataset/d1p01M", header=None)
single_sample.columns = ['tempo', 'frontal', 'vertical', 'lateral', 'antena', 
                         'rssi', 'fase', 'frequencia', 'atividade']

print("Conhecendo a distribuição dos dados por classes (class distribution)")
print(single_sample.groupby('atividade').size())

print("Criando gráficos de dispersão dos dados")
colors_palette = {1: 'red', 2: 'yellow', 3: 'blue', 4: 'green'}
colors = [colors_palette[c] for c in single_sample['atividade']]
scatter_matrix(single_sample[['tempo', 'frontal', 'vertical', 'lateral', 
                              'antena', 'rssi', 'fase', 'frequencia']], 
                c=colors, figsize  = [10, 10])
plt.show()

# Plot 3D da classificação da classe segundo as acelerações nos 3 eixos
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
for x,c,action in [(1,'r', 'sentado na cama'), (2,'g','sentado na cadeira'),
                   (3,'b', 'deitado'), (4,'k', 'em movimento')]:
    xs = single_sample.loc[single_sample['atividade'] == x]['frontal']
    ys = single_sample.loc[single_sample['atividade'] == x]['vertical']
    zs = single_sample.loc[single_sample['atividade'] == x]['lateral']
    ax.scatter(xs, ys, zs, c=c, marker='.', label=action)

ax.legend()
ax.set_xlabel('frontal')
ax.set_ylabel('vertical')
ax.set_zlabel('lateral')    
ax.view_init(None, 30)
plt.show()

# Gráfico da aceleração x tempo, classificando segundo as classes
fig = plt.figure(figsize=(20,10))
ax=fig.add_subplot(111)
for x,c,action in [(1,'r', 'sentado na cama'), (2,'g','sentado na cadeira'),
                   (3,'b', 'deitado'),(4,'k', 'em movimento')]:
    ts = single_sample.loc[single_sample['atividade'] == x]['tempo']
    xs = single_sample.loc[single_sample['atividade'] == x]['frontal']
    ys = single_sample.loc[single_sample['atividade'] == x]['vertical']
    zs = single_sample.loc[single_sample['atividade'] == x]['lateral']
    ax.scatter(ts, xs, c=c, marker='.', label=action)
    ax.scatter(ts, ys, c=c, marker='x', label=action)
    ax.scatter(ts, zs, c=c, marker='^', label=action)
ax.legend()
ax.set_xlabel('tempo (s)')
ax.set_ylabel('aceleração (g)')
plt.show()

X = full_data.values[:, 0:-1]
y = full_data.values[:, -1]

# Split the data into 40% test and 60% training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# Create a random forest classifier
clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)

# Train the classifier
clf.fit(X_train, y_train)

# Lista com importância de cada feature
for feature in zip(full_data.columns, clf.feature_importances_):
    print(feature)
    
