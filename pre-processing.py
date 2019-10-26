#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------
# Projeto Soluções de Mineração de Dados
# ----------------------------------------
# ********** Pré-processamento **********

# Base de dados: 
# Activity recognition with healthy older people using a batteryless
# wearable sensor Data Set
#

#%%
# *********************************
# *** Importação de bibliotecas ***
# *********************************

import glob
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#%%
# ***********************************
# *** Pré-processamento dos dados ***
# ***********************************

# Salvar todos os data paths dos arquivos
data_path_arquivos = sorted(glob.glob('S*_Dataset/d*'))

df = pd.DataFrame()

# Nome dos atributos dos arquivos da base de dados
colunas = ['tempo', 'frontal', 'vertical', 'lateral', 'antena', 'rssi', 
           'fase', 'frequencia', 'atividade']

for data_path in data_path_arquivos:
    pasta = data_path[0:11]         # Salva o nome da pasta, 'S1_Dataset/'
    nome_arquivo = data_path[11:]   # Salva o nome do arquivo, ex.: 'd1p01M'
    
    if nome_arquivo != 'README.txt':
        # Leitura do arquivo
        data = pd.read_csv(data_path, header=None, names=colunas)
        
        # Substituir diretamente os caracteres por valores numéricos, para
        # criação das colunas 'sala' e 'sexo':
        data['sala'] = (0, 1)[nome_arquivo.startswith('d2')] # S1: 0 / S2:1
        data['sexo'] = (0, 1)[nome_arquivo.endswith('F')] # 'M':0 / 'F':1
        
        # Juntando todos os arquivos lidos em um mesmo dataframe
        df = df.append(data, ignore_index=True)

# Reordenamento das colunas
df = df[['sala', 'sexo', 'tempo', 'frontal', 'vertical', 'lateral', 'antena', 
         'rssi', 'fase', 'frequencia', 'atividade']]

# Separação dos dados em atributos e classe
X = df.values[:, 0:-1]
y = df.values[:, -1]

#%% *** Normalização dos dados ***
MinMax = MinMaxScaler(feature_range=(0, 1))

X_escalonado = MinMax.fit_transform(X)

df_Normalizado = pd.DataFrame(X_escalonado)

#%% *** Análise de Componentes Principais (PCA) ***

pca = PCA().fit(X_escalonado)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Número de componentes')
plt.ylabel('Variância explicada cumulativa')
plt.title('Análise da variância explicada com PCA\n(normalização MinMax)', size=16)
plt.show()