import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn import preprocessing
import math
from pickle import dump
base = pd.read_csv('HousingData.csv', sep=',')
print('Registros:', len(base))
 
base = base.dropna()
base = base.reset_index(drop=True) 
print('Registros sem valores ausentes:', len(base))

base_numerica = base.drop(columns=['CHAS']) 
base_categorica = base['CHAS']
nome_categorico = 'CHAS'
 
escalador = preprocessing.MinMaxScaler()

modelo_escalador = escalador.fit(base_numerica)

dump(modelo_escalador, open('imoveis_escalador.model', 'wb'))
 
base_numerica_escalada = modelo_escalador.fit_transform(base_numerica)

base_numerica_escalada = pd.DataFrame(base_numerica_escalada, columns=base_numerica.columns)
 
base_categorica_binaria = pd.get_dummies(base_categorica, prefix=nome_categorico, dummy_na=False, dtype=int)
 
base_final = base_numerica_escalada.join(base_categorica_binaria)
 
erros =[]
valores_k = range(1,100) 
for k in valores_k:
    modelo_kmeans = KMeans(n_clusters=k, random_state=42).fit(base_final)
    erros.append(
        sum(
            np.min(
                cdist(base_final, modelo_kmeans.cluster_centers_, 'euclidean'),
                axis=1)/base_final.shape[0]
            )
    )

k_inicial = valores_k[0]
erro_inicial = erros[0]
k_final = valores_k[-1]
erro_final = erros[-1]
afastamentos = []
for i in range(len(erros)):
    k_atual=valores_k[i]
    erro_atual=erros[i]
    parte_cima = abs(
        (erro_final-erro_inicial)*k_atual - (k_final-k_inicial)*erro_atual + k_final*erro_inicial - erro_final*k_inicial
    )
    parte_baixo= math.sqrt(
        (erro_final-erro_inicial)**2 + (k_final-k_inicial)**2 
    )
    afastamentos.append(parte_cima/parte_baixo)
 
k_ideal =valores_k[afastamentos.index(max(afastamentos))]

agrupador = KMeans(n_clusters=k_ideal, random_state=42).fit(base_final)
 
dump(
    agrupador,
    open('imoveis_agrupador.model','wb')
)
print('Número ótimo de clusters:', k_ideal)
