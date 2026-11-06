nova_casa_num = [[0.00737, 22.0, 7.31, 0.638, 5.785, 72.3, 10.02, 3, 244, 17.3, 396.9, 20.98, 27.0]]

nova_casa_cat = [[1, 0]]
 
cols_num = ['CRIM','ZN','INDUS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
cols_cat = ['CHAS_0.0','CHAS_1.0']
 
import pandas as pd
from pickle import load
 
escalador = load(open('imoveis_escalador.model', 'rb'))

agrupador = load(open('imoveis_agrupador.model', 'rb'))
 
nova_casa_num = escalador.transform(nova_casa_num)
nova_casa_num = pd.DataFrame(nova_casa_num, columns=cols_num)

nova_casa_cat = pd.DataFrame(nova_casa_cat, columns=cols_cat)

nova_casa = nova_casa_num.join(nova_casa_cat)
 
grupo_da_casa = agrupador.predict(nova_casa)
print('Cluster da nova residência:', grupo_da_casa[0])
 
centro = agrupador.cluster_centers_[grupo_da_casa]
centro = pd.DataFrame(centro, columns=cols_num + cols_cat)
 
centro_num = centro.drop(columns=cols_cat)
centro_cat = centro[cols_cat]
 
centro_num_original = escalador.inverse_transform(centro_num)
centro_num_original = pd.DataFrame(centro_num_original, columns=cols_num)
 
centro_cat = centro_cat.round()
centro_cat_original = pd.from_dummies(centro_cat, sep='_')
 
centro_original = centro_num_original.join(centro_cat_original)
print('Centroide desnormalizado:')
print(centro_original.T)
