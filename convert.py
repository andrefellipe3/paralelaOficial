import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

# Carregar o arquivo CSV
df = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")

# Exibir as primeiras linhas para inspecionar os dados
print(df.head())

# Selecionar colunas que podem ser usadas para análise
# Por exemplo, podemos usar colunas numéricas para aplicar o K-means
# Vamos pegar todas as colunas numéricas para o exemplo
df_numeric = df.select_dtypes(include=[float, int])

# Tratar dados ausentes, se houver (por exemplo, removendo as linhas com valores nulos)
df_numeric = df_numeric.dropna()

# Normalizar os dados para garantir que as diferentes escalas não afetem o K-means
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_numeric)

# Salvar os dados processados em um arquivo CSV para o código C
np.savetxt("processed_data_diabetes.csv", scaled_data, delimiter=",")
