import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Read full data and select columns for PCA.
df = pd.read_csv("MaternalHealthRiskDataSet.csv")
print(df)
selected_columns = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']

n_components = len(selected_columns) + 1
dataframe = df[selected_columns]

# Standardization
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(dataframe)
df_scaled = pd.DataFrame(x_scaled, columns=selected_columns)
