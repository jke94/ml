import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("data.csv", index_col=0)
selected_columns = ['Age','AST','BIL','CHE','CREA','GGT']
dataframe = df[selected_columns]

# Normalize data.
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(dataframe)
df_scaled = pd.DataFrame(x_scaled, 
                         columns=selected_columns)

# Select values
x = np.array(df[selected_columns])
y = np.array(df['Category'])

print(f'Shape x: {x.shape}')
print(f'Shape y: {y.shape}')

Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i, random_state=0, n_init="auto") for i in Nc]
score = [kmeans[i].fit(x).score(x) for i in range(len(kmeans))]

x_norm = np.array(df_scaled[selected_columns])
y = np.array(df['Category'])
score_data_scaled = [kmeans[i].fit(x_norm).score(x_norm) for i in range(len(kmeans))]

fig, axs = plt.subplots(1,2)
fig.suptitle('Data scaled vs non scaled')

axs[0].plot(Nc, score_data_scaled)
axs[0].set_ylabel('Score')
axs[0].set_xlabel('Number of Clusters')
axs[0].set_title('Elbow Curve')

axs[1].plot(Nc, score)
axs[1].set_ylabel('Score')
axs[1].set_xlabel('Number of Clusters')
axs[1].set_title('Elbow Curve')

# plt.show()
fig.savefig('ElbowCurve.png')