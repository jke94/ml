import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

d = pd.read_csv("data.csv", index_col=0)

selected_columns = ['AST','BIL','CHE','CREA','GGT']

dataframe = d[selected_columns]

# Normalize data.
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(dataframe)
df_scaled = pd.DataFrame(x_scaled, 
                         columns=selected_columns)

x = np.array(df_scaled[selected_columns])
y = np.array(d['Category'])

print(x.shape)
print(y.shape)

kmeans = KMeans(n_clusters=6).fit(x)
centroids = kmeans.cluster_centers_
print(centroids)

# Predicting the clusters
labels = kmeans.predict(x)
# Getting the cluster centers
C = kmeans.cluster_centers_
colores=['red','green','blue','cyan','yellow','orange']
asignar=[]
for row in labels:
    asignar.append(colores[row])
 
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=asignar,s=60)
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c=colores, s=1000)

# Getting the values and plotting it
f1 = dataframe['AST'].values
f2 = dataframe['BIL'].values

plt.scatter(f1, f2, c=asignar, s=70)
plt.scatter(C[:, 0], C[:, 1], marker='*',c=colores, s=1000)
plt.show()