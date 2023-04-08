import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Read full data and select columns for PCA.
df = pd.read_csv("data.csv", index_col=0)
selected_columns = ['Age','AST','BIL','CHE','CREA','GGT']
n_components = len(selected_columns) + 1
dataframe = df[selected_columns]

# Standardization
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(dataframe)
df_scaled = pd.DataFrame(x_scaled, columns=selected_columns)

# PCA
pca = PCA()
pca.fit(df_scaled)
print(pca.explained_variance_ratio_)

# Plot PCA
# plt.plot(range(1,n_components), 
#          pca.explained_variance_ratio_.cumsum(),
#          marker='o',
#          linestyle='--')

# plt.title('Explained Variance by Components')
# plt.xlabel('Number of Components')
# plt.ylabel('Cumulative Explained Variance')
# plt.grid(True)
# plt.show()

n_components = range(1,len(selected_columns))

for n_comp in n_components:
    pca = PCA(n_components=n_comp)
    # pca = PCA(n_components=3)
    pca.fit(df_scaled)
    scores_pca = pca.transform(df_scaled)

    wcss = []
    for i in range(1, 20):
        kmeans = KMeans(n_clusters=i, 
                        init='k-means++', 
                        max_iter=20, 
                        n_init=10, 
                        random_state=0)
        kmeans.fit(scores_pca)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 20), wcss)
plt.title('Elbow Method with num. of components')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.legend(n_components)
plt.grid(True)
plt.show()