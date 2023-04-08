import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

dataframe = pd.read_csv("data.csv", index_col=0)
corr_matrix = dataframe.corr()
sn.heatmap(corr_matrix, annot=True, vmin=0.30)
plt.show()