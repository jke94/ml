import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Read data
df = pd.read_csv("data.csv", index_col=0)
selected_columns = ['Age','AST','BIL','CHE','CREA','GGT']
dataframe = df[selected_columns]

# Standardization
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(dataframe)
df_scaled = pd.DataFrame(x_scaled, columns=selected_columns)
print(df_scaled)