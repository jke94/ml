import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# Read full data and select columns for PCA.
dataframe = pd.read_csv("forestfires.csv")

numeric_cols = dataframe.select_dtypes(include=['float64', 'int']).columns.to_list()
cat_cols = dataframe.select_dtypes(include=['object', 'category']).columns.to_list()

pipeline = ColumnTransformer(
    [('scale', StandardScaler(), numeric_cols),
    ('onehot', OneHotEncoder(handle_unknown='ignore'), cat_cols),
])
preprocessed_dataset = pipeline.fit_transform(dataframe)
df_transformed = pd.DataFrame(preprocessed_dataset)

print(df_transformed)
