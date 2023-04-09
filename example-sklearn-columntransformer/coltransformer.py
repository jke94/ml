import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


# Read full data and select columns for PCA.
dataframe = pd.read_csv("Stars.csv")
print(dataframe)
num_attrs = ["Temperature", "L", "R"]
text_attrs = ["Color", "Spectral_Class"]

pipeline = ColumnTransformer([
                              ("numeric", StandardScaler(), num_attrs),
                              ("text", OneHotEncoder(), text_attrs)
])
preprocessed_dataset = pipeline.fit_transform(dataframe)

dataframe_transformed = pd.DataFrame(preprocessed_dataset.toarray())

print(dataframe_transformed)
