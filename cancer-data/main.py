import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

np.seterr(divide='ignore', invalid='ignore')

# Read full data and select columns for PCA.
dataframe = pd.read_csv("cancer-data.csv", sep=',')
labels = dataframe['diagnosis']

dataframe = dataframe.drop('id', axis=1)
dataframe = dataframe.drop('diagnosis', axis=1)

numeric_cols = dataframe.select_dtypes(include=['float64', 'int']).columns.to_list()
cat_cols = dataframe.select_dtypes(include=['object', 'category']).columns.to_list()

pipeline = ColumnTransformer(
    [('scale', StandardScaler(), numeric_cols),
    # ('onehot', OneHotEncoder(handle_unknown='ignore'), cat_cols),
])
preprocessed_dataset = pipeline.fit_transform(dataframe)
df_features_transformed = pd.DataFrame(preprocessed_dataset)

# Dataset split: Train & Test.
datasets = train_test_split(df_features_transformed, 
                            labels,
                            test_size=0.25)

train_data, test_data, train_labels, test_labels = datasets

# Classifier
clf = MLPClassifier(solver='lbfgs',
                    max_iter=10000,
                    activation='relu',
                    alpha=1e-6,
                    hidden_layer_sizes=(5,5), 
                    random_state=1)

clf.fit(train_data, train_labels)
score = clf.score(train_data, train_labels)

# Predictions: Train and test dataset.
predictions_train = clf.predict(train_data)
predictions_test = clf.predict(test_data)

# Accuracy score.
train_score = accuracy_score(predictions_train, train_labels)
print(f'score on train data: {train_score}')
test_score = accuracy_score(predictions_test, test_labels)
print(f'score on test data: {test_score}')

# Print confusion matrix.
print(f'Confusion matrix:\n{confusion_matrix(predictions_train, train_labels)}')
print(f'Confusion matrix:\n{confusion_matrix(predictions_test, test_labels)}')

# Print classification
print(classification_report(predictions_test, test_labels))