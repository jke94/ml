import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Read full data and select columns for PCA.

raw_dataframe = pd.read_csv("diabetes.csv")

# Features (dataframe columns selected).

df_selected_columns = [
    'cholesterol',          # Feature 0
    'glucose',              # Feature 1
    'hdl_chol',             # Feature 2
    'chol_hdl_ratio',       # Feature 3
    'age',                  # Feature 4
    'gender',               # Feature 5
    'height',               # Feature 6
    'weight',               # Feature 7
    'bmi',                  # Feature 8
    'systolic_bp',          # Feature 9
    'diastolic_bp',         # Feature 10
    'waist',                # Feature 11
    'hip',                  # Feature 12
    'waist_hip_ratio',      # Feature 13
]

# Labels
labels = raw_dataframe['diabetes']

# Data pre-processing: Converting str values to float.

raw_dataframe['chol_hdl_ratio'] = raw_dataframe['chol_hdl_ratio'].str.replace(',', '.').astype(float)
raw_dataframe['waist_hip_ratio'] = raw_dataframe['waist_hip_ratio'].str.replace(',', '.').astype(float)

# Data pre-processing: Scaling values.

numeric_cols = raw_dataframe.select_dtypes(include=['float64', 'int']).columns.to_list()
pipeline = ColumnTransformer(
    [('scale', StandardScaler(), numeric_cols)])
preprocessed_dataset = pipeline.fit_transform(raw_dataframe)

df_transformed = pd.DataFrame(preprocessed_dataset)

# Dataset spliting: Train & Test.

datasets = train_test_split(df_transformed, labels, test_size=0.2)
train_data, test_data, train_labels, test_labels = datasets

# Selecting Multi-layer Perceptron classifier with values in hyperparameters.

clf = MLPClassifier(solver='lbfgs',
                    max_iter=10000,
                    activation='relu',
                    alpha=1e-6,
                    hidden_layer_sizes=(5,3), 
                    random_state=1)

# Training and prediction model.

clf.fit(train_data, train_labels)
score = clf.score(train_data, train_labels)

predictions_train = clf.predict(train_data)
predictions_test = clf.predict(test_data)

# Getting precision metrics and show info.

train_score = accuracy_score(predictions_train, train_labels)
print("Score on train data: ", train_score)

test_score = accuracy_score(predictions_test, test_labels)
print("Score on test data: ", test_score)

train_confusion_matrix = confusion_matrix(predictions_train, train_labels)
test_confusion_matrix = confusion_matrix(predictions_test, test_labels)

print(classification_report(predictions_test, test_labels))