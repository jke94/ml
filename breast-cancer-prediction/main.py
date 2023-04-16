import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Read full data and select columns for PCA.

raw_dataframe = pd.read_csv("breast-cancer-dataset.csv")

# Features (dataframe columns selected).
df_selected_columns = [
    'radius_mean',              # Feature 00
    'texture_mean',             # Feature 01
    'perimeter_mean',           # Feature 02
    'area_mean',                # Feature 03
    'smoothness_mean',          # Feature 04
    'compactness_mean',         # Feature 05
    'concavity_mean',           # Feature 06
    'concave points_mean',      # Feature 07
    'symmetry_mean',            # Feature 08   
    'fractal_dimension_mean',   # Feature 09
    'radius_se',                # Feature 10
    'texture_se',               # Feature 11
    'perimeter_se',             # Feature 12
    'area_se',                  # Feature 13
    'smoothness_se',            # Feature 14
    'compactness_se',           # Feature 15
    'concavity_se',             # Feature 16
    'concave points_se',        # Feature 17
    'symmetry_se',              # Feature 18
    'fractal_dimension_se',     # Feature 19
    'radius_worst',             # Feature 20
    'texture_worst',            # Feature 21
    'perimeter_worst',          # Feature 22
    'area_worst',               # Feature 23
    'smoothness_worst',         # Feature 24
    'compactness_worst',        # Feature 25
    'concavity_worst',          # Feature 26
    'concave points_worst',     # Feature 27
    'symmetry_worst',           # Feature 28
    'fractal_dimension_worst'   # Feature 29
    ]

# Labels
labels = raw_dataframe['diagnosis']

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