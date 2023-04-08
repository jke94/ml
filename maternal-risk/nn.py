import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Read full data and select columns for PCA.
df = pd.read_csv("MaternalHealthRiskDataSet.csv")

selected_columns = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']

dataframe = df[selected_columns]

# Standardization
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(dataframe)
df_scaled = pd.DataFrame(x_scaled, columns=selected_columns)

# Dataset split: Train & Test.
datasets = train_test_split(df_scaled, 
                            df['RiskLevel'],
                            test_size=0.2)

train_data, test_data, train_labels, test_labels = datasets

clf = MLPClassifier(max_iter=6000,
                    solver='lbfgs',
                    # max_iter = 50,
                    # activation='relu',
                    alpha=1e-5,
                    hidden_layer_sizes=(6,3), 
                    random_state=1)

clf.fit(train_data, train_labels)
score = clf.score(train_data, train_labels)

predictions_train = clf.predict(train_data)
predictions_test = clf.predict(test_data)

train_score = accuracy_score(predictions_train, train_labels)
print("score on train data: ", train_score)
test_score = accuracy_score(predictions_test, test_labels)
print("score on test data: ", test_score)

confusion_matrix(predictions_train, train_labels)
confusion_matrix(predictions_test, test_labels)

print(classification_report(predictions_test, test_labels))

# Save to onnx model.
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
initial_type = [('float_input', FloatTensorType([None, 7]))]
onx = convert_sklearn(classification_report, initial_types=initial_type)
with open("maternal-risk.onnx", "wb") as f:
    f.write(onx.SerializeToString())