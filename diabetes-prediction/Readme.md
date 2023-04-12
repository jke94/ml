# Predict diabetes based on diagnostic measures

## A. Dataset
- URL: https://www.kaggle.com/datasets/houcembenmansour/predict-diabetes-based-on-diagnostic-measures

## B. How to run:

0. Create virtualenv and install requirements.txt dependencies.
1. Changes to 'diabetes-prediction' directorty as base path.

2. Run:
```
python main.py
```

Similar output like this...
```
Score on train data:  0.9839743589743589
Score on test data:  0.8589743589743589
              precision    recall  f1-score   support

    Diabetes       0.50      0.45      0.48        11
 No diabetes       0.91      0.93      0.92        67

    accuracy                           0.86        78
   macro avg       0.71      0.69      0.70        78
weighted avg       0.85      0.86      0.86        78
```