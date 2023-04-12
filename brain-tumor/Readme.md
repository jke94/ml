# Brain tumor

## A. Dataset
- URL: https://www.kaggle.com/datasets/jillanisofttech/brain-tumor

## B. How to run:

0. Create virtualenv and install requirements.txt dependencies.
1. Changes to 'brain-tumor' directorty as base path.

2. Run:
```
python main.py
```

Similar output like this...
```
Score on train data:  0.9642857142857143
Score on test data:  1.0
              precision    recall  f1-score   support

      Normal       1.00      1.00      1.00         5
       tumor       1.00      1.00      1.00         3

    accuracy                           1.00         8
   macro avg       1.00      1.00      1.00         8
weighted avg       1.00      1.00      1.00         8
```