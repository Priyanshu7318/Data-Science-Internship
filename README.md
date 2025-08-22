# Data-Science-Internship

# Data-Science-Internship

## ✅ Problem 1
**Output:**
priyanshushakya@Priyanshus-MacBook-Air DS Internship % /Users/priyanshushakya/.pyenv/versions/3.10.10/bin/python "/Users/priyanshushakya/Desktop/Coding/DS I
nternship/Basic task1 /Basic_level_task1.py"
Scraping page 1...
Scraping page 2...
Scraping page 3...
Scraping page 4...
Scraping page 5...
 Data saved to books_20250822_023648.csv
 Data saved to books_20250822_023648.json
Total books scraped: 100

## ✅ Problem 2
**Output:**
priyanshushakya@Priyanshus-MacBook-Air DS Internship % /Users/priyanshushakya/.pyenv/versions/3.10.10/bin/python "/Users/priyanshushakya/Desktop/Coding/DS I
nternship/Basic task2/Basic_level_task2.py"

=== Original Data ===
     Name   Age  Gender Department    Salary
0    John  28.0    Male         HR   50000.0
1    Anna   NaN  Female         IT   54000.0
2   Peter  35.0    Male    Finance  580000.0
3   Linda  40.0  Female         IT   60000.0
4   James  29.0    Male    Finance       NaN
5  Sophie  32.0  Female         HR   52000.0
/Users/priyanshushakya/Desktop/Coding/DS Internship/Basic task2/Basic_level_task2.py:19: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df['Age'].fillna(df['Age'].mean(), inplace=True)
/Users/priyanshushakya/Desktop/Coding/DS Internship/Basic task2/Basic_level_task2.py:21: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df['Salary'].fillna(df['Salary'].median(), inplace=True)

=== Cleaned & Preprocessed Data ===
     Name       Age  Gender    Salary  Department_Finance  Department_HR  Department_IT
0    John  0.000000       1  0.000000               False           True          False
1    Anna  0.400000       0  0.007547               False          False           True
2   Peter  0.583333       1  1.000000                True          False          False
3   Linda  1.000000       0  0.018868               False          False           True
4   James  0.083333       1  0.007547                True          False          False
5  Sophie  0.333333       0  0.003774               False           True          False

✅ Cleaned data saved to cleaned_dataset.csv
priyanshushakya@Priyanshus-MacBook-Air DS Internship % /Users/priyanshushakya/.pyenv/versions/3.10.10/bin/python "/Users/priyanshushakya/Desktop/Coding/DS I
nternship/Basic task2/Basic_level_task2.py"

=== Original Data ===
     Name   Age  Gender Department    Salary
0    John  28.0    Male         HR   50000.0
1    Anna   NaN  Female         IT   54000.0
2   Peter  35.0    Male    Finance  580000.0
3   Linda  40.0  Female         IT   60000.0
4   James  29.0    Male    Finance       NaN
5  Sophie  32.0  Female         HR   52000.0
/Users/priyanshushakya/Desktop/Coding/DS Internship/Basic task2/Basic_level_task2.py:19: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df['Age'].fillna(df['Age'].mean(), inplace=True)
/Users/priyanshushakya/Desktop/Coding/DS Internship/Basic task2/Basic_level_task2.py:21: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df['Salary'].fillna(df['Salary'].median(), inplace=True)

=== Cleaned & Preprocessed Data ===
     Name       Age  Gender    Salary  Department_Finance  Department_HR  Department_IT
0    John  0.000000       1  0.000000               False           True          False
1    Anna  0.400000       0  0.007547               False          False           True
2   Peter  0.583333       1  1.000000                True          False          False
3   Linda  1.000000       0  0.018868               False          False           True
4   James  0.083333       1  0.007547                True          False          False
5  Sophie  0.333333       0  0.003774               False           True          False

✅ Cleaned data saved to cleaned_dataset.csv



## ✅ Problem 3
**Output:**

priyanshushakya@Priyanshus-MacBook-Air DS Internship % /Users/priyanshushakya/.pyenv/versions/3.10.10/bin/python "/Users/priyanshushakya/Desktop/Coding/DS I
nternship/Basic task3/Basic_level_task3.py"

=== First 5 Rows of Data ===
   Age  Salary  Experience
0   22   25000           1
1   25   30000           3
2   30   40000           5
3   35   50000           7
4   40   60000          10

=== Summary Statistics ===
             Age        Salary  Experience
count   7.000000      7.000000    7.000000
mean   35.285714  52857.142857    7.571429
std    10.355583  23779.743280    5.028490
min    22.000000  25000.000000    1.000000
25%    27.500000  35000.000000    4.000000
50%    35.000000  50000.000000    7.000000
75%    42.500000  67500.000000   11.000000
max    50.000000  90000.000000   15.000000

=== Median Values ===
Age              35.0
Salary        50000.0
Experience        7.0
dtype: float64

=== Variance ===
Age           1.072381e+02
Salary        5.654762e+08
Experience    2.528571e+01
dtype: float64

=== Correlation Matrix ===
                 Age    Salary  Experience
Age         1.000000  0.994428    0.998142
Salary      0.994428  1.000000    0.994584
Experience  0.998142  0.994584    1.000000

=== EDA Insights ===
🔹 Strong positive correlation between Age and Salary.
🔹 Salary increases with Age and Experience in this dataset.


## ✅ Problem 4
**Output:**

priyanshushakya@Priyanshus-MacBook-Air DS Internship % /Users/priyanshushakya/.pyenv/versions/3.10.10/bin/python "/Users/priyanshushakya/Desktop/Coding/DS I
nternship/Intermediate task1/Intermediate_level_task1.py"
⚠ data.csv not found, generating dummy dataset...
✅ Dummy data.csv created and saved!

--- Dataset Preview ---
   feature1  feature2     target
0  3.745401  0.157146  32.101582
1  9.507143  3.182052   4.206998
2  7.319939  1.571780   8.081436
3  5.986585  2.542853  44.927709
4  1.560186  4.537832  30.321453

Linear Regression:
  MSE: 283.1375
  R²: -0.2694

Decision Tree:
  MSE: 658.9477
  R²: -1.9542

Random Forest:
  MSE: 361.4402
  R²: -0.6204

--- Model Comparison ---
               Model         MSE        R²
0  Linear Regression  283.137458 -0.269363
1      Decision Tree  658.947713 -1.954198
2      Random Forest  361.440173 -0.620410


## ✅ Problem 5
**Output:**

priyanshushakya@Priyanshus-MacBook-Air DS Internship % /Users/priyanshushakya/.pyenv/versions/3.10.10/bin/python "/Users/priyanshushakya/Desktop/Coding/DS I
nternship/Intermediate task2/Intermediate_level_task2.py"
=== Logistic Regression ===
Accuracy: 1.0
Precision: 1.0
Recall: 1.0
Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00        10
           1       1.00      1.00      1.00         9
           2       1.00      1.00      1.00        11

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30


=== Random Forest ===
Accuracy: 1.0
Precision: 1.0
Recall: 1.0
Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00        10
           1       1.00      1.00      1.00         9
           2       1.00      1.00      1.00        11

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30


=== SVM ===
Accuracy: 1.0
Precision: 1.0
Recall: 1.0
Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00        10
           1       1.00      1.00      1.00         9
           2       1.00      1.00      1.00        11

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30



## ✅ Problem 6
**Output:**

priyanshushakya@Priyanshus-MacBook-Air DS Internship % /Users/priyanshushakya/.pyenv/versions/3.10.10/bin/python "/Users/priyanshushakya/Desktop/Coding/DS I
nternship/Intermediate task3/Intermediate_level_task3.py"
Sample Data:
    Feature1  Feature2
0  -9.113944  6.813616
1  -9.354576  7.092790
2  -2.015671  8.281780
3  -7.010236 -6.220843
4 -10.061202  6.718671

Silhouette Scores:
k=2 -> Score=0.615
k=3 -> Score=0.799
k=4 -> Score=0.876
k=5 -> Score=0.731
k=6 -> Score=0.585
k=7 -> Score=0.452
k=8 -> Score=0.330
k=9 -> Score=0.338
k=10 -> Score=0.359

 Best k based on Silhouette Score: 4

Cluster Centers (original feature space):
[[-2.60516878  8.99280115]
 [-6.85126211 -6.85031833]
 [ 4.68687447  2.01434593]
 [-8.83456141  7.24430734]]

Cluster Counts:
Cluster
3    75
0    75
1    75
2    75


## ✅ Problem 7
**Output:**

Priyanshushakya@Priyanshus-MacBook-Air DS Internship % /Users/priyanshushakya/.pyenv/versions/3.10.10/bin/python "/Users/priyanshushakya/Desktop/Coding/DS I
nternship/Advanced task1/Advanced_level_task1.py"
Sample Data:
                 Sales
Date                  
2018-01-01  302.000000
2018-02-01  471.408451
2018-03-01  308.816901
2018-04-01  275.225352
2018-05-01  393.633803
/Users/priyanshushakya/.pyenv/versions/3.10.10/lib/python3.10/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/Users/priyanshushakya/.pyenv/versions/3.10.10/lib/python3.10/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/Users/priyanshushakya/.pyenv/versions/3.10.10/lib/python3.10/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/Users/priyanshushakya/.pyenv/versions/3.10.10/lib/python3.10/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/Users/priyanshushakya/Desktop/Coding/DS Internship/Advanced task1/Advanced_level_task1.py:63: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  test['Forecast'] = forecast.values
RMSE: 128.07



## ✅ Problem 8
**Output:**

priyanshushakya@Priyanshus-MacBook-Air DS Internship % /Users/priyanshushakya/.pyenv/versions/3.10.10/bin/python "/Users/priyanshushakya/Desktop/Coding/DS I
nternship/Advanced task2/Advanced_level_task2.py"
[nltk_data] Downloading package stopwords to
[nltk_data]     /Users/priyanshushakya/nltk_data...
[nltk_data]   Unzipping corpora/stopwords.zip.
[nltk_data] Downloading package wordnet to
[nltk_data]     /Users/priyanshushakya/nltk_data...

=== Classification Report ===
/Users/priyanshushakya/.pyenv/versions/3.10.10/lib/python3.10/site-packages/sklearn/metrics/_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
/Users/priyanshushakya/.pyenv/versions/3.10.10/lib/python3.10/site-packages/sklearn/metrics/_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
/Users/priyanshushakya/.pyenv/versions/3.10.10/lib/python3.10/site-packages/sklearn/metrics/_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
              precision    recall  f1-score   support

         ham       0.00      0.00      0.00         2
        spam       0.33      1.00      0.50         1

    accuracy                           0.33         3
   macro avg       0.17      0.50      0.25         3
weighted avg       0.11      0.33      0.17         3


=== Confusion Matrix ===
[[0 2]
 [0 1]]


## ✅ Problem 9
**Output:**

priyanshushakya@Priyanshus-MacBook-Air DS Internship % /Users/priyanshushakya/.pyenv/versions/3.10.10/bin/python "/Users/priyanshushakya/Desktop/Coding/DS I
nternship/Advanced task3/Advanced_level_task3.py"
📥 Loading MNIST dataset...
/Users/priyanshushakya/.pyenv/versions/3.10.10/lib/python3.10/site-packages/keras/src/layers/core/dense.py:92: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
🚀 Training model...
Epoch 1/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 909us/step - accuracy: 0.9298 - loss: 0.2417 - val_accuracy: 0.9606 - val_loss: 0.1295
Epoch 2/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 851us/step - accuracy: 0.9687 - loss: 0.1050 - val_accuracy: 0.9678 - val_loss: 0.1054
Epoch 3/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 855us/step - accuracy: 0.9760 - loss: 0.0767 - val_accuracy: 0.9735 - val_loss: 0.0875
Epoch 4/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 846us/step - accuracy: 0.9829 - loss: 0.0547 - val_accuracy: 0.9709 - val_loss: 0.0989
Epoch 5/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 836us/step - accuracy: 0.9851 - loss: 0.0455 - val_accuracy: 0.9781 - val_loss: 0.0782
Epoch 6/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 832us/step - accuracy: 0.9885 - loss: 0.0356 - val_accuracy: 0.9790 - val_loss: 0.0797
Epoch 7/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 845us/step - accuracy: 0.9902 - loss: 0.0292 - val_accuracy: 0.9747 - val_loss: 0.0945
Epoch 8/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 836us/step - accuracy: 0.9916 - loss: 0.0251 - val_accuracy: 0.9731 - val_loss: 0.1025
Epoch 9/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 836us/step - accuracy: 0.9927 - loss: 0.0223 - val_accuracy: 0.9775 - val_loss: 0.1027
Epoch 10/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 833us/step - accuracy: 0.9931 - loss: 0.0202 - val_accuracy: 0.9788 - val_loss: 0.0895
313/313 - 0s - 415us/step - accuracy: 0.9788 - loss: 0.0895

✅ Test Accuracy: 0.9788
✅ Test Loss: 0.0895
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step

🔍 Sample Predictions:
Image 1: Predicted=7, Actual=7
Image 2: Predicted=2, Actual=2
Image 3: Predicted=1, Actual=1
Image 4: Predicted=0, Actual=0
Image 5: Predicted=4, Actual=4



