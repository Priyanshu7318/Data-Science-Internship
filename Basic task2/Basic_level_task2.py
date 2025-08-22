import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from scipy import stats


data = {
    "Name": ["John", "Anna", "Peter", "Linda", "James", "Sophie"],
    "Age": [28, np.nan, 35, 40, 29, 32],
    "Gender": ["Male", "Female", "Male", "Female", "Male", "Female"],
    "Department": ["HR", "IT", "Finance", "IT", "Finance", "HR"],
    "Salary": [50000, 54000, 580000, 60000, np.nan, 52000]  
}
df = pd.DataFrame(data)

print("\n=== Original Data ===")
print(df)

df['Age'].fillna(df['Age'].mean(), inplace=True)

df['Salary'].fillna(df['Salary'].median(), inplace=True)

z_scores = np.abs(stats.zscore(df[['Salary']]))
df = df[(z_scores < 3).all(axis=1)]  

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

df = pd.get_dummies(df, columns=['Department'])

scaler = MinMaxScaler()
df[['Age', 'Salary']] = scaler.fit_transform(df[['Age', 'Salary']])

print("\n=== Cleaned & Preprocessed Data ===")
print(df)

df.to_csv("cleaned_dataset.csv", index=False)
print("\n✅ Cleaned data saved to cleaned_dataset.csv")
