import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler



data = {
    'Age': [25, 30, np.nan, 35, 40],
    'Salary': [50000, 60000, 55000, np.nan, 65000],
    'Score': [90, 85, 88, 75, np.nan]
}

df = pd.DataFrame(data)
print("Original Data:")
print(df)


df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Salary'] = df['Salary'].fillna(df['Salary'].median())
df['Score'] = df['Score'].fillna(df['Score'].mode()[0])
print("\nData after handling missing values:")
print(df)


df = df.apply(pd.to_numeric)

scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)  

print("\nNormalized data:")
print(df_normalized)