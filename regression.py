import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv('Housing.csv')
print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset Information:")
df.info()

binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
df[binary_cols] = df[binary_cols].apply(lambda x: x.map({'yes': 1, 'no': 0}))

furnishing_dummies = pd.get_dummies(df['furnishingstatus'], drop_first=True, dtype=int)
df = pd.concat([df, furnishing_dummies], axis=1)

df.drop('furnishingstatus', axis=1, inplace=True)

print("\nDataset after handling categorical variables:")
print(df.head())

scaler = MinMaxScaler()

num_vars = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']
df[num_vars] = scaler.fit_transform(df[num_vars])

print("\nDataset after scaling numerical features:")
print(df.head())

plt.figure(figsize=(8, 5))
sns.histplot(df['price'], kde=True)
plt.title('Distribution of House Prices')
plt.xlabel('Price (Scaled)')
plt.ylabel('Frequency')
plt.show()


plt.figure(figsize=(8, 5))
sns.scatterplot(x='area', y='price', data=df)
plt.title('Area vs. Price')
plt.show()


plt.figure(figsize=(16, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of All Features')
plt.show()

y = df.pop('price')
X = df

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")