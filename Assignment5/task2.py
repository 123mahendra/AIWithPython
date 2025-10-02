
#####Task2#####

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("50_Startups.csv", delimiter=',')

print("First rows of the dataset:")
print(df.head())

print("\nInformation of the Dataset:")
print(df.info())

sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True)
plt.tight_layout()
plt.show()

plt.subplot(1,2,1)
plt.scatter(df['R&D Spend'], df['Profit'])
plt.xlabel('R&D Spend')
plt.ylabel('Profit')
plt.title('R&D Spend VS Profit')

plt.subplot(1,2,2)
plt.scatter(df['Marketing Spend'], df['Profit'])
plt.xlabel('Marketing Spend')
plt.ylabel('Profit')
plt.title('Marketing Spend VS Profit')
plt.tight_layout()
plt.show()

X = df[['R&D Spend', 'Marketing Spend']]
y = df['Profit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

lm = LinearRegression()
lm.fit(X_train, y_train)
y_train_predict = lm.predict(X_train)
y_test_predict = lm.predict(X_test)

rmse_train = np.sqrt(mean_squared_error(y_train, y_train_predict))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_predict))

r2_train = r2_score(y_train, y_train_predict)
r2_test = r2_score(y_test, y_test_predict)

print(f"RMSE Train : {rmse_train} and R2 Train: {r2_train}")

print(f"RMSE Test: {rmse_test} and R2 Test: {r2_test}")

"""
Findings:
    Values are:
        RMSE Train : 9358.583115148496 and R2 Train: 0.9436198878593198
        RMSE Test: 7073.857168705303 and R2 Test: 0.9683604384024198
        
Explanation:
    The model predicts profit very well, with R² values of 94% for training 
    and 96% for testing, meaning it explains most of the variation in profit. 
    The RMSE values about 9359 for training and 7074 for testing show prediction 
    errors are small compared to actual profits.

"""