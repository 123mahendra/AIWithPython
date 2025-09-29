
#####Task 2#####

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

df = pd.read_csv('weight-height.csv')

xValue = df[["Height"]].values
yValue = df["Weight"].values

plt.scatter(xValue, yValue,color='red')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('Scatter Plot of Weight and Height')
plt.show()

model = LinearRegression()

model.fit(xValue, yValue)

yValuePredict = model.predict(xValue)

plt.scatter(xValue, yValue,color='red')
plt.plot(xValue, yValuePredict, color='blue', label='Linear Regression')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('Linear Regression of Weight and Height')
plt.show()

RMSE = np.sqrt(mean_squared_error(yValue, yValuePredict))
R2 = r2_score(yValue, yValuePredict)

print("Root Mean Squared Error = ", RMSE)
print("R2 = ", R2)