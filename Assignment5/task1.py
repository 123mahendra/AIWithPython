#####Task 1#####

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

data = load_diabetes(as_frame=True)
df = data['frame']

plt.hist(df['target'],25)
plt.xlabel('target')
plt.show()

sns.heatmap(data=df.corr().round(2),annot=True)
plt.show()

plt.subplot(1,2,1)
plt.scatter(df['bmi'], df['target'])
plt.xlabel('bmi')
plt.ylabel('target')

plt.subplot(1,2,2)
plt.scatter(df['s5'], df['target'])
plt.xlabel('s5')
plt.ylabel('target')
plt.tight_layout()
plt.show()

X = pd.DataFrame(df[['bmi', 's5']], columns=['bmi', 's5'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

lm = LinearRegression()
lm.fit(X_train, y_train)
y_train_predict = lm.predict(X_train)
y_test_predict = lm.predict(X_test)

rmse_train = np.sqrt(mean_squared_error(y_train, y_train_predict))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_predict))

r2_train = r2_score(y_train, y_train_predict)
r2_test = r2_score(y_test, y_test_predict)

newX = pd.DataFrame(df[['bmi', 's5', 'bp']], columns=['bmi', 's5', 'bp'])
X_train, X_test, y_train, y_test = train_test_split(newX, y, test_size=0.2, random_state=5)

newLm = LinearRegression()
newLm.fit(X_train, y_train)
new_y_train_predict = newLm.predict(X_train)
new_y_test_predict = newLm.predict(X_test)

new_rmse_train = np.sqrt(mean_squared_error(y_train, new_y_train_predict))
new_rmse_test = np.sqrt(mean_squared_error(y_test, new_y_test_predict))

new_r2_train = r2_score(y_train, new_y_train_predict)
new_r2_test = r2_score(y_test, new_y_test_predict)

print("Train metrics of bmi and s5: ")
print(f"RMSE: {rmse_train} and R2: {r2_train}")

print("Test metrics of bmi and s5: ")
print(f"RMSE: {rmse_test} and R2: {r2_test}")

print("Train metrics of bmi, s5 and bp: ")
print(f"RMSE: {new_rmse_train} and R2: {new_r2_train}")

print("Test metrics of bmi, s5, and bp: ")
print(f"RMSE: {new_rmse_test} and R2: {new_r2_test}")

"""
Answers:

a) 
=> I choose blood pressure(bp) as a next variable 
    because diabetes progression is closely related 
    to the blood pressure(bp) and it often makes disease worse.

b)
=> After Adding the next variable bp, i got improvement in both RMSE and R2 values.
    RMSE decreases and R2 increases slightly.
    
c)
=> Yes, it help in result after adding more variable which improves the model.
    RMSE decreases from around 56.56 to 55.32 in train metrices where in test metrices
     it decreases from around 57.17 to 56.62.
    R2 increases from around 0.45 to 0.47 in train metrices where in test metrices
     it increases from around 0.48 to 0.49.
    
    So, after adding next variable it's giving the improvement in result.

"""


