
#####Task3#####

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from matplotlib import pyplot as plt

df = pd.read_csv('Auto.csv')
# print(df.head())

X = df.drop(columns=["mpg", "name", "origin"])
y = df["mpg"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

alphas = np.logspace(-3, 3, 50)

ridge_scores = []
lasso_scores = []

for alpha in alphas:
    ridge = Ridge(alpha = alpha)
    ridge.fit(X_train, y_train)
    ridge_scores.append(ridge.score(X_test, y_test))

    lasso = Lasso(alpha = alpha)
    lasso.fit(X_train, y_train)
    lasso_scores.append(lasso.score(X_test, y_test))

plt.plot(alphas, ridge_scores, marker='x', label='Ridge')
plt.plot(alphas, lasso_scores, marker='o', label='Lasso')

plt.show()

best_ridge_index = np.argmax(ridge_scores)
best_ridge_alpha = alphas[best_ridge_index]
best_ridge_score = ridge_scores[best_ridge_index]


best_lasso_index = np.argmax(lasso_scores)
best_lasso_alpha = alphas[best_lasso_index]
best_lasso_score = lasso_scores[best_lasso_index]

print(f"Best Ridge alpha: {best_ridge_alpha}, R2 Score: {best_ridge_score}")
print(f"Best LASSO alpha: {best_lasso_alpha}, R2 Score: {best_lasso_score}")

"""
Findings:
    Values:
        Best Ridge alpha: 0.001, R2 Score: 0.7942
        Best LASSO alpha: 0.001, R2 Score: 0.7941
        
Explanations:
    The results show that both Ridge and LASSO regression achieve their best performance
    at an alpha value of 0.001. The R² score is about 0.7942 for Ridge and 0.7941 for LASSO, 
    which means the models explain nearly 79% of the variation in car mpg.
    
"""