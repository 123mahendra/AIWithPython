# import numpy as np
#
# import matplotlib.pyplot as plt
#
# import pandas as pd
#
# import seaborn as sns
#
# from sklearn.datasets import load_diabetes
#
# from sklearn.linear_model import LinearRegression
#
# from sklearn.metrics import mean_squared_error, r2_score
#
# from sklearn.model_selection import train_test_split
#
# data = load_diabetes(as_frame=True)
#
# print(data.keys())
#
# print(data.DESCR)
#
# df = data['frame']
#
# print(df)
#
# plt.hist(df['target'],25)
#
# plt.xlabel('target')
#
# plt.show()
#
# sns.heatmap(data=df.corr().round(2),annot=True)
#
# plt.show()
#
# plt.subplot(1,2,1)
#
# plt.scatter(df['bmi'], df['target'])
#
# plt.xlabel('bmi')
# plt.ylabel('target')
#
# plt.subplot(1,2,2)
#
# plt.scatter(df['s5'], df['target'])
#
# plt.xlabel('s5')
# plt.ylabel('target')
#
# plt.show()
#
# X = pd.DataFrame(df[['bmi', 's5']], columns=['bmi', 's5'])
# y = df['target']
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=5)
#
# lm = LinearRegression()
#
# lm.fit(X_train, y_train)
#
# y_train_pred = lm.predict(X_train)
#
# rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
# r2 = r2_score(y_train, y_train_pred)
#
# print("Root Mean Squared Error:", rmse)
# print("R2 value:", r2)
#
# y_test_pred = lm.predict(X_test)
#
# rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
# r2_test = r2_score(y_test, y_test_pred)
#
# print("Root Mean Squared Error:", rmse_test)
# print("R2 value:", r2_test)







# import numpy as np
#
# import matplotlib.pyplot as plt
#
# import pandas as pd
#
# from sklearn.linear_model import Ridge
#
# from sklearn.model_selection import train_test_split
#
# df = pd.read_csv('ridgereg_data.csv')
#
# x = df[['x']]
#
# y = df[['y']]
#
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)
#
# alphas = np.linspace(0,2,50)
#
# print(alphas)
#
# r2values = []
#
# for alp in alphas:
#     rr = Ridge(alpha=alp)
#     rr.fit(X_train, y_train)
#     r2_test = rr.score(X_test, y_test)
#     r2values.append(r2_test)
#
# plt.plot(alphas,r2values)
# plt.show()
# best_r2 = max(r2values)
# print(best_r2)
# idx = r2values.index(best_r2)
# best_apl = alphas[idx]
# print(f"Best alpha = {best_apl}, Best r2 = {best_r2}a")







import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('diamonds.csv')
print(df.head())

X = df[['carat','depth','table','x','y','z']]
y = df[['price']]
print(X,y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

alphas = [0.1,0.2,0.3,0.4,0.5,1,2,3,4,5,6,7,8]

scores = []

for alp in alphas:
    lasso = linear_model.Lasso(alpha=alp)
    lasso.fit(X_train, y_train)

    sc = lasso.score(X_test, y_test)
    scores.append(sc)

plt.plot(alphas, scores)
plt.show()

best_r2 = max(scores)

idx = scores.index(best_r2)

best_alp = alphas[idx]

print(f"\nBest R2 = {best_r2}, Best alp = {best_alp}")












