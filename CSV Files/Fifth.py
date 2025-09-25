# import numpy as np
# import pandas as pd
#
# df = pd.read_csv('linreg_data.csv',names=['x','y'], skiprows=0)
# givenX = df['x']
# givenY = df['y']
#
# XY = givenX * givenY
#
# meanOfXY = np.mean(XY)
#
# meanOfX = np.mean(givenX)
# meanOfY = np.mean(givenY)
#
# squareOfX = np.square(givenX)
#
# n = len(givenX)
#
# b = (np.sum(XY) - n * meanOfX * meanOfY) / (np.sum(squareOfX) - n * (meanOfX ** 2))
# a = meanOfY - b * meanOfX
#
# print("Slope (b):", b)
# print("Intercept (a):", a)
#
#
#
# yhat = a+b*givenX
#
# RSS = np.sum((givenY-yhat)**2)
#
# print("RSS:", RSS)
#
# RMSE = np.sqrt((1/n)*np.sum((givenY-yhat)**2))
#
# print("RMSE:", RMSE)
#
# MAE = (1/n)*np.sum(np.abs(givenY-yhat))
#
# print("MAE:", MAE)
#
# MSE = np.sum((givenY-yhat)**2)/n
#
# print("MSE:", MSE)
#
# R2 = 1 - np.sum((givenY-yhat)**2)/np.sum((givenY-meanOfY)**2)
#
# print("R2:", R2)
#
#
# import matplotlib.pyplot as plt
#
# from sklearn.linear_model import LinearRegression
#
# my_data = np.genfromtxt('linreg_data.csv', delimiter=',')
#
# xp = my_data[:,0]
# yp = my_data[:,1]
#
# xp = xp.reshape(-1,1)
# yp = yp.reshape(-1,1)
#
# regr = LinearRegression()
#
# regr.fit(xp, yp)
#
# print(regr.coef_, regr.intercept_)
#
# xval = np.linspace(-1,2,100).reshape(-1,1)
#
# yval = regr.predict(xval)
#
# plt.plot(xval, yval)
#
# plt.scatter(xp,yp,color="red")
#
# plt.show()
#
# from sklearn import metrics
#
# yhat = regr.predict(xp)
#
# print('Mean Absolute Error:', metrics.mean_absolute_error(yp, yhat))
#
# print('Mean Squared Error:', metrics.mean_squared_error(yp, yhat))
#
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(yp, yhat)))
#
# print('R2 value:', metrics.r2_score(yp, yhat))
#



# import numpy as np
# import pandas as pd
#
# df = pd.read_csv('linreg_data.csv',names=['x','y'], skiprows=0)
# xpd = df['x']
# givenY = df['y']