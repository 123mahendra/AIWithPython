# import pandas as pd
#
# df = pd.read_csv('iris.csv')
#
# print(df.head())
#
# df2 = df.sort_values('sepal_width',ascending=False)
#
# print(df2)
#
# print(df[['sepal_width','sepal_length']])
#
# print(df[15:20])
#
# print(df.loc[15:20,['sepal_width']])
#
# print(df.iloc[15:20,[0,1]])
#
# print(df[df.sepal_width>4])
#
# print(df[df['species'].isin(['Iris-setosa'])])
#
# df['sepal_area'] = df['sepal_width']*df['sepal_length']
#
# print(df)
#
# # df.to_csv('Mahendrs-dataset.csv')
#
# to_append = [5.5,6.1,7.5,8.1,'new-line',9.5]
#
# df.loc[len(df)] = to_append
#
# print(df)


import matplotlib.pyplot as plt
import numpy as np
# x = [1,2,3,4]
#
# y = [1,4,9,16]
#
# plt.plot(x,y,"g")
#
# plt.show()

x = np.linspace(0,7,100)
y = np.sin(x)

plt.xlabel('x')
plt.ylabel('y')

plt.title('Picture Sin Graph')
plt.plot(x,y)
plt.show()

plt.subplot(1,2,1)
plt.plot(x,y)
plt.title('First sub plot')
# plt.show()

plt.subplot(1,2,2)
plt.plot(x,2*y)
plt.title('Second sub plot')
plt.show()

print(x)
print(y)

