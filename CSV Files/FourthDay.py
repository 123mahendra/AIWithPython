
# import matplotlib.pyplot as plt
# import numpy as np
# x = np.array([0,1,2,3,4])
#
# y1 = 1 + 2*x
#
# y2 = 1 + 3*x
#
# y3 = 1 + 4*x
#
# plt.title("Linear Regression!")
# plt.xlabel("x-axis")
# plt.ylabel("y-axis")
#
# plt.plot(x,y1,color='red', marker='+')
# plt.plot(x,y2,color='green', marker='+')
# plt.plot(x,y3,color='blue', marker='+')
#
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
#
# df = pd.read_csv('linreg_data.csv')
#
# # df = np.loadtxt("linreg_data.csv", delimiter=",")
#
# df = np.array(df)
#
# x = df[:, 0]
# y = df[:, 1]
#
#
# plt.scatter(x,y,color='red', marker='+')
#
# plt.show()


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('linreg_data.csv',names=['x','y'], skiprows=0)
xgiven = df['x']
ygiven = df['y']
term1xy = xgiven*ygiven
print(term1xy)


