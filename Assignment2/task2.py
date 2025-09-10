
#####Task 2#####

import matplotlib.pyplot as plt
import numpy as np

x = np.array([1,2,3,4,5,6,7,8,9])
y = np.array([-0.57, -2.57, -4.80, -7.36, -8.78, -10.52, -12.85, -14.69, -16.78])

plt.title("Scatter Plot of Given Points")
plt.xlabel('x-axis')
plt.ylabel('y-axis')

plt.scatter(x, y, marker='+', color='b')

plt.show()