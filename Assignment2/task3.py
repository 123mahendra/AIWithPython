
#####Task 3#####

import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("weight-height.csv", delimiter=",", skiprows=1, usecols=(1, 2))

length_inch = data[:, 0]
weight_lb = data[:, 1]

length_cm = length_inch * 2.54
weight_kg = weight_lb * 0.45359237

mean_length = np.mean(length_cm)
mean_weight = np.mean(weight_kg)

print(f"Mean length: {mean_length:.2f} cm")
print(f"Mean weight: {mean_weight:.2f} kg")

plt.title("Histogram of Student Lengths (cm)")
plt.xlabel("Length (cm)")

plt.hist(length_cm, bins=10, color='gray', edgecolor='black')

plt.show()
