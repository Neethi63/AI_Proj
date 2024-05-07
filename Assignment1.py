# Assignment01
"""
First calculate error for various combinations of beta1 and beta2.
Minimum error is then identified
A 3D scatter plot is created finally
"""

# Import libraries
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the arrays
x = [-3, -2, -1, 0, 1, 2, 3]
y = [7, 2, 0, 0, 0, 2, 7]

# Initialize lists to store beta1, beta2, and error
beta1_values = []
beta2_values = []
error_values = []

# Calculate y (est) for beta1 = -3 to 3, beta2 = -3 to 3
for beta1 in range(-3, 4):
    for beta2 in range(-3, 4):
        z = [beta1 * x_val + beta2 * (x_val ** 2) for x_val in x]
        error = sum(abs(y_val - z_val) for y_val, z_val in zip(y, z))
        beta1_values.append(beta1)
        beta2_values.append(beta2)
        error_values.append(error)

# Finding minimum error and corresponding beta1 and beta2
min_error_index = error_values.index(min(error_values))
min_beta1 = beta1_values[min_error_index]
min_beta2 = beta2_values[min_error_index]
min_error = error_values[min_error_index]

print("Minimum Error:", min_error)
print("Corresponding Beta1:", min_beta1)
print("Corresponding Beta2:", min_beta2)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(beta1_values, beta2_values, error_values)
ax.set_xlabel('Beta1')
ax.set_ylabel('Beta2')
ax.set_zlabel('Error')
plt.title('Error, Beta1 and Beta2')
plt.show()
