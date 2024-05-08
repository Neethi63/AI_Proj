# Assignment01c
"""
First calculate error for various combinations of beta1 and beta2.
Beta1 and Beta2 now varies from -1 to 1 with precision 0.001
Minimum error is then identified
A 3D surface plot is created finally
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Define the arrays
x = np.array([-3, -2, -1, 0, 1, 2, 3])
y = np.array([7, 2, 0, 0, 0, 2, 7])

# Initialize lists to store beta1, beta2, and error
beta1_values = []
beta2_values = []
error_values = []

# Calculate y (est) for beta1 = -1 to 1, beta2 = -1 to 1 with precision 0.001
beta_range = np.arange(-1, 1.001, 0.001)
for beta1 in beta_range:
    for beta2 in beta_range:
        z = beta1 * x + beta2 * (x ** 2)
        error = np.sum(np.abs(y - z))
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

# Convert lists to NumPy arrays for plotting
beta1_values = np.array(beta1_values)
beta2_values = np.array(beta2_values)
error_values = np.array(error_values)

# Reshape arrays for surface plot
size = int(np.sqrt(len(beta1_values)))
beta1_values = beta1_values.reshape((size, size))
beta2_values = beta2_values.reshape((size, size))
error_values = error_values.reshape((size, size))

# Plot surface
ax.plot_surface(beta1_values, beta2_values, error_values, cmap='viridis')

ax.set_xlabel('Beta1')
ax.set_ylabel('Beta2')
ax.set_zlabel('Error')
plt.title('Error, Beta1 and Beta2')
plt.show()
