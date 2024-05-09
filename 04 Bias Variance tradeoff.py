import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import random


def calculate_y(x):
    """Function to calculate y values based on a polynomial function with some random noise."""
    return 2 * x ** 4 - 3 * x ** 3 + 7 * x ** 2 - 23 * x + 8 + random.normalvariate(0, 3)


def lagrange_interpolation(x, x_values, y_values):
    """
    Function to perform Lagrange interpolation for a given x value.

    Parameters:
    - x: The x value to interpolate.
    - x_values: Array of x values.
    - y_values: Array of corresponding y values.

    Returns:
    - Interpolated y value corresponding to x.
    """
    n = len(x_values)
    result = 0
    for i in range(n):
        term = y_values[i]
        for j in range(n):
            if j != i:
                term *= (x - x_values[j]) / (x_values[i] - x_values[j])
        result += term
    return result


# Generating data points
x_values = np.arange(-5, 5.1, 0.1)
y_values = np.array([calculate_y(x) for x in x_values])
data = list(zip(x_values, y_values))
random.shuffle(data)

# Splitting data into train and test sets
plot_data = data[:80]
test_data = data[80:]

# Preparing train data for plotting
x_plot, y_plot = zip(*plot_data)
x_plot = np.array(x_plot).reshape(-1, 1)
y_plot = np.array(y_plot).reshape(-1, 1)

# Plotting train data
plt.scatter(x_plot, y_plot, s=10, color='blue', alpha=0.5, label='Train Data')

# Polynomial regression for different degrees
for degree in range(1, 5):
    poly_features = PolynomialFeatures(degree=degree)
    x_values_poly = poly_features.fit_transform(x_values.reshape(-1, 1))
    x_plot_poly = poly_features.fit_transform(x_plot)
    b = np.linalg.inv(np.matmul(x_plot_poly.T, x_plot_poly)).dot(x_plot_poly.T).dot(y_plot)
    y_values_for_plot = np.matmul(x_values_poly, b)
    plt.plot(x_values, y_values_for_plot, label=f'Degree {degree}')

# Lagrange interpolation
lagrange_y_values = [lagrange_interpolation(x, x_values, y_values) for x in x_values]
plt.plot(x_values, lagrange_y_values, label="Lagrange Interpolation", linestyle='--', color='green')

# Plot configurations
plt.title('Polynomial Regression and Lagrange Interpolation')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# Evaluation of models with test data
x_test, y_test = zip(*test_data)
summed_errors_train = []
summed_errors_test = []

for degree in range(1, 11):
    poly_features = PolynomialFeatures(degree=degree)
    x_test_poly = poly_features.fit_transform(np.array(x_test).reshape(-1, 1))
    x_train_poly = poly_features.fit_transform(x_plot)
    b = np.linalg.inv(np.matmul(x_train_poly.T, x_train_poly)).dot(x_train_poly.T).dot(y_plot)
    y_values_for_plot_test = np.matmul(x_test_poly, b)
    y_values_for_plot_train = np.matmul(x_train_poly, b)
    errors_test = (np.array(y_test).reshape(-1, 1) - y_values_for_plot_test) ** 2
    errors_train = (y_plot - y_values_for_plot_train) ** 2
    summed_errors_test.append(np.sum(errors_test) / len(x_test))
    summed_errors_train.append(np.sum(errors_train) / len(x_plot))

# Plotting summed errors for different polynomial degrees
plt.plot(range(1, 11), summed_errors_test, label='Test Data')
plt.plot(range(1, 11), summed_errors_train, label='Training Data')
plt.title('Summed Errors for Different Polynomial Degrees')
plt.xlabel('Polynomial Degree')
plt.ylabel('Summed Errors')
plt.legend()
plt.grid(True)
plt.show()
