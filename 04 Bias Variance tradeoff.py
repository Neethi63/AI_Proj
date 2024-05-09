import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import random

# Define the function to calculate y
def calculate_y(x):
    return 2*x**4 - 3*x**3 + 7*x**2 - 23*x + 8 + random.normalvariate(0, 3)

# Define Lagrange interpolation function
def lagrange_interpolation(x, x_values, y_values):
    n = len(x_values)
    result = 0
    for i in range(n):
        term = y_values[i]
        for j in range(n):
            if j != i:
                term *= (x - x_values[j]) / (x_values[i] - x_values[j])
        result += term
    return result

# Generate x values from -5 to 5 with step size 0.1
x_values = np.arange(-5, 5.1, 0.1)

# Calculate y values corresponding to x values
y_values = np.array([calculate_y(x) for x in x_values])

# Combine x and y values into a list of tuples
data = list(zip(x_values, y_values))

# Shuffle the data
random.shuffle(data)

# Select 80 values for plotting
plot_data = data[:80]

# Store the remaining 20 values for testing purposes
test_data = data[80:]

# Unzip plot_data into separate x and y arrays
x_plot, y_plot = zip(*plot_data)

# Reshape x_plot and y_plot into numpy arrays
x_plot = np.array(x_plot).reshape(-1, 1)
y_plot = np.array(y_plot).reshape(-1, 1)

# Plot the scatter plot
plt.scatter(x_plot, y_plot, s=10, color='blue', alpha=0.5, label='Train Data')

# Generate polynomial regression up to the 4th order
for degree in range(1, 5):
    # Generate polynomial features for plotting data
    poly_features = PolynomialFeatures(degree=degree)
    x_values_poly = poly_features.fit_transform(x_values.reshape(-1, 1))
    
    # Calculate coefficients using the formula b = (x^T x)^(-1) (x^T y)
    x_plot_poly = poly_features.fit_transform(x_plot)
    b = np.linalg.inv(x_plot_poly.T.dot(x_plot_poly)).dot(x_plot_poly.T).dot(y_plot)
    
    # Calculate y values for plotting using the polynomial regression equation
    y_values_for_plot = x_values_poly.dot(b)
    
    # Plot polynomial regression line
    plt.plot(x_values, y_values_for_plot, label=f'Degree {degree}')

# Lagrange Interpolation
lagrange_y_values = [lagrange_interpolation(x, x_values, y_values) for x in x_values]
plt.plot(x_values, lagrange_y_values, label="Lagrange Interpolation", linestyle='--', color='green')

plt.title('Polynomial Regression and Lagrange Interpolation')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# Extract test data
x_test, y_test = zip(*test_data)

# Initialize lists to store summed errors for training and test data
summed_errors_train = []
summed_errors_test = []

# Plot error functions for all degrees
for degree in range(1, 11):  # Change the range to iterate from 1 to 11
    # Generate polynomial features for test data
    poly_features = PolynomialFeatures(degree=degree)
    x_test_poly = poly_features.fit_transform(np.array(x_test).reshape(-1, 1))
    
    # Generate polynomial features for training data
    x_train_poly = poly_features.fit_transform(x_plot)
    
    # Calculate coefficients using the training data
    b = np.linalg.inv(x_train_poly.T.dot(x_train_poly)).dot(x_train_poly.T).dot(y_plot)
    
    # Calculate y values for test data using the polynomial regression equation
    y_values_for_plot_test = x_test_poly.dot(b)
    
    # Calculate y values for training data using the polynomial regression equation
    y_values_for_plot_train = x_train_poly.dot(b)
    
    # Calculate squared differences between y_test and regression values for test data
    errors_test = (np.array(y_test).reshape(-1, 1) - y_values_for_plot_test) ** 2
    
    # Calculate squared differences between y_plot and regression values for training data
    errors_train = (y_plot - y_values_for_plot_train) ** 2
    
    # Sum squared errors over x for both test and training data
    summed_errors_test.append(np.sum(errors_test)/len(x_test))
    summed_errors_train.append(np.sum(errors_train)/len(x_plot))

# Plot summed errors for both test and training data
plt.plot(range(1, 11), summed_errors_test, label='Test Data')  # Change the range to plot against 1 to 10
plt.plot(range(1, 11), summed_errors_train, label='Training Data')  # Change the range to plot against 1 to 10
plt.title('Summed Errors for Different Polynomial Degrees')
plt.xlabel('Polynomial Degree')
plt.ylabel('Summed Errors')
plt.legend()
plt.grid(True)
plt.show()
