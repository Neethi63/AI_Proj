import numpy as np
import matplotlib.pyplot as plt
import random
from tabulate import tabulate

# Function to generate y values with noise
def generate_data(num_points):
    x_values = np.linspace(-5, 5, num_points)  # Generate x values
    y_values = 2 * x_values - 3 + np.random.normal(0, 5, num_points)  # Generate y values with noise
    return x_values, y_values

# Closed-Form Solution for Linear Regression
def closed_form_solution(x_train, y_train):
    X_train = np.column_stack([np.ones_like(x_train), x_train])  # Design matrix
    coefficients = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train  # Coefficients
    return coefficients

# Gradient Descent for Linear Regression
def gradient_descent(x_train, y_train, eta=0.01, max_iterations=1000, convergence=1e-6):
    b0 = random.normalvariate(0, 1)  # Initialize coefficients randomly
    b1 = random.normalvariate(0, 1)
    b0_values = [b0]
    b1_values = [b1]
    errors = []
    for iteration in range(max_iterations):
        y_estimate = b0 + b1 * x_train  # Calculate y estimates
        mse = np.mean((y_train - y_estimate) ** 2)  # Calculate Mean Squared Error (MSE)
        errors.append(mse)
        if iteration > 0 and abs(errors[-1] - errors[-2]) < convergence:
            break
        b0 -= eta * np.mean(-2 * (y_train - y_estimate))  # Update coefficients using gradients
        b1 += eta * np.mean(x_train * (y_train - y_estimate))
        b0_values.append(b0)
        b1_values.append(b1)
    return b0_values, b1_values, errors

# Main function
def main():
    # Generate data
    x_train, y_train = generate_data(800)
    x_test, y_test = generate_data(200)

    # Closed-Form Solution
    b_closed_form = closed_form_solution(x_train, y_train)

    # Gradient Descent
    b0_values, b1_values, errors = gradient_descent(x_train, y_train)

    # Calculate Mean Squared Error (MSE) for test data using closed-form solution
    y_test_predicted_closed_form = b_closed_form[0] + b_closed_form[1] * x_test
    mse_closed_form_test = np.mean((y_test - y_test_predicted_closed_form) ** 2)

    # Calculate Mean Squared Error (MSE) for test data using gradient descent
    y_test_predicted_gradient_descent = b0_values[-1] + b1_values[-1] * x_test
    mse_gradient_descent_test = np.mean((y_test - y_test_predicted_gradient_descent) ** 2)

    # Calculate Mean Squared Error (MSE) for training data using gradient descent
    y_train_predicted_gradient_descent = b0_values[-1] + b1_values[-1] * x_train
    mse_gradient_descent_train = np.mean((y_train - y_train_predicted_gradient_descent) ** 2)

    # Tabulate results
    results = [
        ["Method", "b0", "b1", "MSE (Training)", "MSE (Test)"],
        ["Gradient Descent", f"{b0_values[-1]:.6f}", f"{b1_values[-1]:.6f}", f"{mse_gradient_descent_train:.6f}", f"{mse_gradient_descent_test:.6f}"],
        ["Closed-Form Solution", f"{b_closed_form[0]:.6f}", f"{b_closed_form[1]:.6f}", "N/A", f"{mse_closed_form_test:.6f}"]
    ]
    print(tabulate(results, headers="firstrow", tablefmt="fancy_grid"))

    # Plotting

    # Comparison of Gradient Descent and Closed-Form Solution on Training Data
    plt.figure(figsize=(8, 6))
    plt.scatter(x_train, y_train, s=10, color='blue', alpha=0.5, label='Original Data')
    plt.plot(x_train, b_closed_form[0] + b_closed_form[1] * x_train, color='green', label='Closed-Form Solution')
    plt.plot(x_train, b0_values[-1] + b1_values[-1] * x_train, color='red', label='Gradient Descent')
    plt.title('Comparison of Gradient Descent and Closed-Form Solution on Training Data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Gradient Descent b0 and b1 values
    plt.figure(figsize=(8, 6))
    plt.plot(b0_values, label='b0', color='blue')
    plt.plot(b1_values, label='b1', color='red')
    plt.title('Gradient Descent b0 and b1 values')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Error Values during Gradient Descent
    plt.figure(figsize=(8, 6))
    plt.plot(errors, color='orange')
    plt.title('Error Values during Gradient Descent')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
