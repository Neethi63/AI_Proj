import numpy as np
import matplotlib.pyplot as plt
import random
from tabulate import tabulate
from sklearn.model_selection import train_test_split

# Function to generate y values with noise
def generate_data(num_points):
    x_values = np.linspace(-5, 5, num_points)  # Generate x values
    y_values = 2 * x_values - 3 + np.random.normal(0, 5, num_points)  # Generate y values with noise
    return x_values, y_values

# Closed-Form Solution for Linear Regression
def closed_form_solution(x_train, y_train):
    X_train = np.column_stack([np.ones_like(x_train), x_train])  # Design matrix
    coefficients = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train  # Coefficients
    y_estimate = X_train @ coefficients
    error = np.mean((y_train - y_estimate) ** 2)  # Calculate Mean Squared Error (MSE)
    return coefficients, error

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

def main():
    # Generate data
    x_values, y_values = generate_data(1000)
    x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, test_size=0.2, random_state=42)

    # Closed-Form Solution
    b_closed_form, error_closed_form = closed_form_solution(x_train, y_train)

    # Gradient Descent
    b0_values, b1_values, errors = gradient_descent(x_train, y_train)

    # Find the index of the minimum error encountered during gradient descent
    min_error_index = np.argmin(errors)

    # Use the beta values corresponding to the minimum error
    b0_min_error = b0_values[min_error_index]
    b1_min_error = b1_values[min_error_index]

    # Calculate Mean Squared Error (MSE) for test data using gradient descent
    y_test_predicted_gradient_descent = b0_min_error + b1_min_error * x_test
    mse_gradient_descent_test = np.mean((y_test - y_test_predicted_gradient_descent) ** 2)

    # Calculate Mean Squared Error (MSE) for test data using closed-form solution
    y_test_predicted_closed_form = b_closed_form[0] + b_closed_form[1] * x_test
    mse_closed_form_test = np.mean((y_test - y_test_predicted_closed_form) ** 2)

    # Calculate Mean Squared Error (MSE) for training data using gradient descent
    y_train_predicted_gradient_descent = b0_min_error + b1_min_error * x_train
    mse_gradient_descent_train = np.mean((y_train - y_train_predicted_gradient_descent) ** 2)

    # Tabulate results
    results = [
        ["Method", "b0", "b1", "MSE (Training)", "MSE (Test)"],
        ["Gradient Descent", f"{b0_min_error:.6f}", f"{b1_min_error:.6f}", f"{mse_gradient_descent_train:.6f}", f"{mse_gradient_descent_test:.6f}"],
        ["Closed-Form Solution", f"{b_closed_form[0]:.6f}", f"{b_closed_form[1]:.6f}", f"{error_closed_form:.6f}", f"{mse_closed_form_test:.6f}"]
    ]
    print(tabulate(results, headers="firstrow", tablefmt="fancy_grid"))

    # Plotting

    # Comparison of Gradient Descent and Closed-Form Solution on Training Data
    plt.figure(figsize=(8, 6))
    plt.scatter(x_train, y_train, s=10, color='blue', alpha=0.5, label='Original Data')
    plt.plot(x_train, b_closed_form[0] + b_closed_form[1] * x_train, color='green', label='Closed-Form Solution')
    plt.plot(x_train, b0_min_error + b1_min_error * x_train, color='red', label='Gradient Descent')
    plt.title('Comparison of Gradient Descent and Closed-Form Solution on Training Data')
    plt.xlabel('x')
    plt.ylabel('y')
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
