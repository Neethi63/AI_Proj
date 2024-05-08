import numpy as np
import matplotlib.pyplot as plt

def plot_normal_fixed_mean(std_values, mean, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    for std in std_values:
        x = np.linspace(mean - 3*std, mean + 3*std, 1000)
        y = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-((x - mean)**2) / (2 * std**2))
        plt.plot(x, y, label=f'Standard Deviation = {std}')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_normal_fixed_std(mean_values, std, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    for mean in mean_values:
        x = np.linspace(mean - 3*std, mean + 3*std, 1000)
        y = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-((x - mean)**2) / (2 * std**2))
        plt.plot(x, y, label=f'Mean = {mean}')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

# (a) Plotting normal distribution for fixed mean and changes in standard deviation
std_values = [1, 2, 3]  # Different standard deviation values
mean = 0  # Fixed mean
plot_normal_fixed_mean(std_values, mean, 'Normal Distribution (Fixed Mean, Varying Standard Deviation)', 'X', 'Probability Density')

# (b) Plotting normal distribution for fixed standard deviation with changes in mean
mean_values = [-2, 0, 2]  # Different mean values
std = 1  # Fixed standard deviation
plot_normal_fixed_std(mean_values, std, 'Normal Distribution (Fixed Standard Deviation, Varying Mean)', 'X', 'Probability Density')
