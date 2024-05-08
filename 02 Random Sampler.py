# Assignment02e
"""
RANDOM SAMPLER
"""
import random

def calculate_cmf(pmf):
    cmf = [pmf[0]]  # Initialize the first element of CMF with the first element of PMF
    for i in range(1, len(pmf)):
        cmf.append(cmf[i - 1] + pmf[i])  # Calculate cumulative sum
    return cmf

def generate_sample(population, cmf):
    # Generate a random number between 0 and 1
    random_number = random.random()

    # Find the index where the random number falls in the CMF
    for i in range(len(cmf)):
        if random_number <= cmf[i]:
            return list(population.keys())[i]

def draw_sample(population, n):
    pmf = list(population.values())
    cmf = calculate_cmf(pmf)
    samples = [generate_sample(population, cmf) for _ in range(n)]
    return samples

# Define population dictionary
population = {"ABCD": 0.04, "EFGH": 0.07, "IJKL": 0.15, "MNOP": 0.22, "QRST": 0.4, "UVWXYZ": 0.12}

# Take input for number of samples
n = int(input("Enter the number of samples to generate: "))

# Generate samples using draw_sample function
samples = draw_sample(population, n)

# Print the samples
print("Samples:", samples)
