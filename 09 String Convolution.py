def tokenizeString(s):
    # Split the string into tokens based on whitespace
    return s.split()

def stringConvolution(s1, s2):
    # Tokenize the input strings
    tokens1 = tokenizeString(s1)
    tokens2 = tokenizeString(s2)

    len1 = len(tokens1)
    len2 = len(tokens2)

    # Initialize an array to hold the convolution result
    convolutionResult = []

    # Perform convolution by sliding tokens2 over tokens1
    for i in range(len1 + len2 - 1):
        matchCount = 0
        for j in range(len2):
            k = i - j
            # Ensure indices are within valid ranges
            if 0 <= k < len1:
                # Compare corresponding tokens
                if tokens1[k] == tokens2[j]:
                    matchCount += 1
        convolutionResult.append(matchCount)

    return convolutionResult

# Read input strings from the terminal
s1 = input("Enter the first string: ")
s2 = input("Enter the second string: ")

# Calculate the convolution result
result = stringConvolution(s1, s2)

# Print the convolution result
print("Convolution Result:", result)
