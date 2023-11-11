# Importing numpy library for using exponential function.

import numpy as np

# Parameters for the logistic regression model is mentioned in the given problem.
para = {'B01': 0.1, 'B1': 0.5, 'B2': 0.5, 'B02': 1, 'B03': 0}

# User input for the data

data = {
    'X1': list(map(float, input("Enter values for X1 separated by space: ").split())),
    'X2': list(map(float, input("Enter values for X2 separated by space: ").split())),
    'Sero': list(map(float, input("Enter values for Sero separated by space: ").split()))}

# converting  the inputs into lists for ease
list1 = data['X1']
list2 = data['X2']
list3 = data['Sero']


# Function to calculate probabilities using the logistic function
def logistic_function(logitsv1, logitsv2, logitsv3):
    P1 = np.exp(logitsv1) / (np.exp(logitsv1) + np.exp(logitsv2) + np.exp(logitsv3))
    P2 = np.exp(logitsv2) / (np.exp(logitsv1) + np.exp(logitsv2) + np.exp(logitsv3))
    P3 = np.exp(logitsv3) / (np.exp(logitsv1) + np.exp(logitsv2) + np.exp(logitsv3))
    return P1, P2, P3

# Iterate through the lists and calculate probabilities
for y in range(len(list1)):
    try:
        x1 = list1[y]
        x2 = list2[y]
        x3 = list3[y]
    except IndexError:
        # Handle the expection when the index is out of range
        print(f"p{y + 1}")
        print("List index out of range. Skipping this iteration.")
        print()
        continue

    # Features and weights for logistic regression that are taken from the user
    features = [1, x1, x2]
    weights0 = [para['B01'], para['B1'], para['B2']]
    weights1 = [para['B02'], para['B1'], para['B2']]
    Sero = [1, x3, x3]  # Corrected Sero values
    weights2 = [para['B03'], para['B1'], para['B2']]

    # Calculate logits for each alternative
    logitsv1 = np.dot(features, weights0)
    logitsv2 = np.dot(features, weights1)
    # calculating the dot product using dot function of numpy
    logitsv3 = np.dot(Sero, weights2)

    # Calculate probabilities using the logistic function
    probabilities = logistic_function(logitsv1, logitsv2, logitsv3)

    # Print the results for each alternative
    print(f"P{y + 1} =")
    for idx, prob in enumerate(probabilities, start=1):
        print(f"  P{idx} = {round(prob, 4)}")
    print()