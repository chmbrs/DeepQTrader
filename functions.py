import numpy as np
import math


# prints formatted price
def formatPrice(n):
	return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))


# returns the vector containing stock data from a fixed file
def getStockDataVec(key):
	vec = []
	lines = open("data/" + key + ".csv", "r").read().splitlines()

	for line in lines[1:]:
		vec.append(float(line.split(",")[4]))

	return vec

# returns the sigmoid
def sigmoid(x):
	ans = 1 / (1 + math.exp(-x))
	return ans


# returns an an n-day state representation ending at time t
def getState(data, step, window_size):
	# d = 0 - window_size + 1
	current_position = step - window_size + 1

	if current_position >= 0:
		block = data[current_position:step + 1]
	else:
		block = -current_position * [data[0]] + data[0 : step + 1] # pad with t0

	print(f'block {block}')

	res = []
	for i in range(window_size - 1):
		res.append(sigmoid(block[i + 1] - block[i]))

	print(f'sigmoid {np.array([res])}')
	return np.array([res])

from sklearn import preprocessing
from collections import deque

# returns an an n-day state representation ending at time t
def getState2(data, step, window_size):
    block = []

    current_position = step - window_size + 1

    if current_position >= 0:
        block = data.iloc[current_position : step + 1].copy()
    else:
        block = data.iloc[0 : window_size].copy()

    for col in block.columns:  # go through all of the columns
        block[col] = preprocessing.scale(block[col].values)  # scale between -1 and 1.

    # Create the array
    sequential_data = [] # This is a list that will CONTAIN the sequences
    prev_days = deque(maxlen=window_size) # This will be our actual sequences

    for i in block.values:  # iterate over the values
        prev_days.append([n for n in i])  # store all
        if len(prev_days) == window_size:
            sequential_data.append(np.array(prev_days))

    # For creating the third dimension
    X = []
    for seq in sequential_data:  # going over our new sequential data
        X.append(seq)

    return np.array(X)
