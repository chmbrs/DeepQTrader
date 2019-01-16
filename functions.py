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
	try:
		ans = 1 / (1 + math.exp(-x))
	except OverflowError:
		print('erro')
		ans = 1 / (1 + float('inf'))

	return ans


# returns an an n-day state representation ending at time t
def getState(data, t, n):
	# d = 0 - window_size + 1
	d = t - n + 1

	block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
	print(f'block {block}')

	res = []
	for i in range(n - 1):
		res.append(sigmoid(block[i + 1] - block[i]))

	print(f'sigmoid {res}')

	return np.array([res])
