import keras
from keras.models import load_model

from agent.agent import Agent
from functions import *
import sys

import pandas as pd

if len(sys.argv) != 3:
	print("Usage: python3 evaluate.py [stock] [model]")
	exit()

stock_name, model_name = sys.argv[1], sys.argv[2]
model = load_model("models/" + model_name)
window_size = model.layers[0].input.shape.as_list()[1]

agent = Agent(window_size, True, model_name)

df = pd.read_csv("./data/" + stock_name + ".csv")

data = df.loc[:, ~df.columns.isin(['Action', 'Timestamp'])]

data_lenght = len(data) - 1

batch_size = 32

state = getState2(data, 0, window_size + 1)

total_profit = 0

agent.inventory = []

for step in range(data_lenght):
	action = agent.act(state)

	# no-action
	next_state = getState2(data, step + 1, step + window_size + 1)
	reward = 0

	if action == 1: # buy
		agent.inventory.append(data.iloc[step, 3]) #3 is the close column
		print("Buy: " + formatPrice(data.iloc[step, 3]))

	elif action == 2 and len(agent.inventory) > 0: # sell
		bought_price = agent.inventory.pop(0)

		reward = data.iloc[step, 3] - bought_price

		total_profit += data.iloc[step, 3] - bought_price
		print("Sell: " + formatPrice(data.iloc[step, 3]) +
		 " | Profit: " + formatPrice(data.iloc[step, 3] - bought_price))

	done = True if step == data_lenght - 1 else False

	if not step <= window_size:
		agent.memory.append((state, action, reward, next_state, done))

	state = next_state

	if done:
		print("--------------------------------")
		print(stock_name + " Total Profit: " + formatPrice(total_profit))
		print("--------------------------------")
