from agent.agent import Agent
from functions import *
import sys

if len(sys.argv) != 4:
	print("Usage: python3 train.py [stock] [window] [episodes]")
	exit()

stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

#Instantiate an Agent
agent = Agent(state_size=window_size)

data = getStockDataVec(stock_name)

data_lenght = len(data) - 1

batch_size = 32

last_higher_profit = 0

for episode in range(episode_count + 1):
    print("Episode " + str(episode) + "/" + str(episode_count))

    state = getState(data, 0, window_size + 1)

    total_profit = 0
    agent.inventory = []

    previous_action = 0
    buy_action_count = 0
    no_action_count = 0

    for t in range(data_lenght):
        print(f'{t}/{data_lenght}')

        action = agent.act(state)

        if action == 1 and previous_action == 1:
            buy_action_count += 1
        if action == 0 and previous_action == 0:
            no_action_count += 1

        # no-action
        next_state = getState(data, t + 1, window_size + 1)
        reward = 0

        if action == 1: # buy
            agent.inventory.append(data[t])
            print("Buy: " + formatPrice(data[t]))

        elif action == 2 and len(agent.inventory) > 0: # sell
            bought_price = agent.inventory.pop(0)
            reward = (data[t] - bought_price) * 100
            total_profit += data[t] - bought_price
            print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))

        # Check if there are 20 consecutives buys
        if buy_action_count >= 20:
            reward = -500
            buy_action_count = 0
        # Check if there are 50 consecutives no-actions
        if no_action_count >= 50:
            reward = -500
            no_action_count = 0

        done = True if t == data_lenght - 1 else False

        agent.memory.append((state, action, reward, next_state, done))

        state = next_state
        previous_action = action

        if done:
            print("--------------------------------")
            print("Total Profit: " + formatPrice(total_profit))
            print("--------------------------------")

        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)

    if total_profit > last_higher_profit:
    # if e % 10 == 0:
        last_higher_profit = total_profit
        total_profit = 34.5546456
        profit = str(total_profit).split('.')

        agent.model.save(f"models/model_ep{episode}_profit{profit[0]}")
