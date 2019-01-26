# DeepQTrader

[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

An implementation of Q-learning applied to (short-term) Bitcoin trading. The model uses n-day windows of OHLC data to determine the best action to take at a given time. i.e. buy, sell or sit.

As a result of the short-term state representation, the model is not very good at making decisions over long-term trends, but is quite good at predicting peaks and troughs.

## Motivation
The simple prediction of future prices with RNNs or CNNs is not enough to make **mostly** correct decisions in crypto-trading given the complexity and volatility of such environment. One possible solution could be to use reinforcement learning in combination with some clever deep neural network optimized policies.

This project is an attempt to see if is possible to use reinforcement and Q learning to predict and **act** super-humanly upon   cryptocurrency prices and positions, despite the lack of evidence.

## Results

See EpisodesViz notebook

## Running the Code

To train the model use the bit1.csv or bit2.csv into `data/`
```
mkdir model
python3 train bit1 10 1000
```

Then when training finishes (minimum 200 episodes for results):
```
python3 evaluate.py bit2 model_ep1000
```

## References
Inspired by [Deep Q-Learning](https://github.com/edwardhdlu/q-trader)

[Deep Q-Learning with Keras and Gym](https://keon.io/deep-q-learning/) - Q-learning overview and Agent skeleton code
