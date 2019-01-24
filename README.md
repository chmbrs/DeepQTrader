# Q-Trader

An implementation of Q-learning applied to (short-term) Bitcoin trading. The model uses n-day windows of OHLC to determine if the best action to take at a given time is to buy, sell or sit.

As a result of the short-term state representation, the model is not very good at making decisions over long-term trends, but is quite good at predicting peaks and troughs.

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
