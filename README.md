# DeepQTrader

[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

An implementation of Q-learning applied to (short-term) Bitcoin trading. The model uses n-day windows of OHLC data to determine the best action to take at a given time. i.e. buy, sell or sit.

## Motivation
The simple prediction of future prices with RNNs or CNNs is not enough to make **mostly** correct decisions in the crypto-trading world given the complexity and volatility of such environment. One possible solution could be to use reinforcement learning in combination with some clever deep neural network optimized policies.

This project is an attempt to see if is possible to use reinforcement and Q learning to predict and **act** super-humanly upon   cryptocurrency prices and positions, despite the lack of evidence.

## Screenshots

See EpisodesViz notebook

## Features
- Keras neural network.
- State shape is the window_size x columns.
- Training happens at each state.
- Data sequences can have n-dimensions.
- Visualization plots appear at the end of each episode. 

## Installation

To run in your local machine, follow these steps:

0. Create your working environment. `$ python3 -m venv /path/to/venv`
1. Install the dependencies with `$ pip3 install`.
2. Create the models directory on the project root. `$ mkdir model`
3. Grab the data, choose the window length and the number of episodes.
(**bit1.csv** or **bit2.csv** into `$ data/` are ready-to-use examples.)
4. Train the model with the selected parameters.
```
$ python3 train.py bit1 10 1000
```
The code is going to generate an HTML plot for each episode, containing all the actions it took along the training.


## How to use?

When training finishes (minimum 200 episodes for results):
```
python3 evaluate.py bit2 model_ep200
```

## Contribute
Please

1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :D

Feedbacks, questions, critics, ideas, etc. are extremely welcome.

## References

[Deep Q-Learning with Keras and Gym](https://keon.io/deep-q-learning/) - Q-learning overview and Agent skeleton code
[OpenAI SppiningUp](https://spinningup.openai.com/en/latest/) - Reinforcement learning study material

## Credits
Inspired and initially forked from [Deep Q-Learning](https://github.com/edwardhdlu/q-trader)

## License: MIT
Copyright (c) 2019 Juan José Chambers

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

MIT © [Juan José Chambers](https://github.com/chmbrs/)
