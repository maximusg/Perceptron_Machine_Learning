# Pong using pygame implementation taken from kaggle.com. 

## To install dependencies:
Install pygame: pip install pygame

Clone the PLE repository: git clone https://github.com/ntasfi/PyGame-Learning-Environment

Install PLE from the root directory of the PLE repo:

$ cd PyGame-Learning-Environment\
$ pip install -e .\
(note the "." must be included)

## Description
pong_nn.py is a simple agent vs simple adversary. Agent utilizes a simple strategy of following the balls y-axis.

to run just type: python pong_nn.py

Note: Pong agent on left is the one implemented under the "agent" method in the file.

pong2Clean_main.py is the finalized primary file for this repository that utilizes. It allows two trained agents to face off against each other.
In this file there are three seperate agents that employ different strategies: "agent", "defense" and "random".

"agent" is a perceptron trained pong agent, which uses a 1 layer perceptron w/ 7 inputs to implement it's strategy.

"defense" is a simple pong defense of following the balls y.axis

"random" is a simple pong denfense of moving the paddle in random directions.

To pit the pong implementations off against each other simply type

python pong2_main.py \<agentName> \<agentName>


All other files were different strategies used to figure out ways to employ types of training methods. Note in order to train using the strategies one must just go into the file and set train=true at the botom of the file