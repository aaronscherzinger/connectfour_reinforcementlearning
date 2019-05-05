# Simple Reinforcement Learning Project

This is a very simple _reinforcement learning_ project with the goal of training a neural network to play the _connect four_ game. The network is trained only by playing against itself without any human interaction or prepared training data.

For an overview of the basic concepts of reinforcement learning you can refer to the following paper: 

V.Heidrich-Meisner, M. Lauer, C. Igel, and M. Riedmiller: _Reinforcement Learning in a Nutshell_. ESANN 2007, pp. 277-288, 2007. [https://christian-igel.github.io/paper/RLiaN.pdf]

The implementation of this project follows the approach proposed in the following paper:

V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, D. Wierstra, M. Riedmiller: _Playing Atari with Deep Reinforcement Learning_. CoRR abs/1312.5602 (2013) [https://arxiv.org/pdf/1312.5602.pdf]

## Idea and Approach

This project was a small weekend project in order to experiment a little bit with reinforcement learning and give the tensorflow API a try, which I had not used before. Please note that I am by no means an expert in reinforcement learning, nor am I an experienced tensorflow user. Also, the goal of this project was *not* to create the best possible AI opponent for playing connect four - please note that the problem of connect four has already been mathematically solved (see [https://en.wikipedia.org/wiki/Connect_Four]). Instead, this project was only about implementing a simple deep reinforcement learning strategy in a relatively short amount of time. *(In the end, this took me longer than a weekend as I tried out a few variants, different network topologies, optimizers, and parameter sets. Nevertheless, the basic implementation is rather simple and consists only of  a few lines of code)*.

In the following, I will try to give you an overview of what I have done and what the result looks like.

### Reinforcement Learning

Coming soon...

### Connect Four

Coming soon...

### Implementation

Coming soon...

### Results

Coming soon...

## Prerequisites

The code is implemented using **Python 3.6** and uses the deep learning framework **Tensorflow**. You can simply install Tensorflow for Python via pip. However, for using this code I would advice you to use a machine with a dedicated GPU that is CUDA-capable and install the GPU-version of Tensorflow (otherwise training thr network might take a very long time). You might need to compile Tensorflow yourself, depending on your GPU driver and CUDA version, but you should find sufficient resources for that online.

You will also need some additional Python packages such as numpy, matplotlib, and - for running the unit tests - the unittest package, which sould also be available for your Python distribution, e.g. via pip.

I have only tested this code on Linux (more specifically, Ubuntu 18.04), but it should be straightforward to run it on other platforms, provided that you have Tensorflow available.

## Files
```connect_four.py``` - contains the class for the basic playing board

```test_board.py``` - unit tests for the playing board

```reinforcement_learning.py``` - trains a feedworward neural network (i.e., multilayer perceptron) by playing connect four against itself, exports the trained model afterwards to ```./current_model``` (and overwrites an existing model!)

```reinforcement_learning_cnn.py``` - trains a convolutional neural network (CNN) that begins with convolutional layers and ends with fully connected layers by playing connect four against itself, exports the trained model afterwards to ```./current_model``` (and overwrites an existing model!)

```test_trained_model.py``` - this is a simple test for a multilayer perceptron which confronts it with two playing situations and prints the estimates of the Q-function (i.e., the output of the model) for each situation

```play_against_model.py``` - simple text-based interface for playing against a trained feedforward neural network

```play_against_cnn.py``` - the same interface for playing against a trained CNN

