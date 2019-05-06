# Simple Reinforcement Learning Project

This is a very simple _reinforcement learning_ project with the goal of training a neural network to play the _connect four_ game. The network is trained only by playing against itself without any human interaction or prepared training data.

For an overview of the basic concepts of reinforcement learning you can refer to the following paper: 

V.Heidrich-Meisner, M. Lauer, C. Igel, and M. Riedmiller: _Reinforcement Learning in a Nutshell_. ESANN 2007, pp. 277-288, 2007. [https://christian-igel.github.io/paper/RLiaN.pdf]

The implementation of this project follows the approach proposed in the following paper:

V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, D. Wierstra, M. Riedmiller: _Playing Atari with Deep Reinforcement Learning_. CoRR abs/1312.5602 (2013) [https://arxiv.org/pdf/1312.5602.pdf]

## Idea and Approach

This project was a small weekend project in order to experiment a little bit with reinforcement learning and give the tensorflow API a try, which I had not used before. Please note that I am by no means an expert in reinforcement learning, nor am I an experienced tensorflow user. Also, the goal of this project was *not* to create the best possible AI opponent for playing connect four - especially since the problem of connect four is already mathematically solved (see [https://en.wikipedia.org/wiki/Connect_Four]). Instead, this project was only about implementing a simple deep reinforcement learning strategy in a relatively short amount of time. *(In the end, this took me longer than a weekend as I tried out a few variants, different network topologies, optimizers, and parameter sets. Nevertheless, the basic implementation is rather simple and consists only of  a few lines of code)*.

In the following, I will try to give you an overview of what I have done and what the result looks like.

### Reinforcement Learning

In contrast to *supervised learning*, where a machine learning system is trained using available labeled data (e.g., in fields such as image recognition) or *unsupervised learning*, where an algorithm learns to recognize structure in unlabeled data (e.g., for clustering data, often as a pre-processing step for other methods), *reinforcement learning* follows a vastly different paradigm. In reinforcement learning, an *agent* observes an *environment*, which is in a certain *state*. The agent will then perform an *action* and observe the change in the environment, i.e., its new state, as a consequence of the action. Moreover, at some points the agent will receive feedback in the form of a *reward* (or punishment). However, feedback is not necessarily received directly after each action, but may be sparse. Over time, the agent learns from performing actions and observing the environment to behave in a way that will maximize the reward of his actions in the long run. Reinforcement learning is thus a very general way of stating a machine learning problem, and is very similar to how we learn (although on a very abtract level and vastly simplified, of course). 

### Markov Decision Processes and Q-learning

Formally, reinforcement learning can be modeled as a Markov Decision Process (MDP), which is a time-discrete stochastic state-transition automata. An MDP can be modeled as a tuple *(S,A,P,R)*, where *S* is a set of states, *A* is a set of actions, *R* are the expected (immediate) rewards when transitioning from a state *s* to a state *s'* using action *a*, and *P* is the set of transition probabilities that in state *s*, action *a* will take the agent to state *s'*. At each point in time *t*, the agent is in state *s<sub>t</sub>* and chooses an action *a<sub>t</sub>* from the set of actions *A*, which will take him to a state *s<sub>t+1</sub>* with a probability *p(s<sub>t</sub> -> s<sub>t+1</sub> | a<sub>t</sub>)*. The agent receives a scalar reward (or punishment) *r<sub>t+1</sub>* for choosing action *a* in state *s<sub>t</sub>*. Note that the Markov property requires that the probabilities of arriving in a state *s<sub>t+1</sub>* and receiving a reward *r<sub>t+1</sub>* only depend on the state *s<sub>t</sub>* and the action *a<sub>t</sub>*. They are independent of previous states, actions, and rewards. The goal is now to find a deterministic *policy* that maximizes the sum of accumulated rewards over time.

While it is theoretically possible to explicitly model the entire MDP including all of its states, in reality this is only feasible for extremely simple cases - often the number of states is very large (or inifinite) and the entire set *S* of states might not be known. Furthermore, *P* (the transition probabilities) and *R* (the immediate rewards for performing a specific action *a* in state *s* that leads to state *s'*) might not be known. In such cases, using a model-free reinforcement learning approach (i.e., an approach that does not explicitly model the MDP) is more appropriate. Typically, this is realized by applying a form of *Q-learning*, which is also the approach taken here. 

In Q-learning, the Q-function plays a central role. For a fixed policy, the Q-function models the expected *future discounted reward* for applying action *a* in state *s*, i.e., for a state-action pair *(s<sub>t</sub>, a<sub>t</sub>)* the Q-function is given by:  
*Q(s<sub>t</sub>, a<sub>t</sub>) = r<sub>t+1</sub> + n * max<sub>a'</sub> (Q(s<sub>t+1</sub>,a'))*,  
where *r<sub>t+1</sub>* is the immediate reward for aplying action *a<sub>t</sub>* in state *s<sub>t</sub>*, *s<sub>t+1</sub>* is the resulting state, *n* is a *discount factor < 1* (often chosen as *0.9*) and *max<sub>a'</sub> (Q(s<sub>t+1</sub>,a'))* is the result of the Q function for applying action *a'* in the resulting state *s<sub>t+1</sub>* where *a'* is the action that maximizes the expected reward for *s<sub>t+1</sub>* from the set of all available actions. Note that for simplified notation, I left out the probabilities (which is done later in coding, anyway, so it corresponds to the code quite closely). That means that the Q-function estimates the accumulated rewards in the future when choosing an action *a* in state *s* where the rewards in the future are discounted by a factor *n*. The discount factor *n* can be chosen as a weighting factor between the immediate reward and the expected reward in the future: when choosing a very small value for *n*, a more greedy strategy (i.e., focusing on the maximum immediate reward) will yield higher values, while for larger values of *n*, higher expected rewards in the future will outweigh the immediate reward. For a more detailed explanation of Q-learning and its mathematical foundation, please refer to the aforementioned papers. 

Q-learning now works by iteratively updating the Q-function, i.e., the estimates of the future discounted rewards, when observing a state transition in the form of a tuple *(s<sub>t</sub>, a<sub>t</sub>, r<sub>t+1</sub>, s<sub>t+1</sub>)* by using the following rule:  
*Q(s<sub>t</sub>,a<sub>t</sub>) <- Q(s<sub>t</sub>,a<sub>t</sub>) + l * (r<sub>t+1</sub> + n * max<sub>a'</sub> (Q(s<sub>t+1</sub>,a')) - Q(s<sub>t</sub>,a<sub>t</sub>))*,  
where *l* is a learning rate.

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

