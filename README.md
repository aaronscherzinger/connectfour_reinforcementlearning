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

### Neural Networks and Deep Learning

Over the last decade, neural networks have been undergoing a renaissance due to the capabilities of modern programmable graphics hardware as well as the availability of large amounts of data for training networks with millions of *weights* (i.e., trainable parameters). Since then, neural networks have continuously set new performance benchmarks in many areas, especially in computer vision and pattern recognition. As a detailed review of the basics of *feedforward neural networks*, also referred to as *multilayer perceptrons*, *backpropagation* (the algorithm used for computing the gradient of some *loss function*, i.e. error function, with respect to the weights of the network, usually in order to perform an iterative optimization via some variant of gradient descent), and *convolutional neural networks* (*CNN*s) would go far beyond the scope of this quick overview, I would refer you to the following resources for the details of training neural networks:

[https://www.deeplearningbook.org/]

[http://neuralnetworksanddeeplearning.com/]

[https://www.coursera.org/learn/machine-learning]

### Connect Four

It is probably not necessary to provide a detailed exlanation of the connect four game, as most people should be familiar with it. If not, you can find some information here: [https://en.wikipedia.org/wiki/Connect_Four]

In the implementation, the playing board is represented by a *7x6* numpy array (column-major order), where *0* corresponds to an unoccupied space, *1* corresponds to a disc of player 1, and *-1* corresponds to a disc of player 2 (this helps with easily inverting the roles of the players, as will be used frequently during training the network). For more implementation details, please refer directly to the source files (you will find a list of the files with a quick description of their corresponding contents at the bottom).

### Implementation

The implementation follows the approach proposed by Mnih et al. Please refer to the corresponding paper using the link above. 

The overall concept works as follows:

* First we create a neural network (or CNN) with randomly initialized weights
  * As its input, the neural network receives the playing field (which corresponds to the state *s<sub>t</sub>*), where the neural network is always player 1.
  * The output of the neural network consists of 7 numbers, each representing the result of the Q-function, i.e., an estimate of the expected future discounted reward, for inserting a disc into the corresponding column. The chosen action *a<sub>t</sub>* is thus an integer in the range *[0, 6]*.
* In each step, the network takes a turn, which is chosen using the *epsilon-greedy* strategy: with probability *(1-epsilon)*, we will choose the action where the network expects the highest future discounted reward (*exploitation*), otherwise we will choose a random action (*exploration*). Over time, when the estimates of the Q-function will converge more and more (i.e., the network gets better at its estimations), we will decrease the value of *epsilon* from the initial value of *1* to a value of *0.1*. 
* After performing the turn, one of the following situations can occur:
  1. We won the game - the immediate reward *r<sub>t+1</sub>* will be set to *+1*, the resulting state *s<sub>t+1</sub>* is set to the terminal state that resulted from the action.
  2. If we did not directly win, we will obtain the resulting state *s<sub>t+1</sub>* as the next state that results from the opponent's move - we will thus create a temporary copy of the playing board after making our move, invert the roles of the players on the temporary board and let the neural network make another turn on this temporary board (obviously now as the opponent). The temporary board is again inverted and we obtain the expected resulting state *s<sub>t+1</sub>* which resulted from the opponent's action. If we now lost the game, the immediate reward *r<sub>t+1</sub>* will be set to *-1* (i.e., we punish the network for losing the game), otherwise it will be set to *0*.
  3. If we chose an action that tries to insert a disc into a colummn that is already full, we set the immediate reward *r<sub>t+1</sub>* to *-2* - that means that we will punish the neural network more for trying to break the rules than for losing the game.
* The transition *(s<sub>t</sub>, a<sub>t</sub>, *r<sub>t+1</sub>*, *s<sub>t+1</sub>*)* along with the information if *s<sub>t+1</sub>* is a terminal game state will be stored in a *replay memory* consisting of the last *N* transitions.
* If the playing board is in a terminal state, it will be reset to an empty state. Else, players will be switched, so that in the next turn, the neural network will take the opposite role. The networks thus learns by playing against itself without any exterior input (except of course the parameters such as the fixed rewards, etc.).
* From the replay memory, a *mini-batch* of *M* transitions is randomly assembled to perform a backpropagation pass for updating the weights of the neural network to improve the estimate of the Q-function. For details, please refer to the explanations provided by Mnih et al. in the corresponding paper listed above or directly to the code.

In regular intervals, the current loss is plotted using a simple line plot. After a fixed number of played games, the training ends and the weights, i.e., the parameter corresponding to the current training state of the neural network, are exported.

### Results

My experience during this small project was that the reinforcement learning process (or rather, this specific implementation) is rather sensitive to the hyper parameters chosen for the training, such as the learning rate and specific optimizer. I tried different optimizers, of which RMSProp delivered the best and most stable training results. Despite experimenting with different learning rates, I could not get the Adam optimizer to converge. While stochastic gradient descent with momentum did converge, its results were inferior to RMSProp in terms of the achieved loss.

The results in terms of the overall performance in connect four were not too brilliant, although I did increase the size of the networks by a considerable amount over the course of my experiments. The current feedforward network has a size of 42 inputs (the playing field), four hidden layers with 4096, 2048, 1024, and 512 fully connected units, and 7 outputs (for the columns). The CNN-version has two 3x3 conv layers with 512 and 768 Kernels filter kernels, followed by a fully connected part with 2048, 1024, 512, and 256 neurons. Training the fully connected network takes around two hours on my NVIDIA GTX 1070 GPU with CUDA 10 (however, please take into account that the code is anything but optimized). Of course, you can reduce the size of the neural networks as well as the number of played training games (*episodes*) in the code prior to starting the training in order to reduce the amount of required time. 

The poor performance of the networks might indicate that the problem of playing connect four is not very well-suited for direct processing by a neural net. This might be due to the fact that neural networks are very good in learning continous relations between input and output where the output changes only slightly if the input does so as well - this is for instance the case with the Atari games in the paper of Mnih et al.: between very similar frames, the playing situation, and thus estimates of the expected reward for taking a specific action, does not drastically change, yielding a more or less continous relation. However, for connect four this is not the case, as the playing situations are dicrete states that change very rapidly, not in a smooth and continous fashion. While looking into this, I stumpled upon this post addressing this question: [https://www.quora.com/How-do-I-make-Q-learning-with-ANN-work-for-a-simple-board-game]

However, since my initial goal was not to create the best possible AI for playing connect four, but only to implement a simple working example of reinforcement learning, I did no further investigation (if you know more about this, please feel free to contact me). For implementing an efficient AI for connect four, other solutions would probably be more adequate, anyway. 

What the network **did** learn very well was not to insert discs into a full column (which is better than nothing, I guess). And sometimes, if you do not watch out, it will make some nice strategic moves, as you can see here from the output of my text-based playing example program:

```
(Turn 1 - the AI inserts into column 3):  
 0 1 2 3 4 5 6  
|. . . . . . .|  
|. . . . . . .|  
|. . . . . . .|  
|. . . . . . .|  
|. . . . . . .|  
|. . . o . . .|  
 -------------  
   
(Player inserts into column 3):  
 0 1 2 3 4 5 6  
|. . . . . . .|  
|. . . . . . .|  
|. . . . . . .|  
|. . . . . . .|  
|. . . x . . .|  
|. . . o . . .|  
 -------------  
   
(Turn 1 - the AI inserts into column 2):  
 0 1 2 3 4 5 6  
|. . . . . . .|  
|. . . . . . .|  
|. . . . . . .|  
|. . . . . . .|  
|. . . x . . .|  
|. . o o . . .|  
 -------------  
   
(Player makes a mistake and insert into column 0):  
 0 1 2 3 4 5 6  
|. . . . . . .|  
|. . . . . . .|  
|. . . . . . .|  
|. . . . . . .|  
|. . . x . . .|  
|x . o o . . .|  
 -------------  
   
(Now the AI will use a strategy that leads to a guaranteed win):  
 0 1 2 3 4 5 6  
|. . . . . . .|  
|. . . . . . .|  
|. . . . . . .|  
|. . . . . . .|  
|. . . x . . .|  
|x . o o o . .|  
 -------------  
 
(From here on, the player is lost - he decides to insert into column 5):
 0 1 2 3 4 5 6  
|. . . . . . .|  
|. . . . . . .|  
|. . . . . . .|  
|. . . . . . .|  
|. . . x . . .|  
|x . o o o x .|  
 -------------  
 
(And the AI wins the game):
 0 1 2 3 4 5 6  
|. . . . . . .|  
|. . . . . . .|  
|. . . . . . .|  
|. . . . . . .|  
|. . . x . . .|  
|x o o o o x .|  
 -------------  
```
Overall, you could maybe compare the network to a little kid that sometimes struggles to see an obvious trap, but in other instances has a clever idea and will try to fool you if you do not pay attention :-)

## Prerequisites for Running the Code

The code is implemented using **Python 3.6** and uses the deep learning framework **Tensorflow**. You can simply install Tensorflow for Python via pip. However, for using this code I would advice you to use a machine with a dedicated GPU that is CUDA-capable and install the GPU-version of Tensorflow (otherwise training the network might take a very long time). You might need to compile Tensorflow yourself, depending on your GPU driver and CUDA version, but there are sufficient resources for that online.

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

