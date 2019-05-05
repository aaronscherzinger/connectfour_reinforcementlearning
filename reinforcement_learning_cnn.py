import tensorflow as tf

import connect_four

import numpy as np
import math
import random

import time

import matplotlib.pyplot as plt

import os
import shutil

# export path for trained model
export_path = "./current_model"

# remove an old model if existing
if os.path.isdir(export_path):
    shutil.rmtree(export_path)

# playing board as global variable
playing_board = connect_four.PlayingBoard()

# replay memory which consists of tuples (input state s_t, performed action a_t, observed immediate reward r_t, resulting state s_(t+1), information if s_(t+1) is terminal state)
replay_memory = []

# parameters for the reinforcement learning algorithm
discount_factor = 0.9
# epsilon for epsilon-greedy strategy that is reduced over time
epsilon_start = 1.0
epsilon_end = 0.1
num_epsilon_interpolation_iterations = 100000
# memory cells, total numnber of games to play, batch size for gradient descent optimizer
num_memory_cells = 50000
num_games = 8000
batch_size = 2000

# do tensorflow stuff
tf.reset_default_graph()

# initialize placeholders: x corresponds to the network's input, which is the current state, i.e., the playing board, as a flattened float32 column vector
# y corresponds to the expected, i.e., desired, outputs for training 
input_x = tf.placeholder(dtype = tf.float32, shape = [None, 7, 6, 1], name = "input_x")
input_y = tf.placeholder(dtype = tf.float32, shape = [None, 7], name = "input_y")

global_step = tf.Variable(0, trainable=False)


#num_inputs = 7*6
num_filters_0 = 512
num_filters_1 = 768

# Convolutional Layer #1 - 3x3 kernels with stride 1 and no padding -> output is 5x4
conv0 = tf.layers.conv2d(
  inputs=input_x,
  filters=num_filters_0,
  kernel_size=[3, 3],
  strides=1,
  padding="valid",
  activation=tf.nn.relu)

# Convolutional Layer # 2 - 3x3 kernels with stride 1 and no padding -> output is 3x2
conv1 = tf.layers.conv2d(
  inputs=conv0,
  filters=num_filters_1,
  kernel_size=[3, 3],
  strides=1,
  padding="valid",
  activation=tf.nn.relu)

conv_flat_outputs = 3 * 2 * num_filters_1
# reshape the output for fully connected layers

conv_flat = tf.reshape(conv1, [-1, conv_flat_outputs])


# mlp network architecture:
# 7*6=42 (input) - hidden_0 - ReLU - ... - hidden_N - ReLU - 7 (output)
num_hidden_units_0 = 2048
num_hidden_units_1 = 1024
num_hidden_units_2 = 512
num_hidden_units_3 = 256
#num_hidden_units_4 = 128
#num_hidden_units_5 = 64
num_outputs = 7

# weights and biases for fully connected layers
weights_0 = tf.Variable(tf.random_normal((conv_flat_outputs,num_hidden_units_0), stddev=math.sqrt(2.0/float(conv_flat_outputs + num_hidden_units_0))), dtype = tf.float32) 
bias_0 = tf.Variable(tf.zeros(num_hidden_units_0), dtype = tf.float32) 

weights_1 = tf.Variable(tf.random_normal((num_hidden_units_0,num_hidden_units_1), stddev=math.sqrt(2.0/float(num_hidden_units_0 + num_hidden_units_1))), dtype = tf.float32) 
bias_1 = tf.Variable(tf.zeros(num_hidden_units_1), dtype = tf.float32)

weights_2 = tf.Variable(tf.random_normal((num_hidden_units_1,num_hidden_units_2), stddev=math.sqrt(2.0/float(num_hidden_units_1 + num_hidden_units_2))), dtype = tf.float32) 
bias_2 = tf.Variable(tf.zeros(num_hidden_units_2), dtype = tf.float32) 

weights_3 = tf.Variable(tf.random_normal((num_hidden_units_2,num_hidden_units_3), stddev=math.sqrt(2.0/float(num_hidden_units_2 + num_hidden_units_3))), dtype = tf.float32) 
bias_3 = tf.Variable(tf.zeros(num_hidden_units_3), dtype = tf.float32)

#weights_4 = tf.Variable(tf.random_normal((num_hidden_units_3,num_hidden_units_4), stddev=math.sqrt(2.0/float(num_hidden_units_3 + num_hidden_units_4))), dtype = tf.float32) 
#bias_4 = tf.Variable(tf.zeros(num_hidden_units_4), dtype = tf.float32)

#weights_5 = tf.Variable(tf.random_normal((num_hidden_units_4,num_hidden_units_5), stddev=math.sqrt(2.0/float(num_hidden_units_4 + num_hidden_units_5))), dtype = tf.float32) 
#bias_5 = tf.Variable(tf.zeros(num_hidden_units_5), dtype = tf.float32)

#weights_6 = tf.Variable(tf.random_normal((num_hidden_units_5,num_outputs), stddev=math.sqrt(2.0/float(num_hidden_units_5 + num_outputs))), dtype = tf.float32) 
#bias_6 = tf.Variable(tf.zeros(num_outputs), dtype = tf.float32) 

#trainable_weights = [weights_0, bias_0, weights_1, bias_1, weights_2, bias_2, weights_3, bias_3, weights_4, bias_4, weights_5, bias_5, weights_6, bias_6]

weights_4 = tf.Variable(tf.random_normal((num_hidden_units_3,num_outputs), stddev=math.sqrt(2.0/float(num_hidden_units_3 + num_outputs))), dtype = tf.float32) 
bias_4 = tf.Variable(tf.zeros(num_outputs), dtype = tf.float32)

trainable_weights = [weights_0, bias_0, weights_1, bias_1, weights_2, bias_2, weights_3, bias_3, weights_4, bias_4]

# specify network

# forward pass
hidden_layer_0_output = tf.nn.relu(tf.add(tf.matmul(conv_flat, weights_0), bias_0))
hidden_layer_1_output = tf.nn.relu(tf.add(tf.matmul(hidden_layer_0_output, weights_1), bias_1))
hidden_layer_2_output = tf.nn.relu(tf.add(tf.matmul(hidden_layer_1_output, weights_2), bias_2))
hidden_layer_3_output = tf.nn.relu(tf.add(tf.matmul(hidden_layer_2_output, weights_3), bias_3))
#hidden_layer_4_output = tf.nn.relu(tf.add(tf.matmul(hidden_layer_3_output, weights_4), bias_4))
#hidden_layer_5_output = tf.nn.relu(tf.add(tf.matmul(hidden_layer_4_output, weights_5), bias_5))
#network_output = tf.add(tf.matmul(hidden_layer_5_output, weights_6), bias_6)

network_output = tf.add(tf.matmul(hidden_layer_3_output, weights_4), bias_4)

# specify loss function
loss = tf.losses.mean_squared_error(input_y, network_output)

# base learning rate (for RMSProp)
base_learning_rate = 0.0001

# create a decayed learning rate
#decayed_lr = tf.train.exponential_decay(base_learning_rate, global_step, 10000, 0.5, staircase=True)

# create optimizer -> here, we use RMSProp, for now we keep a constant learning rate
optimizer = tf.train.RMSPropOptimizer(learning_rate = base_learning_rate)
#optimizer = tf.train.RMSPropOptimizer(learning_rate = decayed_lr)
#train_op = optimizer.minimize(loss, var_list = trainable_weights, global_step = global_step)
train_op = optimizer.minimize(loss, global_step = global_step)

# Adam requires lower learning rate (0.00001), but does not converge as well as RMSProp
#optimizer = tf.train.AdamOptimizer(learning_rate = base_learning_rate).minimize(loss, var_list = trainable_weights, global_step = global_step)
#optimizer = tf.train.AdamOptimizer(learning_rate = decayed_lr).minimize(loss, var_list = trainable_weights, global_step = global_step)

# Momentum gradient descent is another (simple) alternative optimizer
#optimizer = tf.train.MomentumOptimizer(learning_rate = base_learning_rate, momentum = 0.9, var_list = trainable_weights)

# create tensorflow session and initialize variables
training_session = tf.Session()
training_session.run(tf.global_variables_initializer())

# TODO: we cannot do this due to the model builder for saving at the end... better solution?
#tf.get_default_graph().finalize()

def board_to_one_channel_map(board):
    '''adds one channel to the board'''
    assert(board.shape == (7,6)), "board does not have shape (7,6)"
    return np.reshape(board, (1, 7, 6, 1))

def epsilon_greedy(in_vec, epsilon):
    '''epsilon greedy strategy for selecting an action'''
    p = random.uniform(0,1)
    random_choice = True
    action = 0
    estimates = None
    if p > epsilon:
        # exploitation: select best action as determined by the classifier                    
        # perform forward pass through network with current state and gather result
        estimates = np.copy(training_session.run( network_output, {input_x : in_vec} )[0])
        
        # select best action by determining the maximum output unit
        action = np.argmax(estimates)

        global num_exploitations 
        num_exploitations += 1

        random_choice = False
    else:
        # with a probability of epsilon: exploration by selecting random action
        action = random.randint(0,6)

        global num_explorations 
        num_explorations += 1
    
    return action, random_choice, estimates

# plotting stuff
plt.ion() 
fig=plt.figure()
plt.axis([0,num_games,0,0.02])
plt.xlabel("Num finished games")
plt.ylabel("MSE (future discounted reward)")
plot_x=list()
plot_y=list()

# training loop
epsilon = epsilon_start
iterations = 0
episode = 0
started_training = False
abort_training = False

# just for tracking the progress
num_wrong_columns = 0
num_exploitations = 0
num_explorations = 0
current_time = time.time()

print("Starting to generate initial replay memory...")

total_num_episodes = 0
while episode <= num_games and (not abort_training):
    # new game: reset the playing board
    playing_board.reset_board()
    
    if started_training:
        episode += 1

    # for every second game, the net starts out as player two with a random insertion by the first player
    total_num_episodes += 1
    if total_num_episodes % 2 == 1:
        start_action = random.randint(0,6)
        playing_board.insert(start_action, -1)
    
    # play the game while it is not in a terminal state
    while not playing_board.check_terminal_state()[0]:
 
        current_state = playing_board.get_board()

        # this is just a sanity check: we are always player one, so there should either be the same amount of -1 and 1 in the array or one more -1 as we should make a move now
        num_stones_player_one = np.count_nonzero(current_state == 1)
        num_stones_player_two = np.count_nonzero(current_state == -1)
        assert((num_stones_player_one == num_stones_player_two) or (num_stones_player_one + 1 == num_stones_player_two)), "playing situation is broken"

        # we create a new transition (current_state, action, reward, resulting_state, is_terminal)
        action = 0
        reward = 0.0
        resulting_state = np.copy(current_state)

        # initialize input vector with current state
        in_vec = board_to_one_channel_map(current_state)
        
        # perform epsilon-greedy strategy to determine action to be taken
        action, random_choice, estimates = epsilon_greedy(in_vec, epsilon)

        # (try to) execute action
        if playing_board.check_column(action):
            # can execute action -> observe environment
            playing_board.insert(action, 1.0)

            # check terminal state to determine reward
            terminal_state = playing_board.check_terminal_state()
            if terminal_state[0]:
                # by our move, we either won or got a draw (i.e., full board)
                resulting_state = playing_board.get_board()
                is_terminal = True
                # positive reinforcement for player 1 (= 1), negative for player 2 (= -1), neutral else (= 0)
                reward = terminal_state[1]
            else:
                # the game is not yet finished
                # the resulting state is not what we directly obtain, but what we get back afterwards from our opponent
                # -> create a temporary board, do the opponent's move and then look at the state from our perspective
                tmp_board = connect_four.PlayingBoard()
                tmp_board.set_board(playing_board.get_board())
                tmp_board.invert_board()

                tmp_action, _, _ = epsilon_greedy(board_to_one_channel_map(tmp_board.get_board()), epsilon)
                # since we are not in a terminal state, there has to be at least one possibility to insert
                # so we let the "opponent" try as long as he needs to find a valid action if he does not select one right away
                while not tmp_board.check_column(tmp_action):
                    tmp_action, _, _ = epsilon_greedy(board_to_one_channel_map(tmp_board.get_board()), epsilon)
                tmp_board.insert(tmp_action, 1.0)
                tmp_board.invert_board()

                # update the terminal state (maybe draw after opponent's move or even defeat)
                terminal_state = tmp_board.check_terminal_state()

                # resulting state is what we observe from our perspective after opponent made a move
                resulting_state = tmp_board.get_board()
                is_terminal = terminal_state[0]
                reward = terminal_state[1]

        else:
            # cannot execute as column is already full - high negative reward
            reward = -2.0
            # we handle a full column as if the game is then terminated because of a wrong move
            is_terminal = True

            # we need to invert the board since we did not insert a stone to avoid wring situations as we invert later on
            playing_board.invert_board()

            # count wrong columns that the net chose itself
            if not random_choice:
                num_wrong_columns += 1

        # store the transition in the replay memory for training
        transition = (current_state, action, reward, resulting_state, is_terminal)
        replay_memory.append(transition)
        # keep the length of the replay memory in check by removing oldest memory
        if len(replay_memory) > num_memory_cells:
            replay_memory.pop(0)
            
        # neural net is always player one and plays against itself -> invert the players for the next turn
        playing_board.invert_board()

        # we do not start the actual training until our replay memory is filled to create meaningful mini-batches
        if len(replay_memory) < num_memory_cells:
            continue

        if (not started_training):
            started_training = True
            elapsed_time = time.time() - current_time
            print("Time for creating initial memory: " + str(elapsed_time) + " seconds")
            current_time = time.time()
            print("Starting iterative training phase...")
            # reset counters
            iterations = 0
            num_explorations = 0
            num_exploitations = 0
            num_wrong_columns = 0

        # now that we have performed the online-phase (i.e., the net playing using greedy-epsilon strategy), we train the network offline from replay memory 

        # assemble random mini-batch
        samples = random.sample(replay_memory, batch_size)

        # samples -> list of tuples 
        # samples_x -> the input states of each samples
        samples_x = np.zeros([batch_size, 7, 6, 1], dtype = np.float32)
        # samples_actions -> the performed action in this situation
        sample_actions = np.zeros(batch_size, dtype = np.uint32)
        # sample_rewards -> the resulting immediate rewards given (input state, action)
        sample_rewards = np.zeros(batch_size, dtype = np.float32)
        # sample_resuting_states -> the states resulting from executing the action in the input state
        sample_resulting_states = np.zeros([batch_size, 7, 6, 1], dtype = np.float32)
        # sample_terminal_states -> determines if the samples' resulting state is terminal
        sample_terminal_states = np.zeros(batch_size, dtype = np.bool_)

        # TODO: this is not very efficient...
        for i in range(0,batch_size):
            current_sample = samples[i]
            samples_x[i,:] = np.reshape(current_sample[0], [7,6,1]) #.flatten()
            sample_actions[i] = current_sample[1]
            sample_rewards[i] = current_sample[2]
            sample_resulting_states[i,:] = np.reshape(current_sample[3], [7,6,1]) #.flatten()
            sample_terminal_states[i] = current_sample[4]
             
        # for each sample in the minibatch: determine samples_y -> desired output vectors for computing loss in order to perform gradient descent
        # note: set the desired output only for the action specified in the tuple, every other output is set to the net prediction as we do not want to (and cannot) update it
        # therefore: perform forward pass to determine the predictions of the net for the state and set the desired values to the predicted ones
        samples_y = np.copy(training_session.run( network_output, {input_x : samples_x} ))

        # now set the y-value for the one action we can update, i.e., the selected one
        # possibility 1: resulting_state in the transition is terminal -> y = reward in the transition
        # possibility 2: perform forward pass for the resulting state, get the maximum max_expected of the expected rewards from the output units of the net and set y = reward in the transition + discount_factor * max_expected

        # in order to make it more efficient, we will first perform the forward pass for all resulting states
        future_net_outs = np.copy(training_session.run( network_output, {input_x : sample_resulting_states} ))

        # now set the y-values for the selected action for each sample
        for i in range(0, batch_size):
            # for a terminal state, there is only the immediate reward
            samples_y[i,sample_actions[i]] = sample_rewards[i]
            
            # for a non-terminal state, we need to estimate the future discounted reward
            # use forward pass of the resulting state to determine net Q-value predictions for computing future discounted reward
            # then add immediate reward (set above) and dicount_factor * estimated_future_reward
            if not sample_terminal_states[i]:
                samples_y[i,sample_actions[i]] = sample_rewards[i] + discount_factor * np.max(future_net_outs[i].flatten())

        # perform gradient step on mean squared error 
        _, current_loss, current_step = training_session.run([train_op, loss, global_step], {input_x: samples_x, input_y: samples_y })

        # adapt epsilon for greedy-epsilon algorithm
        if iterations <= num_epsilon_interpolation_iterations:
            interpolation_factor = iterations / num_epsilon_interpolation_iterations
            epsilon = (1 - interpolation_factor) * epsilon_start + interpolation_factor * epsilon_end
        else:
            epsilon = epsilon_end

        iterations = iterations + 1

        # TODO: better loss plot / plot more results such as average expected reward, etc.
        if (iterations > 0) and (iterations  % 500 == 0) or (playing_board.check_terminal_state()[0] and (episode == num_games)):
            print("-----")
            print("Episode: " + str(episode))
            print("Iteration: " + str(iterations ) + ", Loss: " + str(current_loss))
            #print("Learning rate: " + str(current_lr))
            print("epsilon: " + str(epsilon))
            print("Num exploitations: " + str(num_exploitations))
            num_exploitations = 0
            print("Num explorations: " + str(num_explorations))
            num_explorations = 0
            print("Selected wrong columns: " + str(num_wrong_columns))
            num_wrong_columns = 0
            elapsed_time = time.time() - current_time
            print("Elapsed time: " + str(elapsed_time) + " seconds")
            current_time = time.time()
            print("-----")

            # plotting
            plot_x.append(episode);
            plot_y.append(np.max(current_loss));
            #plt.scatter(episode,current_loss, c="red",marker=".");
            plt.plot(plot_x, plot_y, c="red")
            plt.show()
            plt.pause(0.0001)

            fig.savefig("training_plot.png")

            # optional quality criterion: once the loss drops below a certain value, we assume to have a good model
            #if current_loss < 0.0003:
            #    abort_training = True

# save final model
# 1. create model builder
builder = tf.saved_model.Builder(export_path)
tensor_info_x = tf.saved_model.utils.build_tensor_info(input_x)
tensor_info_out = tf.saved_model.utils.build_tensor_info(network_output)

prediction_signature = (
  tf.saved_model.signature_def_utils.build_signature_def(
      inputs={'input_x': tensor_info_x},
      outputs={'network_output': tensor_info_out},
      method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
)

builder.add_meta_graph_and_variables(
  training_session, [tf.saved_model.tag_constants.SERVING],
  signature_def_map={
      tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
          prediction_signature 
  })
 
# 2. perform actual saving
builder.save()

# close the session
training_session.close()

print("Finished training, saved final model to " + str(export_path))
