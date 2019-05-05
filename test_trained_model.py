import tensorflow as tf

import numpy as np 
import connect_four

def board_to_column_vector(board):
    '''converts a playing board (i.e., 2D numpy array with shape [7,6]) to a column vector with shape [1, 42]'''
    assert(board.shape == (7,6)), "board does not have shape (7,6)"
    return np.reshape(board.flatten(), (1, 42))

# create tensorflow session 
test_session = tf.Session()
tf.reset_default_graph()

# import model
import_path =  './current_model'
meta_graph_def = tf.saved_model.loader.load(
           test_session,
          [tf.saved_model.tag_constants.SERVING],
          import_path)
signature = meta_graph_def.signature_def

# get input and output tensors
signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
input_tensor_name = signature[signature_key].inputs["input_x"].name
output_tensor_name = signature[signature_key].outputs["network_output"].name

input_x = test_session.graph.get_tensor_by_name(input_tensor_name)
network_output = test_session.graph.get_tensor_by_name(output_tensor_name)

tf.get_default_graph().finalize()


# playing board as global variable
playing_board = connect_four.PlayingBoard()
playing_board.insert(5, 1)
playing_board.insert(4, 1)
playing_board.insert(3, 1)
playing_board.insert(1, -1)
playing_board.insert(1, -1)
playing_board.insert(0, -1)

playing_board.invert_board()

# test
feed_dict = { input_x : board_to_column_vector(playing_board.get_board()) }
estimates = test_session.run(network_output, feed_dict)[0]

print(estimates)
playing_board.print_board()
print(np.argmax(estimates))

playing_board.reset_board()

# construct kind of an extreme test situation
# 1. one full column (should yield and expection of -2)
player = -1
for i in range(0, 6):
    playing_board.insert(6, player)
    player *= -1

# 2. one column which allows us to win (should yield expectation of 1)
playing_board.insert(0, 1)
playing_board.insert(0, 1)
playing_board.insert(0, 1)

# 3. one column where the opponent would win if we select anything else
playing_board.insert(2, -1)
playing_board.insert(2, -1)
playing_board.insert(2, -1)

# we are player two -> one more of -1
#playing_board.insert(3, -1)

# test
feed_dict = { input_x : board_to_column_vector(playing_board.get_board()) }
estimates = test_session.run(network_output, feed_dict)[0]

print(estimates)
playing_board.print_board()
print(np.argmax(estimates))
