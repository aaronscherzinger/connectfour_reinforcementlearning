import unittest

import numpy as np
import connect_four

class BoardTest(unittest.TestCase):

    board = connect_four.PlayingBoard()

    def setUp(self):
        self.board = connect_four.PlayingBoard()

    def test_empty(self):
        # 1. test empty self.board
        self.assertEqual(self.board.get_board().shape, (7,6))
        self.assertTrue(np.equal(self.board.get_board(), np.zeros([7,6])).all())

    def test_reset(self):
        # empty self.board reset
        self.board.reset_board()
        self.assertEqual(self.board.get_board().shape, (7,6))
        self.assertTrue(np.equal(self.board.get_board(), np.zeros([7,6])).all())

        # insert beforehand
        self.board.insert(3, 1)
        self.board.insert(1, -1)
        self.board.insert(0, 1)

        self.board.reset_board()
        self.assertEqual(self.board.get_board().shape, (7,6))
        self.assertTrue(np.equal(self.board.get_board(), np.zeros([7,6])).all())

    def test_insert(self):
        # insert in every column the max number of stones
        for column in range(0,7):
            for row in range(0,6):
                self.board.insert(column, 1)
                self.assertEqual(self.board.get_board()[column][row], 1)
        
        # reset board
        self.board.reset_board()

        # the same, but first go through the columns and then the rows
        for row in range(0,6):
            for column in range(0,7):
                self.board.insert(column, -1)
                self.assertEqual(self.board.get_board()[column][row], -1)

    def test_column_check(self):
        # insert into a column the max number of stones and then check if it is full
        for column in range(0,7):
            for stone in range(0,6):
                self.assertTrue(self.board.check_column(column))
                self.board.insert(column, 1)
            self.assertFalse(self.board.check_column(column))

    def test_invert_board(self):
        tmp_board = self.board.get_board()
        self.board.invert_board()
        self.assertTrue(np.equal(tmp_board, -1.0 * self.board.get_board()).all())

        for column in range(0,7):
            for row in range(0,6):
                self.board.insert(column, 1)
                tmp_board = self.board.get_board()
                self.board.invert_board()
                self.assertTrue(np.equal(tmp_board, -1.0 * self.board.get_board()).all())

    def test_terminal_state(self):
        # check empty self.board
        self.assertFalse(self.board.check_terminal_state()[0])
        
        # insert a few times
        self.board.insert(3, 1)
        self.board.insert(2, -1)
        self.board.insert(3, 1)
        self.board.insert(6, -1)
        self.assertFalse(self.board.check_terminal_state()[0])

        # reset
        self.board.reset_board()
        self.assertFalse(self.board.check_terminal_state()[0])

        # now check for columns
        for column in range(0,7):
            for player in [-1, 1]:
                for filler_rows in range(0,2):
                    for num_rows in range(0, filler_rows):
                        self.board.insert(column, -player)
                    
                    self.assertFalse(self.board.check_terminal_state()[0])
                    
                    self.board.insert(column, player)
                    self.assertFalse(self.board.check_terminal_state()[0])
                    
                    self.board.insert(column, player)
                    self.assertFalse(self.board.check_terminal_state()[0])
                    
                    self.board.insert(column, player)
                    self.assertFalse(self.board.check_terminal_state()[0])
                    
                    self.board.insert(column, player)
                    self.assertTrue(self.board.check_terminal_state()[0])
                    self.assertEqual(self.board.check_terminal_state()[1], player)
                    # reset
                    self.board.reset_board()
                    self.assertFalse(self.board.check_terminal_state()[0])

        # now check for bottom rows
        for starting_column in range(0,4):
            for player in [-1, 1]:
                for column_add in range(0,4):
                    self.assertFalse(self.board.check_terminal_state()[0])
                    self.board.insert(starting_column + column_add, player)
                self.assertTrue(self.board.check_terminal_state()[0])
                self.assertEqual(self.board.check_terminal_state()[1], player)
                # reset
                self.board.reset_board()
                self.assertFalse(self.board.check_terminal_state()[0])
        
        # now check for some other row
        for starting_column in range(0,4):
            for player in [-1, 1]:
                for starting_row in range(1, 6):
                    self.assertFalse(self.board.check_terminal_state()[0])
                    # here, we directly set the numbers in the board as a hack as otherwise we would have to fill the bottom rows while not making any unwanted columns or diagonals that evaluate to True
                    for column_add in range(0,4):
                        self.assertFalse(self.board.check_terminal_state()[0])
                        self.board.board[starting_column + column_add, starting_row] = player
                    self.assertTrue(self.board.check_terminal_state()[0])
                    self.assertEqual(self.board.check_terminal_state()[1], player)
                    # reset
                    self.board.reset_board()
                    self.assertFalse(self.board.check_terminal_state()[0])

        # now check ascending diagonals (same hack as before)
        for starting_column in range(0,4):
            for player in [-1, 1]:
                for starting_row in range(0, 3):
                    for i in range(0,4):
                        self.assertFalse(self.board.check_terminal_state()[0])
                        self.board.board[starting_column + i,starting_row + i] = player
                    self.assertTrue(self.board.check_terminal_state()[0])
                    self.assertEqual(self.board.check_terminal_state()[1], player)
                    # reset
                    self.board.reset_board()
                    self.assertFalse(self.board.check_terminal_state()[0])

        # now check descending diagonals (see above)
        for starting_column in range(0,4):
            for player in [-1, 1]:
                for starting_row in range(3, 5):
                    for i in range(0,4):
                        self.assertFalse(self.board.check_terminal_state()[0])
                        self.board.board[starting_column + i,starting_row - i] = player
                    self.assertTrue(self.board.check_terminal_state()[0])
                    self.assertEqual(self.board.check_terminal_state()[1], player)
                    # reset
                    self.board.reset_board()
                    self.assertFalse(self.board.check_terminal_state()[0])

if __name__ == '__main__':
    unittest.main()
