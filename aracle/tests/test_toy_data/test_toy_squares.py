import os, sys
import numpy as np
import unittest
from aracle.toy_data import ToySquares

class TestToySquares(unittest.TestCase):
    """A suite of tests testing the ToySquares class
    
    """
    def test_constructor(self):
        """Test the instantiation of ToySquares

        """
        toy_squares = ToySquares(canvas_size=32, n_objects=20)
        return toy_squares

    def test_increment_time_step(self):
        """Test the time step increment of a ToySquares object

        """
        toy_squares = self.test_constructor()
        toy_squares.increment_time_step()