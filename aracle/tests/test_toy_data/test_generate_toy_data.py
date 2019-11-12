import os, sys
import numpy as np
import shutil
import unittest
from aracle.toy_data import ToySquares

class TestGenerateToyData(unittest.TestCase):
    """A suite of tests testing the ToySquares class
    
    """
    def test_command_line_generate_toy_data(self):
        """Test the execution of the generate script from the command line

        """
        passed = False
        dest_dir = os.path.abspath('./test_toy_generate')
        try:
            subprocess.check_output('generate_toy_data 2 5 {:s}'.format(dest_dir), shell=True)
        except:
            passed = False
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        
        self.assertTrue(passed)