import os
import sys
sys.path.append(os.path.dirname(__file__))

from TestSimpleArray import TestSimpleArray
from TestTrainData import TestTrainData
from TestCompatibility import TestCompatibility
from TestTrainDataGenerator import TestTrainDataGenerator
from TestDJCDataLoader import TestDJCDataLoader
# from TestCFunctions import TestCFunctions

from multiprocessing import freeze_support
import unittest


if __name__ == '__main__':
    freeze_support()
    unittest.main()
