'''
Checks for file compatibility with (only) the previous version.
'''

from djcdata import TrainData, SimpleArray
import numpy as np
import unittest
import os


class TestCompatibility(unittest.TestCase):

    def test_SimpleArrayRead(self):
        print('TestCompatibility SimpleArray')
        a = SimpleArray()
        basedir = os.path.dirname(__file__)
        a.readFromFile(os.path.join(basedir, "simpleArray_previous.djcsa"))

        arr = np.load(os.path.join(basedir, "np_arr.npy"))
        # FIXME: this array was actually wrong
        arr = arr[:100]
        rs = np.load(os.path.join(basedir, "np_rs.npy"))

        b = SimpleArray(arr, rs)

        self.assertEqual(a, b)

    def test_TrainDataRead(self):
        print('TestCompatibility TrainData')
        td = TrainData()
        basedir = os.path.dirname(__file__)
        td.readFromFile(os.path.join(basedir, 'trainData_previous.djctd'))

        self.assertEqual(td.nFeatureArrays(), 1)

        arr = np.load(os.path.join(basedir, "np_arr.npy"))
        # FIXME: this array was actually wrong
        arr = arr[:100]
        rs = np.load(os.path.join(basedir, "np_rs.npy"))

        b = SimpleArray(arr, rs)

        a = td.transferFeatureListToNumpy(False)
        a, rs = a[0], a[1]

        a = SimpleArray(a, np.array(rs, dtype='int64'))

        self.assertEqual(a, b)

        