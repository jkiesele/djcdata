import numpy as np
from djcdata import TrainData, DataCollection
import shutil
import unittest
import os
import sys

sys.path.append(os.path.dirname(__file__))
from TestTrainDataGenerator import TempFileList, TempDirName


class TrainData_weightertest(TrainData):
    '''
    The weighter object stores the number of source files it was created from.
    Every converted feature entry is filled with that number, so the output
    files show which weighter objects were used during the conversion.
    '''
    def __init__(self):
        super(TrainData_weightertest,self).__init__()

    def createWeighterObjects(self, allsourcefiles):
        return {'n_sources': len(allsourcefiles)}

    def convertFromSourceFile(self, filename, weighterobjects, istraining):
        feat = np.full((10,1), weighterobjects['n_sources'], dtype='float32')
        truth = np.zeros((10,1), dtype='float32')
        return [feat],[truth],[]


class TestDataCollection(unittest.TestCase):

    def writeFileList(self, files, path):
        with open(path,'w') as f:
            for fn in files.filenames:
                f.write(fn+'\n')

    def convert(self, files, outdir, takeweightersfrom='', istestdata=False):
        listdir = TempDirName()
        os.mkdir(listdir.path)
        filelist = listdir.path+'/filelist.txt'
        self.writeFileList(files, filelist)

        dc = DataCollection()
        dc.no_copy_on_convert=True #no shm write
        dc.istestdata=istestdata
        dc.convertListOfSourceFiles(filelist, TrainData_weightertest, outdir,
                                    takeweightersfrom=takeweightersfrom)
        shutil.rmtree(listdir.path)
        return outdir+'/dataCollection.djcdc'

    def checkConverted(self, dcfile, n_sources_expected):
        dc = DataCollection(dcfile)
        self.assertEqual({'n_sources': n_sources_expected}, dc.weighterobjects)
        self.assertTrue(len(dc.samples) > 0)
        for s in dc.samples:
            td = TrainData()
            td.readFromFile(dc.getSamplePath(s))
            f = td.transferFeatureListToNumpy(False)
            self.assertTrue(np.all(f[0] == n_sources_expected), s)

    def test_useWeightersFrom(self):
        print('TestDataCollection: use weighters from')

        trainfiles = TempFileList(3)
        testfiles = TempFileList(2)
        trainoutdir = TempDirName()
        testoutdir = TempDirName()

        traindc = self.convert(trainfiles, trainoutdir.path)
        self.checkConverted(traindc, 3)

        #weighters must be taken from the training conversion, not recreated from the 2 test files
        testdc = self.convert(testfiles, testoutdir.path,
                              takeweightersfrom=traindc, istestdata=True)
        self.checkConverted(testdc, 3)

        shutil.rmtree(trainoutdir.path)
        shutil.rmtree(testoutdir.path)

    def test_newWeighters(self):
        print('TestDataCollection: new weighters')

        files = TempFileList(2)
        outdir = TempDirName()

        dcfile = self.convert(files, outdir.path)
        self.checkConverted(dcfile, 2)

        shutil.rmtree(outdir.path)


if __name__ == '__main__':
    unittest.main()
