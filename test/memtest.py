'''
Not part of the usual unit tests, but to check if there is a memory leak in the data pipeline,
in particular when feeding to the training.
'''

from TestTrainDataGenerator import TrainData_test, TempFileList, TempDirName
from djcdata import DataCollection
import shutil

from memory_profiler import profile

class TrainData_mem_test(TrainData_test):
    def __init__(self):
        super(TrainData_mem_test,self).__init__([4000,10000])#large number of samples per file
        
#not within unit tests
def mem_test():
    n_files=11
    n_per_batch=2078
    files = TempFileList(n_files)
    dcoutdir = TempDirName()
    
    n_per_batch=n_per_batch
    
    dc = DataCollection()
    dc.dataclass = TrainData_test
    dc.sourceList = [f for f in files.filenames]
    dc.no_copy_on_convert=True #no shm write
    dc.createDataFromRoot(TrainData_test, outputDir=dcoutdir.path)
    
    gen = dc.invokeGenerator()
    gen.setBatchSize(n_per_batch)
    
    for epoch in range(10000):
        gen.prepareNextEpoch()
        for b in range(gen.getNBatches()):
            d,t = next(gen.feedNumpyData())
            
    shutil.rmtree(dcoutdir.path)

if __name__ == '__main__':
    mem_test()