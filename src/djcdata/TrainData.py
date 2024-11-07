'''
Created on 20 Feb 2017

@author: jkiesele

New (post equals 2.1) version
'''

import os
import numpy as np
import logging

from .compiled import trainData
from .SimpleArray import SimpleArray
import time

def fileTimeOut(fileName, timeOut):
    '''
    simple wait function in case the file system has a glitch.
    waits until the dir, the file should be stored in/read from, is accessible
    again, or the the timeout
    '''
    filepath=os.path.dirname(fileName)
    if len(filepath) < 1:
        filepath = '.'
    if os.path.isdir(filepath):
        return

    counter=0
    print('file I/O problems... waiting for filesystem to become available for '+fileName)
    while not os.path.isdir(filepath):
        if counter > timeOut:
            raise Exception('...file could not be opened within '+str(timeOut)+ ' seconds')
        counter+=1
        time.sleep(1)

#inherit from cpp class, just slim wrapper

class TrainData(trainData):
    '''
    Base class for batch-wise training of the DNN
    '''
    def __init__(self):
        trainData.__init__(self)
        
    
    def getInputShapes(self):
        print('TrainData:getInputShapes: Deprecated, use getNumpyFeatureShapes instead')
        return self.getNumpyFeatureShapes()
        
    
    def readIn(self,fileprefix,shapesOnly=False):
        print('TrainData:readIn deprecated, use readFromFile')
        self.readFromFile(fileprefix,shapesOnly)
    
    
    def _convertToCppType(self,a,helptext):
        saout=None
        if str(type(a)) == "<class 'djcdata.SimpleArray.SimpleArray'>":
            saout = a.sa
        elif str(type(a)) == "<type 'numpy.ndarray'>" or str(type(a)) == "<class 'numpy.ndarray'>":
            rs = np.array([])
            a = SimpleArray(a,rs)
            saout = a.sa
        else:
            raise ValueError("TrainData._convertToCppType MUST produce either a list of numpy arrays or a list of DeepJetCore simpleArrays!")
        
        if saout.hasNanOrInf():
            raise ValueError("TrainData._convertToCppType: the "+helptext+" array "+saout.name()+" has NaN or inf entries")
        return saout
            
    def _store(self, x, y, w):
        for xa in x:
            self.storeFeatureArray(self._convertToCppType(xa, "feature"))
        x = [] #collect garbage
        for ya in y:
            self.storeTruthArray(self._convertToCppType(ya, "truth"))
        y = []
        for wa in w:
            self.storeWeightArray(self._convertToCppType(wa, "weight"))
        w = []    
        
    def readFromSourceFile(self,filename, weighterobjects={}, istraining=False, **kwargs):
        x,y,w = self.convertFromSourceFile(filename, weighterobjects, istraining,  **kwargs)
        self._store(x,y,w)
        

    ################# functions to be defined by the user    
        
    def createWeighterObjects(self, allsourcefiles):
        '''
        Will be called on the full list of source files once.
        Can be used to create weighter objects or similar that can
        then be applied to each individual conversion.
        Should return a dictionary
        '''
        return {}
    
    ### perform a simple and quick check if the file is not corrupt. Can be called in advance to conversion
    # return False if file is corrupt
    def fileIsValid(self, filename):
        return True
    
    ### either of the following need to be defined
    
    ## if direct writeout is useful
    def writeFromSourceFile(self, filename, weighterobjects, istraining, outname):
        self.readFromSourceFile(filename, weighterobjects, istraining)
        self.writeToFile(outname)
    
    ## otherwise only define the conversion rule
    # returns a list of numpy arrays OR simpleArray (mandatory for ragged tensors)
    def convertFromSourceFile(self, filename, weighterobjects, istraining):
        return [],[],[]
    
    ## defines how to write out the prediction
    # must not use any of the stored arrays, only the inputs
    # optionally it can return the output file name to be added to a list of output files
    def writeOutPrediction(self, predicted, features, truth, weights, outfilename, inputfile):
        return None



class TrainData_mock(TrainData):
    def __init__(self,nsamples=[200,500], n_features=10, n_truth=1):
        super(TrainData_mock,self).__init__()
        self.nsamples = nsamples
        self.n_features = n_features
        self.n_truth = n_truth
    
    def convertFromSourceFile(self, filename, weighterobjects, istraining):
        
        import hashlib     
        print('creating mock...',filename) 
        
        seed = int(hashlib.sha1(filename.encode('utf-8')).hexdigest(), 16) % (10 ** 8)
        np.random.seed(seed)
        nsamples = np.random.randint(self.nsamples[0],self.nsamples[1],size=1)
        data = []
        truth = []
        rs = [0]
        for i in range(nsamples[0]):
            n = np.random.randint(20,100)
            data.append(np.random.normal(size=(n,self.n_features)))
            truth.append(np.random.normal(size=(n,self.n_truth)))
            rs.append(n)

        # make them numpy arrays
        data = np.concatenate(data, axis=0)
        #make dtype float32
        data = np.array(data,dtype='float32')
        truth = np.concatenate(truth, axis=0)
        truth = np.array(truth,dtype='float32')

        rs = np.cumsum(np.array(rs))

        farr = SimpleArray(data, rs,name="features_ragged")
        true_arr = SimpleArray(truth, rs,name="truth_ragged")
        farrint = SimpleArray(np.array(data,dtype='int32'), rs, name="features_int_ragged")
        
        return [farr,farrint],[true_arr],[]
