import unittest
import shutil
import torch
import os
import sys
import numpy as np

# Adjust the import path to include the directory containing TestTrainDataGenerator.py
sys.path.append(os.path.dirname(__file__))
from TestTrainDataGenerator import (
    RaggedTester,
    TempFileList,
    TempDirName,
    TrainData_test
)

# Import the DJCDataLoader class
# Adjust the import according to your package structure
from djcdata.torch_interface import DJCDataLoader 
from djcdata import DataCollection, TrainData, SimpleArray

class TestDJCDataLoader(unittest.TestCase):
    def test_djc_dataloader(self):
        print("Testing DJCDataLoader")

        passed = True

        n_files = 11
        batch_size = 2078
        files = TempFileList(n_files)
        dcoutdir = TempDirName()

        # Create a DataCollection using the existing TrainData_test class
        dc = DataCollection()
        dc.dataclass = TrainData_test
        dc.sourceList = [f for f in files.filenames]
        dc.no_copy_on_convert = True  # Avoid copying data to shared memory
        dc.createDataFromSource(TrainData_test, outputDir=dcoutdir.path)

        dc.writeToFile(dcoutdir.path + "/test.djcdc")

        ## dc can be deleted
        del dc

        # Initialize the DJCDataLoader
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data_loader = DJCDataLoader(
            data_path=dcoutdir.path  + "/test.djcdc",
            batch_size=batch_size,
            shuffle=True,
            device=device,
            dict_output=True
        )

        # Split into training and validation loaders
        val_loader = data_loader.split(split_fraction=0.2)

        # Instantiate RaggedTester for data validation
        raggedtester = RaggedTester()

        for epoch in range(2):  # Reduced number of epochs for testing
            print(f"Epoch {epoch + 1}")

            # Training data
            for b, batch_data in enumerate(data_loader):
                # Unpack data
                if len(batch_data) == 2:
                    x, y = batch_data
                    w = None
                elif len(batch_data) == 3:
                    x, y, w = batch_data
                else:
                    print(f"Unexpected batch data length: {len(batch_data)}")
                    passed = False
                    break

                #check if x is really a dictionary
                if not isinstance(x, dict):
                    print(f"Unexpected data type: {type(x)}")
                    passed = False
                    break

                data = x['features_ragged']
                rs = x['features_ragged_rowsplits']
                dint = x['features_int_ragged']
                truth = y['truth_ragged']

                # Move tensors to CPU and convert to numpy arrays for validation
                rs_cpu = rs.cpu().numpy()
                data_cpu = data.cpu().numpy()
                dint_cpu = dint.cpu().numpy()
                truth_cpu = truth.cpu().numpy()

                # Validate data using RaggedTester
                datagood = raggedtester.checkData(data_cpu, rs_cpu)
                datagood = datagood and raggedtester.checkData(dint_cpu, rs_cpu, 'int32')
                datagood = datagood and raggedtester.checkData(truth_cpu, rs_cpu)

                if not datagood:
                    print(f"Epoch {epoch}, batch {b} data check failed")
                    passed = False
                    break

                if rs_cpu[-1] > batch_size:
                    print(f"Maximum batch size exceeded in batch {b}, epoch {epoch}")
                    passed = False
                    break

            # Validation data
            for b, batch_data in enumerate(val_loader):
                # Unpack data
                if len(batch_data) == 2:
                    x, y = batch_data
                    w = None
                elif len(batch_data) == 3:
                    x, y, w = batch_data
                else:
                    print(f"Unexpected batch data length: {len(batch_data)}")
                    passed = False
                    break

                data = x['features_ragged']
                rs = x['features_ragged_rowsplits']
                dint = x['features_int_ragged']
                truth = y['truth_ragged']

                # Move tensors to CPU and convert to numpy arrays for validation
                rs_cpu = rs.cpu().numpy()
                data_cpu = data.cpu().numpy()
                dint_cpu = dint.cpu().numpy()
                truth_cpu = truth.cpu().numpy()

                # Validate data using RaggedTester
                datagood = raggedtester.checkData(data_cpu, rs_cpu)
                datagood = datagood and raggedtester.checkData(dint_cpu, rs_cpu, 'int32')
                datagood = datagood and raggedtester.checkData(truth_cpu, rs_cpu)

                if not datagood:
                    print(f"Epoch {epoch}, validation batch {b} data check failed")
                    passed = False
                    break

                if rs_cpu[-1] > batch_size:
                    print(f"Maximum batch size exceeded in validation batch {b}, epoch {epoch}")
                    passed = False
                    break

        # Clean up the temporary data directory
        shutil.rmtree(dcoutdir.path)

        self.assertTrue(passed)


class TrainData_flat(TrainData):
    '''
    non-ragged test data: each file holds a fixed number of events,
    one row per event
    '''
    def __init__(self, nsamples=[120,350]):
        super(TrainData_flat,self).__init__()
        self.nsamples = nsamples

    def convertFromSourceFile(self, filename, weighterobjects, istraining):
        import hashlib

        seed = int(hashlib.sha1(filename.encode('utf-8')).hexdigest(), 16) % (10 ** 8)
        rng = np.random.default_rng(seed) # do not touch the global random state
        nsamples = rng.integers(self.nsamples[0],self.nsamples[1])
        data = rng.normal(size=(nsamples,4)).astype('float32')
        truth = rng.normal(size=(nsamples,1)).astype('float32')

        return [SimpleArray(data,name="features")],[SimpleArray(truth,name="truth")],[]


class TestDJCDataLoaderNBatches(unittest.TestCase):
    '''
    Checks that the DJCDataLoader produces the correct number of batches
    for a given batch size. The expected number is computed independently
    from the converted files.
    '''

    n_files = 11
    batch_size_flat = 100
    batch_size_ragged = 2078

    def setUp(self):
        self.files = TempFileList(self.n_files)
        self.dcoutdir = TempDirName()

    def tearDown(self):
        # may not exist if the conversion failed, don't hide the original error
        shutil.rmtree(self.dcoutdir.path, ignore_errors=True)

    def _createDataCollection(self, dataclass):
        dc = DataCollection()
        dc.dataclass = dataclass
        dc.sourceList = [f for f in self.files.filenames]
        dc.no_copy_on_convert = True
        dc.createDataFromSource(dataclass, outputDir=self.dcoutdir.path)
        # the expected numbers of batches below assume a plain element count limit
        self.assertFalse(dc.batch_uses_sum_of_squares)
        dc.writeToFile(self.dcoutdir.path + "/test.djcdc")
        return self.dcoutdir.path + "/test.djcdc"

    def _nEvents(self, dc):
        ntotal = 0
        for s in dc.samples:
            td = dc.dataclass()
            td.readFromFile(dc.getSamplePath(s))
            ntotal += td.nElements()
        return ntotal

    def _elementsPerEvent(self, dc):
        '''
        returns the number of ragged elements per event for all files in the
        data collection, in file order (no shuffling)
        '''
        out = []
        for s in dc.samples:
            td = dc.dataclass()
            td.readFromFile(dc.getSamplePath(s))
            rs = td.transferFeatureListToNumpy(False)[1].reshape(-1)
            out.append(rs[1:] - rs[:-1])
        return np.concatenate(out)

    def _expectedNBatchesFlat(self, ntotal, batch_size):
        return int(np.ceil(ntotal / batch_size))

    def _expectedNBatchesRagged(self, nelems, batch_size):
        '''
        greedy filling: an event goes into the current batch as long as the
        total number of elements does not exceed the batch size
        '''
        assert np.all(nelems <= batch_size) # no skipped events in this test
        nbatches = 0
        current = 0
        for n in nelems:
            if current + n > batch_size:
                nbatches += 1
                current = 0
            current += n
        if current > 0:
            nbatches += 1
        return nbatches

    def _iterate(self, loader, ragged):
        '''
        runs one epoch, returns the number of events and the number of
        elements per batch
        '''
        nevents = []
        nelements = []
        for batch_data in loader:
            x = batch_data[0]
            if ragged:
                rs = x[1].cpu().numpy().reshape(-1)
                nevents.append(len(rs) - 1)
                nelements.append(int(rs[-1]))
            else:
                nevents.append(x[0].shape[0])
                nelements.append(x[0].shape[0])
        return nevents, nelements

    def _checkLoader(self, loader, expected_nbatches, expected_nevents, batch_size, ragged, nepochs=2):
        self.assertEqual(len(loader), expected_nbatches,
                         "len(loader) does not match the expected number of batches")
        for epoch in range(nepochs):
            nevents, nelements = self._iterate(loader, ragged)
            self.assertEqual(len(nevents), expected_nbatches,
                             f"epoch {epoch}: number of iterated batches does not match the expected number")
            self.assertEqual(sum(nevents), expected_nevents,
                             f"epoch {epoch}: not all events were used exactly once")
            self.assertLessEqual(max(nelements), batch_size,
                                 f"epoch {epoch}: maximum batch size exceeded")

    def test_nbatches_flat(self):
        print("TestDJCDataLoaderNBatches flat")
        path = self._createDataCollection(TrainData_flat)
        ntotal = self._nEvents(DataCollection(path))
        expected = self._expectedNBatchesFlat(ntotal, self.batch_size_flat)

        for shuffle in [False, True]:
            with self.subTest(shuffle=shuffle):
                loader = DJCDataLoader(data_path=path, batch_size=self.batch_size_flat,
                                       shuffle=shuffle, device='cpu')
                self._checkLoader(loader, expected, ntotal, self.batch_size_flat, ragged=False)

    def test_generator_state_flat(self):
        print("TestDJCDataLoaderNBatches generator state flat")
        path = self._createDataCollection(TrainData_flat)
        dc = DataCollection(path)
        ntotal = self._nEvents(dc)

        gen = dc.invokeGenerator()
        # zero is accepted as placeholder and gives no batches
        gen.setBatchSize(0)
        self.assertEqual(gen.getNBatches(), 0)

        gen.setBatchSize(self.batch_size_flat)
        self.assertEqual(gen.getNBatches(), self._expectedNBatchesFlat(ntotal, self.batch_size_flat))
        self.assertFalse(gen.lastBatch())

        # everything in one batch: the first batch is also the last one
        gen.setBatchSize(ntotal)
        self.assertEqual(gen.getNBatches(), 1)
        self.assertTrue(gen.lastBatch())

    def test_nbatches_ragged(self):
        print("TestDJCDataLoaderNBatches ragged")
        path = self._createDataCollection(TrainData_test)
        nelems = self._elementsPerEvent(DataCollection(path))
        expected = self._expectedNBatchesRagged(nelems, self.batch_size_ragged)

        # without shuffling the order of events is known and so is the exact number of batches
        loader = DJCDataLoader(data_path=path, batch_size=self.batch_size_ragged,
                               shuffle=False, device='cpu')
        self._checkLoader(loader, expected, len(nelems), self.batch_size_ragged, ragged=True)

    def test_nbatches_ragged_shuffled(self):
        print("TestDJCDataLoaderNBatches ragged shuffled")
        path = self._createDataCollection(TrainData_test)
        nelems = self._elementsPerEvent(DataCollection(path))
        batch_size = self.batch_size_ragged

        loader = DJCDataLoader(data_path=path, batch_size=batch_size,
                               shuffle=True, device='cpu')

        for epoch in range(2):
            nevents, nelements = self._iterate(loader, ragged=True)
            self.assertEqual(len(nevents), len(loader),
                             f"epoch {epoch}: number of iterated batches does not match len(loader)")
            self.assertEqual(sum(nevents), len(nelems),
                             f"epoch {epoch}: not all events were used exactly once")
            self.assertEqual(sum(nelements), int(np.sum(nelems)))
            self.assertLessEqual(max(nelements), batch_size)
            # at least as many batches as needed to hold all elements
            self.assertGreaterEqual(len(nevents), int(np.ceil(np.sum(nelems) / batch_size)))

    def test_nbatches_split(self):
        print("TestDJCDataLoaderNBatches split")
        for dataclass, batch_size, ragged in [(TrainData_flat, self.batch_size_flat, False),
                                              (TrainData_test, self.batch_size_ragged, True)]:
            with self.subTest(dataclass=dataclass.__name__):
                if os.path.exists(self.dcoutdir.path):
                    shutil.rmtree(self.dcoutdir.path)
                path = self._createDataCollection(dataclass)

                # reproduce the split on a plain DataCollection to get the expected content
                train_dc = DataCollection(path)
                val_dc = train_dc.split(1. - 0.2)

                loader = DJCDataLoader(data_path=path, batch_size=batch_size,
                                       shuffle=False, device='cpu')
                val_loader = loader.split(split_fraction=0.2)

                for l, dc in [(loader, train_dc), (val_loader, val_dc)]:
                    if ragged:
                        nelems = self._elementsPerEvent(dc)
                        expected = self._expectedNBatchesRagged(nelems, batch_size)
                        nevents = len(nelems)
                    else:
                        nevents = self._nEvents(dc)
                        expected = self._expectedNBatchesFlat(nevents, batch_size)
                    self._checkLoader(l, expected, nevents, batch_size, ragged)


if __name__ == "__main__":
    unittest.main()
