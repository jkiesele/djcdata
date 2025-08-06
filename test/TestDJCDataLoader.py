import unittest
import shutil
import torch
import os
import sys

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
from djcdata import DataCollection 

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

if __name__ == "__main__":
    unittest.main()
