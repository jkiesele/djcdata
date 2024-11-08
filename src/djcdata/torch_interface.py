#try to import torch, if not, complain that this interface requires torch
try:
    import torch
except ImportError:
    raise ImportError("torch not found. Please install pytorch first for this interface.")

from . import DataCollection  # Adjust the import as needed

# just for the mock class
import tempfile
import shutil

class DJCDataLoader:
    def __init__(self, 
                 data_path, 
                 batch_size=32, 
                 shuffle=True, 
                 device=None, 
                 dict_output=False,
                 **kwargs):
        """
        DJCDataLoader that uses TrainDataGenerator under the hood.

        Args:
            data_path (str): Path to the dataset descriptor file (.djcdc).
            batch_size (int): Batch size for training.
            shuffle (bool): Whether to shuffle the data.
            device (torch.device or str): Device to move data to ('cpu' or 'cuda').
            **kwargs: Additional arguments (not supported).
        """
        

        # Load the DataCollection
        self.data_collection = DataCollection(data_path)

        # Create the generator
        self.generator = self.data_collection.invokeGenerator(dict_output=dict_output)

        # Set batch size
        self.generator.setBatchSize(batch_size)

        # Set shuffling
        self.shuffle = shuffle
        if self.shuffle:
            self.generator.shuffleFileList()

        # Set device
        self.device = device

        # Handle other kwargs or raise exceptions if unsupported
        for key in kwargs:
            raise NotImplementedError(f"Argument '{key}' is not supported by DJCDataLoader.")

        # Prepare the generator for the first epoch
        self.generator.prepareNextEpoch()
        self.iterator = iter(self.generator.feedNumpyData())

    def __iter__(self):
        # Reinitialize the iterator for a new epoch
        # Optionally reshuffle the file list
        if self.shuffle:
            self.generator.shuffleFileList()
        self.generator.prepareNextEpoch()
        self.iterator = iter(self.generator.feedNumpyData())

        return self

    def __next__(self):
        data = next(self.iterator)
        # Convert numpy arrays to torch tensors and move to device
        return self._convert_to_tensors(data)

    def _convert_to_tensors(self, data):
        # data can be (x, y) or (x, y, w)
        # Convert numpy arrays to torch tensors and move to device
        converted_data = []
        for item in data:
            if isinstance(item, dict):
                # If data is a dict of arrays
                converted_item = {k: torch.from_numpy(v) for k, v in item.items()}
                if self.device is not None:
                    converted_item = {k: v.to(self.device) for k, v in converted_item.items()}
                converted_data.append(converted_item)
            elif isinstance(item, list):
                # If data is a list of arrays
                converted_item = [torch.from_numpy(arr) for arr in item]
                if self.device is not None:
                    converted_item = [arr.to(self.device) for arr in converted_item]
                converted_data.append(converted_item)
            else:
                converted_item = torch.from_numpy(item)
                if self.device is not None:
                    converted_item = converted_item.to(self.device)
                converted_data.append(converted_item)
        return tuple(converted_data)

    def __len__(self):
        return self.generator.getNBatches()

    def setBatchSize(self, batch_size):
        self.generator.setBatchSize(batch_size)

    def split(self, split_fraction):
        """
        Splits the data loader into training and validation loaders.

        Args:
            split_fraction (float): Fraction of data to be used for validation.

        Returns:
            DJCDataLoader: A new data loader for validation data.
        """
        # Split the data collection
        val_data_collection = self.data_collection.split(1. - split_fraction)

        # Update the generator for training data
        batchsize = self.generator.getBatchSize()
        self.generator = self.data_collection.invokeGenerator(dict_output=self.generator.dict_output)
        self.generator.setBatchSize(batchsize)
        self.generator.prepareNextEpoch()
        self.iterator = iter(self.generator.feedNumpyData())

        # Create a new DJCDataLoader for the validation data
        val_loader = DJCDataLoader.__new__(DJCDataLoader)
        val_loader.data_collection = val_data_collection
        val_loader.generator = val_loader.data_collection.invokeGenerator(dict_output=self.generator.dict_output)
        val_loader.generator.setBatchSize(batchsize)
        val_loader.shuffle = False  # Typically, validation data is not shuffled
        val_loader.device = self.device

        # Prepare the generator for the first epoch
        val_loader.generator.prepareNextEpoch()
        val_loader.iterator = iter(val_loader.generator.feedNumpyData())

        return val_loader




# a mock trainData class for testing
class MockDJCDataLoader(DJCDataLoader):
    """
    A mock DJCDataLoader class that can be used for testing.
    It does not require a data_path and does not load any data.
    """
    def __init__(self, 
                 batch_size=32, 
                 shuffle=True, 
                 device=None, 
                 dict_output=False,
                 **kwargs):
        
        from djcdata.TrainData import TrainData_mock
        
        mock_filenames = [f"mock_file_{i}.input" for i in range(10)]

        # create a temporary output directory
        self.temp_dir = tempfile.mkdtemp()
        shutil.rmtree(self.temp_dir) # will be recreated by conversion

        dc = DataCollection()
        dc.dataclass = TrainData_mock
        dc.sourceList = mock_filenames
        dc.no_copy_on_convert=True #no shm write
        
        dc.createDataFromSource(TrainData_mock, outputDir=self.temp_dir)
        
        dc_path = f"{self.temp_dir}/snapshot.djcdc"

        super().__init__(dc_path, batch_size, shuffle, device, dict_output, **kwargs)

    def __del__(self): #clean up
        shutil.rmtree(self.temp_dir)

