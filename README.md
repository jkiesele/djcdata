# djcdata

**djcdata** is a Python package designed for efficient data handling and preprocessing for deep learning models. It provides tools to convert raw data into a format suitable for training, manage datasets, and feed data into training loops seamlessly, supporting both TensorFlow and PyTorch frameworks.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Pipeline Overview](#pipeline-overview)
- [Quick Start Guide](#quick-start-guide)
  - [Defining a Custom `TrainData` Class](#defining-a-custom-traindata-class)
  - [Running Data Conversion](#running-data-conversion)
  - [Performing Training](#performing-training)
    - [Using TensorFlow](#using-tensorflow)
    - [Using PyTorch](#using-pytorch)
- [Helper Scripts](#helper-scripts)
- [Documentation](#documentation)
- [License](#license)

---

## Features

- **Efficient Data Conversion**: Convert raw data into a format optimized for training with deep learning frameworks.
- **Flexible Data Management**: Manage datasets through the `DataCollection` class, allowing splitting, shuffling, and batching.
- **Support for Variable-Length Data**: Handle datasets with variable-length sequences efficiently.
- **Integration with TensorFlow and PyTorch**: Seamlessly feed data into training loops for both frameworks.
- **Multiprocessing Support**: Utilize multiprocessing for faster data conversion and loading.

## Installation

The package can be installed via pip for linux distributions from python 3.9 - 3.11:
```
pip install djcdata
```

For other distributions, you can install the latest version of `djcdata` from GitHub; run:

```bash
pip install git+https://github.com/jkiesele/djcdata
```

**Note**: The github installation requires `cmake` version 3 or higher. On some systems (e.g., `lxplus7`), you might need to add `cmake` to your `PATH`:

```bash
export PATH=/cvmfs/sft.cern.ch/lcg/contrib/CMake/latest/Linux-x86_64/bin:$PATH
```

This adjustment is only needed during the installation process.

## Pipeline Overview

The data processing pipeline in `djcdata` involves the following steps:

1. **Define Conversion Logic**: Create a custom `TrainData` class to define how raw data is converted.
2. **Run Data Conversion**: Use the provided scripts to convert raw data into the `djcdata` format.
3. **Manage Datasets**: Utilize `DataCollection` to manage and manipulate your dataset.
4. **Perform Training**: Use the data loaders to feed data into your training loop with TensorFlow or PyTorch.

![Pipeline Illustration](https://github.com/jkiesele/djcdata/blob/master/pipeline.png)

## Quick Start Guide

### Defining a Custom `TrainData` Class

To convert your raw data, you need to define a custom class that inherits from `TrainData` and implements the `convertFromSourceFile` method. This method specifies how to read and process your raw data files.

```python
from djcdata import TrainData
from djcdata import SimpleArray  # For handling variable-length data

class YourTrainDataClass(TrainData):
    def __init__(self):
        super(YourTrainDataClass, self).__init__()
        # Initialize any variables or parameters here

    def convertFromSourceFile(self, filename, weighterobjects, istraining):
        # Read your raw data from 'filename'
        import numpy as np

        # Example: Load data from a NumPy file
        data = np.load(filename)

        # Process and prepare your data
        # Split data into features, truth labels, and weights if applicable
        features = data['features']
        truths = data['truths']
        weights = data.get('weights', None)  # Optional

        # If using variable-length data, wrap arrays in SimpleArray
        features_array1 = SimpleArray(features, name="features1")
        features_array2 = SimpleArray(features, name="features2")
        truths_array = SimpleArray(truths, name="truths")

        # Return a tuple of (feature_arrays, truth_arrays, weight_arrays)
        return [features_array, features_array2], [truths_array], [weights] if weights is not None else []
```

- **Features**: Input data for your model.
- **Truths**: Ground truth labels or targets.
- **Weights**: Sample weights (optional).
- **Note**: For variable-length data (e.g., sequences of different lengths), use `SimpleArray` and also pass row splits to handle ragged tensors efficiently.

### Running Data Conversion

Once you've defined your custom `TrainData` class, you can convert your raw data using the `convertDJCFromSource.py` script provided by `djcdata`. For this to work, `YourTrainDataClass` must to be part of a module calles `datastructures`.

**Command Syntax**:

```bash
convertDJCFromSource.py -i input_file_list.txt -o output_directory -c YourTrainDataClass
```

- `-i`: Path to a text file containing a list of your raw data files.
- `-o`: Output directory where the converted data will be stored.
- `-c`: The name of your custom `TrainData` class that handles data conversion.

**Example**:

```bash
convertDJCFromSource.py -i data/input_files.txt -o data/converted -c YourTrainDataClass
```

**Additional Options**:

- `--gpu`: Enable GPU usage for conversion (useful if conversion involves GPU operations).
- `--nothreads`: Use only a single process for conversion.
- `--checkFiles`: Enable file checking (requires `fileIsValid` method in `TrainData` to be defined).
- `--testdata`: Convert as test data (does not create weighter objects).
- `--help`: Display detailed help message with all available options.

### Performing Training

After converting your data, you can use `DataCollection` to manage your dataset and feed data into your training loop.

```python
from djcdata import DataCollection

# Load the data collection
train_data = DataCollection("data/converted/dataCollection.djcdc")

# Optionally split the data for validation
val_data = train_data.split(0.8)  # Use 80% for training and 20% of the data for validation
```

#### Using TensorFlow

For TensorFlow models, you can use the `TrainDataGenerator` to feed data into your training loop.

```python
from djcdata import TrainDataGenerator

# Create generators
traingen = train_data.invokeGenerator()
valgen = val_data.invokeGenerator()

# Set batch size
batch_size = 32
traingen.setBatchSize(batch_size)
valgen.setBatchSize(batch_size)

# Training loop
model.fit(
    traingen.feedNumpyData(),
    steps_per_epoch=traingen.getNBatches(),
    validation_data=valgen.feedNumpyData(),
    validation_steps=valgen.getNBatches(),
    epochs=num_epochs
)
```

#### Using PyTorch

For PyTorch models, use the `DJCDataLoader` class to create data loaders compatible with PyTorch's training loop.

```python
from djcdata import DJCDataLoader
import torch

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize data loaders
train_loader = DJCDataLoader(
    data_path="data/converted/dataCollection.djcdc",
    batch_size=32,
    shuffle=True,
    device=device,
    dict_output=True  # Set to True if the output of the data loader should be dictionaries (convenience function)
)

# Split off validation data, here use 80% for training and 20% for validation. 
# Please note that the syntax is inverted w.r.t. the raw tensorflow interface in the previous example.
val_loader = train_loader.split(split_fraction=0.2)

# Training loop
for epoch in range(num_epochs):
    for batch_data in train_loader:
        # Unpack data
        x_batch, y_batch = batch_data[:2]
        # x_batch and y_batch are already on the device

        features1 = x_batch['features1']
        features2 = x_batch['features2']
        # The dictionary keys are the names defined for the SimpleArrays in the converFromSource function

        # Forward pass, loss computation, etc.
        # ...

    # Validation loop
    with torch.no_grad():
        for batch_data in val_loader:
            x_batch, y_batch = batch_data[:2]
            # Validation logic
            # ...
```

**Key Points**:

- **Automatic Device Transfer**: The `DJCDataLoader` automatically moves data to the specified device.
- **Data Splitting**: Use the `split` method to create validation data loaders.
- **Custom Data Structures**: For convenience, the output can also be given as a dictionary, for that to take effect, set `dict_output=True`. The dictionary keys will be the names of the SimpleArrays defined in the convertFromSource function.

## Helper Scripts

`djcdata` provides several utility scripts to facilitate data conversion and management. All scripts support the `--help` option for detailed usage information.

- **convertDJCFromSource.py**: Converts raw data files into the `djcdata` format using your custom `TrainData` class.
- **createDataCollectionFromTD.py**: Creates a `DataCollection` wrapper from existing converted individual `TrainData` (`.djctd`) files.
- **mergeOrSplitDJCFiles.py**: Merges or splits individual a whole `DataCollection` into more fine or more coarsely grained individual `TrainData` files.
- **validateDJCDataCollection.py**: Validates the integrity of a `DataCollection`.
- **validateDJCFiles.py**: Validates a list (text file) of input files to `convertFromSource.py` (requires `fileIsValid` method in `TrainData` to be defined).

**Usage**:

For detailed usage of each script, run:

```bash
script_name.py --help
```

**Example**:

```
validateDJCDataCollection.py --help
```

## Documentation

For more detailed documentation and advanced usage, please refer to the [DeepJetCore documentation](https://github.com/DL4Jets/DeepJetCore). `djcdata` is based on `DeepJetCore` and shares many of its concepts and functionalities.

### Key Classes and Methods

- **TrainData**: Base class for defining how raw data is converted into the format used for training.
  - **convertFromSourceFile**: Method to be implemented for custom data conversion logic.
  - **createWeighterObjects**: (Optional) Create weighting objects for balancing datasets.
- **DataCollection**: Manages a collection of converted data samples.
  - **createDataFromSource**: Converts and collects data from source files.
  - **split**: Splits the data collection into training and validation sets.
  - **invokeGenerator**: Creates a data generator for feeding data into the training loop.
- **TrainDataGenerator**: Feeds data into the training loop for TensorFlow models.
- **DJCDataLoader**: Custom data loader compatible with PyTorch's `DataLoader` interface.
- **SimpleArray**: Handles variable-length data (ragged tensors) efficiently.

### Handling Variable-Length Data

For datasets with variable-length sequences, wrap your data arrays in `SimpleArray` when returning them from `convertFromSourceFile`. This allows `djcdata` to manage ragged tensors without unnecessary padding.

### Multiprocessing and Performance

`djcdata` utilizes multiprocessing to speed up data conversion and loading. The data generators and loaders handle shuffling, batching, and device transfers to optimize training performance.


