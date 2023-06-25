
pip3 install git+https://github.com/jkiesele/djcdata

Requires cmake>=3., on lxplus7, this does not exist out of the box, but can be added from here:
PATH=/cvmfs/sft.cern.ch/lcg/contrib/CMake/latest/Linux-x86_64/bin:$PATH 

This is only needed to install the package.


For more documentation, please refer to https://github.com/DL4Jets/DeepJetCore

Please notice that the scripts have an additional "DJC" in this repository w.r.t. DeepJetCore. The scripts can be found here: https://github.com/jkiesele/djcdata/tree/master/src/djcdata/bin

![pipeline](https://github.com/jkiesele/djcdata/blob/master/pipeline.png "Data pipeline for training")
