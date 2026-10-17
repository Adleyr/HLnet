# HLnet: cross-modal transformer-CNN hybrid architecture for medical image synthesis via hierarchical frequency integration

Official PyTorch implementation of the paper:  
**"HLnet: cross-modal transformer-CNN hybrid architecture for medical image synthesis via hierarchical frequency integration"**  
*Accepted at Cluster Computing *

[![Paper]([https://img.shields.io/badge/Paper-ISBI2026-blue](https://link.springer.com/article/10.1007/s10586-026-06649-x?utm_source=rct_congratemailt&utm_medium=email&utm_campaign=nonoa_20261016&utm_content=10.1007%2Fs10586-026-06649-x))](https://github.com/)

----
1. [Requirements](#Requirements)
2. [Code Structure](#Structure)
3. [Demo](#Demo)
4. [TODO](#TODO)




## Requirements
----
1. pytorch=1.12.1 versions had been tested 
2. python3.8.17 was tested
3. python dependencies, please see the ```requirements.txt``` file
4. CUDA11.4 had been tested
5. Download preprocessed dataset and our model files: <a href="https://drive.google.com/file/d/11HnWcx6612sTk4CjUbc9HdN4OPVcw7-T/view?usp=sharing">data</a>
6. Download the source code and put the data file to the right location according to the code structure below

## Structure
----
```
HLnet/:
   |--Model/:folder contains the code for our model
   |
   |
   |--Yaml/: experiment configuration folder
   |    |--HLnet.yaml/: HLnet experiment configuration file
   |
   |
   |--data1/:folder contains preprocessed data from our experiments(*.npy)
   |    |--T1T2:
   |        |--test2D/: folder contains test data
   |            |--A: T1 files(.npy)
   |            |--B: T2 files(.npy)
   |        |--train2D/: folder contains training data
   |            |--A:
   |            |--B:
   |        |--val2D/: folder contains validation data
   |            |--A:
   |            |--B:
   |
   |--output/: folder contains models saved during training
   |    |--hlnet/:
   |
   |
   |--trainer/: folder contains code related to model training
   |
   |
   |--test.py: test script that demonstrates the inference workflow and outputs the metric results
   |--train.py: training script that trains models
   |--requirements.txt: python dependency libraries
```

## Demo
----
### Test our Models

Please use the following example settings to test our model:

   ```python test.py```

### Train from Scratch

Please use the "Yaml/HLnet.yaml" configuration file to train your model:

   ```nohup python  -u train.py > train.log 2>&1 &```


## TODO
----
- [x] Traing code example and script of the algorithm  
- [x] Testing code example and script of the algorithm
- [x] Visualization code example and script of the algorithm

