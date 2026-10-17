# HLnet: Medical Image Synthesis by Integrating Transformer and CNN with Effective Fusion of High and Low-Frequency Information



## Contents
----
1. [Requirements](#Requirements)
2. [Code Structure](#Structure)
3. [Demo](#Demo)
4. [TODO](#TODO)
5. [Acknowledgement](#Acknowledgement)



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

   ```python train.py```


## TODO
----
- [x] Traing code example and script of the algorithm  
- [x] Testing code example and script of the algorithm
- [x] Visualization code example and script of the algorithm

