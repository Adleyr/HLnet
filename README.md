# HLnet: cross-modal transformer-CNN hybrid architecture for medical image synthesis via hierarchical frequency integration

Official PyTorch implementation of the paper:  
**"HLnet: cross-modal transformer-CNN hybrid architecture for medical image synthesis via hierarchical frequency integration"**  
*Accepted at Cluster Computing *

[Paper](https://link.springer.com/article/10.1007/s10586-026-06649-x?utm_source=rct_congratemailt&utm_medium=email&utm_campaign=nonoa_20261016&utm_content=10.1007%2Fs10586-026-06649-x)

## 📌 Abstract
Magnetic Resonance Imaging is indispensable in clinical practice, but the synthesis of T1- and T2-weighted images remains a complex
problem. Existing methods often fail to fully exploit inter-task relationships, thereby limiting information sharing and integration. To
address this limitation, a novel framework integrating convolutional neural networks (CNNs) with Transformer architectures is
proposed. First, low- and high-frequency components are extracted by a CNN to represent coarse and fine structural details.
Subsequently, a Transformer module models global contextual dependencies, enhancing the generation of realistic image
transformations. A key component is the Task Transformer module, designed to promote efficient information sharing and integration
across tasks. Results from experiments confirm that the method exceeds the performance of existing approaches in both structural and
textural detail reconstruction, while sustaining superior image quality. This work provides a robust solution for enhancing MRI image
transformation, with significant implications for clinical diagnostics and research.
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

