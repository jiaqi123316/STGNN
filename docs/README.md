# STGNN 

## Introduction
This repository contains implementation of Spatio-Temporal Graph Neural Network, LSTM and seq2seq model for dealing with the nutrition prediction.
We used 24 location and collected the nitrate+nitrite value from 2021 to 2022. Our objective is to predict the last 73 days nutrition variance using different model and compare the model performance using RSME and MAE. 
## Requirements

- Python 3
- Tensorflow 2.*
- Tensorflow GPU (recommended)
- Pytorch

## Main components
There are some components/modules in our code. Please check our documentation in each file for more details.
### Datasets
This module handles:
- Load dataset from CSV file.
- Difference data.
- Split data according to date.
- Rescale data.
- Turn time series data into supervised sequence data for machine learning models.

### Models
This module contains different model classes.
- LSTM.
- Seq2Seq.
- Spatio-Temporal Graph neural network (STGNN).

### Trainer
Trainer contains every information (such as dataset, optimizer, loss function, etc) for training each type of models mentioned above.

### Callbacks
This module consists of callbacks which can be executed before/after some steps of training or testing model.

### Utils
Utility functions that can be used any where in the code.

## Experiments
All running files are stored in `tests` folder. Configuration of experiments can be found in `tests/configs`. Make `results` folder to store output of experiments.
```bash
mkdir tests/results
```
The script to run experiments is
```bash
python tests/test_{model_name}.py
```
The process in a test file is as follow, check each file for more details.
- Import necessary libraries.
- Setup configs and parameters.
- Load and process data.
- Define functions to handle post-prediction to make predictions back to original scale.
- Create model, select loss function and optimizer.
- Create a trainer containing necessary information.
- Train the model via the trainer.
- Make prediction and compute metrics.

After finishing, predictions and metric scores will be stored as CSV file in `tests/results` folder.

## References
1. STGCN in traffic: https://github.com/FelixOpolka/STGCN-PyTorch.
2. Seq2Seq in Neural machine translation: https://www.tensorflow.org/tutorials/text/nmt_with_attention.
