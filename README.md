# Image classification using PyTorch Lightning and EMNIST dataset.
A general introduction to Lightning (PyTorch) framework using the extended version of the well-knwon MNIST dataset. The project pursuits the objective of giving a brief and simple overview of the following Lightning functionalities:

* **Model implementation**.
* **Dataset and Dataloader configuration**.
* **Metrics definition and logging using Tensorboard**.
* **Training pipeline configuration**:
    + Training, validation and testing loops.
    + Lightning Trainer definition and configuration.
    + Callbacks (Early stopping and custom).
    + Profiling. 

## Dependencies
This project has been developed using the following packages:
```
torch==2.0.1
torchmetrics==1.0.1
torchvision==0.15.2
lightning==2.0.5
numpy==1.25.1
matplotlib==3.7.2
```
It is necessary to have Tensorboard installed to execute the very exact code (the logger can be changed to other, e.g.: CSVLogger).

## Structure
The project is divided into 6 Python files:

* `visualize_emnist.py`: downloads the EMNIST dataset in the current working directory and runs an interactive visualization of example images through the CL.
* `main.py`: all global paramters are defined in this file. It runs automatically the training, validation and testing.
* `model.py`: includes the class which defines the PyTorch model using Lightning.
* `dataset.py`: includes the class which defines the PyTorch Dataset using Lightning.
* `trainer.py`: includes a function that returns the Trainer Lightning instance. This file might seem unnecessary, because it probably is. The purpose is to have the code as much modular as possible.
* `callbacks.py`: includes the definition of a customized callback function using Lightning corresponding module.