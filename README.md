# Gesture classification using biosignals. 

[![githubio](https://img.shields.io/badge/EMG-EEG%20Gesture%20Classification)](https://github.com/HumanMachineInterface/Gest-Infer/) 
This repository includes components used in the study titled "Event-Driven Edge Deep Learning Decoder for Real-time Gesture Classification and Neuro-inspired Rehabilitation Device Control". Link to the article will be provided soon [`Dere et al. (2023)`]()

## Table of Contents

- [Backgorund](#background)
- [Project Directory](#project-directory)
- [Installation](#installation)
- [Usage](#usage)
- [Credits](#credits)

## Background

Gesture classification plays a crucial role in the development of neuro-inspired rehabilitation devices. The use of electromyography (EMG) and electroencephalography (EEG) are well-known techniques in this field. However, combining both EMG and EEG signals could provide a more reliable augmented signal, particularly during EMG electrode-shift. In addition to signal fusion, there are other challenges to consider. Cloud-based deep neural network (DNN) inference often introduces latency and raises concerns about data privacy. To address these issues, a sustainable approach is to deploy the DNN model onto embedded devices. In this repository, we aim to demonstrate how to deploy a DNN model onto a field-programmable gate array (FPGA) for real-time gesture classification. 

## Project Directory
- [Dataset](https://ieee-dataport.org/documents/emg-eeg-dataset-upper-limb-gesture-classification)
  - This contain raw EMG and EEG data acquired from 33 subjects [`Dere et al. (2023)`]().
- [Data acquistion software](https://github.com/HumanMachineInterface/Gest-Infer/tree/main/data-acquistion-software)
  - This repository contains custom software that has been developed to collect EMG data from the Myo armband and EEG data from the OpenBCI Ultracortex "Mark IV".
- [Notebooks](https://github.com/HumanMachineInterface/Gest-Infer/tree/main/Notebooks)
  - This repository includes a Jupyter notebook that can be used to obtain offline gesture classification results.
- [src](https://github.com/HumanMachineInterface/Gest-Infer/tree/main/src)
  - Contains the scripts for preprocessing including `data agumentation` and `filtering.`
- [Pretrained models](https://github.com/HumanMachineInterface/Gest-Infer/tree/main/Pretrained_Models)
  - This repository includes models that can be fine-tunned for gesture classification or other applications. 
## Installation

- It is recommended to have the [Vitis-AI](https://xilinx.github.io/Vitis-AI/3.5/html/index.html) toolkit installed in order to facilitate training, quantization, and compilation of the deep learning model for deployment.

- It is recommended to install all dependencies in a virtual environment prior to data acquisition.

- Additionally, it is recommended to install PyCharm to facilitate the running of the data acquisition software for ease of use. Kindly ensure that the data acquisition code is executed within the virtual environment.

```bash
pip install -r requirements.txt
```

## Usage

- After installing all prerequisites, the user can acquire raw data using the custom software provided. However, for ease of use, pre-trained models and raw EMG and EEG datasets are also provided.

- The dataset should be placed in a folder named `data`, and the scripts from `src` as well as the required notebooks from `Notebooks` should be placed in a folder named `code`. The `data` and `code` should be in the same directory.

## Credits

We like to thank the team at [OpenBCI](https://openbci.com/), [BrainFlow](https://brainflow.org/), [AMD Xilinx](https://www.amd.com/en.html) and creators of [Pyomyo](https://github.com/PerlinWarp/pyomyo) 
 
    
<h3> The paper has been submitted for possible publication with IEEE Transcations on Instrumentation and Measurement (TIM). <h3> 
