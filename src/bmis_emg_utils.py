 # -*- coding: utf-8 -*-
"""
// LAB: Biomedical Information and Signal Lab 
// Engineer: Dere Mustapha Deji
// Created on: 2022/05/11 02:07:00 PM
// Design Name: Motor Execution
//Version 2
// 
"""

import os
from biosignal_analysis_tools import * 
from scipy.io import loadmat
import numpy as np
import random
from sklearn.model_selection import train_test_split



#STEP1: The function returns the base folder path and all dataset file names
def get_per_subject_file(subject, no_gesture):
    
    """
    Take in the subject  index and the number of gestures desired gestuers
    
    Paramerer
    -------------
    
    
    Return
    -------------
    
    """
    
    base_path = os.getcwd().strip('\\code') + "/data/mat_data/subject_{}".format(subject)
    data_file = os.listdir(base_path)
    gesture = {}
    for i in range(no_gesture):  # Runs through the total number of gesture
        j = i + 1  
        gesture[str(i)] = [file for inx, file in enumerate(data_file) if data_file[inx][7] == str(j) or data_file[inx][8] == str(j)]

    return base_path, gesture


#STEP2: Concatenate all gesture together and include respective label as dictionary key
def get_data_per_gesture(subject, no_gesture):
    
    path, filename = get_per_subject_file(subject, no_gesture)
    gesture = {}
    
    for i in filename:

        for inx, j in enumerate(filename[i]):  # load individual file
            data = loadmat(os.path.join(path, j))['data'] 

            if inx == 0:
                stack_data = data
             
            else:
                stack_data = np.row_stack((stack_data, data))
                
        gesture[str(i)] = stack_data
        
    return gesture # (samples_per_gesture, channels)


# Preprocess the data
def pre_processing(data, fs, notch_freq, quality_factor, fc, fh, order):
    # 1. Remove DC Offset .. This is optional depending on the bandpass cutoff frequency
    # 2. Notch at 60Hz
    # 3. Bandpass at 5-50Hz
    # 4. Normalize between [-1, 1]


    #rectifed_data = np.abs(dat)
    #scaled_rectified_data = rectifed_data / 128.0 # Mode 3
    #offset_removed_data = lowpass_filter(dat, fs=200, offset=99.0, order=6) # sampling frequency is determined by the device
    
    notched_data = mains_removal(data, fs=fs, notch_freq=notch_freq, quality_factor=quality_factor)
    filtered_data = butter_bandpass_filter(notched_data, lowcut=fc, highcut=fh, fs=fs, order=order)
    
    if filtered_data.any() < 0:
        filtered_data = filtered_data / -128.0
    else:
        filtered_data = filtered_data /127.0

    return filtered_data.transpose() # Return as (channel, samples)


#  Segement the data by using sliding window
def window_with_overlap(data, sampling_frequency=200, window_time=200, overlap=60, no_channel=8):
    samples = int(sampling_frequency * (window_time / 1000))
    num_overlap = int(samples * (overlap/100))

    num_overlap_samples = samples - num_overlap
    idx = [i for i in range(samples, data.shape[1], num_overlap_samples)] # data = (channel, samples)

    data_matrix = np.zeros([len(idx), no_channel, samples]) # (samples, channels, samples

    
    for i, end in enumerate(idx):
        start = end - samples

        if end <= data.shape[1]:
            data_matrix[i] = data[0:no_channel, start:end]
    
    return data_matrix


def create_label(data, inx):
    label = np.zeros([data.shape[0], 1])
    label.fill(int(inx))
    return label


def get_data_subject_specific(subject, no_gesture, fs, notch_freq, quality_factor, fc, fh, order,
                              window_time, overlap, no_channel):
    
    dict_gestures = get_data_per_gesture(subject, no_gesture)
    
    for idx, inx in enumerate(dict_gestures):
        data = pre_processing(dict_gestures[inx], fs, notch_freq, quality_factor, fc, fh, order)
        win_data = window_with_overlap(data=data, sampling_frequency=fs, window_time=window_time, overlap=overlap,
                            no_channel=no_channel)
    
        
        label = create_label(win_data, inx)
        
        if idx == 0:
            data_stack = win_data
            label_stack = label

        else:
            data_stack = np.row_stack((data_stack, win_data))
            label_stack = np.row_stack((label_stack, label))
    
    
    X, y = shuffle_data(data_stack, label_stack)
    return X, y



def shuffle_data(data, label):
    
    idx = np.random.permutation(len(data))
    x,y = data[idx], label[idx]
    
    return x, y
            
def spilt_data(data, label, ratio):
    
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=ratio, random_state=42)
    
    return X_train, y_train, X_test, y_test

