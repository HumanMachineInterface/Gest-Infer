# -*- coding: utf-8 -*-
"""
// LAB: Biomedical Information and Signal Lab 
// Engineer: Dere Mustapha Deji
// Created on: 2022/05/12 11:45:00 PM
// Design Name: Motor Execution
// 
"""

import os
from biosignal_analysis_tools import *
from scipy.io import loadmat
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from scipy import signal

emg_path = "BMIS_EMG_DATA"
eeg_path = "BMIS_EEG_DATA"

class Scale:
    mini_volt = 1000
    micro_volt = 1000000

def get_emg_per_subject_file(subject, no_gesture):

    base_path = os.getcwd().strip('\\code') + emg_path +"/data/mat_data/subject_{}".format(subject)
    data_file = os.listdir(base_path)
    gesture = {}
    for i in range(no_gesture):  # Runs through the total number of gesture
        j = i + 1
        gesture[str(i)] = [file for inx, file in enumerate(data_file) if
                           data_file[inx][7] == str(j) or data_file[inx][8] == str(j)]

    return base_path, gesture

def get_eeg_per_subject_file(subject, no_gesture):

    base_path = os.getcwd().strip('\\code') + eeg_path +"/data/mat_data/subject_{}".format(subject)
    data_file = os.listdir(base_path)
    gesture = {}
    for i in range(no_gesture):  # Runs through the total number of gesture
        j = i + 1
        gesture[str(i)] = [file for inx, file in enumerate(data_file) if
                           data_file[inx][7] == str(j) or data_file[inx][8] == str(j)]

    return base_path, gesture



# STEP2: Concatenate all gesture together and include respective label as dictionary key
def get_emg_data_per_gesture(subject, no_gesture):
    path, filename = get_emg_per_subject_file(subject, no_gesture)
    gesture = {}

    for i in filename:

        for inx, j in enumerate(filename[i]):  # load individual file
            data = loadmat(os.path.join(path, j))['data']

            if inx == 0:
                stack_data = data

            else:
                stack_data = np.row_stack((stack_data, data))

        gesture[str(i)] = stack_data

    return gesture  # (samples_per_gesture, channels)



def get_eeg_data_per_gesture(subject, no_gesture):
    path, filename = get_eeg_per_subject_file(subject, no_gesture)
    gesture = {}

    for i in filename:

        for inx, j in enumerate(filename[i]):  # load individual file
            data = loadmat(os.path.join(path, j))['data'].transpose()

            if inx == 0:
                stack_data = data

            else:
                stack_data = np.row_stack((stack_data, data))
                
                
        stack_data = stack_data / Scale.micro_volt
        gesture[str(i)] = stack_data

    return gesture  # (samples_per_gesture, channels)



# STEP 3: Preprocessing the data 

def pre_processing_emg(data, emg_fs, notch_freq, quality_factor, fc, fh, order):
    # 1. Remove DC Offset. This is optional depending on the bandpass cutoff frequency
    # 2. Notch at 60Hz
    # 3. Bandpass at 5-50Hz
    # 4. Normalize between [-1, 1]

    # rectifed_data = np.abs(dat)
    # scaled_rectified_data = rectifed_data / 128.0 # Mode 3
    # offset_removed_data http://localhost:8888/tree  = lowpass_filter(dat, fs=200, offset=99.0, order=6)
    # sampling frequency is determined by the device



    notched_data = mains_removal(data, fs=emg_fs, notch_freq=notch_freq, quality_factor=quality_factor)
    filtered_data = butter_bandpass_filter(notched_data, lowcut=fc, highcut=fh, fs=emg_fs, order=order)

    if filtered_data.any() < 0:
        filtered_data = filtered_data / -128.0
    else:
        filtered_data = filtered_data / 127.0

    return filtered_data.transpose()  # Return as (channel, samples)


def pre_processing_eeg(data, eeg_fs, notch_freq, quality_factor, fc, fh, order):
    # 1. Remove DC Offset. This is optional depending on the bandpass cutoff frequency
    # 2. Notch at 60Hz
    # 3. Bandpass at 5-50Hz
    # 4. Normalize between [-1, 1]

    # rectifed_data = np.abs(dat)
    # scaled_rectified_data = rectifed_data / 128.0 # Mode 3
    # offset_removed_data http://localhost:8888/tree  = lowpass_filter(dat, fs=200, offset=99.0, order=6)
    # sampling frequency is determined by the device



    notched_data = mains_removal(data, fs=eeg_fs, notch_freq=notch_freq, quality_factor=quality_factor)
    filtered_data = butter_bandpass_filter(notched_data, lowcut=fc, highcut=fh, fs=eeg_fs, order=order)
    filtered_data = filtered_data[250:,:] ##

    return filtered_data.transpose()  # Return as (channel, samples)



# Segementing the data 
#  Segement the data by using sliding window

def window_with_overlap(data, sampling_frequency=250, window_time=1000, overlap=60, no_channel=8):
    samples = int(sampling_frequency * (window_time / 1000))
    num_overlap = int(samples * (overlap / 100))

    num_overlap_samples = samples - num_overlap
    idx = [i for i in range(samples, data.shape[1], num_overlap_samples)]  # data = (channel, samples)

    data_matrix = np.zeros([len(idx), no_channel, samples])  # (samples, channels, samples

    for i, end in enumerate(idx):
        start = end - samples

        if end <= data.shape[1]:
            data_matrix[i] = data[0:no_channel, start:end]

    return data_matrix


def create_label(data, inx):
    label = np.zeros([data.shape[0], 1])
    label.fill(int(inx))
    return label


def shuffle_data(data_emg, data_eeg, label):
    
    size=len(data_eeg)
    classes = len(np.unique(label))
    divs = int(np.floor(size/classes))
    
    emg_data = []
    rev_label = []
    
    for i in range(classes):
        
        random_items = np.random.choice(np.where(np.squeeze(label == i))[0], size=divs, replace=True)
        emg_data.append(data_emg[random_items])
        rev_label.append(label[random_items])
    
    new_emg_data = np.concatenate(emg_data, axis=0)
    new_label = np.concatenate(rev_label, axis=0)
    
    idx = np.random.permutation(len(data_eeg))
    x_emg, x_eeg, y = new_emg_data[idx], data_eeg[idx], new_label[idx]

    return x_emg, x_eeg, y


def get_data_subject_specific(subject, no_gesture, emg_fs, eeg_fs, 
                              notch_freq, quality_factor, emg_fc, emg_fh, 
                              eeg_fc, eeg_fh, order,
                              window_time, overlap, no_channel):
    
    dict_gestures_emg = get_emg_data_per_gesture(subject, no_gesture)
    dict_gestures_eeg = get_eeg_data_per_gesture(subject, no_gesture)
    

    for idx, inx in enumerate(dict_gestures_emg):
        
        
        data_emg = pre_processing_emg(dict_gestures_emg[inx], emg_fs, notch_freq, 
                                  quality_factor, emg_fc, emg_fh, order)
        
        data_eeg = pre_processing_eeg(dict_gestures_eeg[inx], eeg_fs, notch_freq, 
                                  quality_factor, eeg_fc, eeg_fh, order)
        
        
        
        win_data_emg = window_with_overlap(data=data_emg, sampling_frequency=emg_fs, 
                                           window_time=window_time, 
                                       overlap=overlap, no_channel=no_channel)
        
        
        win_data_eeg = window_with_overlap(data=data_eeg, sampling_frequency=eeg_fs, 
                                           window_time=window_time, 
                                       overlap=overlap, no_channel=no_channel)
        
        

        label = create_label(win_data_emg, inx)

        if idx == 0:
            data_stack_emg = win_data_emg
            data_stack_eeg = win_data_eeg
            label_stack = label

        else:
            data_stack_emg = np.row_stack((data_stack_emg, win_data_emg))
            data_stack_eeg = np.row_stack((data_stack_eeg, win_data_eeg))
            label_stack = np.row_stack((label_stack, label))

    X_emg, X_eeg, y = shuffle_data(data_stack_emg, data_stack_eeg, label_stack)

    return X_emg, X_eeg, y

def spilt_data(data, label, ratio):
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=ratio, random_state=42)

    return X_train, y_train, X_test, y_test


def downsample_data(data, nfs, window_time):
    
    n_samples = int(nfs * (window_time/1000))
    downsampled_data = signal.resample(data, n_samples, axis=2)
    
    return downsampled_data


'''
def split_data(emg_data, eeg_data, label, ratio=0.2, random_seed=43):
    """
    Split the data into training and testing sets.
    X: Input features as NxM matrix, N samples of M features
    Y: Corresponding outputs as a vector of length N
    test_perc: Percentage of data to use for testing
    random_seed: Random seed for splitting the data
    Returns: X_train, Y_train, X_test, Y_test
    """

    # Set random seed (if given)
    if random_seed:
        np.random.seed(random_seed)

    # Shuffle the data randomly
    idx = np.random.permutation(len(X))

    # Split the data based on the test_perc parameter
    split_point = int(len(X) * (1 - test_perc))
    train_idx, test_idx = idx[:split_point], idx[split_point:]

    # Separate the data into training and testing sets
    X_train, Y_train = X[train_idx, :], Y[train_idx]
    X_test, Y_test = X[test_idx, :], Y[test_idx]

    # Return the train and test sets
    return X_train, Y_train, X_test, Y_test


'''