# -*- coding: utf-8 -*-
"""
// LAB: Biomedical Information and Signal Lab 
// Engineer: Dere Mustapha Deji
// Created on: 2022/04/27 12:35:00 PM
// Design Name: Motor Execution
// 
"""

import os
from biosignal_analysis_tools import *
from scipy.io import loadmat, savemat
import numpy as np
import random
from sklearn.model_selection import train_test_split


def get_per_subject_file(subject):
    base_path = os.getcwd().strip('\\code')
    data_path = "/data/mat_data/subject_{}".format(subject)
    return base_path + data_path + "/"


class Scale:
    mini_volt = 1000
    micro_volt = 1000000


def segregate_per_gesture(subject, no_gesture=7):
    data_file = os.listdir(get_per_subject_file(subject))
    gesture = {}
    for i in range(no_gesture):
        j = i + 1
        gesture[str(j)] = [file for inx, file in enumerate(data_file) if
                           data_file[inx][7] == str(j) or data_file[inx][8] == str(j)]

    return gesture


def get_data_per_gesture(subject, desired_gesture):
    for i in range(len(segregate_per_gesture(subject)[str(desired_gesture)])):

        # print(get_per_subject_file(subject)+segregate_per_gesture(subject)[str(desired_gesture)][i])
        data = loadmat(get_per_subject_file(subject) + segregate_per_gesture(subject)[str(desired_gesture)][i])['data']
        data = data.transpose()

        if i == 0:
            data_stack = data

        else:
            data_stack = np.row_stack((data_stack, data))

    data_micro_scale = data_stack / Scale.micro_volt
    # data_mini_scale = data_stack / Scale.mini_volt

    data_micro_scale = data_micro_scale.transpose()
    return data_micro_scale


def data_accum(subject, no_gesture=7):
    gesture = {}
    for i in range(no_gesture):
        j = i + 1
        data = get_data_per_gesture(subject, j)
        gesture[str(j)] = data

    return gesture


def create_label(data):
    label = {}
    for indx in data:
        size = data[str(indx)].shape[1]
        lbl = (np.zeros((1, size)) + int(indx)) - 1
        label[indx] = lbl

    return label


def pre_processing(data):
    # 1. Remove DC Offset
    # 2. Notch at 60Hz
    # 3. Bandpass at 5-50Hz

    keys = list(data.keys())
    random.shuffle(keys)

    filtered_data = {}
    for index in keys:
        offset_removed_data = remove_dc_offset(data[index])  # input is channel, samples
        notched_data = mains_removal(offset_removed_data)
        filtered = butter_bandpass_filter(notched_data)
        filtered_data[index] = filtered

    return filtered_data


def whole_data(data, label):
    for idx, indx in enumerate(data):

        if idx == 0:
            data_stack = data[indx].transpose()
            label_stack = label[indx].transpose()

        else:
            data_stack = np.row_stack((data_stack, data[indx].transpose()))
            label_stack = np.row_stack((label_stack, label[indx].transpose()))

    return data_stack.transpose(), label_stack.transpose()


def window_with_overlap(data, label, sampling_frequency=250, window_time=200, overlap=90, no_channel=8):
    samples = int(sampling_frequency * (window_time / 1000))
    num_overlap = int(sampling_frequency * (window_time * (overlap / 100) / 1000))
    # data = data.transpose() Data must be in form (channels, number of samples) if not then transpose
    num_overlap_samples = samples - num_overlap
    idx = [i for i in range(samples, data.shape[1], num_overlap_samples)]

    data_matrix = np.zeros([len(idx), no_channel, samples])
    label_matrix = np.zeros([len(idx), 1])

    for i, end in enumerate(idx):
        start = end - samples

        if end <= data.shape[1]:
            data_matrix[i] = data[0:no_channel, start:end]
            lbl = np.int8(label[start:end]).reshape(-1)
            max_label = np.argmax(np.bincount(lbl))
            label_matrix[i] = max_label

    return data_matrix, label_matrix


def window_with_overlap_transpose(data, label, sampling_frequency=250, window_time=200, overlap=90, no_channel=8):
    samples = int(sampling_frequency * (window_time / 1000))
    num_overlap = int(sampling_frequency * (window_time * (overlap / 100) / 1000))
    data = data.transpose()  # Data must be in form (channels, number of samples) if not then transpose
    num_overlap_samples = samples - num_overlap
    idx = [i for i in range(samples, data.shape[0], num_overlap_samples)]

    data_matrix = np.zeros([len(idx), samples, no_channel])
    label_matrix = np.zeros([len(idx), 1])

    for i, end in enumerate(idx):
        start = end - samples

        if end <= data.shape[0]:
            data_matrix[i] = data[start:end, 0:no_channel]
            lbl = np.int8(label[0:, start:end]).reshape(-1)
            max_label = np.argmax(np.bincount(lbl))
            label_matrix[i] = max_label

    return data_matrix, label_matrix


def get_data_subject_specific(subject):
    data = data_accum(subject)
    data = pre_processing(data)
    label = create_label(data)
    data, label = whole_data(data, label)
    data = standardize_data(data)

    return data, label


def get_data_all_subject(number_of_subject):
    for i in range(number_of_subject):

        idx = i + 1

        X, y = get_data_subject_specific(idx)
        X = X.transpose()
        y = y.transpose()
        if idx == 1:
            data_stack = X
            label_stack = y

        else:
            data_stack = np.row_stack((data_stack, X))
            label_stack = np.row_stack((label_stack, y))

    return data_stack.transpose(), label_stack.transpose()


def standardize_data(data):
    mu = data.mean(axis=0)
    std = data.std(axis=0)
    standarized_data = (data - mu) / std
    return standarized_data


def spilt_data(data, label, ratio):
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=ratio, random_state=42)

    return X_train, y_train, X_test, y_test


def save_hardware_test_data(path, data, label, ratio=0.2, window=200, overlap=70):
    data, label = window_with_overlap(data, label, window_time=window, overlap=overlap)
    _, _, X, y = spilt_data(data, label, ratio)
    mdic = {'data': X, 'label': y}
    savemat(path + "/test_data.mat", mdic)
# data = data_accum(1)
# data = pre_processing(data)
# label = create_label(data)
# data, label = whole_data(data, label)
# X, y = window_with_overlap(data, label)
