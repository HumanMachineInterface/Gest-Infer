# Version 2

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, sosfreqz, lfilter, iirnotch, filtfilt
from scipy import signal
from scipy.signal import stft


def get_time_domain_plot(data):
    """
    This returns bio-signal plot in the time domain.6

    """
    plt.clf()
    plt.title("Time Domain")
    plt.plot(data)


def get_frequency_domain_plot(data, fs):
    fft = np.fft.rfft(data)
    freq = np.fft.rfftfreq(len(data), 1 / fs)
    plt.clf()
    plt.title("Frequency Domain")
    plt.plot(freq, abs(fft))
    plt.xlabel("Frequency")
    plt.ylabel("Number of samples")


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='bandpass', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut=5, highcut=35, fs=250, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    
    filtered_signal = np.zeros_like(data)

    # Loop through each channel of the EMG signal
    for i in range(data.shape[1]):
        filtered_signal[:, i] = sosfilt(sos, data[:,i])

    return filtered_signal


def get_spectogram(data, fs, nfft, overlap):
    plt.title('Spectrogram')
    plt.specgram(data, Fs=fs, NFFT=nfft, noverlap=overlap)
    plt.xlabel("Time Sec")
    plt.ylabel("Frequency Hz")


def remove_dc_offset(data, fs=250, offset=.5, order=3):
    nyq = .5 * fs
    normal_cutoff = offset / nyq
    b, a = butter(N=order, Wn=normal_cutoff, btype='highpass', analog=False)
    fil_data = lfilter(b, a, data)
    return fil_data


def lowpass_filter(data, fs=250, offset=.5, order=3):
    """

    :param data: Input signal in form of (channels, samples)
    :param fs: Sampling Frequency
    :param offset: Amount of offset
    :param order: The order of filter to use
    :return: Returns signal passed through a low pass filter
    """
    nyq = .5 * fs
    normal_cutoff = offset / nyq
    b, a = butter(N=order, Wn=normal_cutoff, btype='lowpass', analog=False)
    fil_data = lfilter(b, a, data)
    return fil_data


def mains_removal(data, fs=250, notch_freq=60.0, quality_factor=30.0):
    
    data = data.transpose() # For faster computation, data must be in channel, samples
    
    b, a = iirnotch(notch_freq, quality_factor, fs)
    fil_data = filtfilt(b, a, data, padlen=len(data) - 1)
    
    return fil_data.transpose() # Return back to channel, samples

"""
def remove_mains_interference(data, fs, fc, Q):

    b, a = iirnotch(fc, Q, fs)
    
    filtered_signal = np.zeros_like(data)

    # Loop through each channel of the EMG signal
    for i in range(data.shape[1]):
        filtered_signal[:, i] = filtfilt(b, a, data[:, i], padlen=len(data) - 1)

    return filtered_signal # Return back to channel, samples

"""


class TimeDomainFeatures:

    def __init__(self, data):

        self.X = data

    def MAV(self, axis=2):  # 1

        mav = np.sum(np.abs(self.X), axis=axis) * 1 / self.X.shape[axis]
        return mav

    def STD(self, axis=2):  # 2
        mu = np.mean(self.X, axis=axis)
        for i, j in enumerate(self.X):
            self.X[i] = (self.X[i].transpose() - mu[i, :]).transpose()

        std = np.sum(np.square(self.X), axis=axis) * (1 / self.X.shape[axis] - 1)
        # std = np.sqrt(std)
        return std

    def VAR(self, axis=2):  # 3
        var = np.sum(np.square(self.X), axis=axis) * (1 / self.X.shape[axis] - 1)
        return var

    def WL(self, axis=2):

        shape = self.X.shape
        X_new = np.zeros(shape)
        for i, j in enumerate(self.X):
            x = self.X[i].transpose()
            for previous in range(x.shape[0]):
                start = previous + 1

                if start == x.shape[0]:
                    xx = x[previous, :]
                    X_new[i, :, previous] = xx
                    break
                else:
                    xx = x[start, :] - x[previous, :]
                    X_new[i, :, previous] = xx

        X_new = np.sum(X_new, axis=axis)
        return X_new

    def RMS(self, axis=2):

        rms = np.sum(np.square(np.abs(self.X)), axis=axis) * 1 / self.X.shape[axis]
        rms = np.sqrt(rms)

        return rms
