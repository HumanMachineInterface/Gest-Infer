import time
import pandas as pd
import numpy as np
import os
from scipy.io import savemat
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from playsound import playsound


def get_eeg_data(seconds_timeout, seconds, csv_filepath, mat_filepath):
    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    params.serial_port = '/dev/ttyUSB0'
    params.timeout = seconds_timeout
    board_type = BoardIds.CYTON_BOARD.value
    eeg_channels = BoardShim.get_eeg_channels(int(board_type))
    # params.mac_address = args.mac_address
    # params.other_info = args.other_info
    # params.serial_number = args.serial_number
    # params.ip_address = args.ip_address
    # params.ip_protocol = args.ip_protocol
    # params.file = args.file
    # params.ip_port = args.ip_port

    fs = 250
    stream_to = 'streaming_board://225.1.1.1:6677'  # This streams to openbci GUI

    no_samples = (fs * seconds)  # param num_samples: size of ring buffer to keep data
    board = BoardShim(board_type, params)
    board.prepare_session()
    board.start_stream(no_samples, stream_to)  # board.start_stream () # use this for default options
    #board.start_stream(no_samples)
    time.sleep(10)
    # data = board.get_current_board_data (256) # get latest 256 packages or less, doesnt remove them from internal buffer
    data = board.get_board_data()  # get all data and remove it from internal buffer
    board.stop_stream()
    board.release_session()

    eeg_current_data = data[eeg_channels, :]
    eeg_df = pd.DataFrame(eeg_current_data)
    eeg_df.to_csv(csv_filepath, index=False)

    mdic = {'data': eeg_current_data}
    savemat(mat_filepath + ".mat", mdic)
    base_path = os.getcwd()
    sound_path = base_path + "/sound.wav"
    playsound(sound_path)

# %%

# %%
