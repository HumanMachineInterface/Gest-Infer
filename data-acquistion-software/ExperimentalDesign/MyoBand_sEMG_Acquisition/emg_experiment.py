'''
The MIT License (MIT)
Copyright (c) 2020 PerlinWarp
Copyright (c) 2014 Danny Zhu


// Lab: Biomedical Information and Signal Lab (BMIS)
// Engineer: Dere Mustapha Deji
// Create Date: 2022/04/13  9:53:00 PM
// Design Name: Modified data acquisition
// Module Name: Myoband
// Tool Versions: Python 3.8
// Description: This script is aim for data acquistion protocol aimed for pilot study
// Dependencies: Pyomyo https://github.com/PerlinWarp/pyomyo
// Revision: 1
// Additional Comments: Credit to Peter Walkington
//
//////////////////////////////////////////////////////////////////////////////////
'''

import time
import multiprocessing
import numpy as np
import pandas as pd
from scipy.io import savemat

from pyomyo import Myo, emg_mode
from playsound import playsound
import os


def get_emg_data(mode, seconds, csv_filepath, mat_filepath):
    collect = True

    # ------------ Myo Setup ---------------
    m = Myo(mode=mode)
    m.connect()

    myo_data = []

    def add_to_queue(emg, movement):
        myo_data.append(emg)

    m.add_emg_handler(add_to_queue)

    def print_battery(bat):
        print("Battery level:", bat)

    m.add_battery_handler(print_battery)

    # Its go time
    # m.set_leds([0, 128, 0], [0, 128, 0])
    # Vibrate to know we connected okay
    # m.vibrate(1)

    print("Data Worker started to collect")
    # Start collecting data.
    start_time = time.time()

    while collect:
        if time.time() - start_time < seconds:
            m.run()  # collect data as long as time is less than seconds
        else:
            collect = False
            acquisition_time = time.time() - start_time
            print("Finished collecting.")
            m.vibrate(1)
            print(f"Total Time of acquisition: {acquisition_time}")
            # print(len(myo_data), "frames collected")

            # Add columns and save to df
            myo_cols = ["Channel_1", "Channel_2", "Channel_3", "Channel_4", "Channel_5", "Channel_6", "Channel_7",
                        "Channel_8"]
            myo_df = pd.DataFrame(myo_data, columns=myo_cols)
            myo_df.to_csv(csv_filepath, index=False)
            # print("CSV Saved at: ", filepath)

            mdic = {'data': myo_data}
            savemat(mat_filepath + ".mat", mdic)

            base_path = os.getcwd()
            sound_path = base_path + "/sound.wav"
            playsound(sound_path)
