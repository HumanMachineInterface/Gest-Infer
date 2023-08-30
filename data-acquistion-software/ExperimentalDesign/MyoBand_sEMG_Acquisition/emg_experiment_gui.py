import tkinter as tk
from tkinter import ttk
from emg_experiment import *
from playsound import playsound
import os
from PIL import ImageTk, Image

base_path = os.getcwd()


def acquire_data():
    sound_path = base_path + "/sound.wav"
    playsound(sound_path)

    seconds = float(acquisition_time.get()) + 0.1
    subject_index = subject_no.get()
    experiment_index = clicked.get().split()[1]
    repetition_index = rep_no.get()

    csv_file_path = base_path + "/BMIS_EMG_DATA/data/csv_data/subject_" + subject_index
    mat_file_path = base_path + "/BMIS_EMG_DATA/data/mat_data/subject_" + subject_index

    print(csv_file_path)
    print(mat_file_path)
    if not os.path.exists(csv_file_path):
        os.makedirs(csv_file_path)
        os.makedirs(mat_file_path)

    csv_file = csv_file_path + '/S{}_R{}_G{}.csv'.format(subject_index, repetition_index, experiment_index)
    mat_file = mat_file_path + '/S{}_R{}_G{}'.format(subject_index, repetition_index, experiment_index)

    mode = emg_mode.RAW
    p = multiprocessing.Process(target=get_emg_data(mode, seconds, csv_file, mat_file))
    p.start()


# function to open a new window
# on a button click
def display_gesture_new_window():
    # Toplevel object which will
    # be treated as a new window
    new_window = tk.Toplevel(window)

    # sets the title of the
    # Toplevel widget
    new_window.title("Gesture Display")

    # sets the geometry of toplevel
    new_window.geometry("2000x2000")

    # A Label widget to show in toplevel
    get_choice = int(clicked.get().split()[1]) - 1
    picture_path = base_path + "/pictures/{}".format(image_list[get_choice])
    print(picture_path)
    images = ImageTk.PhotoImage(Image.open(picture_path))
    image_label = ttk.Label(new_window, image=images, width=100, padding=100)
    image_label.grid(column=200, row=200)
    new_window.mainloop()


window = tk.Tk()
window.title("EMG ACQUISITION GUI")
window.geometry("500x500")


window.columnconfigure(0, weight=200)
window.columnconfigure(1, weight=200)
window.columnconfigure(2, weight=200)

experiment_message = ttk.Label(text="EMG EXPERIMENT")
experiment_message.grid(column=1, row=0, padx=10, pady=10)

time_label = ttk.Label(text="Enter time   ")
time_label.grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)
acquisition_time = tk.Entry(window, width=50, borderwidth=10)
acquisition_time.grid(column=1, row=1, sticky=tk.E, padx=5, pady=5)

subject_label = ttk.Label(text="Enter Subject")
subject_label.grid(column=0, row=2, sticky=tk.W, padx=5, pady=5)
subject_no = tk.Entry(window, width=50, borderwidth=10)
subject_no.grid(column=1, row=2, sticky=tk.W, padx=5, pady=5)

repetition_label = ttk.Label(text="Enter Repetition")
repetition_label.grid(column=0, row=3, sticky=tk.W, padx=5, pady=5)
rep_no = tk.Entry(window, width=50, borderwidth=10)
rep_no.grid(column=1, row=3, sticky=tk.W, padx=5, pady=5)

clicked = tk.StringVar()
clicked.set("Gesture 1")
drop = tk.OptionMenu(window, clicked, "Gesture 1", "Gesture 2", "Gesture 3", "Gesture 4", "Gesture 5",
                     "Gesture 6", "Gesture 7")
drop.grid(column=1, row=4, pady=10)

image_list = ["exp{}.jpg".format(i) for i in range(1, 8)]

start_button = tk.Button(window, text="Start Experiment", command=acquire_data, bg="green")
start_button.grid(column=1, row=5, pady=10)

display_button = tk.Button(window, text="Display Gesture", command=display_gesture_new_window, bg="orange")
display_button.grid(column=1, row=6, pady=10)

window.mainloop()

