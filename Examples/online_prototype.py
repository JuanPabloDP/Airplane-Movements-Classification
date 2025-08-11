#------------------------------------------------------------------------------------------------------------------
#   Online classification of mobile sensor data
#------------------------------------------------------------------------------------------------------------------

import time
import requests
import numpy as np
import threading
from scipy.interpolate import interp1d

##########################################
############ Data properties #############
##########################################

sampling_rate = 20      # Sampling rate in Hz of the input data
window_time = 0.5       # Window size in seconds for each trial window
window_samples = int(window_time * sampling_rate)   # Number of samples in each window

##########################################
##### Load data and train model here #####
##########################################

# YOUR CODE HERE

##########################################
##### Data acquisition configuration #####
##########################################

# Communication parameters
IP_ADDRESS = '192.168.100.4'
COMMAND = 'accX&accY&accZ&acc_time'
BASE_URL = "http://{}/get?{}".format(IP_ADDRESS, COMMAND)

# Data buffer (circular buffer)
max_samp_rate = 5000            # Maximum possible sampling rate
n_signals = 3                   # Number of signals (accX, accY, accZ)
buffer_size = max_samp_rate*5   # Buffer size (number of samples to store)

buffer = np.zeros((buffer_size, n_signals + 1), dtype='float64')    # Buffer for storing data
buffer_index = 0                                                    # Index for the next data point to be written
last_sample_time = 0.0                                              # Last sample time for the buffer

# Flag for stopping the data acquisition
stop_recording_flag = threading.Event()

# Mutex for thread-safe access to the buffer
buffer_lock = threading.Lock()

# Function for continuously fetching data from the mobile device
def fetch_data():    
    sleep_time = 1. / max_samp_rate 
    while not stop_recording_flag.is_set():
        try:
            response = requests.get(BASE_URL, timeout=0.5)
            response.raise_for_status()            
            data = response.json()

            global buffer, buffer_index, last_sample_time
            
            with buffer_lock:  # Ensure thread-safe access to the buffer
                buffer[buffer_index:, 0] = data["buffer"]["acc_time"]["buffer"][0]    
                buffer[buffer_index:, 1] = data["buffer"]["accX"]["buffer"][0]
                buffer[buffer_index:, 2] = data["buffer"]["accY"]["buffer"][0]
                buffer[buffer_index:, 3] = data["buffer"]["accZ"]["buffer"][0]

                buffer_index = (buffer_index + 1) % buffer_size
                last_sample_time = data["buffer"]["acc_time"]["buffer"][0] 

        except Exception as e:
            print(f"Error fetching data: {e}")

        time.sleep(sleep_time)

# Function for stopping the data acquisition
def stop_recording():
    stop_recording_flag.set()
    recording_thread.join()
    
# Start data acquisition
recording_thread = threading.Thread(target=fetch_data, daemon=True)
recording_thread.start()

##########################################
######### Online classification ##########
##########################################

update_time = 0.25
ref_time = time.time()

while True:
        
    time.sleep(update_time)   

    if buffer_index > 2*sampling_rate:  # Update every update_time seconds and only if enough data is available
    
        ref_time = time.time()
        
        ##### Get last data samples #####            
        
        # Get data from circular buffer
        end_index = (buffer_index - 1) % buffer_size
        start_index = (buffer_index - 2) % buffer_size
        
        with buffer_lock:

            while (buffer[end_index, 0] - buffer[start_index, 0]) <= window_time:
                start_index = (start_index-1) % buffer_size

            indices = (buffer_index - np.arange(buffer_size, 0, -1)) % buffer_size            
            last_raw_data = buffer[indices, :]  # Get last data samples from the buffer

        # Calculate time vector for interpolation                    
        t = last_raw_data[:, 0]  # Time vector from the buffer
        t_uniform = np.linspace(last_sample_time-window_time, last_sample_time, int(window_time * sampling_rate))   

        # interpolate each signal to a uniform time vector
        last_data = np.zeros((len(t_uniform), n_signals))  # Array with interpolated data
        for i in range(n_signals):
            interp_x = interp1d(t, last_raw_data[:, i+1], kind='linear', fill_value="extrapolate") # Interpolation function for signal i
            last_data[:,i] = interp_x(t_uniform)  # Interpolate signal i to the uniform time vector
                        
        print ("Window data:\n", last_data)

        #######################################################
        ##### Calculate features of the last data samples #####
        #######################################################

        # YOUR feature calculation code here
        # Features must be the same as the calculated above when the model was trained        
        
        #################################################################
        ##### Evaluate classifier here with the calculated features #####
        #################################################################
        
        # YOUR classification code here
        
     
# Stop data acquisition
stop_recording()

#------------------------------------------------------------------------------------------------------------------
#   End of file
#------------------------------------------------------------------------------------------------------------------