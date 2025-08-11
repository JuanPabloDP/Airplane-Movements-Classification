#------------------------------------------------------------------------------------------------------------------
#   Online classification of mobile sensor data
#------------------------------------------------------------------------------------------------------------------

import time
import requests
import numpy as np
import threading
from scipy.interpolate import interp1d

from scipy import stats

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE


##########################################
############ Data properties #############
##########################################

sampling_rate = 20      # Sampling rate in Hz of the input data
window_time = 2       # Window size in seconds for each trial window
window_samples = int(window_time * sampling_rate)   # Number of samples in each window

##########################################
##### Load data and train model here #####
##########################################

data = np.loadtxt("Programa Final\Proto\data.txt")
x = data[:, 1:]
y = data[:, 0]

clases = [('Nothing', 1), ('Despegue', 2), ('Aterrizaje', 3), ('Vuelo estable', 4), ('Turbulencia', 5), 
              ('Choque', 6), ('Truco', 7), ('Patrones de espera', 8), ('En pista', 9)]
dicc_clases = {num: name for name, num in clases}

opt_hyperparameter=9
opt_features=8

clf = RandomForestClassifier(max_depth=opt_hyperparameter, random_state=1234)
fselection = RFE(clf, n_features_to_select=opt_features)
fselection.fit(x, y)

x_transformed = fselection.transform(x)
clf.fit(x_transformed, y)

##########################################
##### Data acquisition configuration #####
##########################################

# Communication parameters

# Here include the adress of "phyphox" remote access in the used phone
IP_ADDRESS = '10.43.96.222' 

COMMAND = ('acc_time&accX&accY&accZ&' #Accelerometer
           'lin_accX&lin_accY&lin_accZ&' #Linear Accelerometer
           'gyroX&gyroY&gyroZ&' #Gyroscope
           'magX&magY&magZ&' #Magnetometer
           'gpsLat&gpsLon') #Location 
BASE_URL = "http://{}/get?{}".format(IP_ADDRESS, COMMAND)

# Data buffer (circular buffer)
max_samp_rate = 5000            # Maximum possible sampling rate
n_signals = 14                   # Number of signals (accX, accY, accZ, lin_accX, lin_accY, lin_accZ, gyroX, gyroY, gyroZ, magX, magY, magZ, gpsLat, gpsLon)
buffer_size = max_samp_rate*20 #20 de acuerdo a que aparentemente este es 10 veces el window_time = 2 
# Buffer size (number of samples to store)

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
            
            with buffer_lock:  # Ensure thread-safe access to the buffer#Acceletometer acc_time&accX&accY&accZ
                buffer[buffer_index:, 0] = data["buffer"]["acc_time"]["buffer"][0]    
                buffer[buffer_index:, 1] = data["buffer"]["accX"]["buffer"][0]
                buffer[buffer_index:, 2] = data["buffer"]["accY"]["buffer"][0]
                buffer[buffer_index:, 3] = data["buffer"]["accZ"]["buffer"][0]
                #Linear Accelerometer lin_accX&lin_accY&lin_accZ
                buffer[buffer_index:, 4] = data["buffer"]["lin_accX"]["buffer"][0]
                buffer[buffer_index:, 5] = data["buffer"]["lin_accY"]["buffer"][0]
                buffer[buffer_index:, 6] = data["buffer"]["lin_accZ"]["buffer"][0]
                #Gyroscope gyroX&gyroY&gyroZ
                buffer[buffer_index:, 7] = data["buffer"]["gyroX"]["buffer"][0]
                buffer[buffer_index:, 8] = data["buffer"]["gyroY"]["buffer"][0]
                buffer[buffer_index:, 9] = data["buffer"]["gyroZ"]["buffer"][0]
                #Magnetometer magX&magY&magZ
                buffer[buffer_index:, 10] = data["buffer"]["magX"]["buffer"][0]
                buffer[buffer_index:, 11] = data["buffer"]["magY"]["buffer"][0]
                buffer[buffer_index:, 12] = data["buffer"]["magZ"]["buffer"][0]
                #Location gpsLat&gpsLon&gpsZ&gpsV&gpsDir
                buffer[buffer_index:, 13] = data["buffer"]["gpsLat"]["buffer"][0]
                buffer[buffer_index:, 14] = data["buffer"]["gpsLon"]["buffer"][0]

                buffer_index = (buffer_index + 1) % buffer_size
                last_sample_time = data["buffer"]["acc_time"]["buffer"][0] 

                #Buffer

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
                        
        #print ("Window data:\n", last_data)

        #######################################################
        ##### Calculate features of the last data samples #####
        #######################################################

        # YOUR feature calculation code here
        # Features must be the same as the calculated above when the model was trained      

        # Process each trial and build data matrices

        # Unifica x a partir de los datos de la ventana 

        entry = []
        rms = 0
        for column in range(n_signals):
            sensor=[]
            for row in last_data:
                sensor.append(row[column])
            sensor=np.array(sensor)
            entry.append(np.average(sensor))
            entry.append(np.std(sensor))
            entry.append(np.max(sensor))
            entry.append(np.min(sensor))
            entry.append(stats.kurtosis(sensor))
            entry.append(stats.skew(sensor))
            rms += np.sum(sensor**2)
        rms = np.sqrt(rms)    
        entry.append(rms)

        x_online =  np.array(entry)
        
        #################################################################
        ##### Evaluate classifier here with the calculated features #####
        #################################################################
        
        x_online_transformed = fselection.transform([x_online])
        y_online_pred = clf.predict(x_online_transformed)
        clase=dicc_clases.get(y_online_pred[0])
        print("Prediction:", clase)
        
     
# Stop data acquisition
stop_recording() #Still missing stop option 

#------------------------------------------------------------------------------------------------------------------
#   End of file
#------------------------------------------------------------------------------------------------------------------