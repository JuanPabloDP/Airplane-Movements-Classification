#------------------------------------------------------------------------------------------------------------------
#   Real-time plot for acceleration data.
#------------------------------------------------------------------------------------------------------------------
import time
import requests
from collections import deque
from threading import Thread

import matplotlib.animation as animation
import matplotlib.pyplot as plt

# Communication parameters
IP_ADDRESS = '192.168.1.72'
COMMAND = 'accX&accY&accZ&acc_time&lin_accX&lin_accY&lin_accZ'
BASE_URL = "http://{}/get?{}".format(IP_ADDRESS, COMMAND)

# Data acquisition parameters
samps_per_frame = 200           # Number of samples per frame
sampling_rate = 20              # Hz (Sampling rate)
sleep_time = 1. / sampling_rate # Sleep time in seconds for each request

# Data buffers (circular buffers)
t = deque(maxlen = samps_per_frame)

#Acceletometer
acc_x1 = deque(maxlen = samps_per_frame)
acc_x2 = deque(maxlen = samps_per_frame)
acc_x3 = deque(maxlen = samps_per_frame)
#Linear Accelerometer
lin_acc_x1 = deque(maxlen = samps_per_frame)
lin_acc_x2 = deque(maxlen = samps_per_frame)
lin_acc_x3 = deque(maxlen = samps_per_frame)

# Function for continuously fetching data from the mobile device
def fetch_data():
    
    acquire = True
    
    while acquire:
        try:
            response = requests.get(BASE_URL, timeout=1)
            response.raise_for_status()
            data = response.json()

            timestamp = data["buffer"]["acc_time"]["buffer"][0]
            #Acceletometer
            accX = data["buffer"]["accX"]["buffer"][0]
            accY = data["buffer"]["accY"]["buffer"][0]
            accZ = data["buffer"]["accZ"]["buffer"][0]
            #Linear Accelerometer
            lin_accX = data["buffer"]["lin_accX"]["buffer"][0]
            lin_accY = data["buffer"]["lin_accY"]["buffer"][0]
            lin_accZ = data["buffer"]["lin_accZ"]["buffer"][0]

            t.append(timestamp)
            #Acceletometer
            acc_x1.append(accX)
            acc_x2.append(accY)
            acc_x3.append(accZ)
            #Linear Accelerometer        
            lin_acc_x1.append(lin_accX)
            lin_acc_x2.append(lin_accY)
            lin_acc_x3.append(lin_accZ)  

        except Exception as e:
            print(f"Error: {e}")
            acquire = False

        time.sleep(sleep_time)

# Initialize plots
fig, (ax11, ax12, ax13) = plt.subplots(3)
#fig, (ax21, ax22, ax23) = plt.subplots(3)

# 
def animate(i):
    if len(t) > 1:
        #Acceletometer
        ax11.clear()
        ax11.plot(t, acc_x1)
        ax11.set_ylim(min(acc_x1) - 1, max(acc_x1) + 1)
        ax11.set_ylabel("X")

        ax12.clear()
        ax12.plot(t, acc_x2)
        ax12.set_ylim(min(acc_x2) - 1, max(acc_x2) + 1)
        ax12.set_ylabel("Y")

        ax13.clear()
        ax13.plot(t, acc_x3)
        ax13.set_ylim(min(acc_x3) - 1, max(acc_x3) + 1)
        ax13.set_ylabel("Z")
        ax13.set_xlabel("Time (s)")
        
        #Linear Accelerometer 
        """ax21.clear()
        ax21.plot(t, lin_acc_x1)
        ax21.set_ylim(min(lin_acc_x1) - 1, max(lin_acc_x1) + 1)
        ax21.set_ylabel("X")

        ax22.clear()
        ax22.plot(t, lin_acc_x2)
        ax22.set_ylim(min(lin_acc_x2) - 1, max(lin_acc_x2) + 1)
        ax22.set_ylabel("Y")

        ax23.clear()
        ax23.plot(t, lin_acc_x3)
        ax23.set_ylim(min(lin_acc_x3) - 1, max(lin_acc_x3) + 1)
        ax23.set_ylabel("Z")
        ax23.set_xlabel("Time (s)")"""


# Launch data fetching in a separate thread
thread = Thread(target=fetch_data, daemon=True)
thread.start()

# Set up the animation
ani1 = animation.FuncAnimation(fig, animate, interval=200, cache_frame_data=False)
plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------------------------------------------
#   End of file
#------------------------------------------------------------------------------------------------------------------