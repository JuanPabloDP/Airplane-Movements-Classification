#------------------------------------------------------------------------------------------------------------------
#   Mobile sensor data acquisition and processing
#------------------------------------------------------------------------------------------------------------------
import pickle
import numpy as np
from scipy import stats

# Load data
file_name = 'activity_data.obj'
inputFile = open(file_name, 'rb')
experiment_data = pickle.load(inputFile)

# Process each trial and build data matrices
features = []
for tr in experiment_data:
    
    # For each signal (one signal per axis)
    feat = [tr[1]]
    rms = 0
    for s in range(tr[2].shape[1]):
        sig = tr[2][:,s]

        feat.append(np.average(sig))
        feat.append(np.std(sig))
        feat.append(stats.kurtosis(sig))
        feat.append(stats.skew(sig))
        rms += np.sum(sig**2)
        
    rms = np.sqrt(rms)    
    feat.append(rms)
    
    features.append(feat)      

# Build x and y arrays
processed_data =  np.array(features)
x = processed_data[:,1:]
y = processed_data[:,0]

# Save processed data
np.savetxt("activity_data.txt", processed_data)

#------------------------------------------------------------------------------------------------------------------
#   End of file
#------------------------------------------------------------------------------------------------------------------