#------------------------------------------------------------------------------------------------------------------
#   Mobile sensor data acquisition and processing
#   This code was used for every single obj to extract the data and putted in Data folder 
#------------------------------------------------------------------------------------------------------------------
import pickle
import numpy as np
from scipy import stats

# Load data
file_name = 'Programa Final\Objects\Juan2_06_01_2025_18_27_16.obj'
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
        feat.append(np.max(sig))
        feat.append(np.min(sig))
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
np.savetxt("Programa Final\Data\Data_Juan2.txt", processed_data)

#------------------------------------------------------------------------------------------------------------------
#   End of file
#------------------------------------------------------------------------------------------------------------------