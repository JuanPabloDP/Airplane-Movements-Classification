import pickle
import pandas as pd
import numpy as np
from scipy import stats
import os

# Archivos a procesar
file_names = [
    "Camila1_06_01_2025_16_54_25",
    "Camila2_06_01_2025_18_16_10",
    "Gonzalo1_06_01_2025_17_27_54",
    "Gonzalo2_06_01_2025_18_41_13",
    "Juan1_06_01_2025_17_09_27",
    "Juan2_06_01_2025_18_27_16"
]

base_path = "Programa Final\Objects"
all_features = []

for file in file_names:
    with open(os.path.join(base_path, file + ".obj"), 'rb') as inputFile:
        experiment_data = pickle.load(inputFile)

        for tr in experiment_data:
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
            feat.append(np.sqrt(rms))
            all_features.append(feat)

# Convertir a arreglo y guardar
processed_data = np.array(all_features)
np.savetxt("Programa Final\Data\Data1.txt", processed_data)