from sklearn.preprocessing import LabelEncoder
import os
import torch
import numpy as np
#from kymatio.torch import Scattering1D
#from kymatio.torch import Scattering2D
import re
import pandas as pd
from openpyxl import load_workbook

from collections import Counter
import pandas as pd
import pickle
from optuna_MLP import MLP, optuna_objective
from helper_functions import generate_datasets, count_labels

#Directories
data_dir = r"/content/drive/MyDrive/Tesis_code/Adultos/Data/1D_WS_features"
results_dir = '/content/drive/MyDrive/Tesis_code/Adultos/Results/DL_WS_all'
os.makedirs(results_dir, exist_ok=True)

data_files = [
    f for f in os.listdir(data_dir)
    if os.path.isfile(os.path.join(data_dir, f)) and f.endswith(".npz")
]

labels = [1,0,-1,-1,0,1,-1,0,1,1,0,-1,0,1,-1]
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)


X_train_all = []
X_test_all = []
y_train_all = []
y_test_all = []
#Load all the dataset
for idx, file in enumerate(data_files):

  print(f"Sesion {idx}, {file}")
  path = os.path.join(data_dir, file)
  loaded = np.load(path, allow_pickle=True)

  X_train = loaded['data'][0:9]
  y_train = np.array(labels[0:9])
  X_test = loaded['data'][9:15]
  y_test = np.array(labels[9:15])

  X_train, X_test, y_train, y_test = generate_datasets(X_train, X_test, y_train, y_test)

  X_train_all.append(X_train)
  X_test_all.append(X_test)
  y_train_all.append(y_train)
  y_test_all.append(y_test)


#Stackon first dimension
X_train = np.vstack(X_train_all)
X_test = np.vstack(X_test_all)
y_train = np.concatenate(y_train_all, axis=0)
y_test = np.concatenate(y_test_all, axis=0)

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

#Count labels
count_labels(y_train)
count_labels(y_test)
