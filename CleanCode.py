import os
import numpy as np
import pandas as pd
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline
FileDs = r'C:\Users\jhean\OneDrive\Documentos\Tareas\Inteligencia artificial\DataSets\workout_fitness_tracker_data.csv'
data = pd.read_csv(FileDs, delimiter=',')
data["Workout Intensity"] = data["Workout Intensity"].map({"Low": 1, "Medium": 2, "High": 3})
X = data.drop(columns=["User ID", "Calories Burned", "Gender", "Workout Type", "Mood Before Workout", "Mood After Workout", "Body Fat (%)", "VO2 Max", "Water Intake (liters)"]).values
y = data["Calories Burned"].values
m = y.size
data
column_names = ["Age", "Height(cm)", "Weight(kg)", "Workout Duration(mins)", "Heart Rate(bpm)", "Steps Taken", "Distance(km)", "Workout Intensity", "Sleep Hours", "Daily Calories Intake", "Resting Heart Rate(bpm)"]
dataclean = pd.DataFrame(X, columns=column_names)
dataclean
print(y)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)
dataclean_normalized = pd.DataFrame(X_normalized, columns=column_names)
dataclean_normalized
X_intercept = np.c_[np.ones((X_normalized.shape[0], 1)), X_normalized]
dataclean_intercept = pd.DataFrame(X_intercept, columns=["Intercept"] + column_names)
dataclean_intercept