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
n = X_intercept.shape[1]
theta = np.zeros(n)
def hypothesis(X, theta):
    return X.dot(theta)
def compute_cost(X, y, theta):
    m = len(y)
    h = hypothesis(X, theta)
    error = h - y
    costo = (1/(2*m)) * np.dot(error, error)
    return costo
def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    cost_history = []
    for i in range(num_iters):
        h = hypothesis(X, theta)
        error = h - y
        gradient = (1/m) * np.dot(X.T, error)
        theta = theta - alpha * gradient
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)
    return theta, cost_history
alpha = 0.01
num_iters = 1000
theta_final, cost_history = gradient_descent(X_intercept, y, theta, alpha, num_iters)
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.plot(range(num_iters), cost_history, color='blue')
plt.title("Convergencia del Costo durante el Descenso por Gradiente")
plt.xlabel("Número de Iteraciones")
plt.ylabel("Costo")
plt.show()