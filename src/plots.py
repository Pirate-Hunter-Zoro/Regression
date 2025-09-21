import matplotlib.pyplot as plt
import numpy as np
import os

def plot_curve(x_values: np.ndarray, y_values: np.ndarray, xlabel: str, ylabel: str, title: str, savepath: str):
    """Plot the curve with the given values and labels

    Args:
        x_values (np.ndarray): x-values
        y_values (np.ndarray): y-values
        xlabel (str): label for x axis
        ylabel (str): label for y axis
        title (str): title for graph
        savepath (str): place where graph will be saved
    """
    plt.scatter(x_values, y_values)
    plt.plot(x_values, y_values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath)
    plt.close()