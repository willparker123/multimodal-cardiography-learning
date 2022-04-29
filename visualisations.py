from venv import create
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from helpers import create_new_folder
import pandas as pd
import numpy as np
import os
from multiprocessing import Pool
from functools import partial

def histogram(data, bins, title="Histogram", resultspath="results/histograms/"):
    if len(data) == 0:
        return None
    try:
        data = data.numpy().squeeze()
    except:
        if len(np.shape(data)) > 1:
          data = np.squeeze(data)
        else:
          pass
    create_new_folder(resultspath)
    num_of_points = len(data)
    num_of_bins = bins
    fig, ax = plt.subplots()
    hist = ax.hist(data, bins=num_of_bins, edgecolor='black', alpha=0.3, weights=np.array(np.ones(len(data)) / len(data)))
    ax.set_title(title)
    ax.set_xlabel("X axis")
    ax.set_ylabel(f'% of samples')
    ax.set_xlim(np.min(data), np.max(data))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))
    fig.savefig(resultspath+f'histogram_{title.lower()}_n={num_of_points}_bins={bins}.png')
    plt.show()
    return hist