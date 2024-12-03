import os
import numpy as np

def check_dir(path):
  if not os.path.isdir(path):
    os.makedirs(path)

def load_plot_config(use_latex):
  import matplotlib
  import matplotlib.pyplot as plt
  if use_latex:
    plt.rcParams['text.usetex'] = True
  # Configure figure
  #     matplotlib.rc('font', **{'family': 'Helvetica'})
  SMALL_SIZE = 18
  MEDIUM_SIZE = 18
  BIGGER_SIZE = 18
  plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
  plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
  plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
  plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
  plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
  plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
  plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
  plt.rc('xtick', direction='out')
  plt.rc('ytick', direction='out')
  plt.rc('xtick', top=False)
  plt.rc('ytick', right=False)
  plt.rc('axes', labelpad=5)
  plt.rc('axes.spines', right=False)
  plt.rc('axes.spines', top=False)
  plt.rc('patch', facecolor='None')
  plt.rc('axes', facecolor='None')
  plt.rc('axes', linewidth=2)
  plt.rc('ytick.major', size=10)
  plt.rc('xtick.major', width=1.5)
  plt.rc('xtick.major', size=10)
  plt.rc('ytick.major', width=1.5)

def moving_average(values, window_size):
  # numpy.convolve uses zero for initial missing values, so is not suitable.
  numerator = np.nancumsum(values)
  # The sum of the last window_size values.
  numerator[window_size:] = numerator[window_size:] - numerator[:-window_size]
  denominator = np.ones(len(values)) * window_size
  denominator[:window_size] = np.arange(1, window_size + 1)
  smoothed = numerator / denominator
  assert values.shape == smoothed.shape
  return smoothed
