'''
Many of these import statements entail to the jupyter notebook in which I am calling this script
Some of the ar enot necessary for the function definitions in this python file.
'''
import os
import sys
import time
import math
import numpy as np
import qutip as qt
import pandas as pd
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
import importlib
from importlib import reload
from IPython.display import clear_output
import pickle
import copy

plotStyle = 'p' # 'p' for positive 'n' for negative
AxesColor = 'w' if plotStyle == 'n' else 'k'

if __name__ == "__main__":
	# Define plotting parameters
	fontsizes=15

	# plt.style.use("dark_background")
	plt.style.use('default')

	plot_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

	font = {'family' : 'sans-serif',
	    'variant':'normal',#'small-caps',
	        'size'   : fontsizes} # ,        'sans-serif' : [ 'Arial ']}
	# 'Helvetica','Tahoma',
	axes= {'titlesize' : fontsizes,
	    'labelsize' : fontsizes ,
	    'xmargin' : .00   ,
	    'ymargin' : .05,
	}

	legend = {'fontsize' : fontsizes}

	xtick = {'direction':'in', 
	        'bottom':True,
	        'top':True,
	        'color':AxesColor}
	ytick = {'direction':'in', 
	        'left':True,
	        'right':True,
	        'color':AxesColor}
	             
	matplotlib.rcParams['lines.linewidth'] = 2
	# Update the colors of the axis labels and legend
	matplotlib.rcParams['axes.labelcolor'] = AxesColor  # Set the color of axis labels
	matplotlib.rcParams['legend.labelcolor'] = AxesColor  # Set the background color of the legend
	matplotlib.rcParams['axes.titlecolor'] = AxesColor # Set the color of the axistitle
	# matplotlib.rcParams['legend.facecolor'] = 'none' 

	matplotlib.rc('font', **font)
	matplotlib.rc('axes', **axes)
	matplotlib.rc('legend', **legend)
	matplotlib.rc('xtick', **xtick)
	matplotlib.rc('ytick', **ytick)