### Import packages
# Run the file containing the general settings
# exec(open('helper_scripts/Settings.py').read())

# save_folder = r'/Users/sloenen/Dropbox/TaminiauLab/documents/Sjoerd/000_Thesis/06_Chapter_6_SiC_Outlook/01_Simulation_Figs'
# data_publish_folder = r'/Users/sloenen/Dropbox/TaminiauLab/documents/Sjoerd/000_Thesis/06_Chapter_6_SiC_Outlook/04_Data_Published'

# Import fitting functionalities
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

from scipy import constants
from scipy.stats import poisson

from IPython import display
from IPython.display import clear_output

from mpl_toolkits.axes_grid1 import make_axes_locatable

import dill as pickle
pickle.settings['recurse'] = True

plotStyle = 'p' # 'p' for positive 'n' for negative
AxesColor = 'w' if plotStyle == 'n' else 'k'

### Import general parameters
# sys.path.append(os.path.abspath('../../../../')) # Points to the directory that contains the analysis and notebooks folders
param_dict = importlib.import_module('01_scripts.SiV_cluster_params'); importlib.reload(param_dict)
# sys.path.append(r'/Users/sloenen/Documents/00_Github_code/01_PhD')
# from analysis.notebooks.SiV_in_SiC.SiV_cluster.scripts import SiV_cluster_params as param_dict
p = param_dict.params



'''
Unit conversion lists
'''
Pos_unit_list = ['nm', 'um', 'mm', 'm']
Pos_Conv_Fac = [1e-9, 1e-6, 1e-3, 1]

Freq_unit_list = ['Hz', 'kHz', 'MHz', 'GHz']
Freq_Conv_Fac = [1, 1e3, 1e6, 1e9]

Time_unit_list = ['ns', 'us', 'ms', 's']
Time_Conv_Fac = [1e-9, 1e-6, 1e-3, 1]

B_unit_list = ['mT', 'T']
B_Conv_Fac = [1e-3, 1]

FreqGrad_unit_list = ['kHz/nm']
FreqGrad_Conv_Fac = [1e3/1e-9]

Conc_unit_list = ['cm^-3', 'm^-3']
Conc_Conv_Fac = [1e6, 1]

I_0 = 1

Freq_unit_MM_SL = 'MHz'
Pos_unit_MM_SL = 'um'
FreqGrad_unit_MM_SL = 'kHz/nm'
B_unit_MM_SL = 'mT'

def data_to_publish_saver(save_str, data_dictionary):
    # Save data
    save_str = save_str + '.xlsx'
    excel_file_path = os.path.join(data_publish_folder, save_str)
    df = pd.DataFrame.from_dict(data_dictionary, orient='index')
    df = df.transpose()
    df.to_excel(excel_file_path, index = False)

def data_dict_to_pkl(dict_to_save, DataName):
    FolderPath ='/Users/sloenen/Documents/00_PhD/Github_Code/Thesis_SJHLoenen/01_notebooks/06_Chapter_6/01_data_figures' 
    DataSavePath = os.path.join(FolderPath,DataName)
    
    DataFile = open(DataSavePath, 'wb')
    pickle.dump(dict_to_save, DataFile)
    DataFile.close()

### convinience functions
def print_params():
    for key in p:
        print(key, ': ' , p[key])

'''
electron-electron coupling functions
'''
def calculate_delta_positions(coordlist):
    '''
    Input:
    - coordlist     =    np.array(size=(N,4))   : N indicates the spin number. 4 is for x, y, z and r (=sqrt(x**2 + y**2 + z**2) coordinates of that spin.
    
    Output:
    Output is a list of 3 arrays; dx, dy, dz
    - dx            =   np.array(size=(N,N))    : NxN array of the difference in x-position between spin number k and m. If k=m, we set the difference to 1 meter.
                                                  Element (k,m) is i_m - i_k (x-position of spin k - x-position of spin m). Element (m,k) is i_m - i_k
    '''

    dx_matrix = []
    dy_matrix = []
    dz_matrix = []
    for i, coord1 in enumerate(coordlist):
        dx_row = []
        dy_row = []
        dz_row = []
        for j, coord2 in enumerate(coordlist):
            if i == j:
                dx_row.append(1.)
                dy_row.append(1.)
                dz_row.append(1.)
            else:
                dx = (coord2[0] - coord1[0])
                dy = (coord2[1] - coord1[1])
                dz = (coord2[2] - coord1[2])
                dx_row.append(dx)
                dy_row.append(dy)
                dz_row.append(dz)
        dx_matrix.append(dx_row)
        dy_matrix.append(dy_row)
        dz_matrix.append(dz_row)

    dx_array = np.array(dx_matrix)
    dy_array = np.array(dy_matrix)
    dz_array = np.array(dz_matrix)

    return [dx_array, dy_array, dz_array]

def compute_electronic_interactions(coordlist_data, print_progress=False):
    """
    Computes relevant ZZ, ZX and ZY dipolar electron-electron interactions from the coordlist_data list, which contains N_cluster instances of 4xN_spins coordinates (x,y,z,r). 

    Input: 
    - coordlist_data    = list(len=N_clusters)  : A list of length "N_clusters". Each coordlist_data element is a Nx4 list that signifies a spin cluster. Here N is the number of spins in the cluster and 4 is for x, y, z and r (=sqrt(x**2 + y**2 + z**2) coordinates of that spin.

    Output:
    Output is a list of 3 lists; J_zz_data, J_zx_data, J_zy_data. 
    - J_zz_data         = list(len=N_clusters)  : Element "i" contains an NxN array with all the ZZ-couplings between all the spins in the cluster corresponding element "i".
    """

    # Initialize data structures
    N_clusters = len(coordlist_data)

    J_zz_data = [None]*N_clusters
    J_zx_data = [None]*N_clusters
    J_zy_data = [None]*N_clusters

    for i in range(N_clusters):

        # execute calculations
        [J_zz_array, J_zx_array, J_zy_array] = calculate_full_coupling_matrix(coordlist_data[i], gamma = p['gamma_e'])
        J_zz_data[i] = J_zz_array
        J_zx_data[i] = J_zx_array
        J_zy_data[i] = J_zy_array
        
        if(print_progress):
            clear_output(wait=True)
            print('[',i+1,'/',N_clusters,']')

    return [J_zz_data, J_zx_data, J_zy_data]

def calculate_full_coupling_matrix(coordlist, gamma = p['gamma_c'], return_dpos = False ):
    '''
    Calculates the ZZ, ZX and ZY couplings between the spins defined by the spin positions contained in coordlist.

    Input:
    - coordlist     = np.array(size=(N,4))      : a Nx4 array. N indicates the number of spins in a cluster. 4 is for x, y, z and r (=sqrt(x**2 + y**2 + z**2) coordinates of that spin.
    - gamma         = float                     : Indicates what the gyromagnetic ratio is of the spins between which the coupling is calculated

    Output:
    Output is a list of 3 arrays; Czz, Czx, Czy
    - Czz: NxN array of Czz couplings between all spin pairs
    - Czx: NxN array of Czx couplings between all spin pairs
    - Czy: NxN array of Czy couplings between all spin pairs
    '''

    # init variables
    mu0 = p['mu0']   #np.pi * 4e-7 #     # H/m
    hbar =   p['hbar'] #1.0545718e-34 #      # J/(rad/s)

    [dx_array, dy_array, dz_array] = calculate_delta_positions(coordlist)

    r2_array = dx_array**2 + dy_array**2 + dz_array**2
   
    # dipole tensor
    alpha = mu0*gamma**2*hbar/(4*np.pi)

    theta = np.arccos( dz_array/np.sqrt(r2_array) )
    phi   = np.arctan2( dy_array, dx_array)  # set phi to zero if r2_array_xy is zero

    C_zz_array  =      (alpha/np.sqrt(r2_array)**3) * (3*np.cos(theta)**2 - 1)    / (4*np.pi)  # Hz!!!
    C_perp_array =  3 * (alpha/np.sqrt(r2_array)**3) * np.cos(theta)*np.sin(theta) / (4*np.pi)
    C_zx_array   = C_perp_array * np.cos(phi)
    C_zy_array   = C_perp_array * np.sin(phi)  

    if return_dpos == True:
        return [C_zz_array, C_zx_array, C_zy_array, dx_array, dy_array, dz_array]
    else:     
        return [C_zz_array, C_zx_array, C_zy_array]


'''
Helper functions
'''
# Flatten a list of lists to one list with one number per element
def flatten_lists(the_lists):
    '''
    Flatten a list of lists. 

    Input:
    - the_lists     = list(list) : a list containing sub-lists.

    Output:
    - result        = list       : a list where the elements are the concatenated/flattened elements of the sub-lists in "the_lists".
    '''
    result = []
    extend = result.extend
    for _list in the_lists:
        extend(_list)
    return result


'''
Plotting functions
'''
def Hist_plot(ax, data, bins, xlbl, ylbl, c1, lbl1, xscale = 'linear', Scnd_Hist = False, data2 = None, ylbl2 = None, c2 = 'g', lbl2 = None, vline = False, vline_lbl = None):
    '''
    Function to generate a histogram plot in an already axisting figure handle. It accepts a potential second histogram that shares the same y-axis

    Input:
    - ax        = figure handle : axis in which the histogram should be drawn
    - data(2)   = np.array()    : Array of data to be drawn in a histogram
    - bins      = np.array()    : Defines the bins in the histogram
    - xlbl      = string        : x-axis label
    - ylbl1(2)  = string        : y-axis label of y-axis 1 or 2
    - c1(2)     = string        : color of data in corresponding to y-axis 1 or 2
    - lbl1(2)   = string        : legend label of data corresponding to y-axis 1 or 2
    - xscale    = string        : defines the scale of the x-axis. Typically linear or log
    - Scnd_Hist = bool          : Defines whether or not to plot a second histrogram, that will correspond to the second y-axis

    Output:
    - ax        = figure handle : figure handle with a histogram/histograms plotted in it.
    '''

    [y_vals1, _, _] = ax.hist(data, bins, color = c1, alpha = 0.5, label = lbl1)
    ax.set_xscale(xscale)
    ax.set_ylim([0,1.1*y_vals1.max()])
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    
    if vline != False:
        ax.axvline(x = vline, c = 'k', label = vline_lbl)
    
    if Scnd_Hist == True:
        ax2 = ax.twinx()
        [y_vals2, _, _] = ax2.hist(data2, bins, color = c2, alpha = 0.5, label = lbl2)
        ax2.set_ylim([0,1.1*y_vals2.max()])
        ax2.set_ylabel(ylbl2)
        
        # color the left and right axis of the figure according to the data color
        ax.spines['left'].set_color(c1)
        ax.yaxis.label.set_color(c1)
        ax.tick_params(axis='y', colors=c1)
        
        ax2.spines['left'].set_color(c1)
        ax2.spines['right'].set_color(c2)
        ax2.yaxis.label.set_color(c2)
        ax2.tick_params(axis='y', colors=c2)

        # Place legend
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc = 'upper right', prop={'size':7})
        
    else:
        ax.legend()
            

def Scatter_broken_y_plot(fig, ax, x, y, threshold, x_lbl, y_lbl, cb_vals, cb_lbl, y_scale = 'symlog'):
    '''
    Makes a scatterplot of the data in x and y where part of the y-axis can be removed. The scatterplot includes a colorbar that can tell more information about the x,y points than just their x- and y-correlation

    Input:
    fig         = figure        : figure in which to plot the broken scatter plot
    ax          = figure handle : axes handle in which to plot the broken scatter plot
    x           = np.array      : x-data of the broken scatter plot
    y           = np.array      : y-data of the broken scatter plot
    threshold   = float         : y-data below this threshold is not shown. This threshold breaks the scatter plot
    x_lbl       = string        : x-axis label
    y_lbl       = string        : y-axis label
    cb_vals     = np.array      : Array containing floating points that signify colorvalues. The value at index i corresponds to the datapoint referring to x[i] and y[i]. colors are displayed in a colorbar
    cb_lbl      = string        : colorbar label
    y_scale     = string        : definition of how to scale the y-axis.

    Output:
    broken scatter plot.
    '''
    divider = make_axes_locatable(ax)
    ax2 = divider.new_vertical(size="100%", pad=0.1)
    fig.add_axes(ax2)
    
    sc1 = ax.scatter(x, y, s = 1, c = cb_vals, cmap = 'magma')
    ax.axhline(y = -threshold, linestyle = '-', c = 'g')
    
    sc1 = ax2.scatter(x, y, s = 1, c = cb_vals, cmap = 'magma')
    ax2.axhline(y = threshold, linestyle = '-', c = 'g')
    
    fig.colorbar(sc1, ax = ax, label = cb_lbl)
    
    ax.set_yscale('symlog')
    ax2.set_yscale('symlog')

    yticks_lst = list(ax.get_yticks())
    yticks_add = [-threshold, threshold]
    [yticks_lst.append( yticks_add[k] ) for k,val in enumerate(yticks_add) if val not in yticks_lst]
    
    ax.set_yticks(np.array(yticks_lst))
    ax.spines['top'].set_visible(False)
    ax.set_ylim(np.min(y), -0.8*threshold)
    ax.set_xlabel(x_lbl)
    ax.set_ylabel(y_lbl)
    
    ax2.set_yticks(np.array(yticks_lst))
    ax2.spines['bottom'].set_visible(False)
    ax2.set_ylim(0.6*threshold, np.max(y))
    ax2.tick_params(bottom=False, labelbottom=False)
    ax2.set_ylabel(y_lbl)
    
    # From https://matplotlib.org/examples/pylab_examples/broken_axis.html
    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False) # top axes
    ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    
    kwargs.update(transform=ax2.transAxes)        # top axes
    ax2.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal


'''
Coherence time functions
'''
def T2star_calculator(Coupl_Arr, Coupl_Threshold = 0.5, Dist_Threshold_IdxLst = 'all'):
    '''
    Function to calculate the T2star of all spins in a spin cluster based on their couplings to other spins

    Input:
    - Coupl_Arr                 = np.array(size=(NxN))  : NxN array of (ZZ) couplings between all different N-spins. On the diagonal it has 0s that indicate the coupling of a spin to itself
    - Coupl_Threshold           = float                 : If two spins couple stronger than this threshold (so if T2* * pi * coupling > Coupl_Threshold), then erase that spin from how the T2* is calculated. It is considered a strongly coupled spin
    - Dist_Threshold_IdxLst     = list                  : List of indices of spins that are located close enough to the origin to be taken into account in the T2* calculation. You want ot avoid spins on the edges as they see less spins than spins in the middle of the pillar. Hence edge-spins will get an artificially lower T2*.
    
    Output: 
    - T2star_mean   =   float   : Mean value of T2* over all spins closer than a thershold to the origin. T2* = 2/(2pi*sqrt(sum(A_par**2))).
    - T2star_std    =   float   : The std spread in T2* values. This is not the std of the mean.
    - P_FewSpinReg  =   float   : The probability that the calculated T2* originates from a few spin cluster. In that case you cannot really speak of a T2* effect.
    - P_StrCoup     =   float   : The probability that at least one of the spins in a spin cluster couples stronger than Coupl_Threshold. Few spin clusters are excluded.
    '''
    # Sort and take out the first element, which is 0 as it is the coupling with the spin to itself. Only include spins according to the Dist_Threshold_IdxLst (spins closer to the origin than a threshold)
    # So define T2* based on all the couplings to spin 1 in a cluster
    if Dist_Threshold_IdxLst != 'all':
        Coupl_Arr_AbsSrt = np.sort( np.abs(Coupl_Arr) )[Dist_Threshold_IdxLst,1::]
    else:
        Coupl_Arr_AbsSrt = np.sort( np.abs(Coupl_Arr) )[:,1::] 
    T2star = 2./(2*np.pi*np.sqrt( np.sum(Coupl_Arr_AbsSrt**2, axis = 1) ) )
    
    Regime_Matrix = T2star[:,None] * Coupl_Arr_AbsSrt * np.pi
    Regime_mask = np.zeros(Regime_Matrix.shape, dtype = bool)
    
    # Take out couplings that couple stronger than T2star, defined by "Regime_Matrix > Coupl_threshold"
    Regime_Special_Idx = np.where(Regime_Matrix > Coupl_Threshold)
    Regime_mask[ Regime_Special_Idx[0], Regime_Special_Idx[1] ] = True
    
    # "Regime_Matrix > Coupl_threshold" also possible because of a few spin regime so that all spins couple > Coupl_threshold. Calculate the probability that you have a few spin cluster
    FewSpinRegime_bool = np.all(Regime_mask, axis = 1)
    FewSpinRegime_Idx = np.where(FewSpinRegime_bool == True)[0]
    P_FewSpinReg = float(len(FewSpinRegime_Idx))/np.shape(Coupl_Arr_AbsSrt)[0] if np.shape(Coupl_Arr_AbsSrt)[0] != 0 else None
    
    # Get the coupling matrix where the couplings stronger than "Coupl_threshold" are masked (and thus not taken into account), and where the spin clusters with just a few spins are taken out.
    Coupl_Arr_masked = np.ma.masked_where(Regime_mask, Coupl_Arr_AbsSrt)
    T2star = 2./(2*np.pi*np.sqrt( np.sum(Coupl_Arr_masked**2, axis = 1) ) )
    T2star_mean = np.mean(T2star)
    T2star_std = np.std(T2star)
    
    # Calculate what the probability is that there was a coupling stronger than T2star
    MultiSpinRegime_Idx = np.where(FewSpinRegime_bool == False)[0]
    MultiSpinClusters_mask = Regime_mask[MultiSpinRegime_Idx,:]
    N_StrCoupl_bool = np.any(MultiSpinClusters_mask, axis = 1)
    N_StrCoupl_Idx = np.where(N_StrCoupl_bool == True)[0]
    P_StrCoupl = float(len(N_StrCoupl_Idx))/np.shape(Coupl_Arr_AbsSrt)[0] if np.shape(Coupl_Arr_AbsSrt)[0] != 0 else None
    
    return T2star_mean, T2star_std, P_FewSpinReg, P_StrCoupl    
   

'''
Sampling distribution functions
'''
def sampling_distribution(ImpVol_dict, ShowPlot = False):
    '''
    Generate a list that contains radial distances in the XY-plane according to a distribution defined by "Sampling". The Haar measure is taken into account

    The list is generated accoring to the following Monte Carlo algorithm:
    Step 1: To simulate wheter a datapoint with a certain radius should be contained in the dataset, 
            draw a random number within the range of r-values you want to take into account in the simulation (x = np.random.uniform(0,max_r,size=1))
    Step 2: Calculate the probability that the radius of that datatpoint should occur (Prob(x))
    Step 3: Draw a random number between 0 and (at least) the maximum probability of the distribution of radii (y = np.random.uniform( argmax(prob(x) ))
    Step 4: Compare the number of step 3 to the number of step 2. If lower or equal, then accept the number, if higher reject.
    The number in step 3 can be chosen higher than y_max_sampling, but that will reduce the amount of accepted datapoints in total while maintaining the same distribution.

    Input:
    - ImpVol_dict = dict    : Implantation Volumen Dictionary containing information on the implantation volume and sampling. It contains the implantation mask radius, the type of defect sampling in the xy plane and the implantation std in the z-direction.

    Output:
    - List_rxy_Vals     = list  : list containing radial distances in the XY plane according to the distribution defined by "Sampling"
    '''

    sigma_r = ImpVol_dict['mask_r']
    Sampling = ImpVol_dict['Sampling']

    # Define the sampling distributions
    if Sampling == 'Gaussian': 
        max_r = 3*sigma_r
        Normalization = sigma_r/np.sqrt(2.*np.pi)*(-np.exp(-sigma_r**2/(2.*sigma_r**2)) +1.)
        def Prob(r):
            return 1./Normalization * r *1/(sigma_r*np.sqrt(2*np.pi))*np.exp(-r**2/(2*sigma_r**2))
        y_max_sampling = Prob(sigma_r)*1.01 # Max of Prob(r) is at "r=sigma_r"
        
    if Sampling == 'uniform': 
        max_r = sigma_r
        Normalization = 1./2*sigma_r**2
        def Prob(r): 
            return 1./Normalization * r
        y_max_sampling = Prob(sigma_r)*1.01 # Max of Prob(r) is at "r=mask_r"
        
    # Start the Monte Carlo simulation    
    N_Samples_Tries = 50000
    List_rxy_Vals = []
    for Sample_try in range(N_Samples_Tries):
        x = np.random.uniform(0,max_r, size = 1)
        y = np.random.uniform(0,y_max_sampling)
        if y <= Prob(x):
            List_rxy_Vals.append(x[0])

    if ShowPlot == True:
        FigTst_, AxTst_ = plt.subplots(1)
        AxTst_.hist(List_rxy_Vals, 100, density = True)
        AxTst_.set_xlabel('r')
        AxTst_.set_ylabel('Normed occurance')
        AxTst_.set_title('Sampling distribution')

    return List_rxy_Vals

'''
Spin cluster generation functions
'''
def generate_spin_cluster_coordinates(N_spins_array, N_clusters, List_rxy_Vals, ImpVol_dict, plot_clusters):
    '''
    N_clusters are generated with N_spins in each cluster. For each cluster, the x, y, z and r values of each spin in that cluster are saved. For each cluster, the data is stored in the list "coordlist_data"
    This is done for multiple N_spins, defined by N_spins_array. The total data is stored in the list "Nspins_cluster_coord_list", where each element correspons to a number of spins in the cluster and its value is "coordslist_data"

    The algorithm used to generate coordlist_data and doordlists_data_list:
    Step 1: take a number of spins you want to have in your cluster
    Step 2: Generate a constellation of the amount of spins in step 1, and save the defect coordinates of this constellations in an list named "coordlist". This coordlist has dimensions 4xNspins. 4:x,y,z,r
    Step 3: Repeat step 2 "N_clusters" times and for every N_mask iteration, store the "coordlist" in "coordlist_data". "coordlist_data" will have dimensions N_clusters(x1)
    Step 4: perform from step 1 onwards with a different amount of spins in the cluster. For each amount of spins, save the "coordlist_data" in "Nspins_cluster_coord_list". This will have dimensions len(N_spins_array)(x1)
    
    Input:
    - N_spins_array     = np.array(int)    : array with integers that define the number of spins in a cluster that should be analyzed
    - N_clusters        = int              : Number of different cluster configurations. The larger this number the more representative the final result. However, it gets computationally more heavy
    - List_rxy_Vals     = list             : list containing radial distances in the XY plane according to the distribution defined by "Sampling". For each spin the r-distance is sampled from this list
    - ImpVol_dict       = dict             : Dictionary containing information on the implantation volume and sampling. It contains the implantation mask radius, the type of defect sampling in the xy plane and the implantation std in the z-direction.
    - plot_clusters     = bool             : Wheter or not to plot all the positions of the spins within the cluster

    Output:
    - Nspins_cluster_coord_list   = list[ sub_list[ sub_sub_list[] ] ]: sub_sub_list contains a list of spin coordinates for one particular cluster. sub_list is a list of length N_cluster. list contains sub_list for each different amount of N_spins
    '''
    sigma_z = ImpVol_dict['sigma_z']

    Nspins_cluster_coord_list = []

    if plot_clusters:
        fig, ax = plt.subplots(len(N_spins_array),2, figsize = [6,1.5 * len(N_spins_array)], sharex=True)


    for i, N_spins_in in enumerate(tqdm(N_spins_array)):

        coordlist_data = []
        for j in range(N_clusters):

            coordlist = []  
            
            rxy_array = np.array(List_rxy_Vals)[ np.random.randint(0,len(List_rxy_Vals)-1, N_spins_in) ]
            phi_array = np.random.uniform(0,2*np.pi, N_spins_in)

            x_array = rxy_array*np.cos(phi_array)
            y_array = rxy_array*np.sin(phi_array)
            z_array = np.random.uniform(-sigma_z, sigma_z, size = N_spins_in)

            r_array = np.sqrt(x_array**2 + y_array**2 + z_array**2)
            N_spins = len(z_array)

            coordlist = np.concatenate([x_array, y_array, z_array, r_array]).reshape( (4,N_spins) ).transpose()
            
            if plot_clusters:
                ax[i,0].scatter(x_array*1e9, z_array*1e9, marker = '.')
                ax[i,1].scatter(x_array*1e9, y_array*1e9, marker = '.')

                ax[i,0].set_ylim(0,200)
                ax[i,1].set_ylim(-100,100)
                ax[i,0].set_xlim(-100,100)

                ax[i,0].set_ylabel("z (nm)")
                ax[i,1].set_ylabel("y (nm)")
                ax[i,0].set_title(str(N_spins_array[i]) + " defects")
                ax[i,1].set_title(str(N_spins_array[i]) + " defects")


            coordlist_data.append(coordlist)

        Nspins_cluster_coord_list.append(coordlist_data)

    if plot_clusters:    
        ax[-1,0].set_xlabel("x (nm)")
        ax[-1,1].set_xlabel("x (nm)")
        plt.tight_layout()
        plt.show()

    return Nspins_cluster_coord_list

'''
Coupling parameter functions
'''
def calculate_Z_couplings(Nspins_cluster_coord_list, N_spins_array):
    '''
    Based Nspins_cluster_coord_list the ZZ spin couplings and their mean,max and median are calculated per Nspins in a cluster
    
    Input:
    - Nspins_cluster_coord_list     = list[] sub_list[ sub_sub_list ] ] ]   : sub_sub_list is N_spinsx4 array of x,y,z,r spin coordinates in one cluster of N_spins. sub_list contains the sub_sub_list, for all clusters of one N_spins. list contains sub_list for all N_spins.
    - N_spins_array                 = np.array(int)    : array with integers that define the number of spins in a cluster that should be analyzed

    Output:
    - J_zz_data_list                = list[] sub_list[ sub_sub_array ] ] ]               : sub_sub_array is N_spinsxN_spins array of Jzz couplings between all spin pairs in one cluster of N_spins. sub_list contains the sub_sub_list for all clusters of one N_spins. list contains sub_list for all N_spins.    
    - J_mean_array                  = np.array(size=( len(N_spins_array), N_clusters) ) : Each element is the mean of the absolute values of all the couplings between the spins in a specific instance of a cluster
    - J_max_array                   = np.array(size=( len(N_spins_array), N_clusters) ) : Each element is the max of the absolute values of all the couplings between the spins in a specific instance of a cluster
    - J_median_array                = np.array(size=( len(N_spins_array), N_clusters) ) : Each element is the median of the absolute values of all the couplings between the spins in a specific instance of a cluster
    '''

    ## Calculate typical (mean) and maximum couplings that can be expected

    # Calculate the coupling matrices for all different spin cluster configurations. So different configurations for different total number of spins.
    J_zz_data_list = [None]*len(N_spins_array)
    N_clusters = len(Nspins_cluster_coord_list[0])

    # Generate arrays to save 1-number statistical properties of a cluster
    J_mean_array = np.zeros((len(N_spins_array), N_clusters))
    J_max_array = np.zeros((len(N_spins_array), N_clusters))
    J_median_array = np.zeros((len(N_spins_array), N_clusters))

    for i, N_spins_in in enumerate(tqdm(N_spins_array)):
        [J_zz_data, J_zx_data, J_zy_data] = compute_electronic_interactions(Nspins_cluster_coord_list[i], print_progress = False)
        J_zz_data_list[i] = J_zz_data    

        for j in range(N_clusters):
            # Only take statistical properties of upper right triangle (np.triu_indices), which contain all the 1/2*N(N-1) couplings. Exclude diagonal terms (k=1) that are meaningless.
            J_mean_array[i,j] = np.mean( np.abs( J_zz_data[j][np.triu_indices(N_spins_in, k = 1)] ) )  # take absolute values
            J_max_array[i,j] = np.max( np.abs( J_zz_data[j][np.triu_indices(N_spins_in, k = 1)] ) )  # take absolute values
            J_median_array[i,j] = np.median( np.abs( J_zz_data[j][np.triu_indices(N_spins_in, k = 1)] ) )  # take absolute values

    # Calculate 1 number statistical properties averaged (std/median) over all clusters with the same total number of spins
    J_mean_mean_array = np.mean(J_mean_array, axis = 1)
    J_mean_std_array = np.std(J_mean_array, axis = 1)

    J_max_mean_array = np.mean(J_max_array, axis = 1)
    J_max_std_array = np.std(J_max_array, axis = 1)

    J_max_median_array = np.median(J_max_array, axis = 1)
    J_median_median_array = np.median(J_median_array, axis = 1)

    return J_zz_data_list, J_mean_array, J_max_array, J_median_array

def cluster_KPIs(Nspins_cluster_coord_list, J_zz_data_list, J_max_array, N_spins_array, T2star_StrCpl_Threshold, J_threshold, ImpVol_dict, save_data_dict):
    # Calculate the probability of finding at least one coupling above the threshold
    '''
    Definition:
    "few spin cluster": a cluster where all spins couple stronger than T2star_StrCpl_Threshold * pi * 1/T2*, with T2* calculated as in T2star_calculator. In that case the calculation that leads to T2* is not valid and hence one cannot really speak about a T2*.
    Function that, for different numbers of spins in a cluster, 
    - calculates the average and std T2*. This already eliminated clusters that contained that are few spin clusters.
    - calculates the electron spin concentration within the mask radius (mask radius is contained in ImpVol_dict)
    - calculates the probability that the cluster is a few spin cluster. 
    - calculates the probability that there is a spin that couples more strongly than T2*
    - calculates the probability that there is a coupling stronger than the coupling threshold
    - calculates whether there are couplings larger than the set threshold. 
        For clusters that have at least one coupling stronger than the threshold, the function stores 
            - All the couplings in the cluster, 
            - The couplings larger than the threshold
            - The couplings lower than the threshold
            - The positions of the spins that couple stornger than the threshold
            - The delta x, y, z, r of the spins that couple stronger than the threshold

    Input: 
    - Nspins_cluster_coord_list     = list[] sub_list[ sub_sub_list ] ] ]               : sub_sub_list is N_spinsx4 array of x,y,z,r spin coordinates in one cluster of N_spins. sub_list contains the sub_sub_list, for all clusters of one N_spins. list contains sub_list for all N_spins.
    - J_zz_data_list                = list[] sub_list[ sub_sub_list ] ] ]               : sub_sub_list is N_spinsxN_spins array of Jzz couplings between all spin pairs in one cluster of N_spins. sub_list contains the sub_sub_list for all clusters of one N_spins. list contains sub_list for all N_spins.    
    - J_max_array                   = np.array(size=( len(N_spins_array), N_clusters) ) : Each element is the max of the absolute values of all the couplings between the spins in a specific instance of a cluster
    - N_spins_array                 = np.array(int)                                     : array with integers that define the number of spins in a cluster that should be analyzed
    - T2star_trCpl_Threshold        = float                                             : Indicates above which value relative to "pi * T2*" couplings are considered as strong and hence not taken into account in the T2* calculation
    - J_Threshold                   = float                                             : The threshold above which a coupling is considerd strongly coupled. Clusters that contain such a coupling are further analyzed in terms of the spin positions of the spins involved in strong couplings, as well as what all other couplings are
    - ImpVol_dict                   = dict                                              : Implantation Volumen Dictionary containing information on the implantation volume and sampling. It contains the implantation mask radius, the type of defect sampling in the xy plane and the implantation std in the z-direction.
    - save_data_dict                = bool                                              : Boolean indicating whether or not to save the data dictionary generated

    Output:
    - data_dict                     = dict                                              : Dictionary containing the calculated data as described in the function description

    '''

    mask_r = ImpVol_dict['mask_r']
    sigma_z = ImpVol_dict['sigma_z']
    Volume = np.pi*mask_r**2*2*sigma_z
    Dist_Threshold = np.sqrt(3)*np.min([mask_r, sigma_z])

    N_clusters = len(J_zz_data_list[0])
    P_threshold = np.zeros(len(N_spins_array))
    for i in range(len(N_spins_array)): # Calculate what the probability is that at least one coupling in the cluster is above the coupling threshold
        P_threshold[i] = np.sum(J_max_array[i]>J_threshold)/len(J_max_array[i])

    # Initialize lists that store relevant parameters. Each list element pertains to a spin cluster with a specific spin number
    Init_arr = [None]*len(N_spins_array)
    data_dict = {}
    data_dict['Coupl_Thres_Plus_list'], data_dict['Coupl_Thres_Min_list'], data_dict['Coupl_Thres_Tot_list'], data_dict['Coupl_Tot_list'] = Init_arr.copy(), Init_arr.copy(), Init_arr.copy(), Init_arr.copy()
    data_dict['N_spins_J_strong_list'], data_dict['N_spins_J_strong_std_list'] = Init_arr.copy(), Init_arr.copy()
    data_dict['x_array_Plus_list'], data_dict['y_array_Plus_list'], data_dict['z_array_Plus_list'] = Init_arr.copy(), Init_arr.copy(), Init_arr.copy()
    data_dict['dx_array_Plus_list'], data_dict['dy_array_Plus_list'], data_dict['dz_array_Plus_list'], data_dict['dr_array_Plus_list'] = Init_arr.copy(), Init_arr.copy(), Init_arr.copy(), Init_arr.copy()
    data_dict['dx_array_Min_list'], data_dict['dy_array_Min_list'], data_dict['dz_array_Min_list'], data_dict['dr_array_Min_list'] = Init_arr.copy(), Init_arr.copy(), Init_arr.copy(), Init_arr.copy()
    data_dict['T2star_mean_list'], data_dict['T2star_std_list'] = Init_arr.copy(), Init_arr.copy()
    data_dict['Concentration_mean_list'], data_dict['Concentration_std_list'] = Init_arr.copy(), Init_arr.copy()
    data_dict['P_FewSpinReg_list'], data_dict['P_StrCoupl_list'] = Init_arr.copy(), Init_arr.copy()
    data_dict['P_threshold'] = np.zeros(len(N_spins_array))

        
    for i, N_spins_in in enumerate(tqdm(N_spins_array)):
        Coupl_Thres_Plus, Coupl_Thres_Min, Coupl_Thres_Tot, Coupl_Tot = [], [], [], []
        x_array_Plus, y_array_Plus, z_array_Plus, N_spins_J_strong = [], [], [], []
        dx_array_Plus, dy_array_Plus, dz_array_Plus, dr_array_Plus = [], [], [], []
        dx_array_Min, dy_array_Min, dz_array_Min, dr_array_Min = [], [], [], []
        T2star_mean_sublst, T2star_std_sublst, P_FewSpinReg_sublst, P_StrCoupl_sublst = [], [], [], []
        N_spins_in_Volume = [] 

        for j in range(N_clusters):  
            Coupl_Arr = J_zz_data_list[i][j]
            
            # Calculate the mean T2star in a cluster, the std of the T2star, the probability to find a significantly strongly coupled spin and the probability that the cluster consists only of a few spins.
            Dist_Threshold_IdxLst = 'all' if Dist_Threshold == 'inf' else np.where(Nspins_cluster_coord_list[i][j][:,3] < Dist_Threshold)[0]
            T2star_mean, T2star_std, P_FewSpinReg, P_StrCoupl = T2star_calculator(Coupl_Arr, Coupl_Threshold = T2star_StrCpl_Threshold, Dist_Threshold_IdxLst = Dist_Threshold_IdxLst)
            if np.ma.is_masked(T2star_mean) == False: # Check whether the T2* is actually a sensible number, not just a masked value since for this N_mask instance it was a few spin cluster which thus had an undefined T2*.
                T2star_mean_sublst.append(T2star_mean)
                T2star_std_sublst.append(T2star_std)
            if P_FewSpinReg != None: # If there were no spins within the Dist_Threshold to be taken into account in this calculation
                P_FewSpinReg_sublst.append(P_FewSpinReg)
                P_StrCoupl_sublst.append(P_StrCoupl)

            # Calculate the concetration of spins within a cylinder defined by the 2*sigma_z en the implantation mask radius.
            N_spins_in_cylinder = len( np.where( np.sqrt( Nspins_cluster_coord_list[i][j][:,0]**2+Nspins_cluster_coord_list[i][j][:,1]**2 ) < mask_r )[0] ) 
            N_spins_in_Volume.append(N_spins_in_cylinder)

                    
            # Get indices of lower triangles (including diagonal) to remove those indices to prevent double counting and prevent 2-spin parameters between the same spins
            [tril1_idx, tril2_idx] = np.tril_indices_from(Coupl_Arr)
            Tril_Idx = [ tril1_idx[n]*len(Coupl_Arr)+tril2_idx[n] for n in range(len(tril1_idx))]
            Coupl_Tot.append( np.delete(Coupl_Arr, Tril_Idx) )
            
            if J_max_array[i][j] > J_threshold: # Check if the cluster contains a coupling stronger than the J_threshold
                # Get indices of spin pairs that couple stronger than threshold. Then convert index of square matrix format to 1D array
                [spin1_idx, spin2_idx] = np.where( np.triu( np.abs(Coupl_Arr), k=1) > J_threshold) # Get the spin pairs within a specific cluster that couple stronger than the threshold            
                SpinsList_Idx = [ spin1_idx[n]*len(Coupl_Arr)+spin2_idx[n] for n in range(len(spin1_idx))]
                SpinsDeleteList_Idx = SpinsList_Idx + Tril_Idx
                
                            
                # Save couplings above and below threshold separately
                Coupl_Thres_Plus.append( Coupl_Arr.flatten()[SpinsList_Idx] )
                Coupl_Thres_Min.append( np.delete(Coupl_Arr, SpinsDeleteList_Idx) )
                Coupl_Thres_Tot.append( np.delete(Coupl_Arr, Tril_Idx) ) # SJH note
                
                # Save the positions of the spins that couple stronger than the theshold
                SpinNr_Plus_Unique = np.unique( np.concatenate((spin1_idx,spin2_idx)) )
                N_spins_J_strong.append( len(SpinNr_Plus_Unique) )
                coord_array_Plus = Nspins_cluster_coord_list[i][j][SpinNr_Plus_Unique]
                x_array_Plus.append( coord_array_Plus[:,0] )
                y_array_Plus.append( coord_array_Plus[:,1] )
                z_array_Plus.append( coord_array_Plus[:,2] )
                
                # Save delta positions of spin that couple stronger (lower) than threshold separately
                [dx_array, dy_array, dz_array] = calculate_delta_positions( Nspins_cluster_coord_list[i][j] )
                dr_array = np.sqrt( dx_array**2 + dy_array**2 + dz_array**2 )
                
                dx_array_Plus.append( dx_array.flatten()[SpinsList_Idx] )
                dy_array_Plus.append( dy_array.flatten()[SpinsList_Idx] )
                dz_array_Plus.append( dz_array.flatten()[SpinsList_Idx] )
                dr_array_Plus.append( dr_array.flatten()[SpinsList_Idx] )
                
                dx_array_Min.append( np.delete(dx_array, SpinsDeleteList_Idx) )
                dy_array_Min.append( np.delete(dy_array, SpinsDeleteList_Idx) )
                dz_array_Min.append( np.delete(dz_array, SpinsDeleteList_Idx) )
                dr_array_Min.append( np.delete(dr_array, SpinsDeleteList_Idx) )
        
        # Flatten the lists. If there is more than 1 coupling stronger than the threshold, the generated lists are lists of lists
        Coupl_Thres_Plus = flatten_lists(Coupl_Thres_Plus)
        Coupl_Thres_Min = flatten_lists(Coupl_Thres_Min)
        Coupl_Thres_Tot = flatten_lists(Coupl_Thres_Tot) 
        Coupl_Tot = flatten_lists(Coupl_Tot)
        x_array_Plus = flatten_lists(x_array_Plus)
        y_array_Plus = flatten_lists(y_array_Plus)
        z_array_Plus = flatten_lists(z_array_Plus)
        dx_array_Plus = flatten_lists(dx_array_Plus)
        dy_array_Plus = flatten_lists(dy_array_Plus)
        dz_array_Plus = flatten_lists(dz_array_Plus)
        dr_array_Plus = flatten_lists(dr_array_Plus)
        dx_array_Min = flatten_lists(dx_array_Min)
        dy_array_Min = flatten_lists(dy_array_Min)
        dz_array_Min = flatten_lists(dz_array_Min)
        dr_array_Min = flatten_lists(dr_array_Min)
                    
        # print('SJH: Coupl_Thres_Plus = ', Coupl_Thres_Plus)       
        # Store the parameters in their lists
        data_dict['Coupl_Thres_Plus_list'][i] = Coupl_Thres_Plus
        # print('SJH: data_dict[Coupl_Thres_Plus_list][i] = ', data_dict['Coupl_Thres_Plus_list'][i])
        data_dict['Coupl_Thres_Min_list'][i] = Coupl_Thres_Min
        data_dict['Coupl_Thres_Tot_list'][i] = Coupl_Thres_Tot
        data_dict['Coupl_Tot_list'][i] = Coupl_Tot
        data_dict['N_spins_J_strong_list'][i] = np.mean(N_spins_J_strong)
        data_dict['N_spins_J_strong_std_list'][i] = np.std(N_spins_J_strong)
        
        data_dict['x_array_Plus_list'][i] = x_array_Plus
        data_dict['y_array_Plus_list'][i] = y_array_Plus
        data_dict['z_array_Plus_list'][i] = z_array_Plus
        
        data_dict['dx_array_Plus_list'][i] = dx_array_Plus
        data_dict['dy_array_Plus_list'][i] = dy_array_Plus
        data_dict['dz_array_Plus_list'][i] = dz_array_Plus
        data_dict['dr_array_Plus_list'][i] = dr_array_Plus
        
        data_dict['dx_array_Min_list'][i] = dx_array_Min
        data_dict['dy_array_Min_list'][i] = dy_array_Min
        data_dict['dz_array_Min_list'][i] = dz_array_Min
        data_dict['dr_array_Min_list'][i] = dr_array_Min
        
        data_dict['T2star_mean_list'][i] = np.mean(T2star_mean_sublst)
        data_dict['T2star_std_list'][i] = np.mean(T2star_std_sublst)
        data_dict['P_FewSpinReg_list'][i] = np.mean(P_FewSpinReg_sublst)
        data_dict['P_StrCoupl_list'][i] = np.mean(P_StrCoupl_sublst)
        
        data_dict['P_threshold'][i] = np.sum(J_max_array[i]>J_threshold)/len(J_max_array[i])

        data_dict['Concentration_mean_list'][i] = np.mean(N_spins_in_Volume)/Volume
        data_dict['Concentration_std_list'][i] = np.std(N_spins_in_Volume)/Volume

    if save_data_dict == True:
        print('I am saving your data_dict. If the cluster size is large (>2000), then this can take me a while')
        DataDir = 'data_dict_dir/'
        DataFile_Name = 'Diameter_' + str(np.round(2*mask_r*1e9).astype(int)) + '_nm_Jthreshold_' + str(int(J_threshold*1e-3))  + 'kHz_Nclusters_'+str(N_clusters)+'.npz'
        np.savez(DataDir + DataFile_Name, data_dict = data_dict)

    return data_dict


def DataPlotter(N_spins_array, data_dict, J_threshold, ImpVol_dict, Number_Spins_Show = 5):
    '''
    Function to plot the following data:
    - T2* as a function of the number of electron spins created in a cluster
    - The spin concentration as a function of the number of electron spins created in the cluster. The cluster size is defined by the mask radius and the spread in the z-position during implantation
    - For different number of electron spins created in a cluster, it plots the probability that there is at least one coupling stronger than J_threshold in a cluster 
    - For a number of spins in a cluster equal to "Number_Spins_Show", it shows the distribution of coupling strengths
    - For a number of spins in a cluster equal to "Number_Spins_Show", it shows the distribution of coupling strengths for clusters conditioned on having at least one strongly coupled spin pair
    - For a number of spins in a cluster equal to "Number_Spins_Show", it shows the x-y distribution of the electron spin positions
    - For a number of spins in a cluster equal to "Number_Spins_Show", it shows the distribution of delta_x values for spins that couple more strongly than the threshold, and the distribution of the coupling strengts per delta_x and corresponding theta (angle in (r_{xy})-z plane)
    - For a number of spins in a cluster equal to "Number_Spins_Show", it shows the distribution of delta_y values for spins that couple more strongly than the threshold, and the distribution of the coupling strengts per delta_y and corresponding theta (angle in (r_{xy})-z plane)
    - For a number of spins in a cluster equal to "Number_Spins_Show", it shows the distribution of delta_z values for spins that couple more strongly than the threshold, and the distribution of the coupling strengts per delta_z and corresponding theta (angle in (r_{xy})-z plane)
    - For a number of spins in a cluster equal to "Number_Spins_Show", it shows the distribution of delta_r values for spins that couple more strongly than the threshold, and the distribution of the coupling strengts per delta_r and corresponding theta (angle in (r_{xy})-z plane)
    - For a number of spins in a cluster equal to "Number_Spins_Show", it shows the distribution of delta_r_{xy} values for spins that couple more strongly than the threshold, and the distribution of the coupling strengts per delta_r_{xy} and corresponding theta (angle in (r_{xy})-z plane)

    Input: 
    - N_spins_array         = np.array()    : Number of spins in cluster that have been analysed
    - data_dict             = dict          : dictionary containing, per number of spins in a cluster, data on e.g. coupling strenghts and probability to have at least one strongly coupled spin pair.
    - J_threshold           = float         : The threshold above which a coupling is defined as strong
    - ImpVol_dict           = dict          : Implantation Volumen Dictionary containing information on the implantation volume and sampling. It contains the implantation mask radius, the type of defect sampling in the xy plane and the implantation std in the z-direction.
    - Number_Spins_Show     = int           : the number of spins in a cluster for which to show the conditioned distribution of coupling strengths

    Output:
    - A figure with the plots as described above
    '''

    plot_TotJzzDist = True
    Freq_unit = 'kHz'
    Pos_unit = 'nm'
    Time_unit = 'us'
    Concentration_unit = 'cm^-3'

    Pos_Conv = Pos_Conv_Fac[ Pos_unit_list.index(Pos_unit) ]
    Freq_Conv = Freq_Conv_Fac[ Freq_unit_list.index(Freq_unit) ] 
    Time_Conv = Time_Conv_Fac[ Time_unit_list.index(Time_unit) ]
    Conc_Conv = Conc_Conv_Fac[ Conc_unit_list.index(Concentration_unit) ]

    if Number_Spins_Show not in N_spins_array:
        N_spins_new = np.abs(N_spins_array - Number_Spins_Show); N_spins_min = np.where( N_spins_new == np.min(N_spins_new) )[0][0]
        raise ValueError('\n%i spins is not in N_spins_array. Choose a different number for "Number_Spins_Show". \nE.g choose %i, it is closest to your choice of %i spins.' % (Number_Spins_Show, N_spins_array[N_spins_min],Number_Spins_Show))
    Idx = np.where(N_spins_array == Number_Spins_Show)[0][0]


    # Convert threshold
    J_threshold_Conv = J_threshold/Freq_Conv

    # Convert times
    T2star_mean_Conv = np.array(data_dict['T2star_mean_list'])/Time_Conv
    T2star_std_Conv = np.array(data_dict['T2star_std_list'])/Time_Conv

    # Convert concentrations
    Concentration_mean_Conv = np.array(data_dict['Concentration_mean_list'])/Conc_Conv
    Concentration_std_Conv = np.array(data_dict['Concentration_std_list'])/Conc_Conv

    # Convert couplings
    Coupl_Thres_Min_Conv = np.array(data_dict['Coupl_Thres_Min_list'][Idx])/Freq_Conv
    Coupl_Thres_Plus_Conv = np.array(data_dict['Coupl_Thres_Plus_list'][Idx])/Freq_Conv
    Coupl_Thres_Tot_Conv = np.array(data_dict['Coupl_Thres_Tot_list'][Idx])/Freq_Conv
    Coupl_Tot_Conv = np.array(data_dict['Coupl_Tot_list'][Idx])/Freq_Conv

    # Convert distances
    xpos = np.array(data_dict['x_array_Plus_list'][Idx])/Pos_Conv
    ypos = np.array(data_dict['y_array_Plus_list'][Idx])/Pos_Conv
    mask_r = ImpVol_dict['mask_r']
    mask_rad = mask_r/Pos_Conv

    dx_arr = np.array(data_dict['dx_array_Plus_list'][Idx])/Pos_Conv
    dy_arr = np.array(data_dict['dy_array_Plus_list'][Idx])/Pos_Conv
    dz_arr = np.array(data_dict['dz_array_Plus_list'][Idx])/Pos_Conv
    dr_arr = np.array(data_dict['dr_array_Plus_list'][Idx])/Pos_Conv
    drxy_arr = np.sqrt( dx_arr**2 + dy_arr**2 )
    theta_arr = 0.5 - np.arctan2(dz_arr,drxy_arr)*1./np.pi 

    xscale = 'log'
    y_lbl_hst = 'occurance'
    c_hst = 'g'
    y_lbl = 'J (' + Freq_unit + ')'
    cb_lbl = r'$\Theta$ (pi rad)'

    vline = J_threshold_Conv
    vline_lbl = 'J threshold'
    Coupl_bins = np.logspace(np.log10(1/Freq_Conv), np.log10(int(np.max(Coupl_Thres_Tot_Conv)+1)), 100)



    '''
    START PART 0
    1) Figure of T2star as a function of number of defects
    2) Figure of probability to have defect above threshold vs. number of defects
    3) Optional: figure of total J_zz distribution
    '''

    xmin = N_spins_array[0]-0.2
    xmax = N_spins_array[-1]+0.2
    # Plot of T2*
    fig_T2star, ax_T2star = plt.subplots(1, figsize = (7,2.5))
    ax_T2star.errorbar(N_spins_array,T2star_mean_Conv, T2star_std_Conv, color = 'k')
    ax_T2star.grid()
    ax_T2star.set_xlabel('Number of defects')
    ax_T2star.set_ylabel('$T_2^*$ (' + Time_unit +')')
    ax_T2star.set_xlim([xmin, xmax])
    ax_T2star.set_yscale('log')
    plt.tight_layout()
    plt.show()

    # Plot of spin concentration
    Conc_scaling = 1e15
    fig_Conc, ax_Conc = plt.subplots(1, figsize = (7,2.5))
    ax_Conc.errorbar(N_spins_array,Concentration_mean_Conv/Conc_scaling, Concentration_std_Conv/Conc_scaling, color = 'k')
    ax_Conc.grid()
    ax_Conc.set_xlabel('Number of defects')
    ax_Conc.set_ylabel('spin concentration (' + format(1e15, '.0e') +'cm-3)')
    ax_Conc.set_xlim([xmin, xmax])
    plt.tight_layout()
    plt.show()

    # Plot of probability to find a coupling above the coupling threshold
    fig_Pthres, ax_Pthres = plt.subplots(1, figsize = (7,2.5))

    cmap = ['k']*len(N_spins_array)
    cmap[ np.where(N_spins_array == Number_Spins_Show)[0][0] ] = 'g'

    ax_Pthres.scatter(N_spins_array, data_dict['P_threshold'] * 100, c = cmap)
    ax_Pthres.grid()
    ax_Pthres.set_axisbelow(True)
    ax_Pthres.set_xlabel("Number of defects created")
    ax_Pthres.set_ylabel("$P(J_{\mathrm{max}} > $" + str(int(J_threshold_Conv)) + " " + Freq_unit +") (%)")
    ax_Pthres.set_xlim([xmin,xmax])

    x_ticks = ax_Pthres.get_xticks()
    x_ticks = np.sort(np.append(x_ticks, Number_Spins_Show)) if Number_Spins_Show not in x_ticks else x_ticks
    ax_Pthres.set_xticks(x_ticks)
    x_ticks_idx = np.where(x_ticks == Number_Spins_Show)[0][0]
    ax_Pthres.xaxis.get_ticklabels()[x_ticks_idx].set_color('g')

    plt.tight_layout()
    plt.show()

    # Plot of the total distribution of couplings
    if plot_TotJzzDist == True:
        fig_Dist, ax_Dist = plt.subplots(1, figsize = (7,2.5))

        data = Coupl_Tot_Conv
        x_lbl = '$J_{zz}$ Coupling ('+Freq_unit+')'
        y_lbl = y_lbl_hst
        lbl = '$J_{zz}$'
        Hist_plot(ax_Dist, data, Coupl_bins, x_lbl, y_lbl, 'k', lbl, xscale)
        
        plt.tight_layout()
        plt.show()
    '''
    END PART 0
    '''


    fig, ax = plt.subplots(6,2, figsize = (7,15))
    '''
    START PART 1
    Below the couplings will be plot as well as a scatterplot of the locations of the defects
    '''
    data1 = Coupl_Thres_Min_Conv
    data2 = Coupl_Thres_Plus_Conv
    xlbl = '$J_{zz}$ Coupling ('+Freq_unit+')'
    ylbl1 = y_lbl_hst
    ylbl2 = y_lbl_hst
    lbl1 = 'J <= Threshold'
    lbl2 = 'J > Threshold'
    c1 = 'r'
    c2 = 'g'
    Hist_plot(ax[0,0], data1, Coupl_bins, xlbl, ylbl1, c1, lbl1, xscale, Scnd_Hist = True, data2 = data2, ylbl2 = ylbl2, c2 = c2, lbl2 = lbl2, vline = vline, vline_lbl = vline_lbl)

    scat_size = 1
    ax[0,1].scatter(xpos, ypos, marker = '.', s = scat_size)
    ax[0,1].add_patch(plt.Circle((0, 0), mask_rad, facecolor='#FF000011', edgecolor='red'))
    ax[0,1].set_xlim([-1.3*mask_rad, 1.3*mask_rad])
    ax[0,1].set_ylim([-1.3*mask_rad, 1.3*mask_rad])
    ax[0,1].set_xlabel('x (' + Pos_unit + ')')
    ax[0,1].set_ylabel('y (' + Pos_unit + ')')
    '''
    END PART 1
    '''

    '''
    START PART 2
    Histogram and scatterplot of delta_x and the spins that couple strongly 
    '''
    dx_bins = np.linspace(np.min(dx_arr)-1, np.max(dx_arr)+1,100)
    x_lbl = r'$\delta$x (' + Pos_unit + ')'
    lbl = r'$\delta$x for J > Threshold'
    Hist_plot(ax[1,0], dx_arr, dx_bins, x_lbl, y_lbl_hst, c_hst, lbl)

    x_lbl = r'$\delta$x (' + Pos_unit + ')'
    Scatter_broken_y_plot(fig, ax[1,1], dx_arr, Coupl_Thres_Plus_Conv, J_threshold_Conv, x_lbl, y_lbl, theta_arr, cb_lbl, y_scale = 'symlog')
    '''
    END PART 2
    '''

    '''
    START PART 3
    Histogram and scatterplot of delta_y and the spins that couple strongly 
    '''
    dy_bins = np.linspace(np.min(dy_arr)-1, np.max(dy_arr)+1,100)
    x_lbl = r'$\delta$y (' + Pos_unit + ')'
    lbl = r'$\delta$y for J > Threshold'
    Hist_plot(ax[2,0], dy_arr, dy_bins, x_lbl, y_lbl_hst, c_hst, lbl)

    x_lbl = r'$\delta$y (' + Pos_unit + ')'
    Scatter_broken_y_plot(fig, ax[2,1], dy_arr, Coupl_Thres_Plus_Conv, J_threshold_Conv, x_lbl, y_lbl, theta_arr, cb_lbl, y_scale = 'symlog')
    '''
    END PART 3
    '''

    '''
    START PART 4
    Histogram and scatterplot of delta_z and the spins that couple strongly 
    '''
    dz_bins = np.linspace(np.min(dz_arr)-1, np.max(dz_arr)+1,100)
    x_lbl = r'$\delta$z (' + Pos_unit + ')'
    lbl = r'$\delta$z for J > Threshold'
    Hist_plot(ax[3,0], dz_arr, dz_bins, x_lbl, y_lbl_hst, c_hst, lbl)

    x_lbl = r'$\delta$z (' + Pos_unit + ')'
    Scatter_broken_y_plot(fig, ax[3,1], dz_arr, Coupl_Thres_Plus_Conv, J_threshold_Conv, x_lbl, y_lbl, theta_arr, cb_lbl, y_scale = 'symlog')
    '''
    END PART 4
    '''

    '''
    START PART 5
    Histogram and scatterplot of delta_r and the spins that couple strongly 
    '''
    dr_bins = np.linspace(np.min(dr_arr)-1, np.max(dr_arr)+1,100)
    x_lbl = r'$\delta$r (' + Pos_unit + ')'
    lbl = r'$\delta$r for J > Threshold'
    Hist_plot(ax[4,0], dr_arr, dr_bins, x_lbl, y_lbl_hst, c_hst, lbl)

    x_lbl = r'$\delta$r (' + Pos_unit + ')'
    Scatter_broken_y_plot(fig, ax[4,1], dr_arr, Coupl_Thres_Plus_Conv, J_threshold_Conv, x_lbl, y_lbl, theta_arr, cb_lbl, y_scale = 'symlog')
    '''
    END PART 5
    '''

    '''
    START PART 6
    Histogram and scatterplot of delta_rxy and the spins that couple strongly 
    '''
    drxy_bins = np.linspace(np.min(drxy_arr)-1, np.max(drxy_arr)+1,100)
    x_lbl = r'$\delta r_{xy}$ (' + Pos_unit + ')'
    lbl = r'$\delta r_{xy}$ for J > Threshold'
    Hist_plot(ax[5,0], dy_arr, dy_bins, x_lbl, y_lbl_hst, c_hst, lbl)

    x_lbl = r'$\delta r_{xy}$ (' + Pos_unit + ')'
    Scatter_broken_y_plot(fig, ax[5,1], drxy_arr, Coupl_Thres_Plus_Conv, J_threshold_Conv, x_lbl, y_lbl, theta_arr, cb_lbl, y_scale = 'symlog')
    '''
    END PART 6
    '''

    plt.tight_layout()
    plt.show()


def DoseAnalysis(dose_array, implantation_yield, ImpVol_dict, N_spins_array, data_dict, J_threshold, save_data):
    '''
    Analyse and plot what, for a certain implantation dose:
    - The mean number of defects in a cluster is
    - the probability that there is a coupling larger than J_threshold in a cluster
    - the number of unique spins with a coupling larger than J_threshold in a cluster, conditioned on having at least one coupling larger than J_threshold

    Input:
    - dose_array            = np.array()    : Array with implantation doses to be analysed
    - implantation_yield    = float         : Number of electron spins created per implanted ion
    - ImpVol_dict           = dict          : Implantation Volumen Dictionary containing information on the implantation volume and sampling. It contains the implantation mask radius, the type of defect sampling in the xy plane and the implantation std in the z-direction.
    - N_spins_array         = np.array()    : Number of spins in cluster that have been analysed
    - data_dict             = dict          : dictionary containing, per number of spins in a cluster, data on e.g. coupling strenghts and probability to have at least one strongly coupled spin pair.
    - J_threshold           = float         : The threshold above which a coupling is defined as strong
    - save_data             = bool          : boolean to indicate whether or not data should be saved

    Output:
    - A figure with the plots as described above
    '''

    mask_r = ImpVol_dict['mask_r']
    mask_area = np.pi * mask_r**2

    N_ions_array = dose_array * mask_area
    mean_defects_array = N_ions_array * implantation_yield

    P_threshold_dose = np.zeros(len(dose_array))

    P_threshold = data_dict['P_threshold']
    N_spins_J_strong_list = data_dict['N_spins_J_strong_list']
    N_spins_J_strong_std_list = data_dict['N_spins_J_strong_std_list']

    N_spins_array01 = np.concatenate((np.array([0,1]), N_spins_array))
    P_threshold01 = np.concatenate((np.array([0,0]), P_threshold)) # Include 0 spins and 1 spin --> Will never have a strong coupling

    N_spins_J_strong_list_dose = np.zeros(len(dose_array))
    N_spins_J_strong_std_list_dose = np.zeros(len(dose_array))
    for i in range(len(dose_array)):
        P_N_defects = poisson.pmf(N_spins_array01, mean_defects_array[i])
        P_threshold_dose[i] = np.dot(P_N_defects, P_threshold01)
        
        N_spins_J_strong_list_dose[i] = np.dot(P_N_defects[2::], N_spins_J_strong_list)/np.sum(P_N_defects[2::]) # Exclude the 0 and 1 spin case, hence you need to renormalize
        N_spins_J_strong_std_list_dose[i] = np.dot(P_N_defects[2::], N_spins_J_strong_std_list)/np.sum(P_N_defects[2::]) # Exclude the 0 and 1 spin case, hence you need to renormalize
        
        
        if np.sum(P_N_defects) < 0.9:
            print('WATCH OUT: at dose %3.1fe11/cm^2 P_threshold_dose is underestimated. Namely np.sum(P_N_defects) = %3.2f' %(dose_array[i]* 1e-4 * 1e-11,np.sum(P_N_defects)) )
        
    fig, ax = plt.subplots(3, figsize = (7,9))
    ax[0].plot(dose_array * 1e-4*1e-11, mean_defects_array)
    ax[0].set_ylabel("Mean $N_{defects}$")
    ax[0].set_xlabel("Dose (1e11 cm $^{-2}$)")
    ax[0].grid()
    # ax[0].axhline(y = Number_Spins_Show, c = 'g', linestyle = '--')
    ax[1].plot(dose_array * 1e-4 * 1e-11, P_threshold_dose * 100)
    ax[1].set_xlabel("Dose (1e11 cm $^{-2}$)")
    ax[1].grid()
    ax[1].set_ylabel("$P(J_{\mathrm{max}} > $" + str(int(J_threshold*1e-3)) + "kHz) (%)")

    ax[2].errorbar(dose_array * 1e-4 * 1e-11, N_spins_J_strong_list_dose, N_spins_J_strong_std_list_dose)
    ax[2].set_xlabel("Dose (1e11 cm $^{-2}$)")
    ax[2].grid()
    ylabel2 = "$N_{J>%3.0fkHz}$" %(J_threshold*1e-3) 
    ax[2].set_ylabel(ylabel2)
    ax[2].set_title("conditioned on at least one coupling larger than " + str(int(J_threshold*1e-3)) + "kHz")

    ax[0].set_title("Mask diameter = " + str(np.round(2*mask_r*1e9)) + " nm")


    plt.tight_layout()

    if save_data == True:
        DataDir = 'Implantation_Parameters/'
        DataFile_Name = 'Diameter_' + str(np.round(2*mask_r*1e9).astype(int)) + '_nm_Jthreshold_' + str(int(J_threshold*1e-3))  + 'kHz.npz'
        np.savez(DataDir + DataFile_Name,
                dose_array_norm = dose_array * 1e-4*1e-11,
                mean_defects_array = mean_defects_array,
                P_threshold_dose_norm = P_threshold_dose * 100,
                N_spins_J_strong_list_dose_norm = N_spins_J_strong_list_dose,
                N_spins_J_strong_std_list_dose_norm = N_spins_J_strong_std_list_dose)


def CombinedDataPlotter(diameters_to_plot, J_threshold_to_plot):
    '''
    Function to plot combined data for different implantation parameters and a certain coupling thershold to be investigated
    More concretely, what is plotted:
    - ax 1: mean number of defects created as a function of implantation dose for a certain implantation mask diameter
    - ax 2: Probability to find at least one coupling above J_threshold_to_plot in a spin cluster, as a function of the implantation dose for a certain implantation mask diameter
    - ax 3: The average number of defects to investigate before finding a defect with a coupling stronger than J_threshold_to_plot. Defined as 1 over the probability to find a strongly coupled spin in a spin cluster, multiplied by the ratio of total spins over the amount of unique spins that couple stronger than J_threshold_to_plot
    - ax 4: Number of unique spins that have a coupling stronger than J_threshold_to_plot in clusters conditioned on having at least one coupling larger than J_treshold. This is plotted as a function of the implantation dose and for a certain implantation mask diameter

    Input:
    - diameters_to_plot     = np.array()    : Array indicating for which implantation mask diameter you want to plot the data
    - J_threshold_to_plot   = float         : Threshold above which the coupling you are interested in should be

    Output:
    4 plots with the information as described in the function description
    - CombinedDataDict      = dict          : A dictionary containing all the plotted data
    '''
    DataDir = 'Implantation_Parameters/'
    Jthreshold_plot = int(J_threshold_to_plot*1e-3)
    DataName = [0]*len(diameters_to_plot)

    CombinedDataDict = {}
    DoseDict = {}
    NDefectsDict = {}
    PcouplDict = {}
    NcouplDict = {}
    NcouplStdDict = {}

    fig_comb, ax_comb = plt.subplots(4, figsize = (7,9))
    for i, d in enumerate(diameters_to_plot):
        DataDir = 'Implantation_Parameters/'
        DataFile_Name = 'Diameter_'+str(d)+'_nm_Jthreshold_'+str(Jthreshold_plot)+'kHz.npz'
        DataName[i] = 'Data'+str(d)+'nm'
        CombinedDataDict[DataName[i]] = np.load(DataDir+DataFile_Name)
        
        DoseDict[d] = CombinedDataDict[DataName[i]]['dose_array_norm']
        NDefectsDict[d] = CombinedDataDict[DataName[i]]['mean_defects_array']
        PcouplDict[d] = CombinedDataDict[DataName[i]]['P_threshold_dose_norm']
        NcouplDict[d] = CombinedDataDict[DataName[i]]['N_spins_J_strong_list_dose_norm']
        NcouplStdDict[d] = CombinedDataDict[DataName[i]]['N_spins_J_strong_std_list_dose_norm']
        
        ax_comb[0].plot(DoseDict[d], NDefectsDict[d], label = str(d)+'nm')
        ax_comb[1].plot(DoseDict[d], PcouplDict[d], label = str(d)+'nm')
        N_average_measure = 1./(PcouplDict[d]/100) * (NDefectsDict[d]/NcouplDict[d])
        ax_comb[2].plot(DoseDict[d], N_average_measure, label = str(d)+'nm') # division by 100 to convert the probability from % to a number between 0 and 1
        ax_comb[3].errorbar(DoseDict[d], NcouplDict[d], NcouplStdDict[d], label = str(d)+'nm')

    ax_comb[0].legend(); ax_comb[0].grid(); ax_comb[0].set_ylabel("Mean number of defects")
    ax_comb[1].legend(); ax_comb[1].grid(); ax_comb[1].set_ylabel("$P(J_{\mathrm{max}}) > $" + str(np.round(Jthreshold_plot)) + "kHz (%)")
    ax_comb[2].legend(); ax_comb[2].grid(); ax_comb[2].set_ylabel("N measure before finding J>%3.0fkHz" %(Jthreshold_plot))
    ax_comb[3].legend(); ax_comb[3].grid(); ax_comb[3].set_ylabel("$N_{J>%3.0fkHz}$" %(Jthreshold_plot) )
    ax_comb[1].set_xlabel("Dose (1e11 cm $^{-2}$)")
    plt.tight_layout()

    return CombinedDataDict


'''
Magnetic field functions
'''
def compute_Bx(x, z , M_es, w, h, c): # c: center point of the micromagnet. Useful if you want to simulate multiple micromagnets
    '''
    Equations derived from Eq (20) for Bx from: https://iopscience.iop.org/article/10.1088/0022-3727/39/9/003/pdf. Note my factor 2 definition difference in width and height of the micromagnet
    
    Input:
    - x     = float        : x-position where the Bx-field is to be calculated relative to the center of the micromagnet
    - z     = float        : z-position where the Bx-field is to be calculated relative to the bottom of the micromagnet
    - M_es  = float        : Volume magnetisation of the micromagnet material
    - w     = float        : Width of the micromagnet
    - h     = float        : height/thickness of the micromagnet
    - c     = float        : center position of the micromagnet in the x-direction

    Output:
    - Bx    = float        : Bx-field at position (x,z)
    '''

    w2 = 0.5*w
    h2 = 0.5*h
    Bx = constants.mu_0 * M_es/(4*np.pi) * ( np.log( ( (x-c+w2)**2 + (z-h2)**2 ) /  ( (x-c+w2)**2 + (z)**2 ) ) - np.log( ( (x-c-w2)**2 + (z-h2)**2 ) / ( (x-c-w2)**2 + (z)**2 ) ) )
    return Bx

def compute_Bz(x, z , M_es, w, h, c):
    '''
    Equations derived from Eq (21) for By (y in this paper is z in our coordinate system) from: https://iopscience.iop.org/article/10.1088/0022-3727/39/9/003/pdf. Note my factor 2 definition difference in width and height of the micromagnet

    Input:
    - x     = float        : x-position where the Bz-field is to be calculated relative to the center of the micromagnet
    - z     = float        : z-position where the Bz-field is to be calculated relative to the bottom of the micromagnet
    - M_es  = float        : Volume magnetisation of the micromagnet material
    - w     = float        : Width of the micromagnet
    - h     = float        : height/thickness of the micromagnet
    - c     = float        : center position of the micromagnet in the x-direction

    Output:
    - Bz    = float        : Bz-field at position (x,z)
    '''

    w2 = 0.5*w
    h2 = 0.5*h 
    Bz = constants.mu_0 * M_es/(2*np.pi) * ( np.arctan(2*h2*(x-c+w2)/( (x-c+w2)**2 +(z-h2)**2 - h2**2 ) ) - np.arctan( 2*h2*(x-c-w2)/ ( (x-c-w2)**2 + (z-h2)**2 - h2**2 ) ) )
    return Bz

def compute_B_stripline(x, z, I_0, w, c, N_pts = 100):
    '''
    Magnetic field funtion based on integral form of Maxwell's equations. We assume the stripline is built up from "N_pts"
    1D wires that guide the total current I_0. 
    In this way the total B field is the sum of the B-field created by the N_pts wires and each wire creates a field equal to
    B_wire * 2*pi*r = mu_0 * J * dx.
    Here J is the current density, which equals the total current divided by the width (w) of the stripline.
    r is the distance from the center of the wire.

    This approach is allowed as long as r >> dx.

    Input:
    - x         = float     : x-position where the B-field of the stripline is to be calculated
    - z         = float     : z-position where the B-field of the stripline is to be calculated
    - I_0       = float     : current send through the stripline
    - w         = float     : width of the stripline
    - c         = float     : center position of the stripline
    - N_pts     = float     : Number of points use to discretise the stripline into wires. This number should preferably be large

    Output:
    - B_list    = list      : List containing the B_x and B_z field generated by the stripline at position (x,z)
    '''

    J = I_0 / w    # Current density (W/m)
    dx = w / N_pts   # Width of 1 stripline element
    
    stripline_x = np.linspace(-w/2,w/2,N_pts) + c
    
    B_x = 0.
    B_z = 0.
    for i, x_pt in enumerate(stripline_x):
        r = np.sqrt( (x - x_pt)**2 + z**2 )
        theta = np.arctan2( z,(x-x_pt))
        B_rad = constants.mu_0 * J * dx / (2*np.pi * r)
        B_x += B_rad * np.sin(theta)
        B_z += B_rad * -np.cos(theta)

    B_list = [B_x, B_z]
    return B_list

def Bx_field_depth_NV_current_1A(depth_NV, d_SIL_NV, w_stripline_NV):
    '''
    Function to calculate the Bx field created on a spin (here NV-center) at a depth "depth_NV" under the (diamond) surface embedded in a SIL of diameter d_SIL_NV to which a stripline of width w_stripline_NV borders that conducts a current of 1A

    Input:
    - depth_NV          = np.array()    : Array contaning different depths of the NV relative to the (diamond) surface
    - d_SIL_NV          = float         : Diameter of the SIL in which the NV is centered
    - w_stripline_NV    = float         : Width of the stripline that borders the SIL, conducts a current of 1A and thereby creates the Bx field on the NV

    Output:
    A plot showing the Bx field at the NV as a function of the depth under the diamond surface for a specific SIL diameter and stripline width
    '''

    # We define all the coordinates relative to the center of the stripline
    x_NV = 0.5 * w_stripline_NV + 0.5 * d_SIL_NV
    z_NV = -depth_NV
    [B_x_NV, B_z_NV] = compute_B_stripline(x_NV, z_NV, I_0, w_stripline_NV, 0, 500)

    fig, ax = plt.subplots()
    ax.plot(depth_NV*1e6,B_x_NV*1e3)
    ax.set_xlabel(r'Depth NV ($\mu$m)')
    ax.set_ylabel('$B_x$ field (mT)')
    ax.set_title('$B_x$ field that we generate in regular NV samples relative to 1A current')
    plt.tight_layout()

def centers_calculation(NP_dict, MM_dict, SL_dict):
    '''
    Function that:
    - calculates the centers of the nanopillars, micromagnets and striplines. 

    Input:
    - NP_dict       = dict  : Dictionary containting data about the nanopillars (NP); number of NPs, spacing between NPs & width of NPs
    - MM_dict       = dict  : Dictionary containting data about the micromagnets (MM); spacing between the edges of the MM and the NP, the volume magnetisation (M_es) of the MM material, the width and the height of the MM and the number of MMs (1; single-, or 2; double-sided)
    - SL_dict       = dict  : Dictionary containting data about the stripline (SL); spacing between the edges of a MM and a SL, width of a SL, number of striplines (1; single-, or 2; double-sided) and spacing between the edges of striplines in case 2 are defined.

    Output (all elements in the dictionaries are in SI units, e.g. m, Hz, A/m):
    - NP_dict       = dict  : Compared to the dictionary that is in the input, the centers of the nanopillars are added
    - MM_dict       = dict  : Compared to the dictionary that is in the input, the centers of the micromagnets are added
    - SL_dict       = dict  : Compared to the dictionary that is in the input, the centers of the stripline are added

    '''
    # Nanopillar parameters
    N_NP = NP_dict['N_NP'] # Number of nanopillars
    Spacing_NP = NP_dict['Spacing_NP'] # Spacing between the edges of the nanopillars
    Width_NP = NP_dict['Width_NP'] # Width of a single nanopillar

    # Micromagnet parameters
    Spacing_MM_NP = MM_dict['Spacing_MM_NP'] # Spacing between the edges of the micromagnet and the nanopillar 
    M_es = MM_dict['M_es'] # Page 2 second reference
    w_MM = MM_dict['w_MM'] # Width of a single micromagnet
    h_MM = MM_dict['h_MM'] # Height of a single micromagnet
    Nr_MM = MM_dict['Nr_MM'] # Number of micromagnets: Only 1 and 2 are currently allowed

    # Stripline parameters
    Spacing_SL_MM = SL_dict['Spacing_SL_MM'] # Spacing between the edges of a micromagnet and a stripline
    w_SL = SL_dict['w_SL'] # Width of a single stripline
    Nr_SL = SL_dict['Nr_SL'] # Number of striplines: Only 1 and 2 are currently allowed
    Spacing_SL = SL_dict['Spacing_SL'] # Spacing between the edges of two striplines


    # Calculate centers of nanopillars
    if 'Cntrs' not in NP_dict:
        NanoPillar_Cntrs = np.zeros(N_NP)
        for i in range(N_NP):
            NanoPillar_Cntrs[i] = 0.5*w_MM + Spacing_MM_NP + 0.5*Width_NP + i*Width_NP + i*Spacing_NP
        NP_dict['Cntrs'] = NanoPillar_Cntrs

    # Calculate centers of the micromagnets
    if 'x_Cntrs' not in MM_dict:
        MicroMagnet_x_Cntrs = [0] if Nr_MM == 1 else [0, 2*(0.5*w_MM + Spacing_MM_NP) + N_NP*Width_NP + (N_NP-1)*Spacing_NP] # [0, 3.8e-6]
        MicroMagnet_x_Cntrs = [0] if Nr_MM == 1 else [0, NP_dict['Cntrs'][-1] + 0.5*Width_NP + Spacing_MM_NP + 0.5*w_MM]
        MM_dict['x_Cntrs'] = MicroMagnet_x_Cntrs

    # Calculate centers of the striplines
    Stripline_Cntrs = [-(0.5*w_MM + Spacing_SL_MM + 0.5*w_SL)] if Nr_SL == 1 else [-(0.5*w_MM + Spacing_SL_MM + 0.5*w_SL), -(Spacing_SL_MM + 0.5*w_MM) + Spacing_SL + 0.5*w_SL]
    # Stripline_Cntrs = [-(0.5*w_MM + Spacing_SL_MM + 0.5*w_SL)] if Nr_SL == 1 else [-(0.5*w_MM + Spacing_SL_MM + 0.5*w_SL), 2*(0.5*w_MM + Spacing_MM_NP) + N_NP*Width_NP + (N_NP-1)*Spacing_NP + 0.5*w_MM + Spacing_SL_MM + 0.5*w_SL] # To define the second stripline right next to the second micromagnet.
    SL_dict['Cntrs'] = Stripline_Cntrs

    return NP_dict, MM_dict, SL_dict

def calculate_magnetic_fields_and_gradients(NP_dict, MM_dict, SL_dict):
    '''
    - calculates the frequency gradients created by the micromagnets at different locations in x and z from the micromagnets
    - calculates the magnetic field created by the stripline at different locations in x and z from the stripline
    
    Input:
    - NP_dict       = dict  : Dictionary containting data about the nanopillars (NP); number of NPs, spacing between NPs, width of NPs and the center locations of the nanopillars
    - MM_dict       = dict  : Dictionary containting data about the micromagnets (MM); spacing between the edges of the MM and the NP, the volume magnetisation (M_es) of the MM material, the width and the height of the MM and the number of MMs (1; single-, or 2; double-sided) and the center locations of the micromagnets
    - SL_dict       = dict  : Dictionary containting data about the stripline (SL); spacing between the edges of a MM and a SL, width of a SL, number of striplines (1; single-, or 2; double-sided), spacing between the edges of striplines in case 2 are defined  and the center locations of the striplines

    Output:
    - location_dict = dict  : Dictionary containint the x- and z-positions where the frequencies and magnetic fields are calculated
    - freq_dict     = dict  : Dictionary containing data about the frequency and frequency gradients at (x,y) positions created by the micromagnets
    - B_dict        = dict  : Dictionary containing data about the magnetic field at (x,z) positions created by the stripline
    '''

    # Nanopillar parameters
    Width_NP = NP_dict['Width_NP'] # Width of a single nanopillar
    NanoPillar_Cntrs = NP_dict['Cntrs']

    # Micromagnet parameters
    Spacing_MM_NP = MM_dict['Spacing_MM_NP'] # Spacing between the edges of the micromagnet and the nanopillar 
    M_es = MM_dict['M_es'] # Page 2 second reference
    w_MM = MM_dict['w_MM'] # Width of a single micromagnet
    h_MM = MM_dict['h_MM'] # Height of a single micromagnet
    Nr_MM = MM_dict['Nr_MM'] # Number of micromagnets: Only 1 and 2 are currently allowed
    MicroMagnet_x_Cntrs = MM_dict['x_Cntrs']

    # Stripline parameters
    Spacing_SL_MM = SL_dict['Spacing_SL_MM'] # Spacing between the edges of a micromagnet and a stripline
    w_SL = SL_dict['w_SL'] # Width of a single stripline
    Nr_SL = SL_dict['Nr_SL'] # Number of striplines: Only 1 and 2 are currently allowed
    Spacing_SL = SL_dict['Spacing_SL'] # Spacing between the edges of two striplines
    Stripline_Cntrs = SL_dict['Cntrs']


    # z_max = 1.5e-6 #500e-9
    # x_max = 10e-6 # MicroMagnet_x_Cntrs[-1] if Nr_MM != 1 else NanoPillar_Cntrs[-1]+1.3*Width_NP
    # dz = z_max/300
    # dx = x_max/300

    # x = np.arange(-1.5e-6, x_max, dx)
    # z = np.arange(0.01e-6, h_MM + z_max, dz)

    xmin = MicroMagnet_x_Cntrs[0] + 0.5*w_MM
    xmax = MicroMagnet_x_Cntrs[1] - 0.5*w_MM if Nr_MM == 2 else xmin + 5e-6
    zmin = -1.5e-6
    zmax = -1*zmin

    x_sample = 300
    z_sample = 300
    x = np.linspace(xmin, xmax, x_sample)
    z = np.linspace(zmin, zmax, z_sample)
    dx = x[1]-x[0]
    dz = z[1]-z[0]
    xx, zz = np.meshgrid(x, z, sparse=True)

    '''
    Calculate magnetic fields, frequencies and gradients generated by the micromagnets
    '''
    Bx_MM = np.zeros((zz.shape[0], xx.shape[1]))
    Bz_MM = np.zeros((zz.shape[0], xx.shape[1]))

    for cntr in MicroMagnet_x_Cntrs:
        Bx_MM += compute_Bx(xx, zz, M_es, w_MM, h_MM, cntr) # xx+w_MM to have x = 0 at the boundary of the strip. zz to have z=0 at the bottom of the strip.
        Bz_MM += compute_Bz(xx, zz, M_es, w_MM, h_MM, cntr) # xx+w_MM to have x = 0 at the boundary of the strip. zz to have z=0 at the bottom of the strip.

    gamma_e = constants.value('electron gyromag. ratio in MHz/T') * 1e6 # Hz / T
    fz_MM = Bz_MM * gamma_e
    fx_MM = Bx_MM * gamma_e
    dfz_MM_dx = np.gradient(fz_MM, axis = 1)/dx
    dfz_MM_dz = np.gradient(fz_MM, axis = 0)/dz
    dfx_MM_dx = np.gradient(fx_MM, axis = 1)/dx
    dfx_MM_dz = np.gradient(fx_MM, axis = 0)/dz

    '''
    Calculate magnetic fields, frequencies and gradients generated by the stripline
    '''
    Bx_SL = np.zeros((zz.shape[0], xx.shape[1]))
    Bz_SL = np.zeros((zz.shape[0], xx.shape[1]))

    for counter, cntr in enumerate(Stripline_Cntrs):
        I_0_temp = I_0 if np.mod(counter,2) != 1 else -I_0 # need -I_0 for alternating striplines if a meandering stripline structure is considered
        [Bx_SL_temp, Bz_SL_temp] = compute_B_stripline(xx, zz, I_0_temp, w_SL, cntr, N_pts = 500) 
        Bx_SL += Bx_SL_temp
        Bz_SL += Bz_SL_temp

    location_dict = {}
    location_dict['x'] = x
    location_dict['z'] = z

    freq_dict = {}
    freq_dict['fz_MM'] = fz_MM
    freq_dict['fx_MM'] = fx_MM
    freq_dict['dfz_MM_dz'] = dfz_MM_dz
    freq_dict['dfz_MM_dx'] = dfz_MM_dx
    freq_dict['dfx_MM_dz'] = dfx_MM_dz
    freq_dict['dfx_MM_dx'] = dfx_MM_dx

    B_dict = {}
    B_dict['Bx_SL'] = Bx_SL
    B_dict['Bz_SL'] = Bz_SL

    return location_dict, freq_dict, B_dict
    
def x_Idxs_fxmax_calculator(fx_MM, zplot_index, fx_max):
    return np.where( np.abs(fx_MM[zplot_index,:]) < fx_max )[0]

def graphical_frequency_representation(zslice_array, NP_dict, MM_dict, SL_dict, location_dict, freq_dict, fx_max, y_scale_log = False, save_fig = False, save_data_to_publish = False):
    '''
    Function that plots the nanopillars (green), micromagnet (red) and stripline (orange) x-positions as shaded areas. 
    It plots the f_z frequency and the f_x frequency created by the micromagnets as a function of x for different z-positions

    Input: 
    - zslice_array      = np.array()    : Array that contains the z-positions for which the frequencies created by the micromagnets should be shown
    - NP_dict           = dict          : Dictionary containting data about the nanopillars (NP); number of NPs, spacing between NPs, width of NPs and the center locations of the nanopillars.
    - MM_dict           = dict          : Dictionary containting data about the micromagnets (MM); spacing between the edges of the MM and the NP, the volume magnetisation (M_es) of the MM material, the width and the height of the MM and the number of MMs (1; single-, or 2; double-sided) and the center locations of the micromagnets
    - SL_dict           = dict          : Dictionary containting data about the stripline (SL); spacing between the edges of a MM and a SL, width of a SL, number of striplines (1; single-, or 2; double-sided), spacing between the edges of striplines in case 2 are defined  and the center locations of the striplines
    - location_dict     = dict          : Dictionary containint the x- and z-positions where the frequencies and magnetic fields are calculated
    - freq_dict         = dict          : Dictionary containing data about the frequency and frequency gradients at (x,z) positions created by the micromagnets

    Output:
    A plot showing the frequency created by the mircomagnet for different z-positions as a function of the x-position. The plot indicates the nanopillar (green), micromagnet (red) and stripline (orange) positions.
    '''

    Pos_Conv_MM_SL = Pos_Conv_Fac[ Pos_unit_list.index(Pos_unit_MM_SL) ]
    Freq_Conv_MM_SL = Freq_Conv_Fac[ Freq_unit_list.index(Freq_unit_MM_SL) ] 
    FreqGrad_Conv_MM_SL = FreqGrad_Conv_Fac[ FreqGrad_unit_list.index(FreqGrad_unit_MM_SL) ]
    B_Conv_MM_SL = B_Conv_Fac[ B_unit_list.index(B_unit_MM_SL) ]

    Width_NP = NP_dict['Width_NP'] 
    NanoPillar_Cntrs = NP_dict['Cntrs']
    Nr_MM = MM_dict['Nr_MM']
    w_MM = MM_dict['w_MM'] 
    MicroMagnet_x_Cntrs = MM_dict['x_Cntrs']
    Nr_SL = SL_dict['Nr_SL']
    w_SL = SL_dict['w_SL'] 
    Stripline_Cntrs = SL_dict['Cntrs']

    x = location_dict['x']
    z = location_dict['z']

    fz_MM = freq_dict['fz_MM']
    fx_MM = freq_dict['fx_MM']

    fig_slice, ax_slice = plt.subplots(1,2, figsize = (9,3))
    # colors_z = cm.seismic( np.linspace(0,1,len(zslice_array)) )
        
    data_dict = {}
    data_dict['zslice_array'] = zslice_array
    data_dict['MicroMagnet_x_Cntrs'] = MicroMagnet_x_Cntrs
    data_dict['Stripline_Cntrs'] = Stripline_Cntrs
    data_dict['w_MM'] = w_MM
    data_dict['w_SL'] = w_SL
    data_dict['Nr_MM'] = Nr_MM
    data_dict['Pos_Conv_MM_SL'] = Pos_Conv_MM_SL
    data_dict['Freq_Conv_MM_SL'] = Freq_Conv_MM_SL
    data_dict['Pos_unit_MM_SL'] = Pos_unit_MM_SL
    data_dict['Freq_unit_MM_SL'] = Freq_unit_MM_SL

    for z_cntr, z_plot in enumerate(zslice_array):
        data_dict[z_plot] = {}
        # Find the index closest to where z, in the reference frame of the micromagnet, is equal to the difference in defect position (top of nanopillar) and the bottom of the nanopillar (z=0)
        zplotmin_new = np.abs(z) if MM_dict['zMM_is_zNP'] else np.abs(z-z_plot)
        zplot_index = np.where( zplotmin_new == np.min(zplotmin_new) )[0][0]

        x_Idxs = x_Idxs_fxmax_calculator(fx_MM, zplot_index, fx_max)

        ax_slice[0].plot(x[x_Idxs]/Pos_Conv_MM_SL, fz_MM[zplot_index, x_Idxs]/Freq_Conv_MM_SL, color = 'k', label = "z = " + str(np.round(z[zplot_index]/Pos_Conv_MM_SL,2) ) + " " + Pos_unit_MM_SL, zorder = 3)
        ax_slice[1].plot(x[x_Idxs]/Pos_Conv_MM_SL, fx_MM[zplot_index, x_Idxs]/Freq_Conv_MM_SL, color = 'k', label = "z = " + str(np.round(z[zplot_index]/Pos_Conv_MM_SL,2) ) + " " + Pos_unit_MM_SL, zorder = 3)
        
        data_dict[z_plot]['x'] = x[x_Idxs]
        data_dict[z_plot]['fz_MM'] = fz_MM[zplot_index, x_Idxs]
        data_dict[z_plot]['fx_MM'] = fx_MM[zplot_index, x_Idxs]

    [ylim_min0, ylim_max0] = ax_slice[0].get_ylim()
    [ylim_min1, ylim_max1] = ax_slice[1].get_ylim()
    data_dict['ylim_min0'] = ylim_min0
    data_dict['ylim_max0'] = ylim_max0
    data_dict['ylim_min1'] = ylim_min1
    data_dict['ylim_max1'] = ylim_max1


    for MM_cntr in MicroMagnet_x_Cntrs:
        ax_slice[0].fill_betweenx( [ylim_min0, ylim_max0] , (MM_cntr-0.5*w_MM)/Pos_Conv_MM_SL, (MM_cntr+0.5*w_MM)/Pos_Conv_MM_SL, facecolor ='grey', alpha = 0.2)
        ax_slice[1].fill_betweenx( [ylim_min1, ylim_max1] , (MM_cntr-0.5*w_MM)/Pos_Conv_MM_SL, (MM_cntr+0.5*w_MM)/Pos_Conv_MM_SL, facecolor ='grey', alpha = 0.2)
        
    for SL_cntr in Stripline_Cntrs:
        ax_slice[0].fill_betweenx( [ylim_min0, ylim_max0] , (SL_cntr-0.5*w_SL)/Pos_Conv_MM_SL, (SL_cntr+0.5*w_SL)/Pos_Conv_MM_SL, facecolor ='y', alpha = 0.1)
        ax_slice[1].fill_betweenx( [ylim_min1, ylim_max1] , (SL_cntr-0.5*w_SL)/Pos_Conv_MM_SL, (SL_cntr+0.5*w_SL)/Pos_Conv_MM_SL, facecolor ='y', alpha = 0.1)

    # for NP_cntr in NanoPillar_Cntrs:
        # ax_slice[0].fill_betweenx( [ylim_min0, ylim_max0] , (NP_cntr-0.5*Width_NP)/Pos_Conv_MM_SL, (NP_cntr+0.5*Width_NP)/Pos_Conv_MM_SL, facecolor ='g', alpha = 0.1)
        # ax_slice[1].fill_betweenx( [ylim_min1, ylim_max1] , (NP_cntr-0.5*Width_NP)/Pos_Conv_MM_SL, (NP_cntr+0.5*Width_NP)/Pos_Conv_MM_SL, facecolor ='g', alpha = 0.1)

    ax_slice[0].axvline(x = 1.3e-6/Pos_Conv_MM_SL, color = 'g', linestyle = 'dotted'); ax_slice[0].axvline(x = (MicroMagnet_x_Cntrs[1] - 1.3e-6)/Pos_Conv_MM_SL, color = 'g', linestyle = 'dotted')
    ax_slice[1].axvline(x = 1.3e-6/Pos_Conv_MM_SL, color = 'g', linestyle = 'dotted'); ax_slice[1].axvline(x = (MicroMagnet_x_Cntrs[1] - 1.3e-6)/Pos_Conv_MM_SL, color = 'g', linestyle = 'dotted')

    xlim_edge_MM = True
    xlim_min = (MicroMagnet_x_Cntrs[0] + 0.4*w_MM)/Pos_Conv_MM_SL if xlim_edge_MM == True else (Stripline_Cntrs[0]+0.45*w_SL)/Pos_Conv_MM_SL
    xlim_max = (MicroMagnet_x_Cntrs[1] - 0.4*w_MM)/Pos_Conv_MM_SL if xlim_edge_MM == True and Nr_MM!=1 else (MicroMagnet_x_Cntrs[-1]+0.55*w_MM)/Pos_Conv_MM_SL if Nr_MM !=1 else x[-1]/Pos_Conv_MM_SL # To set the xlim close to the second micromagnet

    ax_slice[0].set_xlabel('x (' + Pos_unit_MM_SL + ')')
    ax_slice[0].set_ylabel(r'$H_z$ (' + Freq_unit_MM_SL + ')')
    # ax_slice[0].legend(facecolor = 'w')
    ax_slice[0].set_xlim([xlim_min,xlim_max])
    # ax_slice[0].set_xlim([1.5,2.5])
    ax_slice[0].set_ylim([ylim_min0, ylim_max0])

    ax_slice[1].set_xlabel('x (' + Pos_unit_MM_SL + ')')
    ax_slice[1].set_ylabel(r'$H_x$ (' + Freq_unit_MM_SL + ')')
    # ax_slice[1].legend(facecolor = 'w')
    ax_slice[1].set_xlim([xlim_min,xlim_max])
    ax_slice[1].set_ylim([ylim_min1, ylim_max1])

    if y_scale_log == True:
            ax_slice[0].set_yscale('symlog'); ax_slice[0].set_yticks([-10**3, -10**1,0, 10**1, 10**3])
            ax_slice[1].set_yscale('symlog'); ax_slice[1].set_yticks([-10**1, -10**1,0, 10**1, 10**3])

    plt.tight_layout()

    if save_fig == True:
        save_str = 'MM_frequency.pdf'
        plt.savefig(os.path.join(save_folder,save_str),format='pdf',bbox_inches = 'tight',transparent=True,pad_inches=0.1 , dpi=400)

    if save_data_to_publish:
        save_str = 'fig_5_fields.pkl'
        data_dict_to_pkl(data_dict, save_str)

        # save_str = 'fig_5e_fz'
        # data_dictionary = {'x':x[x_Idxs]/Pos_Conv_MM_SL, 'fz':fz_MM[zplot_index, x_Idxs]/Freq_Conv_MM_SL}
        # data_to_publish_saver(save_str,data_dictionary)

        # save_str = 'fig_5f_fx'
        # data_dictionary = {'x':x[x_Idxs]/Pos_Conv_MM_SL, 'fx':fx_MM[zplot_index, x_Idxs]/Freq_Conv_MM_SL}
        # data_to_publish_saver(save_str,data_dictionary)

def graphical_delta_frequency_Bfield_representation(zslice_array, NP_dict, MM_dict, SL_dict, location_dict, freq_dict, fx_max, B_dict, plot_B_field = True, y_scale_log = False, save_fig = False, save_data_to_publish = False):
    '''
    Function that plots the nanopillars (green), micromagnet (red) and stripline (orange) x-positions as shaded areas. 
    It plots the df_z/dz, df_z/dx, df_x/dz, d_fx/dx frequency gradients created by the micromagnets as a function of x for different z-positions.
    Simultaneously it plots the B-field created by a 1A current running through the stripline.

    Input: 
    - zslice_array      = np.array()    : Array that contains the z-positions for which the frequency gradients created by the micromagnets should be shown
    - NP_dict           = dict          : Dictionary containting data about the nanopillars (NP); number of NPs, spacing between NPs, width of NPs and the center locations of the nanopillars
    - MM_dict           = dict          : Dictionary containting data about the micromagnets (MM); spacing between the edges of the MM and the NP, the volume magnetisation (M_es) of the MM material, the width and the height of the MM and the number of MMs (1; single-, or 2; double-sided) and the center locations of the micromagnets
    - SL_dict           = dict          : Dictionary containting data about the stripline (SL); spacing between the edges of a MM and a SL, width of a SL, number of striplines (1; single-, or 2; double-sided), spacing between the edges of striplines in case 2 are defined  and the center locations of the striplines
    - location_dict     = dict          : Dictionary containint the x- and z-positions where the frequencies and magnetic fields are calculated
    - freq_dict         = dict          : Dictionary containing data about the frequency and frequency gradients at (x,z) positions created by the micromagnets
    - B_dict            = dict          : Dictionary containing data about the magnetid field at (x,z) positions created by the stripline

    Output:
    A plot showing the freqeuncy gradients created by the mircomagnet for different z-positions as a function of the x-position. The plot indicates the nanopillar (green), micromagnet (red) and stripline (orange) positions.
    '''

    Pos_Conv_MM_SL = Pos_Conv_Fac[ Pos_unit_list.index(Pos_unit_MM_SL) ]
    Freq_Conv_MM_SL = Freq_Conv_Fac[ Freq_unit_list.index(Freq_unit_MM_SL) ] 
    FreqGrad_Conv_MM_SL = FreqGrad_Conv_Fac[ FreqGrad_unit_list.index(FreqGrad_unit_MM_SL) ]
    B_Conv_MM_SL = B_Conv_Fac[ B_unit_list.index(B_unit_MM_SL) ]

    Width_NP = NP_dict['Width_NP'] 
    NanoPillar_Cntrs = NP_dict['Cntrs']

    Nr_MM = MM_dict['Nr_MM']
    w_MM = MM_dict['w_MM'] 
    MicroMagnet_x_Cntrs = MM_dict['x_Cntrs']

    Nr_SL = SL_dict['Nr_SL']
    w_SL = SL_dict['w_SL'] 
    Stripline_Cntrs = SL_dict['Cntrs']

    x = location_dict['x']
    z = location_dict['z']

    fz_MM = freq_dict['fz_MM']
    fx_MM = freq_dict['fx_MM']
    dfz_MM_dz = freq_dict['dfz_MM_dz']
    dfz_MM_dx = freq_dict['dfz_MM_dx']
    dfx_MM_dz = freq_dict['dfx_MM_dz']
    dfx_MM_dx = freq_dict['dfx_MM_dx']

    Bx_SL = B_dict['Bx_SL']
    Bz_SL = B_dict['Bz_SL']

    data_dict = {}
    data_dict['z'] = z
    data_dict['zslice_array'] = zslice_array
    data_dict['MicroMagnet_x_Cntrs'] = MicroMagnet_x_Cntrs
    data_dict['Stripline_Cntrs'] = Stripline_Cntrs
    data_dict['w_MM'] = w_MM
    data_dict['w_SL'] = w_SL
    data_dict['Nr_MM'] = Nr_MM
    data_dict['Pos_Conv_MM_SL'] = Pos_Conv_MM_SL
    data_dict['Freq_Conv_MM_SL'] = Freq_Conv_MM_SL
    data_dict['Pos_unit_MM_SL'] = Pos_unit_MM_SL
    data_dict['Freq_unit_MM_SL'] = Freq_unit_MM_SL
    data_dict['FreqGrad_unit_MM_SL'] = FreqGrad_unit_MM_SL
    data_dict['FreqGrad_Conv_MM_SL'] = FreqGrad_Conv_MM_SL
    data_dict['B_Conv_MM_SL'] = B_Conv_MM_SL
    data_dict['B_unit_MM_SL'] = B_unit_MM_SL

    fig_slice, ax_slice = plt.subplots(2,2, figsize = (9,6))
    # colors_z = cm.seismic( np.linspace(0,1,len(zslice_array)) )

    if plot_B_field:
        ax2_slice00 = ax_slice[0,0].twinx()
        ax2_slice01 = ax_slice[0,1].twinx()
        ax2_slice10 = ax_slice[1,0].twinx()
        ax2_slice11 = ax_slice[1,1].twinx()
        # ax2_slice00.set_yscale('symlog')
        # ax2_slice01.set_yscale('symlog')
        # ax2_slice10.set_yscale('symlog')
        # ax2_slice11.set_yscale('symlog')

    if y_scale_log == True:
        ax_slice[0,0].set_yscale('symlog');
        ax_slice[0,1].set_yscale('symlog');
        ax_slice[1,0].set_yscale('symlog');
        ax_slice[1,1].set_yscale('symlog');

    B_field_color = 'r'
        
    for z_cntr, z_plot in enumerate(zslice_array):
        data_dict[z_plot] = {}
        # Find the index closest to where z, in the reference frame of the micromagnet, is equal to the difference in defect position (top of nanopillar) and the bottom of the nanopillar (z=0)
        zplotmin_new = np.abs(z) if MM_dict['zMM_is_zNP'] else np.abs(z-z_plot)
        zplot_index = np.where( zplotmin_new == np.min(zplotmin_new) )[0][0]

        x_Idxs = x_Idxs_fxmax_calculator(fx_MM, zplot_index, fx_max)
        
        ax_slice[0,0].plot(x[x_Idxs]/Pos_Conv_MM_SL, dfz_MM_dz[zplot_index, x_Idxs]/FreqGrad_Conv_MM_SL, color = 'k', label = "z = " + str(np.round(z[zplot_index]/Pos_Conv_MM_SL,2) ) + " " + Pos_unit_MM_SL, zorder = 3)
        ax_slice[0,1].plot(x[x_Idxs]/Pos_Conv_MM_SL, dfz_MM_dx[zplot_index, x_Idxs]/FreqGrad_Conv_MM_SL, color = 'k', label = "z = " + str(np.round(z[zplot_index]/Pos_Conv_MM_SL,2) ) + " " + Pos_unit_MM_SL, zorder = 3)
        ax_slice[1,0].plot(x[x_Idxs]/Pos_Conv_MM_SL, dfx_MM_dz[zplot_index, x_Idxs]/FreqGrad_Conv_MM_SL, color = 'k', label = "z = " + str(np.round(z[zplot_index]/Pos_Conv_MM_SL,2) ) + " " + Pos_unit_MM_SL, zorder = 3)
        ax_slice[1,1].plot(x[x_Idxs]/Pos_Conv_MM_SL, dfx_MM_dx[zplot_index, x_Idxs]/FreqGrad_Conv_MM_SL, color = 'k', label = "z = " + str(np.round(z[zplot_index]/Pos_Conv_MM_SL,2) ) + " " + Pos_unit_MM_SL, zorder = 3)
        
        data_dict[z_plot]['x'] = x[x_Idxs]
        data_dict[z_plot]['dfz_MM_dz'] = dfz_MM_dz[zplot_index, x_Idxs]
        data_dict[z_plot]['dfz_MM_dx'] = dfz_MM_dx[zplot_index, x_Idxs]
        data_dict[z_plot]['dfx_MM_dz'] = dfx_MM_dz[zplot_index, x_Idxs]
        data_dict[z_plot]['dfx_MM_dx'] = dfx_MM_dx[zplot_index, x_Idxs]

        # Find the index closest to z_plot in z. The stripline is level with the bottom of the nanopillars.
        zplotmin_new_Bfield = np.abs(z - z_plot); 
        zplot_index_Bfield = np.where( zplotmin_new_Bfield == np.min(zplotmin_new_Bfield) )[0][0]

        if plot_B_field:
            ax2_slice00.plot(x/Pos_Conv_MM_SL, Bx_SL[zplot_index_Bfield,:]/B_Conv_MM_SL, color = B_field_color, zorder = 3)
            ax2_slice01.plot(x/Pos_Conv_MM_SL, Bx_SL[zplot_index_Bfield,:]/B_Conv_MM_SL, color = B_field_color, zorder = 3)
            ax2_slice10.plot(x/Pos_Conv_MM_SL, Bx_SL[zplot_index_Bfield,:]/B_Conv_MM_SL, color = B_field_color, zorder = 3)
            ax2_slice11.plot(x/Pos_Conv_MM_SL, Bx_SL[zplot_index_Bfield,:]/B_Conv_MM_SL, color = B_field_color, zorder = 3)

            data_dict[z_plot]['x_Bx'] = x
            data_dict[z_plot]['Bx_SL'] = Bx_SL[zplot_index_Bfield,:]

    [ylim_min00, ylim_max00] = ax_slice[0,0].get_ylim()
    [ylim_min01, ylim_max01] = ax_slice[0,1].get_ylim()
    [ylim_min10, ylim_max10] = ax_slice[1,0].get_ylim()
    [ylim_min11, ylim_max11] = ax_slice[1,1].get_ylim()

    data_dict['ylim_min00'] = ylim_min00; data_dict['ylim_max00'] = ylim_max00
    data_dict['ylim_min01'] = ylim_min01; data_dict['ylim_max01'] = ylim_max01
    data_dict['ylim_min10'] = ylim_min10; data_dict['ylim_max10'] = ylim_max10
    data_dict['ylim_min11'] = ylim_min11; data_dict['ylim_max11'] = ylim_max11

    for MM_cntr in MicroMagnet_x_Cntrs:
        ax_slice[0,0].fill_betweenx( [ylim_min00, ylim_max00] , (MM_cntr-0.5*w_MM)/Pos_Conv_MM_SL, (MM_cntr+0.5*w_MM)/Pos_Conv_MM_SL, facecolor ='grey', alpha = 0.2)
        ax_slice[0,1].fill_betweenx( [ylim_min01, ylim_max01] , (MM_cntr-0.5*w_MM)/Pos_Conv_MM_SL, (MM_cntr+0.5*w_MM)/Pos_Conv_MM_SL, facecolor ='grey', alpha = 0.2)
        ax_slice[1,0].fill_betweenx( [ylim_min10, ylim_max10] , (MM_cntr-0.5*w_MM)/Pos_Conv_MM_SL, (MM_cntr+0.5*w_MM)/Pos_Conv_MM_SL, facecolor ='grey', alpha = 0.2)
        ax_slice[1,1].fill_betweenx( [ylim_min11, ylim_max11] , (MM_cntr-0.5*w_MM)/Pos_Conv_MM_SL, (MM_cntr+0.5*w_MM)/Pos_Conv_MM_SL, facecolor ='grey', alpha = 0.2)

    for SL_cntr in Stripline_Cntrs:        
        ax_slice[0,0].fill_betweenx( [ylim_min00, ylim_max00] , (SL_cntr-0.5*w_SL)/Pos_Conv_MM_SL, (SL_cntr+0.5*w_SL)/Pos_Conv_MM_SL, facecolor ='y', alpha = 0.1)
        ax_slice[0,1].fill_betweenx( [ylim_min01, ylim_max01] , (SL_cntr-0.5*w_SL)/Pos_Conv_MM_SL, (SL_cntr+0.5*w_SL)/Pos_Conv_MM_SL, facecolor ='y', alpha = 0.1)
        ax_slice[1,0].fill_betweenx( [ylim_min10, ylim_max10] , (SL_cntr-0.5*w_SL)/Pos_Conv_MM_SL, (SL_cntr+0.5*w_SL)/Pos_Conv_MM_SL, facecolor ='y', alpha = 0.1)
        ax_slice[1,1].fill_betweenx( [ylim_min11, ylim_max11] , (SL_cntr-0.5*w_SL)/Pos_Conv_MM_SL, (SL_cntr+0.5*w_SL)/Pos_Conv_MM_SL, facecolor ='y', alpha = 0.1)


    # for NP_cntr in NanoPillar_Cntrs:
    #     ax_slice[0,0].fill_betweenx( [ylim_min00, ylim_max00] , (NP_cntr-0.5*Width_NP)/Pos_Conv_MM_SL, (NP_cntr+0.5*Width_NP)/Pos_Conv_MM_SL, facecolor ='g', alpha = 0.1)
    #     ax_slice[0,1].fill_betweenx( [ylim_min01, ylim_max01] , (NP_cntr-0.5*Width_NP)/Pos_Conv_MM_SL, (NP_cntr+0.5*Width_NP)/Pos_Conv_MM_SL, facecolor ='g', alpha = 0.1)
    #     ax_slice[1,0].fill_betweenx( [ylim_min10, ylim_max10] , (NP_cntr-0.5*Width_NP)/Pos_Conv_MM_SL, (NP_cntr+0.5*Width_NP)/Pos_Conv_MM_SL, facecolor ='g', alpha = 0.1)
    #     ax_slice[1,1].fill_betweenx( [ylim_min11, ylim_max11] , (NP_cntr-0.5*Width_NP)/Pos_Conv_MM_SL, (NP_cntr+0.5*Width_NP)/Pos_Conv_MM_SL, facecolor ='g', alpha = 0.1)

    ax_slice[0,0].axvline(x = 1.3e-6/Pos_Conv_MM_SL, color = 'g', linestyle = 'dotted'); ax_slice[0,0].axvline(x = (MicroMagnet_x_Cntrs[1] - 1.3e-6)/Pos_Conv_MM_SL, color = 'g', linestyle = 'dotted')
    ax_slice[1,0].axvline(x = 1.3e-6/Pos_Conv_MM_SL, color = 'g', linestyle = 'dotted'); ax_slice[1,0].axvline(x = (MicroMagnet_x_Cntrs[1] - 1.3e-6)/Pos_Conv_MM_SL, color = 'g', linestyle = 'dotted')
    ax_slice[0,1].axvline(x = 1.3e-6/Pos_Conv_MM_SL, color = 'g', linestyle = 'dotted'); ax_slice[0,1].axvline(x = (MicroMagnet_x_Cntrs[1] - 1.3e-6)/Pos_Conv_MM_SL, color = 'g', linestyle = 'dotted')
    ax_slice[1,1].axvline(x = 1.3e-6/Pos_Conv_MM_SL, color = 'g', linestyle = 'dotted'); ax_slice[1,1].axvline(x = (MicroMagnet_x_Cntrs[1] - 1.3e-6)/Pos_Conv_MM_SL, color = 'g', linestyle = 'dotted')


    xlim_edge_MM = True
    xlim_min = (MicroMagnet_x_Cntrs[0] + 0.4*w_MM)/Pos_Conv_MM_SL if xlim_edge_MM == True else (Stripline_Cntrs[0]+0.45*w_SL)/Pos_Conv_MM_SL
    xlim_max = (MicroMagnet_x_Cntrs[1] - 0.4*w_MM)/Pos_Conv_MM_SL if xlim_edge_MM == True and Nr_MM!=1 else (MicroMagnet_x_Cntrs[-1]+0.55*w_MM)/Pos_Conv_MM_SL if Nr_MM !=1 else x[-1]/Pos_Conv_MM_SL # To set the xlim close to the second micromagnet
    # xlim_max = (Stripline_Cntrs[-1]-0.45*w_SL)/Pos_Conv_MM_SL if Nr_SL !=1 else x[-1]/Pos_Conv_MM_SL # To set the xlim close to the second stripline 

    ax_slice[0,0].set_yticks([-3e3,-1e3])
    ax_slice[0,1].set_yticks([-1e4,-1e2,0,1e2,1e4])
    ax_slice[1,0].set_yticks([-1e3,-1e1,0,1e1,1e3])
    ax_slice[1,1].set_yticks([1e2,1e3])

    ax_slice[0,0].set_xlabel('x(um)')
    ax_slice[0,0].set_ylabel(r'$\frac{dH_z}{dz}$ (' + FreqGrad_unit_MM_SL + ')')
    # ax_slice[0,0].legend(facecolor = 'w')
    ax_slice[0,0].set_xlim([xlim_min,xlim_max])
    ax_slice[0,0].set_ylim([ylim_min00, ylim_max00])
    ax_slice[0,0].set_yticks([-3e3,-1e3])

    ax_slice[0,1].set_xlabel('x(um)')
    ax_slice[0,1].set_ylabel(r'$\frac{dH_z}{dx}$ (' + FreqGrad_unit_MM_SL + ')')
    # ax_slice[0,1].legend(facecolor = 'w')
    ax_slice[0,1].set_xlim([xlim_min,xlim_max])
    ax_slice[0,1].set_ylim([ylim_min01, ylim_max01])

    ax_slice[1,0].set_xlabel('x(um)')
    ax_slice[1,0].set_ylabel(r'$\frac{dH_x}{dz}$ (' + FreqGrad_unit_MM_SL + ')')
    # ax_slice[1,0].legend(facecolor = 'w')
    ax_slice[1,0].set_xlim([xlim_min,xlim_max])
    ax_slice[1,0].set_ylim([ylim_min10, ylim_max10])

    ax_slice[1,1].set_xlabel('x(um)')
    ax_slice[1,1].set_ylabel(r'$\frac{dH_x}{dx}$ (' + FreqGrad_unit_MM_SL + ')')
    # ax_slice[1,1].legend(facecolor = 'w')
    ax_slice[1,1].set_xlim([xlim_min,xlim_max])
    # ax_slice[1,1].set_ylim([ylim_min11, ylim_max11])

    if plot_B_field:
        ax2_slice00.spines['right'].set_color(B_field_color); ax2_slice00.yaxis.label.set_color(B_field_color); ax2_slice00.tick_params(colors = B_field_color, which = 'major')
        ax2_slice01.spines['right'].set_color(B_field_color); ax2_slice01.yaxis.label.set_color(B_field_color); ax2_slice01.tick_params(colors = B_field_color, which = 'major')
        ax2_slice10.spines['right'].set_color(B_field_color); ax2_slice10.yaxis.label.set_color(B_field_color); ax2_slice10.tick_params(colors = B_field_color, which = 'major')
        ax2_slice11.spines['right'].set_color(B_field_color); ax2_slice11.yaxis.label.set_color(B_field_color); ax2_slice11.tick_params(colors = B_field_color, which = 'major')
        
        ax2_slice00.set_xlabel('x(um)')
        ax2_slice00.set_ylabel(r'$B_x$ (' + B_unit_MM_SL + ')')
        # ax2_slice00.legend(facecolor = 'w')
        ax2_slice00.set_xlim([xlim_min,xlim_max])

        ax2_slice01.set_xlabel('x(um)')
        ax2_slice01.set_ylabel(r'$B_x$ (' + B_unit_MM_SL + ')')
        # ax2_slice01.legend(facecolor = 'w')
        ax2_slice01.set_xlim([xlim_min,xlim_max])

        ax2_slice10.set_xlabel('x(um)')
        ax2_slice10.set_ylabel(r'$B_x$ (' + B_unit_MM_SL + ')')
        # ax2_slice10.legend(facecolor = 'w')
        ax2_slice10.set_xlim([xlim_min,xlim_max])

        ax2_slice11.set_xlabel('x(um)')
        ax2_slice11.set_ylabel(r'$B_x$ (' + B_unit_MM_SL + ')')
        # ax2_slice11.legend(facecolor = 'w')
        ax2_slice11.set_xlim([xlim_min,xlim_max])


    plt.tight_layout()  

    if save_fig == True:
        save_str = 'MM_frequency_gradient_SL_Bfield.pdf'
        plt.savefig(os.path.join(save_folder,save_str),format='pdf',bbox_inches = 'tight',transparent=True,pad_inches=0.1 , dpi=400)

    if save_data_to_publish:
        save_str = 'fig_5_gradients.pkl'
        data_dict_to_pkl(data_dict, save_str)

        # save_str = 'fig_5a_dfz_dz'
        # data_dictionary = {'x':x[x_Idxs]/Pos_Conv_MM_SL, 'dfz_dz':dfz_MM_dz[zplot_index, x_Idxs]/FreqGrad_Conv_MM_SL, 'Bx_SL':Bx_SL[zplot_index_Bfield,:]/B_Conv_MM_SL}
        # data_to_publish_saver(save_str,data_dictionary)

        # save_str = 'fig_5b_dfz_dx'
        # data_dictionary = {'x':x[x_Idxs]/Pos_Conv_MM_SL, 'dfz_dx':dfz_MM_dx[zplot_index, x_Idxs]/FreqGrad_Conv_MM_SL, 'Bx_SL':Bx_SL[zplot_index_Bfield,:]/B_Conv_MM_SL}
        # data_to_publish_saver(save_str,data_dictionary)

        # save_str = 'fig_5c_dfx_dz'
        # data_dictionary = {'x':x[x_Idxs]/Pos_Conv_MM_SL, 'dfx_dz':dfx_MM_dz[zplot_index, x_Idxs]/FreqGrad_Conv_MM_SL, 'Bx_SL':Bx_SL[zplot_index_Bfield,:]/B_Conv_MM_SL}
        # data_to_publish_saver(save_str,data_dictionary)

        # save_str = 'fig_5d_dfx_dx'
        # data_dictionary = {'x':x[x_Idxs]/Pos_Conv_MM_SL, 'dfx_dx':dfx_MM_dx[zplot_index, x_Idxs]/FreqGrad_Conv_MM_SL, 'Bx_SL':Bx_SL[zplot_index_Bfield,:]/B_Conv_MM_SL}
        # data_to_publish_saver(save_str,data_dictionary)

def thesis_figure_A(diameters_to_plot, J_threshold_to_plot, N_spins_array, data_dict, ImpVol_dict, save_fig = False, save_data_to_publish = False):
    '''
    Function to plot combined data for different implantation parameters and a certain coupling thershold to be investigated. Also the T2* and defect concentration is plotted for different number of defects in a cluster.
    - 1st subfigure: mean number of defects created as a function of implantation dose for a certain implantation mask diameter
    - 2nd subfigure: Probability to find at least one coupling above J_threshold_to_plot in a spin cluster, as a function of the implantation dose for a certain implantation mask diameter
    - 3th subfigure: The average number of defects to investigate before finding a defect with a coupling stronger than J_threshold_to_plot. Defined as 1 over the probability to find a strongly coupled spin in a spin cluster, multiplied by the ratio of total spins over the amount of unique spins that couple stronger than J_threshold_to_plot
    - 4th subfigure: T2* as a function of the number of electron spins created in a cluster & the spin concentration as a function of the number of electron spins created in the cluster. The cluster size is defined by the mask radius and the spread in the z-position during implantation

    Input: 
    - diameters_to_plot     = np.array()    : Array indicating for which implantation mask diameter you want to plot the data
    - J_threshold_to_plot   = float         : Threshold above which the coupling you are interested in should be
    - N_spins_array         = np.array()    : Number of spins in cluster that have been analysed
    - data_dict             = dict          : dictionary containing, per number of spins in a cluster, data on e.g. coupling strenghts and probability to have at least one strongly coupled spin pair.

    Output:
    - A figure with the plots as described above
    '''
    DataDir = 'Implantation_Parameters/'
    Jthreshold_plot = int(J_threshold_to_plot*1e-3)
    DataName = [0]*len(diameters_to_plot)

    mask_diameter = 2*ImpVol_dict['mask_r']

    Time_unit = 'us'
    Concentration_unit = 'cm^-3'

    Time_Conv = Time_Conv_Fac[ Time_unit_list.index(Time_unit) ]
    Conc_Conv = Conc_Conv_Fac[ Conc_unit_list.index(Concentration_unit) ]

    # Convert times
    T2star_mean_Conv = np.array(data_dict['T2star_mean_list'])/Time_Conv
    T2star_std_Conv = np.array(data_dict['T2star_std_list'])/Time_Conv

    # Convert concentrations
    Concentration_mean_Conv = np.array(data_dict['Concentration_mean_list'])/Conc_Conv
    Concentration_std_Conv = np.array(data_dict['Concentration_std_list'])/Conc_Conv


    fig, ax = plt.subplots(2,2, figsize = (12,8))
    ax_Nmean = ax[0,0]
    ax_P = ax[0,1]
    ax_Nmeasure = ax[1,0]
    ax_times = ax[1,1]

    data_dict = {}
    data_dict['diameters_to_plot'] = diameters_to_plot

    # For different implantation mask diameters, plot 1) the mean number of created defects, 2) probability to find a defect with a strong coupling, 3) the number of defects to measure before finding a strongly coupled spin
    for i, d in enumerate(diameters_to_plot):
        data_dict[d] = {}
        DataDir = 'Implantation_Parameters/'
        DataFile_Name = 'Diameter_'+str(d)+'_nm_Jthreshold_'+str(Jthreshold_plot)+'kHz.npz'
        DataName[i] = 'Data'+str(d)+'nm'
        temp_dict = np.load(DataDir+DataFile_Name)

        DoseDict = temp_dict['dose_array_norm']
        NDefectsDict = temp_dict['mean_defects_array']
        PcouplDict = temp_dict['P_threshold_dose_norm']
        NcouplDict = temp_dict['N_spins_J_strong_list_dose_norm']
        # NcouplStdDict = temp_dict['N_spins_J_strong_std_list_dose_norm']
        
        data_dict[d]['DoseDict'] = DoseDict
        data_dict[d]['NDefectsDict'] = NDefectsDict
        data_dict[d]['PcouplDict'] = PcouplDict
        data_dict[d]['NcouplDict'] = NcouplDict

        ax_Nmean.plot(DoseDict, NDefectsDict, label = str(d)+'nm')
        ax_P.plot(DoseDict, PcouplDict, label = str(d)+'nm')
        N_average_measure = 1./(PcouplDict/100) * (NDefectsDict/NcouplDict)
        ax_Nmeasure.plot(DoseDict, N_average_measure, label = str(d)+'nm') # division by 100 to convert the probability from % to a number between 0 and 1

    ax_Nmean.legend(framealpha = 1); ax_Nmean.grid(); ax_Nmean.set_ylabel("Mean number of defects")
    ax_Nmean.set_xlabel("Dose (1e11 cm$^{-2}$)")

    ax_P.legend(framealpha = 1); ax_P.grid(); ax_P.set_ylabel(r"$P(J_{\mathrm{max}}) > $" + str(np.round(Jthreshold_plot)) + "kHz (%)")
    ax_P.set_xlabel("Dose (1e11 cm$^{-2}$)")

    ax_Nmeasure.legend(framealpha = 1); ax_Nmeasure.grid(); ax_Nmeasure.set_ylabel("N measure before finding J>%3.0fkHz" %(Jthreshold_plot))
    ax_Nmeasure.set_xlabel("Dose (1e11 cm$^{-2}$)")

    data_dict['Jthreshold_plot'] = Jthreshold_plot

    # Plot of T2* and spin concentration
    xmin = N_spins_array[0]-0.2
    xmax = N_spins_array[-1]+0.2

    ax_times.errorbar(N_spins_array,T2star_mean_Conv, T2star_std_Conv, color = 'k')
    ax_times.grid()
    ax_times.set_xlabel('Number of defects')
    Time_unit_label = Time_unit if Time_unit != 'us' else r'$\mu$s'
    ax_times.set_ylabel('$\overline{T_2^*}$ (' + Time_unit_label +')')
    ax_times.set_xlim([xmin, xmax])
    ax_times.set_yscale('log')

    diameter_str = r'$d_{mask}$ = %3.0f nm' %(mask_diameter*1e9) 
    props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=1)
    ax_times.text(0.3, 0.9, diameter_str, transform=ax_times.transAxes, fontsize=15,
    verticalalignment='top', bbox=props)

    data_dict['N_spins_array'] = N_spins_array
    data_dict['T2star_mean_Conv'] = T2star_mean_Conv
    data_dict['T2star_std_Conv'] = T2star_std_Conv
    data_dict['diameter_str'] = diameter_str

    Conc_color = 'r'
    Conc_scaling = 1e15
    ax_times2 = ax_times.twinx()
    ax_times2.errorbar(N_spins_array,Concentration_mean_Conv/Conc_scaling, Concentration_std_Conv/Conc_scaling, color = Conc_color)
    ax_times2.set_ylabel('spin concentration (1e15 cm$^{-3}$)')
    ax_times2.spines['right'].set_color(Conc_color)
    ax_times2.yaxis.label.set_color(Conc_color)
    ax_times2.tick_params(colors = Conc_color, which = 'major')
    # decimal_min = np.floor(0.9*norm_min*10)/10
    # decimal_max = np.floor(1.0*norm_max*10)/10
    # ax_norm.set_yticks(np.linspace(decimal_min,decimal_max,3))

    data_dict['C_scaling'] = Concentration_mean_Conv/Conc_scaling
    data_dict['C_std_scaling'] = Concentration_std_Conv/Conc_scaling

    fig.tight_layout()

    if save_fig == True:
        save_str = 'Parameters_vs_Dose.pdf'
        plt.savefig(os.path.join(save_folder,save_str),format='pdf',bbox_inches = 'tight',transparent=True,pad_inches=0.1 , dpi=400)

    if save_data_to_publish:
        save_str = 'fig_2.pkl'
        data_dict_to_pkl(data_dict, save_str)

        # save_str = 'fig_2a_mean_defects'
        # data_dictionary = {'Dose':DoseDict, 'Mean_N_defects':NDefectsDict}
        # # data_to_publish_saver(save_str,data_dictionary)
        # data_dict_to_pkl(data_dictionary, save_str+'.pkl')

        # save_str = 'fig_2b_P'
        # data_dictionary = {'Dose':DoseDict, 'P':PcouplDict}
        # # data_to_publish_saver(save_str,data_dictionary)
        # data_dict_to_pkl(data_dictionary, save_str+'.pkl')

        # save_str = 'fig_2c_N_measure'
        # data_dictionary = {'Dose':DoseDict, 'N_measure':N_average_measure}
        # # data_to_publish_saver(save_str,data_dictionary)
        # data_dict_to_pkl(data_dictionary, save_str+'.pkl')

        # save_str = 'fig_2d_T2star'
        # data_dictionary = {'N_defects':N_spins_array, \
        #                     'T2star':T2star_mean_Conv, 'T2star_err':T2star_std_Conv, \
        #                     'spin_conc':Concentration_mean_Conv/Conc_scaling, 'spin_conc_err':Concentration_std_Conv/Conc_scaling}
        # # data_to_publish_saver(save_str,data_dictionary)
        # data_dict_to_pkl(data_dictionary, save_str+'.pkl')


def thesis_figure_B(N_spins_array, data_dict, ImpVol_dict, J_threshold, save_fig = False, save_data_to_publish = False):
    '''
    Function to plot the following data:
    - For a number of spins in a cluster equal to "Number_Spins_Show", it shows the distribution of delta_x values for spins that couple more strongly than the threshold, and the distribution of the coupling strengts per delta_x and corresponding theta (angle in (r_{xy})-z plane)
    - For a number of spins in a cluster equal to "Number_Spins_Show", it shows the distribution of delta_y values for spins that couple more strongly than the threshold, and the distribution of the coupling strengts per delta_y and corresponding theta (angle in (r_{xy})-z plane)
    - For a number of spins in a cluster equal to "Number_Spins_Show", it shows the distribution of delta_z values for spins that couple more strongly than the threshold, and the distribution of the coupling strengts per delta_z and corresponding theta (angle in (r_{xy})-z plane)
    - For a number of spins in a cluster equal to "Number_Spins_Show", it shows the distribution of delta_r values for spins that couple more strongly than the threshold, and the distribution of the coupling strengts per delta_r and corresponding theta (angle in (r_{xy})-z plane)
    - For a number of spins in a cluster equal to "Number_Spins_Show", it shows the distribution of delta_r_{xy} values for spins that couple more strongly than the threshold, and the distribution of the coupling strengts per delta_r_{xy} and corresponding theta (angle in (r_{xy})-z plane)

    Input: 
    - N_spins_array         = np.array()    : Number of spins in cluster that have been analysed
    - data_dict             = dict          : dictionary containing, per number of spins in a cluster, data on e.g. coupling strenghts and probability to have at least one strongly coupled spin pair.
    - J_threshold           = float         : The threshold above which a coupling is defined as strong
    - ImpVol_dict           = dict          : Implantation Volumen Dictionary containing information on the implantation volume and sampling. It contains the implantation mask radius, the type of defect sampling in the xy plane and the implantation std in the z-direction.
    - Number_Spins_Show     = int           : the number of spins in a cluster for which to show the conditioned distribution of coupling strengths

    Output:
    - A figure with the plots as described above
    '''

    Number_Spins_Show = np.max(N_spins_array)
    Idx = np.where(N_spins_array == Number_Spins_Show)[0][0]

    Freq_unit = 'kHz'
    Pos_unit = 'nm'

    Pos_Conv = Pos_Conv_Fac[ Pos_unit_list.index(Pos_unit) ]
    Freq_Conv = Freq_Conv_Fac[ Freq_unit_list.index(Freq_unit) ]

    # Convert threshold
    J_threshold_Conv = J_threshold/Freq_Conv

    # Convert couplings
    Coupl_Thres_Min_Conv = np.array(data_dict['Coupl_Thres_Min_list'][Idx])/Freq_Conv
    Coupl_Thres_Plus_Conv = np.array(data_dict['Coupl_Thres_Plus_list'][Idx])/Freq_Conv
    Coupl_Thres_Tot_Conv = np.array(data_dict['Coupl_Thres_Tot_list'][Idx])/Freq_Conv
    Coupl_Tot_Conv = np.array(data_dict['Coupl_Tot_list'][Idx])/Freq_Conv

    # Convert distances
    xpos = np.array(data_dict['x_array_Plus_list'][Idx])/Pos_Conv
    mask_r = ImpVol_dict['mask_r']
    mask_rad = mask_r/Pos_Conv

    dx_arr = np.array(data_dict['dx_array_Plus_list'][Idx])/Pos_Conv
    dy_arr = np.array(data_dict['dy_array_Plus_list'][Idx])/Pos_Conv
    dz_arr = np.array(data_dict['dz_array_Plus_list'][Idx])/Pos_Conv
    J_arr  = np.array(data_dict['Coupl_Thres_Plus_list'][Idx])/Freq_Conv

    fig, ax = plt.subplots(1,2, figsize=(9,4.5))
    ax_coupl = ax[0]
    ax_frac = ax[1]

    cb1_lbl = r'J (kHz)' # r'$\Theta$ (pi rad)'
    # xy_J_scatter = ax.scatter(dx_arr, dz_arr, s = 1, c = np.abs(dy_arr), cmap = 'magma')
    ax_coupl.grid(alpha = 0.4)
    xz_J_scatter = ax_coupl.scatter(dx_arr, dz_arr, s = 1, c = np.abs(J_arr), cmap = 'magma', norm = matplotlib.colors.LogNorm())
    cbar1 = fig.colorbar(xz_J_scatter, ax = ax_coupl, label = cb1_lbl)
    ax_coupl.set_xlabel(r'$\delta_x$ (nm)')
    ax_coupl.set_ylabel(r'$\delta_z$ (nm)')
    


    discretisation = 100
    dx_arr_plot = np.linspace(0,max(dx_arr),discretisation)
    dz_arr_plot = np.linspace(0,max(dz_arr),discretisation)
    DX, DZ = np.meshgrid(dx_arr_plot, dz_arr_plot)
    frac_matrix = np.zeros(DX.shape)
    N_couplings = len(dx_arr)
    for i,x in enumerate(tqdm(dx_arr_plot)):
        for j,z in enumerate(dz_arr_plot):
            N_leq_x = set( np.where( np.abs(dx_arr) <= x )[0] )
            N_leq_z = set( np.where( np.abs(dz_arr) <= z )[0] )
            len_set = float( len( N_leq_x & N_leq_z ) )
            frac_matrix[i,j] = 1-len_set/N_couplings

    frac_plot = ax_frac.pcolor(DX,DZ,frac_matrix, cmap = 'magma')
    cbar2 = fig.colorbar(frac_plot, ax = ax_frac, label = '')
    ax_frac.set_xlabel(r'$|\delta_{x,thres}|$ (nm)')
    ax_frac.set_ylabel(r'$|\delta_{z,thres}|$ (nm)')    
    # ax_frac.contour(frac_matrix, levels = [0.1])

    fig.tight_layout()

    # Calculate probability to find a randomly sampled coupling that involves spins that you can spectrally distringuish, given you can create a sufficient field over lx in Bx and lz in Bz
    lx = 7
    lz = 8
    x_Idx = min(np.where(DX[1,:]>lx)[0])
    z_Idx = min(np.where(DZ[:,1]>lz)[0])
    P = frac_matrix[z_Idx,x_Idx]
    print(P)
    Str = 'Probability to find a randomly sampled coupling which involves spins that \nyou can spectrally distinguish, given you are able to create a sufficiently large field over \nan x-distance of %2.1f nm and a z-distance of %2.1f nm. P = %3.1f%%' %(lx,lz,100*P)
    print(Str)


    if save_fig == True:
        save_str = 'xz_positions_strongly_coupled.pdf'
        plt.savefig(os.path.join(save_folder,save_str),format='pdf',bbox_inches = 'tight',transparent=True,pad_inches=0.1 , dpi=400)

    if save_data_to_publish:
        # save_str = 'fig_3a_CouplingStrength_2D_xz'
        # np.savez(os.path.join(data_publish_folder,save_str), delta_x = dx_arr, delta_z = dz_arr, colorcode = np.abs(J_arr)) 

        # save_str = 'fig_3b_CouplingStrength_2D_abs_xz'
        # np.savez(os.path.join(data_publish_folder,save_str), abs_delta_x = DX, abs_delta_z = DZ, P = frac_matrix)    


        save_str = 'fig_3a_CouplingStrength_2D_xz.pkl'
        data_dictionary = {'dx_arr':dx_arr, \
                            'dz_arr':dz_arr, \
                            'colorcode':np.abs(J_arr)}
        data_dict_to_pkl(data_dictionary, save_str)

        save_str = 'fig_3b_CouplingStrength_2D_abs_xz.pkl'
        data_dictionary = {'DX':DX, \
                            'DZ':DZ, \
                            'P':frac_matrix}
        data_dict_to_pkl(data_dictionary, save_str)




















