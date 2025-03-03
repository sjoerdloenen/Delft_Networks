import numpy as np
### general parameters of the nv-carbon system
params = {}

# physical constants
params['gamma_e']	   	= 28.031e9 * (2*np.pi)		    # Electron gyromagnetic ratio rad/s/T
params['gamma_c']      	= 1.0705e7 * (2*np.pi)  			# Carbon gyromagnetic ratio rad/s/T
params['mu0']          	= np.pi * 4e-7                   # H/m
params['hbar']   	   	= 1.0545718e-34     				# J/(rad/s)	
params['a0'] 	   		= 3.5668e-10       				# lattice constant diamond

# system specific parameters
params['13C_density']  	= 0.0107                         # density of 13C isotope

# variable parameters
params['B_z']      = 0.0403553       		# Tesla
