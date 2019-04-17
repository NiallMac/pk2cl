import numpy as np
import scipy.interpolate as interp

#read in P(k,z)
z,k,Pk = (np.loadtxt("test_output_pkhighres/matter_power_lin/z.txt"),
	      np.loadtxt("test_output_pkhighres/matter_power_lin/k_h.txt"),
	      np.loadtxt("test_output_pkhighres/matter_power_lin/p_k.txt"))

#select closest k to k_growth and output D(z) = P(k)
k_growth=1e-3
ind_k_growth = (np.abs(k-k_growth)).argmin()

growth = np.sqrt(Pk[:,ind_k_growth] / Pk[0,ind_k_growth])
np.savetxt("growth.dat", growth)
