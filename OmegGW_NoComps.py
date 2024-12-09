import numpy as np
import matplotlib.pyplot as plt
#%%

#Loading all the datafiles
otab = np.load("datafiles/jaxOmeg.npy")
xt, te = otab[:,0], otab[:,1]

data = [
    [0.2, 1.68575e-6], [0.218547, 2.09599e-6], [0.238813, 2.60489e-6], 
    [0.260959, 3.28666e-6], [0.285159, 4.17816e-6], [0.311603, 5.20304e-6],
    [0.340499, 6.28363e-6], [0.372075, 7.53625e-6], [0.406578, 9.18055e-6], 
    [0.444282, 1.12799e-5], [0.485482, 1.35966e-5], [0.530502, 1.5915e-5], 
    [0.579697, 1.89958e-5], [0.633455, 2.30602e-5], [0.692197, 2.5934e-5], 
    [0.756387, 2.94267e-5], [0.82653, 3.50427e-5], [0.903177, 3.80892e-5], 
    [0.986931, 4.40926e-5], [1.07845, 4.77267e-5], [1.17846, 5.48365e-5],
    [1.28774, 5.84677e-5], [1.40716, 6.52861e-5], [1.53765, 7.09824e-5], 
    [1.68025, 7.60901e-5], [1.83606, 8.20633e-5], [2.00632, 8.83102e-5], 
    [2.19238, 9.38795e-5], [2.39569, 9.80632e-5], [2.61785, 1.01594e-4], 
    [2.86061, 1.04401e-4], [3.12588, 1.06852e-4], [3.41576, 1.08298e-4], 
    [3.73251, 1.09993e-4], [4.07864, 1.10491e-4], [4.45687, 1.11707e-4], 
    [4.87017, 1.12119e-4], [5.3218, 1.12159e-4], [5.8153, 1.12411e-4], 
    [6.35458, 1.12717e-4], [6.94386, 1.12775e-4], [7.58779, 1.12734e-4], 
    [8.29143, 1.12672e-4], [9.06033, 1.12485e-4], [9.90052, 1.12151e-4], 
    [10.8186, 1.12031e-4], [11.8219, 1.11643e-4], [12.9182, 1.11308e-4], 
    [14.1161, 1.10919e-4], [15.4252, 1.10489e-4], [16.8556, 1.09984e-4], 
    [18.4187, 1.09314e-4], [20.1267, 1.0859e-4], [21.9931, 1.07804e-4], 
    [24.0326, 1.06799e-4], [26.2612, 1.05714e-4], [28.6965, 1.044e-4], 
    [31.3577, 1.02936e-4], [34.2656, 1.0123e-4], [37.4431, 9.93014e-5], 
    [40.9154, 9.70536e-5], [44.7096, 9.45595e-5], [48.8557, 9.17463e-5], 
    [53.3862, 8.86179e-5], [58.3369, 8.50337e-5], [63.7467, 8.11187e-5], 
    [69.6582, 7.67686e-5], [76.1178, 7.21193e-5], [83.1765, 6.70595e-5], 
    [90.8898, 6.16338e-5], [99.3183, 5.59066e-5], [108.528, 4.99202e-5], 
    [118.593, 4.38206e-5], [129.59, 3.77656e-5], [141.608, 3.1894e-5], 
    [154.739, 2.63081e-5], [169.089, 2.11025e-5], [184.769, 1.64908e-5], 
    [201.903, 1.26811e-5], [220.627, 9.85718e-6], [241.086, 7.9531e-6], 
    [263.443, 6.73748e-6], [287.873, 6.01661e-6], [314.568, 5.66114e-6], 
    [343.739, 5.57368e-6], [375.615, 5.44249e-6], [410.448, 4.87802e-6], 
    [448.51, 4.03016e-6], [490.102, 3.27005e-6], [535.551, 2.64619e-6], 
    [585.214, 2.11117e-6], [639.483, 1.8003e-6], [698.785, 1.83914e-6], 
    [763.586, 1.59903e-6], [834.396, 1.36485e-6], [911.772, 1.24355e-6], 
    [996.324, 9.94898e-7], [1088.72, 8.61792e-7], [1189.68, 8.04384e-7], 
    [1300.0, 7.45641e-7]
]

# Split data into x and y vectors
xdat, ydat = np.array(data).T

PTA = np.loadtxt("/Users/alisha/Documents/Vec_DM/datafiles/sensitivity_curves_NG15yr_fullPTA.txt",delimiter=',', skiprows=1)
freqs = PTA[:, 0]
h_c = PTA[:, 1] # characteristic strain
S_eff = PTA[:, 2] #strain power spectral density
Omega_GW = PTA[:, 3] #calculated using H0=67.4 km/s/Mpc = 67.4*1e3/1/3e22 s^-1 to match the paper 


#%%
# Case 1 compared with Nano and Ska

fgw1 = 1e-11

def case1(y):
    res = y*1e-6
    return res

def freq1(f):
    res = f*fgw1
    return res

def OmegAnalytical(f):
    a1 = 2
    a2 = 3
    
    n1 = 2.6
    n2 = -0.05
    n3 = -2
    
    f1 = 1
    f2 = 90
    
    t1 = (f/f1)**n1
    t2 = (1+(f/f1)**a1)**((-n1+n2)/a1)
    t3 = (1+(f/f2)**a2)**((-n2+n3)/a2)
    return 1.25e-4*t1*t2*t3

analytical = np.array(list(map(OmegAnalytical, xt)))


#For case 1 using the analytical fit we get
anaomeg1 = np.array(list(map(case1, analytical)))
anafreq1 = np.array(list(map(freq1, xt)))

plt.figure(figsize=(5, 7))
# plt.loglog(frequency, PLS, label = "PLS", color = "orangered")
plt.loglog(freqs, Omega_GW, label = r"Nanograv", color = "indigo")
# plt.loglog(anafreq1, case1(te), label ="numerical")
plt.loglog(freqs, Omega_GW*1e-2, label = r"SKA", color = "black")
plt.loglog(anafreq1, anaomeg1, label = "Our Model (Case 1)", color = "blue")
plt.ylabel(r"$\Omega_{GW}$", fontsize = 14)
plt.xlabel(r"$f (Hz)$", fontsize = 14)
plt.legend(loc = 0, fontsize = 12)
plt.grid(True)
# plt.savefig('OverlayOmegGW_skanano1.png', bbox_inches='tight')
plt.show()

#%%
#Case 2 compared with SKA and Nano
fgw2 = 7e-11

def case2(y):
    res = y*1e-7
    return res

def freq2(f):
    res = f*fgw2
    return res

def OmegAnalytical(f):
    a1 = 2
    a2 = 3
    
    n1 = 2.6
    n2 = -0.05
    n3 = -2
    
    f1 = 1
    f2 = 90
    
    t1 = (f/f1)**n1
    t2 = (1+(f/f1)**a1)**((-n1+n2)/a1)
    t3 = (1+(f/f2)**a2)**((-n2+n3)/a2)
    return 1.25e-4*t1*t2*t3

analytical = np.array(list(map(OmegAnalytical, xt)))

# plt.loglog(xt, analytical)
# plt.show()

#For case 1 using the analytical fit we get
anaomeg2 = np.array(list(map(case2, analytical)))
anafreq2 = np.array(list(map(freq2, xt)))
# #%%
plt.figure(figsize=(5, 7))
# plt.loglog(frequency, PLS, label = "PLS", color = "orangered")
plt.loglog(freqs, Omega_GW, label = r"Nanograv", color = "indigo")
# plt.loglog(anafreq1, case1(te), label ="numerical")
plt.loglog(freqs, Omega_GW*1e-2, label = r"SKA", color = "black")
plt.loglog(anafreq2, anaomeg2, label = "Our Model (Case 2)", color = "green")
plt.ylabel(r"$\Omega_{GW}$", fontsize = 14)
plt.xlabel(r"$f (Hz)$", fontsize = 14)
plt.legend(loc = 0, fontsize = 12)
plt.grid(True)
# plt.savefig('OverlayOmegGW_skanano2.png', bbox_inches='tight')
plt.show()
#%%