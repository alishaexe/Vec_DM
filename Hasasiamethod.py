import numpy as np
import pandas as pd
import hasasia
import hasasia.sensitivity as hsen
import hasasia.sim as hsim
from hasasia.sim import Pulsar
import matplotlib.pyplot as plt
from scipy.integrate import quad
#%%

sensitivity_file  = ("/Users/alisha/Documents/Vec_DM/sensitivity_curves_NG15yr_fullPTA.txt")
strain_file = "/Users/alisha/Documents/Vec_DM/characteristic_strain_noise_power_spectra_NG15yr_psrs.txt"
noise_file = "/Users/alisha/Documents/Vec_DM/residual_noise_power_spectra_NG15yr_psrs.txt"

# Load the characteristic strain noise spectra
strain_df = pd.read_csv(strain_file, sep=",")
freqs_strain = strain_df.iloc[:, 0].values  # Frequencies
strain_data = strain_df.iloc[:, 1:].values  # Noise data for each pulsar
pulsar_names = strain_df.columns[1:].tolist()  # Pulsar names from header

# Load the residual noise power spectra
residual_noise_df = pd.read_csv(noise_file, sep=",")
freqs_residual = residual_noise_df.iloc[:, 0].values  # Frequencies
residual_noise_data = residual_noise_df.iloc[:, 1:].values

# Load the sensitivity curves
sensitivity_df = pd.read_csv(sensitivity_file, sep=",")
freqs_sensitivity = sensitivity_df.iloc[:, 0].values
characteristic_strain = sensitivity_df.iloc[:, 1].values
#strain power spectral density
spsd = sensitivity_df.iloc[:, 2].values
Omeggw = sensitivity_df.iloc[:, 3].values

#%%
# Example pulsar: Create a Pulsar object for each pulsar
pulsars = []
for i, pulsar_name in enumerate(pulsar_names):
    pulsar = hsen.Pulsar(
        toas=15.0,  # Nanograv 15-year data
        # sigma=np.mean(strain_data[:, i]),  # Approximate timing noise
        # cad=1.0,  # Approximate cadence in years (adjust as necessary)
    )
    pulsars.append(pulsar)

pta = hasasia.PTA(pulsars)

# Compute PTA sensitivity using hasasia
frequencies = hasasia.Frequencies(logf_range=(-9, -7), Nf=100)  # Adjust range if needed
pta_sensitivity = pta.sensitivity(frequencies)

# Plot the computed vs provided sensitivity curve
plt.loglog(freqs_sensitivity, characteristic_strain, label="Nanograv Data")
plt.loglog(frequencies.f, pta_sensitivity, label="Hasasia Sensitivity", linestyle="--")

plt.xlabel("Frequency [Hz]")
plt.ylabel("Characteristic Strain")
plt.legend()
plt.grid(which="both", linestyle="--", linewidth=0.5)
plt.title("Comparison of Nanograv and Hasasia Sensitivity")
plt.show()

#%%
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [5,3]
# plt.rcParams['text.usetex'] = True

phi = np.random.uniform(0, 2*np.pi,size=34)
cos_theta = np.random.uniform(-1,1,size=34)
#This ensures a uniform distribution across the sky.
theta = np.arccos(cos_theta)

timespan=[11.4 for ii in range(10)]
timespan.extend([3.0 for ii in range(24)])


freqs = np.logspace(np.log10(5e-10),np.log10(5e-7),500)

psrs = hsim.sim_pta(timespan=timespan, cad=23, sigma=1e-7,
                    phi=phi,theta=theta)

psrs2 = hsim.sim_pta(timespan=timespan,cad=23,sigma=1e-7,
                     phi=phi,theta=theta,
                     A_rn=6e-16,alpha=-2/3.,freqs=freqs)

spectra = []
for p in psrs:
    sp = hsen.Spectrum(p, freqs=freqs)
    sp.NcalInv
    spectra.append(sp)


spectra2 = []
for p in psrs2:
    sp = hsen.Spectrum(p, freqs=freqs)
    sp.NcalInv
    spectra2.append(sp)


sc1a = hsen.GWBSensitivityCurve(spectra)
sc1b = hsen.DeterSensitivityCurve(spectra)
sc2a = hsen.GWBSensitivityCurve(spectra2)
sc2b = hsen.DeterSensitivityCurve(spectra2)

#%%
plt.figure(figsize=(6, 7))
plt.loglog(sc1a.freqs,sc1a.Omega_gw,label='w/o GWB')
plt.loglog(sc2a.freqs,sc2a.Omega_gw,'--',label='w/ GWB')
plt.loglog(frequency, PLS, label = "PLS")
plt.loglog(freqs, Omega_GW, label = r"$\Omega_{GW}$")
plt.loglog(anafreq1, anaomeg1, label = "Case 1 (analytical)", color = "black")
plt.xlabel(r'$Frequency [Hz]$')
plt.ylabel(r'$Energy Density, \Omega_{gw}$')
plt.legend()
plt.show()
#%%

hgw = hsen.Agwb_from_Seff_plaw(sc1a.freqs, Tspan=sc1a.Tspan, SNR=3,
                               S_eff=sc1a.S_eff)

#We calculate the power law across the frequency range for plotting.
fyr = 1/(365.25*24*3600)
plaw_h = hgw*(sc1a.freqs/fyr)**(-2/3)

PI_sc, plaw = hsen.PI_Omega_GW(freqs=sc1a.freqs, Tspan=sc1a.Tspan,
                         SNR=3, S_eff=sc1a.S_eff, N=30)

for ii in range(plaw.shape[1]):
    plt.loglog(sc1a.freqs,plaw[:,ii],
               color='gray',lw=0.5)
plt.loglog(sc1a.freqs,plaw_h,color='C1',lw=2,
           label=r'SNR=3, $\alpha=-2/3$')
plt.loglog(sc1a.freqs,sc1a.h_c, label='Stochastic Sensitivity')
plt.loglog(sc1a.freqs,PI_sc, linestyle=':',color='k',lw=2,
           label='PI Stochastic Sensitivity')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Characteristic Strain, $h_c$')
plt.axvline(fyr,linestyle=':')
plt.title('Power Law Integrated Stochastic Senstivity Curve')
plt.ylim(hgw*0.75,2e-11)
plt.text(x=4e-8,y=3e-16,
         s=r'$A_{\rm GWB}$='+'{0:1.2e}'.format(hgw),
         bbox=dict(facecolor='white', alpha=0.9))
plt.legend(loc='upper left')
plt.show()
#%%
plt.loglog(sc1a.freqs,(sc1a.h_c)**2*(2*pi**2)/(3*H0**2)*sc1a.freqs**2, label='Stochastic Sensitivity')
plt.loglog(sc1a.freqs,(PI_sc)**2*(2*pi**2)/(3*H0**2)*sc1a.freqs**2, linestyle=':',color='k',lw=2,
           label='PI Stochastic Sensitivity')
# plt.loglog(sc1a.freqs,sc1a.Omega_gw,label='w/o GWB')
# plt.loglog(sc2a.freqs,sc2a.Omega_gw,'--',label='w/ GWB')
plt.loglog(frequency, PLS, label = "PLS")
plt.loglog(freqs, Omega_GW, label = r"$file \Omega_{GW}$")
plt.loglog(anafreq1, anaomeg1, label = "Case 1 (analytical)", color = "black")
plt.xlabel(r'$Frequency [Hz]$')
plt.ylabel(r'$Energy Density, \Omega_{gw}$')
plt.legend()
plt.show()