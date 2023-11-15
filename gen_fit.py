# -*- coding: utf-8 -*-
"""
Created Mar 10th, 2021

Spec_fit.py:
Fits absorbance of 3 species spectrum to HITRAN model

@author: Carl
"""

import hapi
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model, Parameters, report_fit
import math
from scipy import interpolate

def runfit(wvn,data,T_i,P_i,x_co2_i,x_co_i,x_ch4_i):
    """
    ~Creates model and runs fit on 3 species data set
    wvn: Array of wavenumbers to fit over
    data: Array of data to fit
    T_i: initial guess of Temperature (assumes same temp on all molecules)
    P_i: initial guess of Pressure (assumes same pressure on all molecules)
    x_i: array of concentraions for each molecule
    molecules: array of strings with molecules names
    molID: array of molID numbers (from HITRAN)
    **NOTE**: x_i, molecules, and molID should have same molecule order

    """
    #pulls linelist data from HITRAN, assumes 1st Isotopologue
    isotopologueID = 1
    mol_id = 2
    hapi.fetch('CO2',mol_id,isotopologueID,wvn[0],wvn[-1])
    #hapi.fetch('CO2',mol_id,isotopologueID,wvn[0],wvn[-1],ParameterGroups=['SDVoigt'])
    mol_id = 5
    hapi.fetch('CO',mol_id,isotopologueID,wvn[0],wvn[-1])
    mol_id = 6
    hapi.fetch('CH4',mol_id,isotopologueID,wvn[0],wvn[-1])
    #hapi.fetch_by_ids('CH4',[1,2],wvn[0],wvn[-1])

    #initialize parameters
    params = Parameters()

    #add relevent parameters for model
    params.add('T',value = T_i, vary = True, min = 0.0, max = 2000)
    params.add('P',value = P_i, vary = True, min = 0.0, max = 10)
    params.add('x_co2',value = x_co2_i, vary = True, min = 0.0, max = 1)
    params.add('x_co',value = x_co_i, vary = True, min = 0.0, max = 1)
    params.add('x_ch4',value = x_ch4_i, vary = True, min = 0.0, max = 1)
    params.add('v_co2', value = 0, vary = True, min = -1, max = 1)
    params.add('v_co', value = 0, vary = True, min = -1, max = 1)
    params.add('v_ch4', value = 0, vary = True, min = -1, max = 1)

    #create Model and fit
    gmodel = Model(SpecSim)

    bl =750#90


    etalons=[]

    weights = np.ones(len(data))
    for i in range(bl):
        weights[i] = 0
        weights[-1 - i] = 0
    for pair in etalons:
        for i in range(pair[1]-pair[0]):
            weights[pair[0]+i] = 0
            weights[-1*(pair[0]+i)] = 0
    hapi.db_begin()
    results = gmodel.fit(data, params=params, wvn = wvn, weights=weights)

    print(results.fit_report(results.params))

    #fit results
    T_bestfit = results.best_values['T']
    P_bestfit = results.best_values['P']
    x_co2_bestfit = results.best_values['x_co2']
    x_co_bestfit = results.best_values['x_co']
    x_ch4_bestfit = results.best_values['x_ch4']
    #print('\t\tBest fits\nT: ',T_bestfit,'\nP: ',P_bestfit,'\nCO2 mol: ',x_co2_bestfit,'\nCO mol: ',x_co_bestfit,'\nCH4 mol: ',x_ch4_bestfit)

    #Plot
    bestfit_spectrum = results.best_fit
    originalData = results.data
    base = (1 - results.weights)*(originalData - bestfit_spectrum)

#for td_plots

    fig,ax = plt.subplots()
    plt.plot(originalData,label='Original Data')
    plt.plot(bestfit_spectrum,label='Fit')
    plt.plot(results.weights,label='weighting function')
    plt.legend()

# #for fd_plots
    fig2,ax2 = plt.subplots(2,1,sharex='col')
    ax2[0].plot(wvn,np.fft.rfft(results.data - base),label='Original Data') #does data vs orginalData make a difference
    ax2[0].plot(wvn,np.fft.rfft(bestfit_spectrum),label = 'Fit')
    #ax2[0].plot(wvn,np.fft.rfft(results.data-base),label='Original Data') #added 21.831
    ax2[0].set_ylabel('Absorbance')
    ax2[1].plot(wvn,(np.fft.rfft((results.data-base) - bestfit_spectrum) ))
    ax2[1].set_ylabel('Residuals')
    ax2[0].legend()

    # fig2,ax2 = plt.subplots(2,1,sharex='col')
    # ax2[0].plot(wvn,np.fft.rfft(results.data),label='Original Data') #does data vs orginalData make a difference
    # ax2[0].plot(wvn,np.fft.rfft(bestfit_spectrum),label = 'Fit')
    # #ax2[0].plot(wvn,np.fft.rfft(results.data-base),label='Original Data') #added 21.831
    # ax2[0].set_ylabel('Absorbance')
    # ax2[1].plot(wvn,(np.fft.rfft((results.data) - bestfit_spectrum) ))
    # ax2[1].set_ylabel('Residuals')
    # ax2[0].legend()

    # #td_test
    # fig2,ax2 = plt.subplots(2,1,sharex='col')
    # ax2[0].plot(wvn,(originalData-bl),label='Original Data') #does data vs orginalData make a difference
    # ax2[0].plot(wvn,bestfit_spectrum,label = 'Fit')
    # ax2[0].set_ylabel('Absorbance')
    # ax2[1].plot(wvn,((originalData-bl) - bestfit_spectrum ))
    # ax2[1].set_ylabel('Residuals')
    # ax2[0].legend()
    plt.show()

def SpecSim(wvn,T,P,x_co2,x_co,x_ch4,v_co2,v_co,v_ch4):
    """
    ~Defines function to be modeled
    wvn: range of Wavenumbers being considered
    T: inital Temperature guess
    P: initial Pressure guess
    x_i: concentraion guess for ith species
    v_i:freqencey shift of ith species
    """

    #print(T,P,x_co2,x_co,x_ch4)
    c = 2.99792458e8
    res = 0.1 #GHz
    res =res/c*1e7
    isotopologueID = 1
    path_length = 316#cm
    hapi.db_begin() #finds HITRAN data
    #CO2
    mol_id = 2
    mol_abundance = 0.984204 #abundance of Isotopologue from HITRAN
    #MIDS = (mol_id,isotopologueID,mol_abundance)
    MIDS = (mol_id,isotopologueID,x_co2)
    #nu, coeff = hapi.absorptionCoefficient_Voigt(Components = [MIDS], SourceTables = 'CO2', Environment = {'p':P,'T':T}, Diluent = {'self':x_co2, 'air':1-x_co2}, OmegaStep=res, HITRAN_units = False, OmegaGrid = wvn , OmegaWingHW = 50.)
    #nu, coeff = hapi.absorptionCoefficient_Voigt(Components = [MIDS], SourceTables = 'CO2', Environment = {'p':P,'T':T}, Diluent = {'self':x_co2, 'air':1-x_co2}, OmegaStep=res, HITRAN_units = False, OmegaGrid = wvn + v_co2, OmegaWingHW = 50.)
    nu, coeff = hapi.absorptionCoefficient_SDVoigt(Components = [MIDS], SourceTables = 'CO2', Environment = {'p':P,'T':T}, Diluent = {'self':x_co2, 'air':1-x_co2}, OmegaStep=res, HITRAN_units = False, OmegaGrid = wvn + v_co2, OmegaWingHW = 50.)
    nu, trans = hapi.transmittanceSpectrum(nu, coeff, Environment = {'l':path_length,'p': P, 'T':T})

    absorb_co2 = -1*np.log(trans)


    #CO
    del nu, coeff, trans, MIDS, mol_abundance
    mol_id = 5
    mol_abundance = 0.986544 #abundance of Isotopologue from HITRAN
    #MIDS = (mol_id,isotopologueID,mol_abundance)
    MIDS = (mol_id,isotopologueID,x_co)
    #nu, coeff = hapi.absorptionCoefficient_Voigt(Components = [MIDS], SourceTables = 'CO', Environment = {'p':P,'T':T}, Diluent = {'self':x_co, 'air':(1-x_co)}, OmegaStep=res, HITRAN_units = False, OmegaGrid = wvn , OmegaWingHW = 50.)
    nu, coeff = hapi.absorptionCoefficient_Voigt(Components = [MIDS], SourceTables = 'CO', Environment = {'p':P,'T':T}, Diluent = {'self':x_co, 'air':(1-x_co)}, OmegaStep=res, HITRAN_units = False, OmegaGrid = wvn + v_co, OmegaWingHW = 50.)
    nu, trans = hapi.transmittanceSpectrum(nu, coeff, Environment = {'l':path_length,'p': P, 'T':T})
    absorb_co = -1*np.log(trans)

    #CH4_1
    del nu, coeff, trans, MIDS, mol_abundance
    #hapi.fetch('CH4',mol_id,1,wvn[0],wvn[-1])
    mol_id = 6
    mol_abundance = 0.988274 #abundance of Isotopologue from HITRAN
    #MIDS = (mol_id,isotopologueID,mol_abundance)
    MIDS = (mol_id,1,x_ch4)
    #nu, coeff = hapi.absorptionCoefficient_Voigt(Components = [MIDS], SourceTables = 'CH4', Environment = {'p':P,'T':T}, Diluent = {'self':x_ch4, 'air':(1-x_ch4)}, OmegaStep=res, HITRAN_units = False, OmegaGrid = wvn , OmegaWingHW = 50.)
    nu, coeff = hapi.absorptionCoefficient_Voigt(Components = [MIDS], SourceTables = 'CH4', Environment = {'p':P,'T':T}, Diluent = {'self':x_ch4, 'air':(1-x_ch4)}, OmegaStep=res, HITRAN_units = False, OmegaGrid = wvn + v_ch4, OmegaWingHW = 50.)
    nu, trans = hapi.transmittanceSpectrum(nu, coeff, Environment = {'l':path_length,'p': P, 'T':T})
    absorb_ch4 = -1*np.log(trans)

    # #CH4_2
    # del nu, coeff, trans, MIDS, mol_abundance
    # hapi.fetch('CH4',mol_id,2,wvn[0],wvn[-1])
    # mol_id = 6
    # mol_abundance = 0.988274 #abundance of Isotopologue from HITRAN
    # #MIDS = (mol_id,isotopologueID,mol_abundance)
    # MIDS = (mol_id,2,x_ch4)
    # #nu, coeff = hapi.absorptionCoefficient_Voigt(Components = [MIDS], SourceTables = 'CH4', Environment = {'p':P,'T':T}, Diluent = {'self':x_ch4, 'air':(1-x_ch4)}, OmegaStep=res, HITRAN_units = False, OmegaGrid = wvn , OmegaWingHW = 50.)
    # nu, coeff = hapi.absorptionCoefficient_Voigt(Components = [MIDS], SourceTables = 'CH4', Environment = {'p':P,'T':T}, Diluent = {'self':x_ch4, 'air':(1-x_ch4)}, OmegaStep=res, HITRAN_units = False, OmegaGrid = wvn + v_ch4, OmegaWingHW = 50.)
    # nu, trans = hapi.transmittanceSpectrum(nu, coeff, Environment = {'l':path_length,'p': P, 'T':T})
    # absorb_ch4_2 = -1*np.log(trans)

    absorb = absorb_co2 + absorb_co + absorb_ch4# + absorb_ch4_2
    return np.fft.irfft(absorb)

def maxPrimeFactors (n):
    # Initialize the maximum prime factor
    maxPrime = -1
    # Print the number of 2s that divide n
    while n % 2 == 0:
        maxPrime = 2
        n >>= 1     # equivalent to n /= 2
    # n must be odd at this point,  # thus skip the even numbers and  # iterate only for odd integers
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        while n % i == 0:
            maxPrime = i
            n = n / i
    # This condition is to handle the      # case when n is a prime number      # greater than 2
    if n > 2:
        maxPrime = n
    return int(maxPrime)

#wave_num = np.arange(6170,6300,.007)
isotopologueID = 1
mol_id_co2 = 2
mol_id_co = 5
mol_id_ch4 = 6

##~~for real data

Data = np.loadtxt('C:/Users/Carl/Desktop/hcf/new_data/last_mix_dat.txt')
wvn_full = Data[:,0]
data_full = Data[:,1]


bandwitdh = [6125,6200]

print(wvn_full[-1],wvn_full[0])
#~~trim data to relevant parts
k = False
d_range = 0
for i in range(len(wvn_full)):
    #increasing or decreasing wavenumber
    if wvn_full[0] < bandwitdh[0]:
        #start counting where to trim
        if wvn_full[i]>= bandwitdh[0] and wvn_full[i] <= bandwitdh[-1]:
            while k == False:
                ind_start = i
                k = True
            d_range += 1

    elif wvn_full[0] > bandwitdh[-1]:
        if wvn_full[i]<= bandwitdh[0] and wvn_full[i] >= bandwitdh[-1]:
            while k == False:
                ind_start = i
                k = True
            d_range += 1

#~~ test to find length with largest prime factor
# looks in last 100 data incdices to find which length gives largest prime factor

#for i in range(100):
#    d_len = len(data) - i
#    print(i,maxPrimeFactors(d_len))


# found n from largest prime factor test
n = 15
wave_num = np.empty(d_range-n)
data = np.empty(d_range-n)
for i in range(len(wave_num)):
    wave_num[i] = wvn_full[ind_start + i]
    data[i] = data_full[ind_start + i]

#~ Pulls in relevent HITRAN data (un comment if not running fits just modeling)
#wave_num = np.arange(6100,6300, 0.001)
#hapi.fetch('CO2',mol_id_co2,isotopologueID,wave_num[0],wave_num[-1])
#hapi.fetch('CO',mol_id_co,isotopologueID,wave_num[0],wave_num[-1])
#hapi.fetch('CH4',mol_id_ch4,isotopologueID,wave_num[0],wave_num[-1])

#~ To create simulation model
#data_sim = SpecSim(wave_num,295,2.11,.221,.195,.032,0,0,0)
#data_sim = SpecSim(wave_num,295,1.86,.999)

data = np.fft.irfft(-np.log(data))
runfit(wave_num,data,295,1.9,.191,.168,.046)

#%%
