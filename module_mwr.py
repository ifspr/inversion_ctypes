#!/usr/bin/env python


import platform
import numpy as np
import math
import matplotlib.pyplot as plt

zeros = np.zeros
array = np.array
exp = np.exp
pi = np.pi
tile = np.tile

def abs_param_l87(contcorr):
  """
  Adapted from abs_param_l87.pro (UL)
  Reads numbers (to calculate absoprtion coefficients) from files,
    given the choice for continuum correction.  
  """
  if (platform.system() == 'Linux'): homedir = '/home/stephanie/'
  elif (platform.system() == 'Darwin'): homedir = '/Volumes/stephanie/'
  oxygen = np.loadtxt(homedir+'uwave/stp/coefficients/oxygen_l87.dat',skiprows=2)
  water = np.loadtxt(homedir+'uwave/stp/coefficients/water_l87.dat',skiprows=2)

  A_l87 = oxygen[:,1:7]
  A_l87 = A_l87.transpose()
  F0O2_l87 = oxygen[:,0]
  B_l87 = water[:,1:4]
  B_l87 = B_l87.transpose() 
  F0water_l87 = water[:,0]
  
  return [F0O2_l87, A_l87, F0water_l87, B_l87]  # a list of np array


def abwvl87(rho,temp,pres,freq,f0water,b_coeff,contcorr):
  """
  Adapted from abwvl87.pro (UL)
  Computes absorption coefficient due to water vapor acc. to Liebe 1987,
    given profiles of humidity[kg/m3], temperature[K], pressure[Pa], frequency channels[GHz]
    and other model-related input.
  """
  normtemp = 300./temp                       
  wvpres = rho * temp * 461.52 * 1.0e-2      
  pres = pres * 1.0E-2                       
  drypres = pres -wvpres                     
  
  zn = complex(0.,0.)
  
  for i in range(30):
    s = b_coeff[0,i]*wvpres*(normtemp**3.5)*exp(b_coeff[1,i]*(1.-normtemp))*1e-1
    gamh = b_coeff[2,i]*(drypres*(normtemp**0.6) + 4.8*wvpres*(normtemp**1.1))*1e-4
    delh = 0.
    term1 = complex(1.,-delh)/((f0water[i]-freq)+(-gamh*1j))
    term2 = complex(1.,delh)/((f0water[i]+freq)+(gamh*1j))
    zf = freq/f0water[i]*(term1-term2)
  
    if (i == 0): zn = s*zf
    if (i > 0): zn += s*zf

  alpha1 = 0.000182*freq*zn.imag/4.343
 
  #compute water vapor continuum absorption
  #The foreign and self broadened water vapor continuum coeff
  orig_BF = 1.13e-6
  orig_BS = 3.57e-5
  if (contcorr == 'tur09'): 
    multiplier_BF = 1.105
    multiplier_BS = 0.79
  else:
    multiplier_BF = multiplier_BS = 1.0

  #scale the foreign and self broadened water vapor continuum coeffs, 
  BF = orig_BF * multiplier_BF
  BS = orig_BS * multiplier_BS

  alpha2 = (BF*wvpres*drypres*(normtemp**3.)+BS*wvpres*wvpres*(normtemp**10.5))*freq*1e-2
  alpha2 = 0.000182*freq*alpha2/4.343

  return alpha1 + alpha2


def abo2l87(rho,temp,pres,freq,f0O2,a_coeff):
  """
  Adapted from abo2l93.pro (UL)
  Computes absorption coefficient of oxygen acc. to Liebe 1993,
    given profiles of humidity[kg/m3], temperature[K], pressure[Pa], frequency channels[GHz]
    and other model-related input.  
  """
  normtemp = 300./temp                       #norm. temperature
  wvpres = rho * temp * 461.52 * 1.0e-2      #water vapor pressure hPa
  pres = pres * 1.0E-2                       #pressure in hPa
  drypres = pres - wvpres                     #dry air pressure

  for i in range(48):
    gamma = 0.
    s = a_coeff[0,i] * drypres * (normtemp**3.0) * exp(a_coeff[1,i]*(1.-normtemp)) * 1.e-7
    gamma = a_coeff[2,i] * (drypres * normtemp**(0.8-a_coeff[5,i]) + 1.1 * wvpres * normtemp)*1.e-4
    delta = a_coeff[3,i] * drypres * normtemp**(a_coeff[4,i]) * 1e-4

    term1 = complex(1.,-delta)/((f0O2[i]-freq)+(-gamma*1j))
    term2 = complex(1.,delta)/((f0O2[i]+freq)+(gamma*1j))
    zf = freq/f0O2[i]*(term1 - term2)
    if (i == 0): zn = s*zf
    if (i > 0): zn += s*zf
  
  #oxygen line absorption
  alpha1 = 0.000182*freq*zn.imag/4.343
  alpha1[alpha1<0]=0.0       # or [0.0 if i < 0.0 else i for i in alpha1]

  #dry air continuum
  ao = 6.14e-4
  go = 4.8e-3*(drypres+1.1*wvpres)*(normtemp**0.8)
  trm1 = ao/(go*(1.+((freq*freq)/(go*go))))
  trm2 = 1.4e-10*(1.-(1.2e-5*(freq**1.5))) * drypres * (normtemp**1.5)
  alpha2 = (trm1 + trm2) * drypres * normtemp * normtemp * freq * 1e-2
  alpha2 = 0.000182*freq*alpha2/4.343
  return alpha1 + alpha2


def abs_param_l93(contcorr):
  """
  Adapted from abs_param_l93.pro (UL)
  Reads numbers (to calculate absoprtion coefficients) from files 
  """
  if (platform.system() == 'Linux'): homedir = '/home/stephanie/'
  elif (platform.system() == 'Darwin'): homedir = '/Volumes/stephanie/'

  if (contcorr  == 'tur09'): filewater = open(homedir+'uwave/stp/coefficients/water_l93_tur09.dat') 
  if (contcorr  == 'org'): filewater = open(homedir+'uwave/stp/coefficients/water_l93_org.dat')

  A_l93 = zeros([6,44])
  F0O2_l93 = zeros(44)
  B_l93 = zeros([6,35])
  F0water_l93 = zeros(35)
  
  fileO2 = open (homedir+'uwave/stp/coefficients/oxygen_l93.dat','r')

  nl=0               
  for line in fileO2:          
    if (nl!=0 and nl!=1): 
      p = line.split()
      for i in range(6): 
        F0O2_l93[nl-2] = float(p[0])
        A_l93[i,nl-2] = float(p[i+1])  
    nl+=1
  fileO2.close()

  nl=0
  for line in filewater:
    if (nl!=0 and nl!=1):
      p = line.split()
      for i in range(6): 
        F0water_l93[nl-2] = float(p[0])
        B_l93[i,nl-2] = float(p[i+1])  
    nl+=1
  filewater.close()

  return [F0O2_l93, A_l93, F0water_l93, B_l93]



def abwvl93(rho,temp,pres,freq,f0water,b_coeff):
  """
  Adapted from abwvl93.pro (UL)
  Calculates abs coeff. of water vapor acc. to Liebe 1993,
    given profiles of humidity[kg/m3], temperature[K], pressure[Pa], frequency channels[GHz]
    and other model-related input.
  """
  normtemp = 300./temp                       #normalized temperature
  wvpres = rho * temp * 461.52 * 1.0e-2      #water vapor pressure hPa
  pres = pres * 1.0E-2                       #pressure in hPa
  drypres = pres -wvpres                     #dry air pressure
  
  zn = complex(0.,0.)
  
  for i in range(35):
    gamh = 0.  
    s = b_coeff[0,i]*wvpres*(normtemp**3.5)*exp(b_coeff[1,i]*(1.-normtemp))
  
    #Doppler approximation
    gamh = b_coeff[2,i]*(drypres*(normtemp**b_coeff[4,i]) + b_coeff[3,i]*wvpres*(normtemp**b_coeff[5,i]))*1.e-3
    gamd2 = 1e-12/normtemp*(1.46*f0water[i])**2
    gamh = 0.535*gamh + (0.217*gamh**2 + gamd2)**0.5
    delh = 0.
    
    term1 = complex(1.,-delh)/((f0water[i]-freq)+(-gamh*1j))
    term2 = complex(1.,delh)/((f0water[i]+freq)+(gamh*1j))
    zf = freq/f0water[i]*(term1-term2)
  
    if (i == 0): zn = s*zf
    if (i > 0): zn += s*zf
  
  return 0.000182*freq*zn.imag/4.343



def abo2l93(rho,temp,pres,freq,f0O2,a_coeff):
  """
  Adapted from abo2l93.pro (UL) 
  Calculates abs. coeff. of oxygen acc. to Liebe 1993,
    given profiles of humidity[kg/m3], temperature[K], pressure[Pa], frequency channels[GHz]
    and other model-related input.
  """
  normtemp = 300./temp                       #norm. temperature
  wvpres = rho * temp * 461.52 * 1.0e-2    #water vapor pressure hPa
  pres = pres * 1.0E-2                 #pressure in hPa
  drypres = pres -wvpres                       #dry air pressure

  for i in range(44):
    gamma = 0.
    s = a_coeff[0,i] * drypres * (normtemp**3.0) * exp(a_coeff[1,i]*(1.-normtemp)) * 1.e-6
    gamma = a_coeff[2,i] * (drypres * normtemp**(0.8-a_coeff[3,i]) + 1.1 * wvpres * normtemp)*1.e-3
    gamma = (gamma*gamma + (25*0.6e-4)*(25*0.6e-4))**(0.5)
    delta = (a_coeff[4,i] + a_coeff[5,i]*normtemp)*(drypres + wvpres)*(normtemp**0.8)*1.e-3

    term1 = complex(1.,-delta)/((f0O2[i]-freq)+(-gamma*1j))
    term2 = complex(1.,delta)/((f0O2[i]+freq)+(gamma*1j))
    zf = freq/f0O2[i]*(term1 - term2)
    if (i == 0): zn = s*zf
    if (i > 0): zn += s*zf
  
  #oxygen line absorption
  alpha1 = 0.000182*freq*zn.imag/4.343
  alpha1[alpha1<0]=0.0       

  #dry air continuum
  so = 6.14e-5*drypres*normtemp*normtemp
  gammao = 0.56e-3*(drypres+wvpres)*normtemp**(0.8)
  zfo = -(freq+0j)/(freq+gammao*1j) 
  sn = 1.4e-12*drypres*drypres*(normtemp**3.5)
  ima = freq/(1.93e-5*freq**1.5+1.)
  zfn = ima*1j
  zn = so*zfo + sn*zfn
  
  #nonresonant dry air absorption
  alpha2 = 0.000182*freq*zn.imag/4.343  

  return alpha1 + alpha2 


def abliq(water,freq,temp):
  """
  A python version of abliq.pro (UL)
  Calculates the abs.coeff. for liquid water (cloud) acc. to Liebe 93,
    given LWC profile[g/m3], frequency channels[GHz] and temperature profile[K].
  """
  ttemp = tile(temp,(freq.size,1)).T  
  twater = tile(water,(freq.size,1)).T 
  tfreq = tile(freq,(temp.size,1))    
  theta1 = 1.- 300./ttemp
  eps0 = 77.66-103.3*theta1
  eps1 = 0.0671*eps0
  eps2 = 3.52                    #from MPM93  
  fp = 20.1*exp(7.88*theta1)     
  fs = 39.8*fp

  eps = (eps0-eps1)/(1.+(tfreq/fp)*1j) + (eps1-eps2)/(1.+(tfreq/fs)*1j) + eps2
  re = (eps-1.)/(eps+2.)
  return -0.06286*re.imag*tfreq*twater
 


def abwvr98(rho,temperature,pressure,frequency,cont_corr='tur09',linew_22='lil05'):
  """
  Adapted from abwvr98.pro (UL)
  Calculates absorption coefficient for oxygen
    given profiles of humidity[kg/m3], temperature[K], pressure[Pa], frequency channels[GHz]
    and other model-related input.
  """
  rho = rho*1000.                     #convert to g/m3
  pressure = pressure/100.            #convert to hPa
  nlines=15
  nf = frequency.size
  df = zeros(shape = [2,nf])  

  #line frequencies:
  fl = array([22.2351, 183.3101, 321.2256, 325.1529, 380.1974, 439.1508,
        443.0183, 448.0011, 470.8890, 474.6891, 488.4911, 556.9360,
        620.7008, 752.0332, 916.1712])

  #line intensities at 300K:
  s1 = array([.1310E-13, .2273E-11, .8036E-13, .2694E-11, .2438E-10,
        .2179E-11, .4624E-12, .2562E-10, .8369E-12, .3263E-11, .6659E-12,
        .1531E-08, .1707E-10, .1011E-08, .4227E-10])

  #T coeff. of intensities:
  b2 = array([2.144, .668, 6.179, 1.541, 1.048, 3.595, 5.048, 1.405,
        3.597, 2.379, 2.852, .159, 2.391, .396, 1.441])

  #T-exponent of air-broadening:
  xx = array([.69, .64, .67, .68, .54, .63, .60, .66, .66, .65, .69, .69,
       .71, .68, .70])

  if (linew_22 == 'lil05'):
    #Air-broadened width parameters at 300K:
    w3 = array([.002656, .00281, .0023, .00278, .00287, .0021, .00186,
          .00263, .00215, .00236, .0026, .00321, .00244, .00306, .00267])

    #Self-broadened width parameters at 300K:
    ws = array([.0127488, .01491, .0108, .0135, .01541, .0090, .00788,
          .01275, .00983, .01095, .01313, .01320, .01140, .01253, .01275])
  else:
    #Air-broadened width parameters at 300K:
    w3 = array([.00281, .00281, .0023, .00278, .00287, .0021, .00186,
          .00263, .00215, .00236, .0026, .00321, .00244, .00306, .00267])
    
    #Self-broadened witdh parameters at 300K
    ws = array([.01349, .01491, .0108, .0135, .01541, .0090, .00788,
          .01275, .00983, .01095, .01313, .01320, .01140, .01253, .01275])

  #T-exponent of self-broadening:
  xs = array([.61, .85, .54, .74, .89, .52, .50, .67, .65, .64, .72,
        1.0, .68, .84, .78])
  
  if (rho > 0.):
    pvap = rho*temperature/217.
    pda = pressure-pvap 
    den = 3.335e16*rho
    ti = 300./temperature
    ti2 = ti**2.5

  #Continuum terms
    bf_org = 5.43E-10 
    bs_org = 1.8E-8
    
    if (cont_corr == 'tur09'): 
      bf_mult = 1.105
      bs_mult = 0.79
    else: bf_mult = bs_mult = 1.0

    bf = bf_org*bf_mult  
    bs = bs_org*bs_mult
   
    con = (bf*pda*ti*ti*ti + bs*pvap*ti**7.5)*pvap*frequency*frequency

    #Add resonances
    thesum = 0.

    for i in range(nlines):
      width = w3[i]*pda*ti**xx[i] + ws[i]*pvap*ti**xs[i]
      wsq = width*width
      ss = s1[i]*ti2*exp(b2[i]*(1.-ti))
      df[0, :] = frequency - fl[i]
      df[1, :] = frequency + fl[i]

      #Use Clough's definition of local line contribution
      base = width/(562500. + wsq)

      #For both positive and negative resonances
      res = zeros(nf)
      for ii in range(nf):
        for jj in range(2):
          if (abs(df[jj, ii]) < 750.): res[ii] = res[ii] + width/(df[jj,ii]*df[jj,ii]+wsq)-base
      thesum += ss*res*(frequency/fl[i])**2.
    alpha = 0.3183e-4*den*thesum + con
  else:
    alpha = zeros(nf) 

  return alpha  


def abo2r98(temperature,pressure,rho,frequency):
  """
  Adapted from abo2r98.pro (UL)
  Computes absorption coefficient due to oxygen acc.to Rosenkranz 1998
    given profiles of temperature[K], pressure[Pa], humidity[kg/m3] and frequency channels[GHz]
    and other model-related input.
  """ 
  pressure = pressure*0.01    #convert to hPa  
  rho = rho*1000.             #convert to g/m3

  ff = array([118.7503, 56.2648, 62.4863, 58.4466, 60.3061, 59.5910,
        59.1642, 60.4348, 58.3239, 61.1506, 57.6125, 61.8002,
        56.9682, 62.4112, 56.3634, 62.9980, 55.7838, 63.5685,
        55.2214, 64.1278, 54.6712, 64.6789, 54.1300, 65.2241,
        53.5957, 65.7648, 53.0669, 66.3021, 52.5424, 66.8368,
        52.0214, 67.3696, 51.5034, 67.9009, 368.4984, 424.7632,
        487.2494, 715.3931, 773.8397, 834.1458])
  
  s300 = array([.2936E-14, .8079E-15, .2480E-14, .2228E-14,
          .3351E-14, .3292E-14, .3721E-14, .3891E-14,
          .3640E-14, .4005E-14, .3227E-14, .3715E-14,
          .2627E-14, .3156E-14, .1982E-14, .2477E-14,
          .1391E-14, .1808E-14, .9124E-15, .1230E-14,
          .5603E-15, .7842E-15, .3228E-15, .4689E-15,
          .1748E-15, .2632E-15, .8898E-16, .1389E-15,
          .4264E-16, .6899E-16, .1924E-16, .3229E-16,
          .8191E-17, .1423E-16, .6494E-15, .7083E-14,
          .3025E-14, .1835E-14, .1158E-13, .3993E-14])

  be = array([.009, .015, .083, .084, .212, .212, .391, .391, .626,
        .626, .915, .915, 1.260, 1.260, 1.660, 1.665, 2.119,
         2.115, 2.624, 2.625, 3.194, 3.194, 3.814, 3.814,
         4.484, 4.484, 5.224, 5.224, 6.004, 6.004, 6.844,
         6.844, 7.744, 7.744, .048, .044, .049, .145, .141, .145])

  wb300 = .56
  xx = .8
  w300 = array([1.63, 1.646, 1.468, 1.449, 1.382, 1.360,
          1.319, 1.297, 1.266, 1.248, 1.221, 1.207, 1.181, 1.171,
          1.144, 1.139, 1.110, 1.108, 1.079, 1.078, 1.05, 1.05,
          1.02, 1.02, 1.00, 1.00, .97, .97, .94, .94, .92, .92, .89,
          .89, 1.92, 1.92, 1.92, 1.81, 1.81, 1.81])

  y300 = array([-0.0233,  0.2408, -0.3486,  0.5227,
          -0.5430,  0.5877, -0.3970,  0.3237, -0.1348,  0.0311,
          0.0725, -0.1663,  0.2832, -0.3629,  0.3970, -0.4599,
          0.4695, -0.5199,  0.5187, -0.5597,  0.5903, -0.6246,
          0.6656, -0.6942,  0.7086, -0.7325,  0.7348, -0.7546,
          0.7702, -0.7864,  0.8083, -0.8210,  0.8439, -0.8529,
          0., 0., 0., 0., 0., 0.])

  vv = array([0.0079, -0.0978,  0.0844, -0.1273,
        0.0699, -0.0776,  0.2309, -0.2825,  0.0436, -0.0584,
        0.6056, -0.6619,  0.6451, -0.6759,  0.6547, -0.6675,
        0.6135, -0.6139,  0.2952, -0.2895,  0.2654, -0.2590,
        0.3750, -0.3680,  0.5085, -0.5002,  0.6206, -0.6091,
        0.6526, -0.6393,  0.6640, -0.6475,  0.6729, -0.6545,
        0., 0., 0., 0., 0., 0.])

  th = 300./temperature
  th1 = th - 1.
  bb = th**xx
  preswv = rho*temperature/217.
  presda = pressure - preswv
  den = 0.001*(presda*bb + 1.1*preswv**th)
  dens = 0.001*(presda + 1.1*preswv)*th
  dfnr = wb300*den
  thesum = 1.6e-17*frequency*frequency*dfnr/(th*(frequency*frequency + dfnr*dfnr))

  for k in range(40):
    if (k == 0): df = w300[0]*dens
    else: df = w300[k]*den
    yy = 0.001*pressure*bb*(y300[k]+vv[k]*th1)
    st = s300[k]*exp(-be[k]*th1)
    sf1 = (df + (frequency-ff[k])*yy)/((frequency-ff[k])*(frequency-ff[k]) + df*df)
    sf2 = (df - (frequency+ff[k])*yy)/((frequency+ff[k])*(frequency+ff[k]) + df*df)
    thesum += st*(sf1+sf2)*(frequency/ff[k])*(frequency/ff[k])
  
  return 0.5034e12*thesum*presda*th*th*th/3.14159


def absn2(temperature, pressure, frequency):
  """
  Adapted from absn2.pro (UL)
  Computes absorption coefficient due to nitrogen acc.to Rosenkranz 1998
    given profiles of temperature[K], pressure[Pa] and frequency channels[GHz].
  """ 
  pressure = pressure*0.01
  
  th = 300./temperature
  return 6.4e-14*pressure*pressure*frequency*frequency*th**3.55

  

def calc_tau_gas(height,temperature,pressure,humidity,frequency, abs_mod='l93',linew_22='lil05', cont_corr='tur09'):
  """
  Calculates optical depth due to gas (water vapor, O2, N2) absorption
  """
  nlev = height.size   #number of the layer boundaries/levels
  
  nlayers = nlev-1
  abs_gas = zeros(shape=(nlayers,frequency.size))
  abs_O2 = zeros(shape=(nlayers,frequency.size))
  abs_wv = zeros(shape=(nlayers,frequency.size))
  tau_gas = zeros(shape=(nlayers,frequency.size))
  tau_O2 = zeros(shape=(nlayers,frequency.size))
  tau_wv = zeros(shape=(nlayers,frequency.size))
  
  for level in range(nlev-1):
    deltaz = height[nlev-1-level]-height[nlev-2-level]
    T_mean = (temperature[nlev-1-level]+temperature[nlev-2-level])/2.   #mean T in the layer between nlev-1-level and nlev-2-level
    if (pressure[nlev-1-level] == pressure[nlev-2-level]):
      P_mean = pressure[nlev-1-level]
    else:
      xp = -np.log(pressure[nlev-1-level]/pressure[nlev-2-level])/deltaz
      P_mean = -pressure[nlev-2-level]/xp*(exp(-xp*deltaz)-1.0)/deltaz
    hmdt_mean = (humidity[nlev-1-level]+humidity[nlev-2-level])/2.
  
    if (abs_mod == 'l93'):
      res = abs_param_l93(cont_corr)
      ab_wv = abwvl93(hmdt_mean,T_mean,P_mean,frequency,res[2],res[3])
      ab_O2 = abo2l93(hmdt_mean,T_mean,P_mean,frequency,res[0],res[1])  
      abs_gas[nlev-2-level,:] = ab_wv + ab_O2

    elif (abs_mod == 'r98'):
      ab_wv = abwvr98(hmdt_mean,T_mean,P_mean,frequency,cont_corr,linew_22)/1000.
      ab_O2 = abo2r98(T_mean,P_mean,hmdt_mean,frequency)/1000.
      ab_N2 = absn2(T_mean,P_mean,frequency)/1000.
      abs_gas[nlev-2-level,:] = ab_wv + ab_O2 + ab_N2

    else:     #abs_mod == 'l87'
      res = abs_param_l87(cont_corr)
      ab_wv = abwvl87(hmdt_mean,T_mean,P_mean,frequency,res[2],res[3],cont_corr)
      ab_O2 = abo2l87(hmdt_mean,T_mean,P_mean,frequency,res[0],res[1])  
      abs_gas[nlev-2-level,:] = ab_wv + ab_O2
        
    abs_wv[nlev-2-level,:] = ab_wv
    abs_O2[nlev-2-level,:] = ab_O2

    if (level == 0):
      tau_gas[nlev-2-level,:] =  abs_gas[nlev-2-level]*deltaz
      tau_wv[nlev-2-level,:] =  abs_wv[nlev-2-level]*deltaz
      tau_O2[nlev-2-level,:] =  abs_O2[nlev-2-level]*deltaz
    else:
      tau_gas[nlev-2-level,:] =  tau_gas[nlev-1-level]+ (abs_gas[nlev-2-level]*deltaz)
      tau_wv[nlev-2-level,:] =  tau_wv[nlev-1-level]+ (abs_wv[nlev-2-level]*deltaz)
      tau_O2[nlev-2-level,:] =  tau_O2[nlev-1-level]+ (abs_O2[nlev-2-level]*deltaz)

  return [tau_gas, tau_O2, tau_wv]  



def calc_tau_cloud_tb(height,temperature,lwc,frequency,mu,tau_gas,cloud_abs='lie'):
  """
  Compute brightness temperature[K] for each frequency channel[GHz], 
    given temperature[K] and gas optical depth at each height level[m].
  """
  nlev = height.size
  nlayers = nlev-1

  deltaz = height[nlayers:0:-1]-height[nlayers-1::-1]
  T_mean = 0.5*(temperature[nlayers:0:-1]+temperature[nlayers-1::-1])

  if(cloud_abs == 'lie'): ab_cloud = abliq(lwc[nlayers-1::-1],frequency,T_mean)   
  abs_cloud = ab_cloud*1.e-3

  tau_cloud = (abs_cloud.T*deltaz).T
  for level in range(1,nlayers):
    tau_cloud[level,:] = tau_cloud[level,:]+tau_cloud[level-1,:]   
  tau_cloud = np.flipud(tau_cloud)      
  
  tau_all = tau_cloud+tau_gas          

  hPl = 6.6262e-34             #Planck constant
  kB = 1.3806e-23              #Boltzman constant
  c_li = 2.997925e8            #speed of light (m/s)
  temp_cmb = 2.73
  
  freq_si = frequency*1e9      #Hz   #shape (14,)
  nlev = temperature.size      #nlev is for example, 60

  tempvar1 = 2*hPl*freq_si*freq_si*freq_si/(c_li*c_li)   
  tempvar2 = hPl*freq_si/kB
  
  tau_top1 = np.zeros(freq_si.size)  
  tau_top2 = tau_all[nlev-2:0:-1,:]
  tau_top = np.vstack([tau_top1,tau_top2])  
  tau_bot = tau_all[nlev-2::-1,:]

  delta_tau = tau_bot - tau_top
  if (np.any(delta_tau)<=0.):
    print 'zero or negative absorption coefficient, exiting...'
    exit()

  tempvar3 = exp(-delta_tau/mu)
  AA = np.ones(freq_si.size) - tempvar3   
  BB = delta_tau - mu + mu*tempvar3

  temperature1 = temperature[nlev-1:0:-1]
  temperature2 = temperature[nlev-2::-1]
  ttemperature1 = tile(temperature1,(freq_si.size,1)).T  
  ttemperature2 = tile(temperature2,(freq_si.size,1)).T  
  ttempvar1 = tile(tempvar1,(temperature1.size,1))       
  ttempvar2 = tile(tempvar2,(temperature2.size,1))       
  T_pl2 = ttempvar1/(exp(ttempvar2/ttemperature2)-1.)    
  T_pl1 = ttempvar1/(exp(ttempvar2/ttemperature1)-1.)    
  diff = (T_pl2 - T_pl1)/delta_tau

  pl_in = tempvar1/(exp(tempvar2/temp_cmb)-1.)
  for lev in range(temperature1.size):
    pl_in = pl_in*exp(-delta_tau[lev,:]/mu) + T_pl1[lev,:]*AA[lev,:] + diff[lev,:]*BB[lev,:]
 
  TB = tempvar2/np.log(tempvar1/pl_in+1.)                 

  return TB      



