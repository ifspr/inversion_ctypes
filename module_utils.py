#!/usr/bin/env python

import numpy as np
from scipy import linalg

zeros = np.zeros
exp = np.exp
pi = np.pi


def subad_frac(z,cdepth,alpha,hnorm):
  """
  Input: height levels (above cloud base), cloud depth (in the same unit as height levels), 
  alpha parameter, hnorm parameter (see Boers et al. 2006 (eq. C1))
  Computes vertical variation of LWC as a function of altitude
  Returns values of the function in an array
  """
  znorm = z/cdepth
  e1 = exp(-alpha*hnorm)
  e2 = exp(hnorm*(znorm-1))
  e3 = exp(-hnorm)
  term1 = (e2-e3)/(1.-e3)
  return abs((1.-e1)*(1.-term1))    


def calc_nre(z,fz,Nad,nu,Gamma_l,rho_air,mod):
  """
  Input: height levels (w.r.t. cloud base) in m, subad func, adiabatic number concentration (1/m3), 
  shape parameter, pressure (Pa) and temperature (K) at cloud base, mixing model
  Computes the number concentration (1/m3), effective radius (m) and extinction coefficient (1/m) 
  as a function of height above cloud base
  Returns a list containing 3 arrays of number concentration, effective radius and extinction coefficient
  """
  k23 = (nu*(nu+1)/(nu+2)/(nu+2))**(1./3.)  
  pinad = pi*Nad
  valz = 0.75*rho_air/1000.*Gamma_l*z
  if (mod==0):
    N = Nad*fz
    re = (1./k23)*(valz/pinad)**(1./3.) 
    sigma = 2.*k23*fz*(pinad**(1./3.))*(valz**(2./3.)) 
  else:
    N = Nad*np.ones(fz.size)
    re = (1./k23)*(valz*fz/pinad)**(1./3.) 
    sigma = 2.*k23*(pinad**(1./3.))*((valz*fz)**(2./3.))  
  return [N,re,sigma]


def ext_coeff_air(zstart=1,zrel=1,wav=1,pressure=1,temperature=1):
  """
  Computes the extinction coefficient of air due to (Rayleigh) molecular scattering
  Examples:
  ext_air = ext_coeff_air(wav=wav,pressure=presdata[0:ind_ct+1],temperature=tempdata[0:ind_ct+1])
  ext_air = ext_coeff_air(0.,zbeta_cld)    #exponential profile

  """
  if (hasattr(pressure,"__len__") and hasattr(temperature,"__len__")): 
    boltzmann = 1.3806488e-23             #kg m^2 / K / s^2                                  
    rho = pressure/boltzmann/temperature  #molecule/m^3             
    wavmic = wav*1e6                      #microns
    if (wavmic <= 0.55): xpar = 0.389*wavmic + 0.09426/wavmic - 0.3228   
    else: xpar = 0.04
    ray_sigma = 4.02e-28/(wavmic**(4+xpar))   
    ray_sigma = ray_sigma*1e-4            #m^2/sr/molecule
    ext_co_air = rho*ray_sigma            #[1/m]
  else: 
    surf = 8*np.pi*(1.6e-6)/3.
    ext_co_air = surf/np.exp((zstart+zrel)/8000.)     # [1/m]
  return ext_co_air


def cov_mat_msr(d_tb,d_beta,d_radref,beta,radref):
  """
  Input: d_tb: random error of brightness temperature (array)
         d_beta: random error of attenuated lidar backscatter (array)
         d_radref: random error of radar reflectivity (array)
         beta: attenuated lidar backscatter without random error (array)
         radref: radar reflectivity without random error (array)
  Return measurement error covariance matrix
  """

  sig_rcal = 1.e-8              
  sig_lcal = 1.e-8              

  nmwr = d_tb.size
  nlid = d_beta.size
  nrad = d_radref.size

  cov_mat = np.zeros(shape=(nmwr+nlid+nrad,nmwr+nlid+nrad))

  for j in range(0,nmwr):
    for k in range(0,nmwr):
      if (j == k): cov_mat[j,k] = d_tb[j]*d_tb[k]
  for j in range(0,nlid):
    for k in range(0,nlid):
       if (j == k): cov_mat[j+nmwr,k+nmwr] = d_beta[j]*d_beta[k]+sig_lcal*sig_lcal*beta[j]*beta[k]
       else: cov_mat[j+nmwr,k+nmwr] = sig_lcal*sig_lcal*beta[j]*beta[k]
  for j in range(0,nrad):
    for k in range(0,nrad):
       if (j == k): cov_mat[j+nmwr+nlid,k+nmwr+nlid] = d_radref[j]*d_radref[k]+sig_rcal*sig_rcal*radref[j]*radref[k]
       else: cov_mat[j+nmwr+nlid,k+nmwr+nlid] = sig_rcal*sig_rcal*radref[j]*radref[k]
  return cov_mat



def invert_lu(A):
  """
  Invert a given matrix A through LU decomposition
  Assumption: matrix A is square, symmetric 
  """
  id = np.zeros(shape=(A.shape[0],A.shape[1]))   
  for i in range(A.shape[0]): id[i,i]=1.0       
  Ainv = np.empty(shape=(A.shape[0],A.shape[1]))
  lu = linalg.lu_factor(A)
  for i in range(A.shape[0]):
    Ainv[:,i] = linalg.lu_solve(lu,id[:,i])
  return Ainv


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
  nlev = temperature.size      

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



