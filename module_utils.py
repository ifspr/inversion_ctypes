#!/usr/bin/env python

import numpy as np
from scipy import linalg

zeros = np.zeros
exp = np.exp
pi = np.pi


def cloud_ad(Pcb, Tcb):
  """
  Input: Pcb[Pa] and Tcb[K]
  Computes adiabatic lapse rate of LWC mixing ratio[] and the density of air[kg/m3]
  Returns a list containing the two computed values 
  """
  cp=1005.              #J/kg/K
  eso = 610.7           #Pa
  Lo=2.501e6            #J/kg
  To=273.15             #K
  Rv=461.5              #J/kg/K
  Rd=287.               #J/kg/K
  ep=0.622         

  #pre-calculated:
  pcv1 = 29.29          #k/m/g
  pcv2 = 2358.0         #cl-cpv  
  pcv3 = 3145087.70     #Lo+(cl-cpv)*To
  pcv4 = 0.00366        #1/To
  Gamma_d = 0.00975     #g/cp
  H = pcv1*Tcb          #k*Tcb/m/g         
  e_sat = eso*exp((pcv3*(1.0/To-1./Tcb)-pcv2*np.log(Tcb/To))/Rv)
  L = Lo+pcv2*(To-Tcb)
  ws = ep*(e_sat/(Pcb-e_sat))            
  pcv5 = L*ws/Rd
  Gamma_s = Gamma_d*(1.0+pcv5/Tcb)/(1.0+ep*L*pcv5/(cp*Tcb*Tcb))    
  Gamma_l = (ep+ws)*pcv5/(Tcb*Tcb)*Gamma_s      
  Gamma_l = Gamma_l-ws*Pcb/(Pcb-e_sat)/H                                  
  rho_air = (Pcb-e_sat)/(Rd*Tcb)                                          
  return [Gamma_l, rho_air]


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
  k23 = (nu*(nu+1)/(nu+2)/(nu+2))**(1./3.)  #k1 in Boers et al.
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
    boltzmann = 1.3806488e-23             # kg m^2 / K / s^2                                  
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
  id = np.zeros(shape=(A.shape[0],A.shape[1]))   #construct identity matrix
  for i in range(A.shape[0]): id[i,i]=1.0       
  Ainv = np.empty(shape=(A.shape[0],A.shape[1]))
  lu = linalg.lu_factor(A)
  for i in range(A.shape[0]):
    Ainv[:,i] = linalg.lu_solve(lu,id[:,i])
  return Ainv




