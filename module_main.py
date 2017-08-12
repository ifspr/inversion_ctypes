#!/usr/bin/env python

import numpy as np
import ConfigParser
import sys
import pickle

from time import clock
from scipy.optimize import differential_evolution
from ctypes import c_int, c_double, POINTER, cdll, Structure, byref
from numpy.ctypeslib import as_ctypes
from platform import system

from module_util import invert_lu,cov_mat_msr,cloud_ad,subad_frac,calc_nre,ext_coeff_air
from module_mwr import calc_tau_gas,calc_tau_cloud_tb


class ms_config(Structure):
    _fields_ = [("small_angle_algorithm",c_int),
                  ("wide_angle_algorithm",c_int),
                  ("options",c_int),
                  ("max_scattering_order",c_int),
                  ("max_theta",c_double),
                  ("first_wide_angle_gate",c_int),
                  ("coherent_backscatter_enhancement",c_double),
                  ("small_angle_lag",POINTER(c_double)),   #used only for -lag (the option that is not relevant for the application of interest)
                  ("total_src",c_double),
                  ("total_reflected",c_double),
                  ("ss_multiplier",c_double)]

class ms_instrument(Structure):
    _fields_ = [("receiver_type",c_int),
                  ("altitude", c_double),
                  ("wavelength",c_double),
                  ("rho_transmitter",c_double),
                  ("rho_receiver",POINTER(c_double)),
                  ("nfov",c_int)]
      
class ms_surface(Structure):
    _fields_ = [("sigma0",c_double),
                ("diffuse_albedo",c_double),
                ("direct_to_diffuse_albedo",c_double),
                ("diffuse_to_direct_backscatter",c_double),
                ("range",c_double)]

    
def calc_beta(klett_param,_ms,configms,instrument,surface,sfac_cld,extcoeff,ra,z_ms,icld,ext_co_air,beta_mean):
  """
  Calculate lidar attenuated backscatter using "multiscatter" 
  """
  indd,alp0,s_air,s_arsl = klett_param
  z0 = z_ms[indd]           
  beta0 = beta_mean[indd]   
  ext_arsl0 = 10.**(alp0)    
  beta_air = ext_co_air/s_air 
  beta_air0 = beta_air[indd] 
  ext_tot0 = ext_arsl0+s_arsl*beta_air0 

  transmission = zeros(indd+1)
  transmission[0] = 2.*(ext_co_air[0]-s_arsl*beta_air[0])*(z_ms[0]-0.0)   
  for i in range(1,indd+1):
    transmission[i] = transmission[i-1]+(2.*(ext_co_air[i]-s_arsl*beta_air[i])*(z_ms[i]-z_ms[i-1]))  
  ext_tot = zeros(indd+1)
  ext_tot[-1] = ext_tot0
  ext_arsl = zeros(indd+1)
  ext_arsl[-1] = ext_arsl0
  pz0 = beta0*exp(transmission[-1])   
  pz = zeros(indd+1)         
  pz[-1] = pz0
  pz_cum = 2.*pz0*(z0-z_ms[indd-1])  
  
  denominator1 = pz0/ext_tot0
  for i in range(indd-1,-1,-1):  
    pz[i] = beta_mean[i]*exp(transmission[i])
    if (i == 0): pz_cum = pz_cum+2.*pz[i]*(z_ms[i]-0.0)                     
    else: pz_cum = pz_cum+2.*pz[i]*(z_ms[i]-z_ms[i-1])           

    ext_tot[i] = pz[i]/(denominator1+pz_cum)
    ext_arsl[i] = ext_tot[i]-s_arsl*beta_air[i]

  #klett calculation ends here
    
  nrange = z_ms.size  
  ra_ms = zeros(nrange)+1e-8
  extco_ms = zeros(nrange)
  cldfrac = zeros(nrange)  

  s_arsl = 50.
  lidarratio = zeros(nrange)
  lidarratio[icld] = sfac_cld
  lidarratio[0:ext_arsl.size-1] = s_arsl
  
  ra_ms[icld] = ra
  extco_ms[icld] = extcoeff
  extco_ms[0:ext_arsl.size]=ext_arsl[0:] 

  cldfrac[icld[1:]] = 1.0

  extco_ms[extco_ms < 0.0] = 0.0           
  ra_ms[ra_ms <= 0.0] = 1.e-8              
  extco_air_in_cld = ext_coeff_air(0.0,z_ms)
  sfactor =  ((extco_ms*lidarratio)+(extco_air_in_cld*8*pi/3.)) / (extco_ms + extco_air_in_cld)
  
  albedo=zeros(nrange)
  albedo_air=zeros(nrange)
  asym=zeros(nrange)+0.85
  icefrac=zeros(nrange)
  backscatter = zeros(nrange)                                                            
  _ms.multiscatter(nrange,
                   nrange,
                   byref(configms),   
                   instrument,
                   surface,
                   as_ctypes(zeros(nrange)+z_ms),
                   as_ctypes(ra_ms),
                   as_ctypes(extco_ms),
                   as_ctypes(albedo),
                   as_ctypes(asym),
                   as_ctypes(sfactor),
                   as_ctypes(ext_co_air),
                   as_ctypes(albedo_air),
                   as_ctypes(cldfrac),
                   as_ctypes(icefrac),
                   as_ctypes(backscatter),   #output
                   None)      
  
  tau_blwcld = zeros(ext_arsl.size)
  tau_blwcld[0] = (ext_arsl[0]+ext_co_air[0])*(z_ms[3]-z_ms[2])  
  tau_cum = tau_blwcld[0]

  for i in range(1,ext_arsl.size):
    tau_blwcld[i] = ((ext_arsl[i]+ext_co_air[i])*(z_ms[3]-z_ms[2]))+tau_cum   
    tau_cum = tau_blwcld[i]

  ext_tot_blwcld = (ext_arsl+s_arsl*ext_co_air[0:ext_arsl.size]*3/(8.*np.pi))
  cl_blwcld = s_arsl*beta_mean[0:ext_arsl.size]*np.exp(2*tau_blwcld)/ext_tot_blwcld
  clid = np.median(cl_blwcld)

  return clid, backscatter*clid   
 

def callback_func(xk,convergence):
  """ 
  Print out the progress of the optimization and allow for an early termination
    xk: state vector value at k-th iteration 
    convergence: fractional value of the convergence
  """
  global Niter,cost_value,progf
  progf.write(str(Niter)+'  '+str(convergence)+'  '+str('%.2f'%(xk[0]))+'  '+str('%.4f'%(xk[1]))+'  '+str('%.2f'%(xk[2]))+'  '+str('%.2e'%(xk[3]))+'  '+str('%.3f'%(xk[4]))+'  '+str('%.1f'%(xk[5]))+'  '+str('%.2f'%(xk[6]))+'  '+str('%.3f'%(xk[7]))+'  '+str('%.4f'%(xk[8]))+'  '+str('%.2f'%(xk[9]))+'  '+str('%.2f'%(xk[10]))+'  '+str('%.1f'%(xk[11]))+'  '+str('%.3f'%(xk[12]))+'\n')
  Niter += 1
#  if (convergence > 0.05): return True   #early termination based on convergence level


def cost2(xv,*args):
  """
  For the case where drizzle is not present below cloud base, allow the possibility that there is no drizzle at all
  Drizzle is derived using excess reflectivity; lwc_dzl within cloud is parametrized (Boers model)
  Cloud base is optimized(&smoothed)
  State vector:
  #        0       1        2       3     4      5         6           7         8       9       10        11       ,   12
  # x = [nu_cld,hhat_cld,alph_cld,Nad_cld,rcal,klettfac,cldtop_mod,cldbase_mod,nu_dzl,hhat_dzl,alph_dzl,lwcscale_dzl,weight] 

  """
  global cost_value,progf, Niter,Nfeval,clid
  global z_common,zcb,zct,dtop              
  global Ftb, Fbeta, Fradref, Fradref_dzl, Fradref_att               #observables
  global ext_cld_common,re_cld_common,lwc_cld_common,Nz_cld_common   #cloud
  global ext_dzl_common,re_dzl_common,lwc_dzl_common,Nz_dzl_common   #dzl

  ctop, gridres, cbase_mod_lo, cbase_mod_hi, tpqheight, presdata, tempdata, height_beta,mmod,indl_cbase,height_z,Z_mean,indr_cbase2,cbase,re_dzl_lo,cff,dbase,re_dzl_hi,indl_dbase,n30,height_incld,n_blwcb,ind_beta_incld,height30,temp30,freq_brt,liq_abs,tau_gas_total,mu,beta_mean,ext_air,s_air,s_arsl,wav,alt,div,fov,ratio,ntb,nbeta,nrref,indl_200,kappa1_l,yy,cov_msr_inv,configms,instrument,surface,_ms = args[0:52]

  nu_cld = xv[0]
  nu_dzl = xv[8]
  
  Nfeval = Nfeval+1
  zct = ctop+(xv[6]*gridres)                                       
  zcb = cbase_mod_lo+(xv[7]*(cbase_mod_hi-cbase_mod_lo))           
  cdepth = zct-zcb
  Pcb = interp(zcb,tpqheight,presdata)
  Tcb = interp(zcb,tpqheight,tempdata)
  Gamma_l, rho_air = cloud_ad(Pcb,Tcb)

  #1st, compute LWC_ad and LWC_cld
  ind_cloud = [b for b,x in enumerate(height_beta) if (x>=zcb and x<=zct)] 
  zdata_cld = height_beta[ind_cloud]
  if (zdata_cld[-1] < zct): zdata_cld = append(zdata_cld,zct)   #adding zct to the grid
  zdata_cld_cb = zdata_cld-zcb
  fz=subad_frac(zdata_cld_cb,cdepth,xv[2],xv[1])
  lwc_ad = rho_air*Gamma_l*zdata_cld_cb     #kg/m3 
  lwc_ad = lwc_ad*1000.                     #g/m3
  lwc_cld = fz*lwc_ad                       #g/m3; see eq. A2 of Boers et al.                
  Nz_cld,re_cld,ext_cld = calc_nre(zdata_cld_cb,fz,xv[3],xv[0],Gamma_l,rho_air,mmod)

  #2nd, compute weights and perform smoothing
  nwidth = 2*(ind_cloud[0]-1-indl_cbase)+1    
  if (nwidth==1):    
    lwc_smooth= append(0,lwc_cld)
    Nz_smooth = append(Nz_cld[0],Nz_cld)
    z_smooth = append(height_beta[ind_cloud[0]-1],zdata_cld)
  else:              
    x=np.linspace(0,0.5*(nwidth-1),0.5*(nwidth+1))
    y = exp(-xv[12]*x)       
    weight = zeros(nwidth)  
    weight[int(0.5*(nwidth-1)):] = y[:]
    weight[range(int(0.5*(nwidth-1))-1,-1,-1)] = y[1:]

    lwc2 = zeros(lwc_cld.size+nwidth) 
    lwc2[nwidth:] = lwc_cld
    lwc_smooth = zeros(lwc_cld.size+0.5*(nwidth+1))  
    extragates=0
    for i in range(nwidth+extragates):   
      lwc_smooth[i] = np.average(lwc2[i:i+nwidth],weights=weight)
    lwc_smooth[nwidth+extragates:] = lwc_cld[nwidth+extragates-int(0.5*(nwidth+1)):]  
  
    ind_extension = np.arange(int(0.5*(nwidth+1)),0,-1)    
    ind_temp = list(ind_cloud[0]-ind_extension)
    ind_smooth = ind_temp+ind_cloud
    z_smooth = height_beta[ind_smooth]      
    if (z_smooth[-1] < zct): z_smooth = append(z_smooth,zct)  
    ind_lo = int(0.5*(nwidth+1))
    Nz_smooth = zeros(Nz_cld.size+ind_lo)
    Nz_smooth[ind_lo:] = Nz_cld[:]
    Nz_smooth[0:ind_lo] = Nz_cld[0]                   
  
  #3rd, compute other microphysical properties of cloud (ra, rn, ext)
  re_smooth = (1.5*lwc_smooth*(nu_cld+2)*(nu_cld+2)/(2*pi*Nz_smooth*1.e6*nu_cld*(nu_cld+1)))**(1./3)
  if (mmod == 0): re_smooth[-1] = re_smooth[-2]+(re_smooth[-2]-re_smooth[-3])  
  ra_smooth = re_smooth*sqrt(nu_cld*(nu_cld+1))/(nu_cld+2)
  rn_smooth = re_smooth/(nu_cld+2)
  ext_smooth = 2*pi*Nz_smooth*ra_smooth*ra_smooth
  
  #4th, interpolate to radar grid and compute cloud reflectivity
  rn_interp = zeros(height_z.size)
  Nz_interp = zeros(height_z.size)
  indgrid = list(where(height_z>=z_smooth[0])[0])     
  rn_interp[indgrid] = interp(height_z[indgrid],z_smooth,rn_smooth)   
  Nz_interp[indgrid] = interp(height_z[indgrid],z_smooth,Nz_smooth) 

  r6 = ((rn_interp*1e6)**6.)*(nu_cld)*(nu_cld+1)*(nu_cld+2)*(nu_cld+3)*(nu_cld+4)*(nu_cld+5)   
  Fradref=64.*Nz_interp*r6*1.e-18*xv[4]         
  maxpos_mod = where(Fradref == np.max(Fradref[-4:-1]))[0][0] 
  maxpos_obs = where(Z_mean == np.max(Z_mean[-4:-1]))[0][0]

  #5th, compute drizzle reflectivity 
  height_z2 = append(dbase,height_z)   
  residual = Z_mean-Fradref               
  residual[residual<0.0] = 0.0
  Fradref_dzl = zeros(residual.size)
  Fradref_dzl[0] = residual[0]  
  Fradref_dzl[-1] = residual[-1]  
  for i in range(1,residual.size-1):  
    Fradref_dzl[i] = np.mean(residual[i-1:i+2])    
  ind_nonzero = array(where(Fradref_dzl > 0.0)[0])   
  ind_nonzero1 = ind_nonzero[0:-1]   
  ind_nonzero2 = ind_nonzero[1:]      
  diff_ind = ind_nonzero2-ind_nonzero1   
  ind_diff_ind = list(where(diff_ind == 1)[0])
  condition1 = (len(ind_diff_ind) >= 2) 
  condition2 = (residual[0] > 0.0)     
  
  #6th, compute drizzle microphysical properties
  if (condition1 and condition2):   
    if (ind_nonzero[-1]+1==residual.size): dtop = zct   
    else: dtop = height_z[ind_nonzero[-1]+1]      
    ddepth = dtop-dbase              
    zdata_dzl_db = height_z2[height_z2<=dtop]-dbase  
    if (height_z2[-1] < dtop): zdata_dzl_db = append(zdata_dzl_db,dtop-dbase)   
    fz=subad_frac(zdata_dzl_db, ddepth,xv[10],xv[9])
    lwc_ad = rho_air*Gamma_l*zdata_dzl_db     #kg/m3 
    lwc_ad = lwc_ad*1000.                     #g/m3
    lwc_dzl = fz*lwc_ad*(10.**xv[11])
  
    k36_2 = nu_dzl*(nu_dzl+1)*(nu_dzl+2)/(nu_dzl+3)/(nu_dzl+4)/(nu_dzl+5)    
    nominator = pi*1.e6*Fradref_dzl[height_z<dtop]*1.e-18*(nu_dzl+2)*(nu_dzl+2)*(nu_dzl+2)
    denominator = lwc_dzl[1:-1]*48.*(nu_dzl+3)*(nu_dzl+4)*(nu_dzl+5)   
    re_dzl = (nominator/denominator)**(1./3.)   #m
    
    re_dzl_top = re_dzl[-1]+((zdata_dzl_db[-1]-zdata_dzl_db[-2])*(re_dzl[-1]-re_dzl[-2])/(zdata_dzl_db[-2]-zdata_dzl_db[-3])) 
    re_dzl0 = re_dzl[0]+(re_dzl[0]-re_dzl[1])    

    re_dzl = append(append(re_dzl0,re_dzl),re_dzl_top)          
    ext_dzl = 1.5*lwc_dzl/1.e6/re_dzl
    rn_dzl = re_dzl/(nu_dzl+2)
    ra_dzl = sqrt(rn_dzl*rn_dzl*nu_dzl*(nu_dzl+1))
    Nz_dzl = ext_dzl/(2.*pi*ra_dzl*ra_dzl)
    height_z3 = zdata_dzl_db+dbase
    re_dzl[lwc_dzl == 0.0] = 0.0      
    ra_dzl[lwc_dzl == 0.0] = 0.0   
    rn_dzl[lwc_dzl == 0.0] = 0.0   

    violation1 = (np.min(re_dzl[1:-1]) < re_dzl_lo)
    violation2 = (np.max(re_dzl[1:-1]) > re_dzl_hi)   
    radrefratio = Fradref_dzl[-2:]/Fradref[-2:]
    violation3 = (np.any(radrefratio > 1.0))   
    if (violation1 or violation2 or violation3): return 100.*cost_value    
    
  #7th, interpolate to a common grid 
    height_extras = height_beta[(height_beta>dbase)&(height_beta<z_smooth[0])]
    z_common = np.sort(append(height_extras,append(dbase,z_smooth)))   
    #------cloud-------
    lwc_cld_common = zeros(z_common.size)
    ra_cld_common = zeros(z_common.size)
    re_cld_common = zeros(z_common.size)
    rn_cld_common = zeros(z_common.size)
    ext_cld_common = zeros(z_common.size)
    Nz_cld_common = zeros(z_common.size)
    lwc_cld_common = interp(z_common,z_smooth,lwc_smooth)     
    ra_cld_common = interp(z_common,z_smooth,ra_smooth)
    re_cld_common = interp(z_common,z_smooth,re_smooth)
    rn_cld_common = interp(z_common,z_smooth,rn_smooth)
    ext_cld_common = interp(z_common,z_smooth,ext_smooth)
    Nz_cld_common = interp(z_common,z_smooth,Nz_smooth)

    #------drizzle-------
    lwc_dzl_common = zeros(z_common.size)
    ra_dzl_common = zeros(z_common.size)
    re_dzl_common = zeros(z_common.size)
    rn_dzl_common = zeros(z_common.size)
    ext_dzl_common = zeros(z_common.size)
    Nz_dzl_common = zeros(z_common.size)
    indgrid = list(where((z_common>=dbase) & (z_common<=dtop))[0])
    lwc_dzl_common[indgrid] = interp(z_common[indgrid],height_z3,lwc_dzl)
    ra_dzl_common[indgrid] = interp(z_common[indgrid],height_z3,ra_dzl)
    re_dzl_common[indgrid] = interp(z_common[indgrid],height_z3,re_dzl)
    rn_dzl_common[indgrid] = interp(z_common[indgrid],height_z3,rn_dzl)
    ext_dzl_common[indgrid] = interp(z_common[indgrid],height_z3,ext_dzl)
    Nz_dzl_common[indgrid] = interp(z_common[indgrid],height_z3,Nz_dzl)

  else:   #no drizzle present!
    
    height_extras = height_beta[(height_beta>dbase)&(height_beta<z_smooth[0])]
    z_common = np.sort(append(height_extras,append(dbase,z_smooth)))   
    #------cloud-------
    lwc_cld_common = zeros(z_common.size)
    ra_cld_common = zeros(z_common.size)
    re_cld_common = zeros(z_common.size)
    rn_cld_common = zeros(z_common.size)
    ext_cld_common = zeros(z_common.size)
    Nz_cld_common = zeros(z_common.size)
    lwc_cld_common = interp(z_common,z_smooth,lwc_smooth)     
    ra_cld_common = interp(z_common,z_smooth,ra_smooth)
    re_cld_common = interp(z_common,z_smooth,re_smooth)
    rn_cld_common = interp(z_common,z_smooth,rn_smooth)
    ext_cld_common = interp(z_common,z_smooth,ext_smooth)
    Nz_cld_common = interp(z_common,z_smooth,Nz_smooth)

    lwc_dzl_common = zeros(z_common.size)
    ra_dzl_common = zeros(z_common.size)
    re_dzl_common = zeros(z_common.size)
    rn_dzl_common = zeros(z_common.size)
    ext_dzl_common = zeros(z_common.size)
    Nz_dzl_common = zeros(z_common.size)
    Fradref_dzl = Fradref_dzl*0.0

  #8th, combine cloud and srizzle parameters 
  lwc_all_common = lwc_cld_common+lwc_dzl_common
  ext_all_common =ext_cld_common+ext_dzl_common 
  ra_all_common = (ra_cld_common*ext_cld_common+ra_dzl_common*ext_dzl_common)/(ext_cld_common+ext_dzl_common) 
  re_all_common = (re_cld_common*ext_cld_common+re_dzl_common*ext_dzl_common)/(ext_cld_common+ext_dzl_common)  
  rn_all_common = (rn_cld_common*ext_cld_common+rn_dzl_common*ext_dzl_common)/(ext_cld_common+ext_dzl_common)  
  ind0 = list(where(ext_all_common == 0.0)[0])
  re_all_common[ind0] = 0.0
  rn_all_common[ind0] = 0.0
  ra_all_common[ind0] = 0.0
  Nz_all_common = Nz_cld_common+Nz_dzl_common     

    
  #9th, compute brightness temperature  
  lwc30 = zeros(n30)
  z_com_ic = zeros(z_common.size)    
  lwc_com_ic = zeros(z_common.size)
  z_com_ic[:] = z_common[:]           
  lwc_com_ic[:] = lwc_all_common[:]   

  bad_elements = list(set(z_com_ic)-set(height_incld))  
  for element in bad_elements:   
    ind = where(element == z_com_ic)[0][0]
    z_com_ic = np.delete(z_com_ic,ind)    
    lwc_com_ic = np.delete(lwc_com_ic,ind)

  lwc30[n_blwcb:n_blwcb+len(ind_beta_incld)] = lwc_com_ic
  Ftb = calc_tau_cloud_tb(height30,temp30,lwc30,freq_brt,mu,tau_gas_total,liq_abs)


  #10th, apply Klett method 
  indbeta = list(where((height_beta >= cbase) & (height_beta <= zct))[0])
  z_com_ic = zeros(z_common.size)
  ext_total = zeros(z_common.size)
  ra_total = zeros(z_common.size)
  z_com_ic[:] = z_common[:]
  ext_total = ext_all_common[:]
  ra_total = ra_all_common[:]
  bad_elements = list(set(z_com_ic)-set(height_beta[indbeta]))     
  for element in bad_elements:   
    ind = where(element == z_com_ic)[0][0]
    z_com_ic = np.delete(z_com_ic,ind)    
    ext_total = np.delete(ext_total,ind)
    ra_total = np.delete(ra_total,ind)

  indd = indl_dbase-1 if (dbase < cbase) else indl_cbase  
    
  clid,Fbeta = calc_beta((indd,xv[5],s_air,s_arsl),_ms,configms,instrument,surface,ratio,ext_total,ra_total,height_beta,indbeta,ext_air,beta_mean)

  #11th, compute the cost function value
  ffx = zeros(ntb+nbeta+nrref)                                                         
  ffx[0:ntb] = Ftb
  ffx[ntb:ntb+nbeta] = Fbeta[0:indl_200+1]

  lwc_heightz = interp(height_z,z_common,lwc_all_common) 

  kappac = kappa1_l * lwc_heightz       #[dB/km]
  kappac = kappac/(10.*np.log10(np.e))  #convert to [Np/km] 
  opac = zeros(lwc_heightz.size)
  kappac_cum = 0.0
  for i in range(1,opac.size):
    kappac_cum = kappac_cum + kappac[i-1]
    opac[i] = kappac_cum*(gridres*0.001)  

  Fradref_att = (Fradref+Fradref_dzl)*np.exp(-2.*opac)
  ffx[ntb+nbeta:ntb+nbeta+nrref] = Fradref_att 

  diff = yy-ffx

  sumtb = 0.0
  for i in range(ntb):
    for j in range(ntb):
       term = cov_msr_inv[i,j]*diff[i]*diff[j]
       sumtb = sumtb + term
   
  sumbeta = 0.0
  for i in range(ntb,ntb+nbeta):
    for j in range(ntb,ntb+nbeta):
       term = cov_msr_inv[i,j]*diff[i]*diff[j]
       sumbeta = sumbeta + term

  sumz = 0.0
  for i in range(ntb+nbeta,ntb+nbeta+nrref):
    for j in range(ntb+nbeta,ntb+nbeta+nrref):
       term = cov_msr_inv[i,j]*diff[i]*diff[j]
       sumz = sumz + term
       
  absdifftb = np.sum(abs(yy[0:ntb]-ffx[0:ntb]))
  absdiffbeta = np.sum(abs(yy[ntb:ntb+nbeta]-ffx[ntb:ntb+nbeta]))
  absdiffrref = np.sum(abs(yy[ntb+nbeta:ntb+nbeta+nrref]-ffx[ntb+nbeta:ntb+nbeta+nrref]))
 
  ind_new_cld = list(where((z_common >= cbase) & (z_common <= zct))[0]) 
  delta_re = re_dzl_common[ind_new_cld[1]:ind_new_cld[-1]+1]-re_dzl_common[ind_new_cld[0]:ind_new_cld[-1]]
  ind_pos = where(delta_re > 0.0)[0]
  npenalty3 = len(ind_pos)   
  cost_value = sumtb+sumbeta+sumz*(1+npenalty3)
  
  cff.write(str(Nfeval)+'  '+str(absdifftb)+'  '+str(sumtb)+'  '+str(absdiffbeta)+'  '+str(sumbeta)+'  '+str(absdiffrref)+'  '+str(sumz)+'  '+str(np.sum(abs(yy-ffx)))+'  '+str(cost_value)+'\n')

  return cost_value

  
  
def cost1(xv,*args):
  """
  For the case where drizzle extends below cloud base
  Drizzle is derived using excess reflectivity; re_dzl within cloud is parametrised (exponential)
  Cloud base is optimized(&smoothed)
  State vector:
  #        0       1        2       3     4      5         6           7         8           9           10      ,    11       12
  #x = [nu_cld,hhat_cld,alph_cld,Nad_cld,rcal,klettfac,cldtop_mod,cldbase_mod,nu_dzl,ext_dzl_dbase,ext_dzl_cbase, ext_dzl_peak,weight] 
  #  x[9] = -3 to -0.001  (to force it to be smaller than ext_dzl_cbase; ext_dzl_dbase = x[9] * ext_dzl_cbase
  #  x[10] = -9 to -6  (ext_dzl_cbase = 10.**x[10])
  #  x[11] = -4 to -2  (ext_dzl_peak = x[11]*np.max(ext_cld)

  """
  global cost_value,progf, Niter,Nfeval,flag_nan,clid
  global z_common,zcb,zct,dtop              
  global Ftb, Fbeta, Fradref, Fradref_dzl, Fradref_att               #observables
  global ext_cld_common,re_cld_common,lwc_cld_common,Nz_cld_common   #cloud
  global ext_dzl_common,re_dzl_common,lwc_dzl_common,Nz_dzl_common   #dzl

  ctop, gridres, cbase_mod_lo, cbase_mod_hi, tpqheight, presdata, tempdata, height_beta,mmod,indl_cbase,height_z,Z_mean,indr_cbase2,Z_cbase,cbase,re_dzl_lo,cff,dbase,indr_cbase1,re_dzl_hi,indl_dbase,n30,height_incld,n_blwcb,ind_beta_incld,height30,temp30,freq_brt,liq_abs,tau_gas_total,mu,beta_mean,ext_air,s_air,s_arsl,wav,alt,div,fov,ratio,ntb,nbeta,nrref,indl_200,kappa1_l,yy,cov_msr_inv,configms,instrument,surface,_ms = args[0:51]

  nu_cld = xv[0]
  rcal = xv[4]
  nu_dzl = xv[8]  
  
  if (np.isnan(xv[6]) and seed_de==8):
    flag_nan = 1
    return True    #early return to the main function (due to failure to converge)
  elif (np.isnan(xv[6]) and seed_de==10):
    flag_nan = 10
    return True    #early return to the main function (due to failure to converge)
   
  Nfeval = Nfeval+1
  zct = ctop+(xv[6]*gridres)                                       
  zcb = cbase_mod_lo+(xv[7]*(cbase_mod_hi-cbase_mod_lo))            
  cdepth = zct-zcb
  Pcb = interp(zcb,tpqheight,presdata)
  Tcb = interp(zcb,tpqheight,tempdata)
  Gamma_l, rho_air = cloud_ad(Pcb,Tcb)
 
  #1st, compute LWC_ad and LWC_cld using lidar grid
  ind_cloud = [b for b,x in enumerate(height_beta) if (x>=zcb and x<=zct)] 
  zdata_cld = height_beta[ind_cloud]
  if (zdata_cld[-1] < zct): zdata_cld = append(zdata_cld,zct)   
    
  zdata_cld_cb = zdata_cld-zcb
  fz=subad_frac(zdata_cld_cb,cdepth,xv[2],xv[1])
  lwc_ad = rho_air*Gamma_l*zdata_cld_cb     #kg/m3 
  lwc_ad = lwc_ad*1000.                     #g/m3
  lwc_cld = fz*lwc_ad                       #g/m3; see eq. A2 of Boers et al.                
  Nz_cld,re_cld,ext_cld = calc_nre(zdata_cld_cb,fz,xv[3],xv[0],Gamma_l,rho_air,mmod) 

  #2nd, weights and smoothing
  nwidth = 2*(ind_cloud[0]-1-indl_cbase)+1    
  if (nwidth==1):    
    lwc_smooth= append(0,lwc_cld)
    Nz_smooth = append(Nz_cld[0],Nz_cld)
    z_smooth = append(height_beta[ind_cloud[0]-1],zdata_cld)
  else:              
    x=np.linspace(0,0.5*(nwidth-1),0.5*(nwidth+1))
    y = exp(-xv[12]*x)       
    weight = zeros(nwidth)  
    weight[int(0.5*(nwidth-1)):] = y[:]
    weight[range(int(0.5*(nwidth-1))-1,-1,-1)] = y[1:]

    lwc2 = zeros(lwc_cld.size+nwidth) 
    lwc2[nwidth:] = lwc_cld
    lwc_smooth = zeros(lwc_cld.size+0.5*(nwidth+1))   
    extragates=0
    for i in range(nwidth+extragates):   
      lwc_smooth[i] = np.average(lwc2[i:i+nwidth],weights=weight)
    lwc_smooth[nwidth+extragates:] = lwc_cld[nwidth+extragates-int(0.5*(nwidth+1)):]  
  
    ind_extension = np.arange(int(0.5*(nwidth+1)),0,-1)    
    ind_temp = list(ind_cloud[0]-ind_extension)
    ind_smooth = ind_temp+ind_cloud
    z_smooth = height_beta[ind_smooth]      
    if (z_smooth[-1] < zct): z_smooth = append(z_smooth,zct)  
    ind_lo = int(0.5*(nwidth+1))
    Nz_smooth = zeros(Nz_cld.size+ind_lo)
    Nz_smooth[ind_lo:] = Nz_cld[:]
    Nz_smooth[0:ind_lo] = Nz_cld[0]                   
    
  #3rd, compute other microphysical properties of cloud (ra, rn, ext)
  re_smooth = (1.5*lwc_smooth*(nu_cld+2)*(nu_cld+2)/(2*pi*Nz_smooth*1.e6*nu_cld*(nu_cld+1)))**(1./3)
  if (mmod == 0): re_smooth[-1] = re_smooth[-2]+(re_smooth[-2]-re_smooth[-3])  
  ra_smooth = re_smooth*sqrt(nu_cld*(nu_cld+1))/(nu_cld+2)
  rn_smooth = re_smooth/(nu_cld+2)
  ext_smooth = 2*pi*Nz_smooth*ra_smooth*ra_smooth
  
  #4th, interpolate to radar grid and compute Z_cld
  rn_interp = zeros(height_z.size)
  Nz_interp = zeros(height_z.size)
  indgrid = list(where(height_z>=z_smooth[0])[0])     
  rn_interp[indgrid] = interp(height_z[indgrid],z_smooth,rn_smooth)   
  Nz_interp[indgrid] = interp(height_z[indgrid],z_smooth,Nz_smooth) 

  r6 = ((rn_interp*1e6)**6.)*(nu_cld)*(nu_cld+1)*(nu_cld+2)*(nu_cld+3)*(nu_cld+4)*(nu_cld+5)   
  Fradref=64.*Nz_interp*r6*1.e-18*rcal         
  maxpos_mod = where(Fradref == np.max(Fradref[-4:-1]))[0][0] 
  maxpos_obs = where(Z_mean == np.max(Z_mean[-4:-1]))[0][0]

  #5th, compute excess reflectivity and Z_dzl
  residual = Z_mean-Fradref   
  residual[residual<0.0] = 0.0
  Fradref_dzl = zeros(residual.size)
  Fradref_dzl[0:indr_cbase2] = residual[0:indr_cbase2]  
  Fradref_dzl[-1] = residual[-1] 
  for i in range(indr_cbase2,residual.size-1):  
    Fradref_dzl[i] = np.mean(residual[i-1:i+2])    #mm6/m3  
  ind_posres = where(Fradref_dzl > 0.0)[0][-1]  
  radrefratio = Fradref_dzl[-2:]/Fradref[-2:]
  
  #6th, construct drizzle profile using Z_dzl and parametrized re_dzl
  ext_dzl_cbase = 10.**(xv[10])
  nominator = 2*pi*((nu_dzl+2)**3)*Z_cbase*1.e-18/rcal
  denominator= 64.*ext_dzl_cbase*(nu_dzl+3)*(nu_dzl+4)*(nu_dzl+5)  
  re_dzl_cbase = (nominator/denominator)**(0.25)  

  ind150 = where(height_z-cbase <=np.min([150.,height_z[ind_posres-1]-cbase]))[0][-1]   
  ext150 = interp(height_z[ind150],z_smooth,ext_smooth)
  ext_dzl_peak = (10.**(xv[11]))*ext150 
  nominator = 2*pi*((nu_dzl+2)**3)*Fradref_dzl[ind150]*1.e-18/rcal
  denominator= 64.*ext_dzl_peak*(nu_dzl+3)*(nu_dzl+4)*(nu_dzl+5)  
  re_dzl_peak = (nominator/denominator)**(0.25)    
  
  violation1 = (maxpos_mod != maxpos_obs)   
  violation2 = (np.max(re_cld) > re_dzl_lo) 
  violation3 = (np.any(radrefratio > 1.0))  
  violation4 = (re_dzl_peak > re_dzl_cbase) 
  violation6 = (np.all(Fradref_dzl[indr_cbase2:] == 0.0)) 
  if (violation1 or violation2 or violation3 or violation4 or violation6):
    cost_value = 100.*cost_value
    return cost_value

  if (ind_posres < Fradref.size-1):
    dtop = height_z[ind_posres+1]                  
    height_z2 = append(cbase,height_z[indr_cbase2:ind_posres+2])  
  else:
    dtop = zct                                   
    height_z2 = append(cbase,append(height_z[indr_cbase2:],dtop))  
  norm_height = (height_z2-cbase)/(dtop-cbase)    

  ind_Zpeak = where(Fradref_dzl == np.max(Fradref_dzl[indr_cbase2:]))[0][-1]  
  y_top = (height_z[ind150]-cbase)/(dtop-cbase)
  bslope = ((-1*np.log(re_dzl_peak/re_dzl_cbase))**2)/y_top
  re_dzl_incld = re_dzl_cbase*exp(-sqrt(bslope*norm_height))  
  
  ext_dzl_blw = (10.**(xv[9])) * ext_dzl_cbase   
  nominator = 2*pi*((nu_dzl+2)**3)*Fradref_dzl[0]*1.e-18/rcal
  denominator= 64.*ext_dzl_blw*(nu_dzl+3)*(nu_dzl+4)*(nu_dzl+5)  
  re_dzl_blw = (nominator/denominator)**(0.25)    
  
  norm_height = (append(dbase,height_z[0:indr_cbase1+1])-dbase)/(cbase-dbase)  
  y_base = (height_z[0]-dbase)/(cbase-dbase)   
  bslope = np.log(re_dzl_blw/re_dzl_cbase)/np.log(y_base)  
  re_dzl_blwcld = re_dzl_cbase*(norm_height**bslope)   
  re_dzl_blwcld[0] = 0.0
  
  re_dzl = append(re_dzl_blwcld,re_dzl_incld)    

  if (indr_cbase2-indr_cbase1 == 2): nominator = 2*pi*((nu_dzl+2)**3)*Fradref_dzl[0:ind_posres+1]*1.e-18/rcal          
  elif (indr_cbase2-indr_cbase1 == 1): 
    radref_dzl = np.insert(Fradref_dzl[0:ind_posres+1],indr_cbase2,Z_cbase) 
    nominator = 2*pi*((nu_dzl+2)**3)*radref_dzl*1.e-18/rcal       
  denominator= 64.*(re_dzl[1:-1]**4)*(nu_dzl+3)*(nu_dzl+4)*(nu_dzl+5)   
  ext_dzl = append(append(0.0,nominator/denominator),0.0)       

  violation5 = (np.isnan(re_dzl).any() or np.max(re_dzl) > re_dzl_hi)   
  if (y_base > 0.5): violation7 = (re_dzl_blw < 0.0)     
  else: violation7 = (re_dzl_blw > re_dzl_cbase)        
  violation8 = (np.min(re_dzl[1:-1]) < re_dzl_lo)   
  if (violation7 or violation8 or violation5):
    cost_value = 100.*cost_value 
    return cost_value
  
  rn_dzl = re_dzl/(nu_dzl+2)                      
  ra_dzl = sqrt(rn_dzl*rn_dzl*nu_dzl*(nu_dzl+1))
  lwc_dzl = 1.e6*re_dzl*ext_dzl/1.5              
  Nz_dzl = ext_dzl/(2.*pi*ra_dzl*ra_dzl)         
  Nz_dzl[ra_dzl == 0.0] = 0.0

  re_dzl[lwc_dzl == 0.0] = 0.0   
  ra_dzl[lwc_dzl == 0.0] = 0.0   
  rn_dzl[lwc_dzl == 0.0] = 0.0   
  
 
  #7th, combine drizzle and cloud into a common grid
  #------drizzle-------
  z_common = append(height_beta[indl_dbase:indl_cbase],z_smooth) 
  if (z_common[0]>dbase):z_common = append(dbase,z_common) 
  if (dtop != zct and dtop not in z_common):
    ztemp = append(z_common,dtop)
    z_common = np.sort(ztemp)

  lwc_dzl_common = zeros(z_common.size)
  ra_dzl_common = zeros(z_common.size)
  re_dzl_common = zeros(z_common.size)
  rn_dzl_common = zeros(z_common.size)
  ext_dzl_common = zeros(z_common.size)
  Nz_dzl_common = zeros(z_common.size)
  indgrid = list(where(z_common<=dtop)[0])

  height_z3 = append(append(dbase,height_z[0:indr_cbase1+1]),height_z2)
  lwc_dzl_common[indgrid] = interp(z_common[indgrid],height_z3,lwc_dzl)

  ra_dzl_common[indgrid] = interp(z_common[indgrid],height_z3,ra_dzl)
  re_dzl_common[indgrid] = interp(z_common[indgrid],height_z3,re_dzl)
  rn_dzl_common[indgrid] = interp(z_common[indgrid],height_z3,rn_dzl)
  ext_dzl_common[indgrid] = interp(z_common[indgrid],height_z3,ext_dzl)  
  Nz_dzl_common[indgrid] = interp(z_common[indgrid],height_z3,Nz_dzl)

  #------cloud-------
  lwc_cld_common = zeros(z_common.size)
  ra_cld_common = zeros(z_common.size)
  re_cld_common = zeros(z_common.size)
  rn_cld_common = zeros(z_common.size)
  ext_cld_common = zeros(z_common.size)
  Nz_cld_common = zeros(z_common.size)
  indgrid = list(where((z_common>=cbase) & (z_common<=zct))[0])
  lwc_cld_common[indgrid] = interp(z_common[indgrid],z_smooth,lwc_smooth)
  ra_cld_common[indgrid] = interp(z_common[indgrid],z_smooth,ra_smooth)
  re_cld_common[indgrid] = interp(z_common[indgrid],z_smooth,re_smooth)
  rn_cld_common[indgrid] = interp(z_common[indgrid],z_smooth,rn_smooth)
  ext_cld_common[indgrid] = interp(z_common[indgrid],z_smooth,ext_smooth)
  Nz_cld_common[indgrid] = interp(z_common[indgrid],z_smooth,Nz_smooth)
  
  #-------combined cloud and drizzle properties
  lwc_all_common = lwc_cld_common+lwc_dzl_common
  ext_all_common = ext_cld_common+ext_dzl_common 
  ra_all_common = (ra_cld_common*ext_cld_common+ra_dzl_common*ext_dzl_common)/(ext_cld_common+ext_dzl_common) 
  re_all_common = (re_cld_common*ext_cld_common+re_dzl_common*ext_dzl_common)/(ext_cld_common+ext_dzl_common)  
  rn_all_common = (rn_cld_common*ext_cld_common+rn_dzl_common*ext_dzl_common)/(ext_cld_common+ext_dzl_common)  
  ind0 = list(where(ext_all_common == 0.0)[0])
  re_all_common[ind0] = 0.0
  rn_all_common[ind0] = 0.0
  ra_all_common[ind0] = 0.0
  Nz_all_common = Nz_cld_common+Nz_dzl_common     

  #8th, compute Ftb    
  lwc30 = zeros(n30)
  z_com_ic = zeros(z_common.size)    
  lwc_com_ic = zeros(z_common.size)
  z_com_ic[:] = z_common[:]           
  lwc_com_ic[:] = lwc_all_common[:]   

  bad_elements = list(set(z_com_ic)-set(height_incld))  
  for element in bad_elements:   
    ind = where(element == z_com_ic)[0][0]
    z_com_ic = np.delete(z_com_ic,ind)  
    lwc_com_ic = np.delete(lwc_com_ic,ind)

  lwc30[n_blwcb:n_blwcb+len(ind_beta_incld)] = lwc_com_ic
  Ftb = calc_tau_cloud_tb(height30,temp30,lwc30,freq_brt,mu,tau_gas_total,liq_abs)

  
  #9th, Klett method 
  indbeta = list(where((height_beta >= dbase) & (height_beta <= zct))[0])
  z_com_ic = zeros(z_common.size)
  ext_total = zeros(z_common.size)
  ra_total = zeros(z_common.size)
  z_com_ic[:] = z_common[:]
  ext_total = ext_all_common[:]
  ra_total = ra_all_common[:]
  bad_elements = list(set(z_com_ic)-set(height_beta[indbeta]))  
  for element in bad_elements:   
    ind = where(element == z_com_ic)[0][0]
    z_com_ic = np.delete(z_com_ic,ind)    
    ext_total = np.delete(ext_total,ind)
    ra_total = np.delete(ra_total,ind)

  indd = indl_dbase-1
  
  clid,Fbeta = calc_beta((indd,xv[5],s_air,s_arsl),_ms,configms,instrument,surface,ratio,ext_total,ra_total,height_beta,indbeta,ext_air,beta_mean)

  ffx = zeros(ntb+nbeta+nrref)         #beta from ground to cloud top                                                        
  extras = zeros(Fradref_dzl.size)
  extras[-2:] = Fradref_dzl[-2:]
  
  ffx[0:ntb] = Ftb
  ffx[ntb:ntb+nbeta] = Fbeta[0:indl_200+1]

  lwc_heightz = interp(height_z,z_common,lwc_all_common) 

  kappac = kappa1_l * lwc_heightz               #[dB/km]
  kappac = kappac/(10.*np.log10(np.e))  #convert to [Np/km] to be integrated into optical depth
  opac = zeros(lwc_heightz.size)
  kappac_cum = 0.0
  for i in range(1,opac.size):
    kappac_cum = kappac_cum + kappac[i-1]
    opac[i] = kappac_cum*(gridres*0.001)  #gridres converted to km

  Fradref_att = (Fradref+Fradref_dzl)*np.exp(-2.*opac)
  ffx[ntb+nbeta:ntb+nbeta+nrref] = Fradref_att + extras

  diff = yy-ffx

  sumtb = 0.0
  for i in range(ntb):
    for j in range(ntb):
       term = cov_msr_inv[i,j]*diff[i]*diff[j]
       sumtb = sumtb + term
   
  sumbeta = 0.0
  for i in range(ntb,ntb+nbeta):
    for j in range(ntb,ntb+nbeta):
       term = cov_msr_inv[i,j]*diff[i]*diff[j]
       sumbeta = sumbeta + term

  sumz = 0.0
  for i in range(ntb+nbeta,ntb+nbeta+nrref):
    for j in range(ntb+nbeta,ntb+nbeta+nrref):
       term = cov_msr_inv[i,j]*diff[i]*diff[j]
       sumz = sumz + term
       
  absdifftb = np.sum(abs(yy[0:ntb]-ffx[0:ntb]))
  absdiffbeta = np.sum(abs(yy[ntb:ntb+nbeta]-ffx[ntb:ntb+nbeta]))
  absdiffrref = np.sum(abs(yy[ntb+nbeta:ntb+nbeta+nrref]-ffx[ntb+nbeta:ntb+nbeta+nrref]))
  cost_value = sumtb+sumbeta+sumz
  
  cff.write(str(Nfeval)+'  '+str(absdifftb)+'  '+str(sumtb)+'  '+str(absdiffbeta)+'  '+str(sumbeta)+'  '+str(absdiffrref)+'  '+str(sumz)+'  '+str(np.sum(abs(yy-ffx)))+'  '+str(cost_value)+'\n')   #'  '+str(term_ap)+'\n')

  return cost_value


def main():
  global progf, Niter, Nfeval, flag_nan, cost_value
  
  theta = 0.0                           #MWR points to zenith
  s_air = 8.*pi/3.                      #lidar ratio for molecules
  s_arsl = 50.                          #lidar ratio for aerosols
      
  re_dzl_lo = 13.e-6     #lower threshold of drizzle droplet effective radius [m]
  re_dzl_hi = 150.e-6    #upper threshold of drizzle droplet effective radius [m]

  if (len(sys.argv) == 1):
    print "--------- INCOMPLETE ARGUMENTS: provide the profile number to invert ----------"
    exit()
    
  elif (len(sys.argv) == 2):   #invert only one profile
    atnumber = int(sys.argv[1])
    endcount = atnumber
    
  elif (len(sys.argv) > 2):    #invert multiple profiles 
    atnumber = int(sys.argv[1])
    endcount = int(sys.argv[2])
    
#---------------------Read configfile-------------------------
  config = ConfigParser.ConfigParser()
  config.read('configfile')

  infilename = config.get('options','infile')
  tpqfilename = config.get('options','tpqfile')
  infix = config.get('options','infix')
  calc_error = int(config.get('options','calc_error'))
  
  mmod = int(config.get('mixmodel','mmod'))

  wav = float(config.get('lidar','wavelength'))        #m 
  alt = float(config.get('lidar','altitude'))          #m
  div = float(config.get('lidar','divergence'))        #rad
  fov = float(config.get('lidar','fieldview'))         #rad
  ratio = float(config.get('lidar','lidarratio'))      #sr
  small_ms=int(config.get('lidar','small_ms'))   
  wide_ms=int(config.get('lidar','wide_ms'))   
 
  radarfreq = float(config.get('radar','frequency'))   #GHz
  
  liq_abs = config.get('MWR','liq_abs')
  gas_abs = config.get('MWR','gas_abs')

  max_it = int(config.get('DE','max_it'))
  recomb = float(config.get('DE','recomb'))
  popsizef = int(config.get('DE','popsizef'))
  b_str = config.get('DE','mutationf')
  b_list = [float(n) for n in b_str.split(',')] 
  mutationf = tuple(b_list)
  tolerance = float(config.get('DE','tolerance'))
  
  b_str = config.get('boundaries1','lb1')
  b_list = [float(n) for n in b_str.split(',')]
  lb1 = array(b_list)

  b_str = config.get('boundaries1','ub1')
  b_list = [float(n) for n in b_str.split(',')]
  ub1 = array(b_list)

  b_str = config.get('boundaries2','lb2')
  b_list = [float(n) for n in b_str.split(',')]
  lb2 = array(b_list)

  b_str = config.get('boundaries2','ub2')
  b_list = [float(n) for n in b_str.split(',')]
  ub2 = array(b_list)
  

#---------------for radar attenuation calculation----------------------

  if (radarfreq < 100):          #[0 - 100) GHz        
    x0 = 0.0
    x1 = 2.18e-3
    x2 = 3.9e-4
  else:                          #[100-1000] GHz          
    x0 = -2.24
    x1 = 7.02e-2
    x2 = -2.05e-5
  #y coefficient
  if (radarfreq < 250):          #[0 - 250] GHz
    y0 = 9.73
    y1 = -8.92e-2
    y2 = 1.73e-4
  else:                          #[250-1000] GHz
    y0 = -1.12
    y1 = -3.04e-3
    y2 = 3.6e-7
  poly_a = x0 + x1*radarfreq + x2*radarfreq*radarfreq  #scalar
  poly_b = y0 + y1*radarfreq + y2*radarfreq*radarfreq  #scalar         


#---------------------------set up multiscatter------------------------


  #small angle algorithms:
  (MS_SINGLE_AND_SMALL_ANGLE_NONE,MS_SINGLE_SCATTERING,MS_SMALL_ANGLE_PVC_ORIGINAL,MS_SMALL_ANGLE_PVC_FAST,MS_SMALL_ANGLE_PVC_EXPLICIT,MS_SMALL_ANGLE_PVC_FAST_LAG,MS_NUM_SMALL_ANGLE_ALGORITHMS) = (0,1,2,3,4,5,6)
  
  #wide angle algorithms:
  (MS_WIDE_ANGLE_NONE,MS_WIDE_ANGLE_TDTS_FORWARD_LOBE,MS_WIDE_ANGLE_TDTS_NO_FORWARD_LOBE,MS_NUM_WIDE_ANGLE_ALGORITHMS) = (0,1,2,3)
  
  #receiver type:
  (MS_TOP_HAT,MS_GAUSSIAN) = (0,1)
  
  if (small_ms==0):small=MS_SMALL_ANGLE_PVC_ORIGINAL
  elif (small_ms==1):small=MS_SMALL_ANGLE_PVC_FAST
  elif (small_ms==2):small=MS_SMALL_ANGLE_PVC_EXPLICIT

  if (wide_ms==0):wide=MS_WIDE_ANGLE_NONE
  elif (wide_ms==1):wide=MS_WIDE_ANGLE_TDTS_FORWARD_LOBE

  opti = 0
  scatt_order = 4
  theta_ms = 0.1
  first_gate = 0
  coherent = 1.0
  lag = None
  source = -1.0
  reflected = -1.0
  multiplier = 1.0

  configms = ms_config(small,
                     wide,
                     opti,
                     scatt_order,
                     theta_ms,
                     first_gate,
                     coherent,
                     lag,
                     source,
                     reflected,
                     multiplier)

  surface = ms_surface(0.0,
                       0.0,
                       0.0,
                       0.0,
                       0.0)

  receiver = MS_TOP_HAT
  ffov = np.array([fov])   
  nfov = 1
  instrument = ms_instrument(receiver,
                             alt,
                             wav,
                             div,
                             as_ctypes(ffov),
                             nfov)

  _ms = cdll.LoadLibrary("libmultiscatter.so")
  
  
#---------------------------Load data------------------------------
  count = 0  

  #prepare file containing radar, lidar and MWR data 
  infile = open(infilename,'r')
   
  #prepare file containing T,P,Q data (up to at least 30 km)
  tpqfile = open(tpqfilename,'r')
  tpqheight = pickle.load(tpqfile)


  sys.stderr.write("Start: count "+str(atnumber)+"\n")
  sys.stderr.write("End: count "+str(endcount)+"\n")

  while 1:
    try:    
      along_track,height_z,Z_mean,Z_err,height_beta,beta_mean,beta_err,freq_brt,tb_mean,tb_err,cbase_mean = pickle.load(infile)  
      along_track_tpq,tempdata,presdata,humdata = pickle.load(tpqfile)

      if (count < atnumber):
         count += 1
         continue    #next profile
      elif (count > endcount):
        print ("The end")
        sys.stderr.write("The end\n")        
        infile.close()
        tpqfile.close()
        exit()
       
      print '\n============Count: ',count, '================='

      temp_heightz = interp(height_z,tpqheight,tempdata)
      gridres = height_z[1]-height_z[0]

      #for radar attenuation calculation:"
      Tnorm = 300./temp_heightz
      kappa1_l = poly_a*(Tnorm**poly_b)     #[dB m3/km /g] 
      
      if (height_beta[0] >= height_z[0]):    
        sys.stderr.write("--------- ERROR: beta data not available below the first radar gate; count "+str(count)+" ----------\n")

        count += 1
        continue    

      cffile = 'cf'+infix+str(count)             
      sgnfile = 'sgn'+infix+str(count)           
      progfile = 'prog'+infix+str(count)         
      retfile = 'ret'+infix+str(count)           

      
      #-----estimate the cloud base height of the mean beta profile-----
      ind_max =list(where(beta_mean == np.max(beta_mean))[0])[0] 
      ind_search = range(ind_max-10,ind_max)  #assume that cloud base is located within 10 range gates from the peak
      beta1 = beta_mean[0:-1]
      beta2 = beta_mean[1:]
      ind_ratio=list(where(beta2/beta1 > 1.5)[0])
      ind_intersect = sorted(list(set(ind_ratio)&set(ind_search)))
      flag_cbase = 0    #cbase is not yet found for the given profile
      for i in range(len(ind_intersect)):
        indl_cbase = ind_intersect[i]             
        beta1 = beta_mean[indl_cbase:ind_max]                                                                                    
        beta2 = beta_mean[indl_cbase+1:ind_max+1]                                                                                
        if (np.all(beta2/beta1 > 1.0)):                                                                                        
          cbase = height_beta[indl_cbase]
          flag_cbase = 1
          break
        else:
          continue     

      if (flag_cbase == 0):
        sys.stderr.write("--------- ERROR: cloud base is not found; count "+str(count)+" ----------\n")

        count += 1
        continue   #the next profile  
        
      if (ind_max-indl_cbase <  3):
        cbase_mod_lo = cbase
        cbase_mod_hi = height_beta[indl_cbase+1]
      else:
        cbase_mod_lo = height_beta[indl_cbase+1]
        cbase_mod_hi = height_beta[indl_max-1]

        count += 1
        continue   #the next profile  

      indr_cbase2 = where(height_z > cbase)[0][0]   
       
      dbase = height_z[0]-gridres
      ctop = height_z[-1] + gridres  

      indl_ctop = where(height_beta < ctop)[0][-1]    
      indl_dbase = where(height_beta > dbase)[0][0]   
      indl_200 = where(height_beta < cbase+200.)[0][-1]  
      
      indices = list(where(tpqheight<=np.min([dbase,cbase]))[0])  
      height_blwcb = tpqheight[indices]                        
      temp_blwcb = tempdata[indices]  
      pres_blwcb = presdata[indices]  
      hum_blwcb =  humdata[indices]   
      n_blwcb = height_blwcb.size
      
      indices = list(where(tpqheight>=ctop)[0])      
      height_abvct = tpqheight[indices]    
      temp_abvct = tempdata[indices]  
      pres_abvct = presdata[indices]  
      hum_abvct =  humdata[indices]   
      
      ind_beta_incld = list(where((height_beta>np.min([dbase,cbase])) & (height_beta<=height_z[-1]))[0])  
      height_incld = height_beta[ind_beta_incld]
      temp_incld = interp(height_incld,tpqheight,tempdata)
      pres_incld = interp(height_incld,tpqheight,presdata)
      hum_incld = interp(height_incld,tpqheight,humdata)

      #stitch TPQ profiles together 
      height30 = append(append(height_blwcb,height_incld),height_abvct)
      pres30 = append(append(pres_blwcb,pres_incld),pres_abvct)
      temp30 = append(append(temp_blwcb,temp_incld),temp_abvct)
      hum30 = append(append(hum_blwcb,hum_incld),hum_abvct)
      n30 = height30.size

      presdata_extair = interp(height_beta,tpqheight,presdata) 
      tempdata_extair = interp(height_beta,tpqheight,tempdata) 
      ext_air = ext_coeff_air(wav=wav,pressure=presdata_extair,temperature=tempdata_extair)
  
#      nbt = height_beta.size  
#      ra_ms = zeros(nbt)+1e-8
#      extco_ms = zeros(nbt)
#      sfactor = sfactor_cld(0.0,height_beta,extco_ms,ratio) 
#      beta_air = zeros(nbt)
#
#      #for beta_air calculation, the albedo, asym, albedo_air, cldfrac, icefrac can be the same values as used in the Fbeta calculation (not relevant) as long as extco_ms = 0.0
#      _ms.multiscatter(nbt,
#                       nbt,
#                       byref(configms),   
#                       instrument,
#                       surface,
#                       as_ctypes(zeros(nbt)+height_beta),
#                       as_ctypes(ra_ms),
#                       as_ctypes(extco_ms),
#                       None, #as_ctypes(albedo),
#                       None, #as_ctypes(asym),
#                       as_ctypes(sfactor),
#                       as_ctypes(ext_air),
#                       None, #as_ctypes(albedo_air),
#                       None, #as_ctypes(cldfrac),
#                       None, #as_ctypes(icefrac),
#                       as_ctypes(beta_air),   #output
#                       None)      


      tau_gas = calc_tau_gas(height30,temp30,pres30,hum30,freq_brt,gas_abs)
      tau_gas_total = tau_gas[0]
      
      theta = theta*pi/180.0                                  #convert to radians      
      mu = np.cos(theta) + 0.025*exp(-11.*np.cos(theta))      #Rozenberg approximation for 1/airmass

      cov_msr = cov_mat_msr(tb_err,beta_err[0:indl_200+1],Z_err,beta_mean[0:indl_200+1],Z_mean)
      cov_msr_inv = invert_lu(cov_msr)                  #Se^{-1}

      ntb = tb_mean.size
      nbeta = indl_200+1
      nrref = Z_mean.size  

      yy = zeros(ntb+nbeta+nrref)         
      yy[0:ntb] = tb_mean
      yy[ntb:ntb+nbeta] = beta_mean[0:indl_200+1]
      yy[ntb+nbeta:ntb+nbeta+nrref] = Z_mean


      cff = open(cffile,'w')
      cff.write('# Nfeval    T_B(abs.diff.)     T_B(term)      beta(abs.diff.)      beta(term)    Z(abs.diff.)    Z(term)    total abs.diff.    total_cost_value\n')

      Nfeval=0
      Niter = 0
      flag_nan = 0
        
      cost_value = 1.e6     #random large number      
      seed_de = 8

      progf = open(progfile,'w')
      progf.write('#along track count = '+str(atnumber)+'\n')

      if (height_z[0] >= cbase):    
        fargs=(ctop, gridres, cbase_mod_lo, cbase_mod_hi, tpqheight, presdata, tempdata, height_beta,mmod,indl_cbase,height_z,Z_mean,indr_cbase2,cbase,re_dzl_lo,cff,dbase,re_dzl_hi,indl_dbase,n30,height_incld,n_blwcb,ind_beta_incld,height30,temp30,freq_brt,liq_abs,tau_gas_total,mu,beta_mean,ext_air,s_air,s_arsl,wav,alt,div,fov,ratio,ntb,nbeta,nrref,indl_200,kappa1_l,yy,cov_msr_inv,configms,instrument,surface,_ms)

        print " ******** There is no radar signal below cloud base; count ", count
        bounds2 = []
        for iix in range(lb2.size): bounds2 = bounds2+[tuple([lb2[iix],ub2[iix]])]
        
        progf.write('#seed(DE): '+str(seed_de)+'; maxiter(DE): '+str(max_it)+'\n')
        progf.write('#recombination(DE): '+str(recomb)+'; popsize(DE): '+str(popsizef)+'\n')
        progf.write('#mutation(DE): '+str(mutationf)+'; tolerance(DE): '+str(tolerance)+'\n')
        progf.write('#DE upper bound: '+str(ub2[0:4])+'\n')
        progf.write('#                '+str(ub2[4:8])+'\n')
        progf.write('#                '+str(ub2[8:])+'\n')
        progf.write('#DE lower bound: '+str(lb2[0:4])+'\n')
        progf.write('#                '+str(lb2[4:8])+'\n')
        progf.write('#                '+str(lb2[8:])+'\n')
        progf.write('#mixing model: '+str(mmod)+'\n')
        progf.write('#Radar grid resolution [m]: '+str(gridres)+'\n')
        progf.write('#Absorption model [liquid,gas]: ['+liq_abs+','+gas_abs+']\n')
        progf.write('#Zenith angle (degrees): '+str(theta)+'\n')
        progf.write('iter  frac.conv.  x[0]  x[1]  x[2]  x[3]  x[4]  x[5]  x[6]  x[7]  x[8]  x[9]  x[10]  x[11]  x[12]\n')
 
        c0 = clock()
        res = differential_evolution(cost2, bounds2, args=fargs,seed=seed_de, tol=tolerance, maxiter=max_it, polish=1,init='latinhypercube',callback=callback_func,mutation=mutationf,popsize=popsizef, recombination=recomb)
        c1 = clock()
        cff.close()

        z_res = np.copy(z_common)
        lwc_cld_res = np.copy(lwc_cld_common)
        re_cld_res = np.copy(re_cld_common)
        Nz_cld_res = np.copy(Nz_cld_common)
        ext_cld_res = np.copy(ext_cld_common)
        lwc_dzl_res = np.copy(lwc_dzl_common)
        re_dzl_res = np.copy(re_dzl_common)
        Nz_dzl_res = np.copy(Nz_dzl_common)
        ext_dzl_res = np.copy(ext_dzl_common)
        Ftb_res = np.copy(Ftb)
        Fradref_res = np.copy(Fradref)
        Fradref_dzl_res = np.copy(Fradref_dzl)
        Fbeta_res = np.copy(Fbeta)

        retf = open(retfile,'w')
        retf.write('#time[hrs] = '+str(along_track/3600.)+'\n')   #time in hrs
        retf.write('#-----cloud parameters-----\n')       
        retf.write('#boundaries   = '+str(cbase)+' , '+str(zcb)+' ; '+str(zct)+' [m]\n')
        retf.write('#nu           = '+str(xres[0])+'\n')
        retf.write('#hhat         = '+str(xres[1])+'\n')
        retf.write('#alpha        = '+str(xres[2])+'\n')
        retf.write('#Nad          = '+str(xres[3])+' [1/m3]\n')
        retf.write('#rcal         = '+str(xres[4])+'\n')
        retf.write('#lcal         = '+str(clid)+'  ;  ext_dbase: '+str(10.**xres[5])+'  ;  '+str(xres[5])+'\n')
        retf.write('#cldtop       = '+str(ctop+(xres[6]*gridres))+'  ;  '+str(xres[6])+'\n')
        retf.write('#cbasemod     = '+str(cbase_mod_lo+(xres[7]*(cbase_mod_hi-cbase_mod_lo)))+'  ;  '+str(xres[7])+'\n')      
        retf.write('#weight_cbase = '+str(xres[12])+'\n')
        retf.write('#-----drizzle parameters-----\n')       
        retf.write('#boundaries   = '+str(dbase)+' ; '+str(dtop)+' [m]\n')   
        retf.write('#nu           = '+str(xres[8])+'\n')
        retf.write('#hhat         = '+str(xres[9])+'\n')
        retf.write('#alpha        = '+str(xres[10])+'\n')
        retf.write('#lwc_scale    = '+str(10.**xres[11])+'  ;  '+str(xres[11])+'\n')
        retf.write('#===============================================================\n')
              
        
      else: 
        indr_cbase1 = where(height_z < cbase)[0][-1]  #index of the highest radar gate located below cbase  
        if (indr_cbase2-indr_cbase1 == 1): Z_cbase = interp(cbase, [height_z[indr_cbase1],height_z[indr_cbase2]], [Z_mean[indr_cbase1],Z_mean[indr_cbase2]]) 
        elif (indr_cbase2-indr_cbase1 == 2): Z_cbase = Z_mean[indr_cbase2-1]

        fargs=(ctop, gridres, cbase_mod_lo, cbase_mod_hi, tpqheight, presdata, tempdata, height_beta,mmod,indl_cbase,height_z,Z_mean,indr_cbase2,Z_cbase,cbase,re_dzl_lo,cff,dbase,indr_cbase1,re_dzl_hi,indl_dbase,n30,height_incld,n_blwcb,ind_beta_incld,height30,temp30,freq_brt,liq_abs,tau_gas_total,mu,beta_mean,ext_air,s_air,s_arsl,wav,alt,div,fov,ratio,ntb,nbeta,nrref,indl_200,kappa1_l,yy,cov_msr_inv,configms,instrument,surface,_ms)
        bounds1 = []
        for iix in range(lb1.size): bounds1 = bounds1+[tuple([lb1[iix],ub1[iix]])]

        progf.write('#seed(DE): '+str(seed_de)+'; maxiter(DE): '+str(max_it)+'\n')
        progf.write('#recombination(DE): '+str(recomb)+'; popsize(DE): '+str(popsizef)+'\n')
        progf.write('#mutation(DE): '+str(mutationf)+'; tolerance(DE): '+str(tolerance)+'\n')
        progf.write('#DE upper bound: '+str(ub1[0:4])+'\n')
        progf.write('#                '+str(ub1[4:8])+'\n')
        progf.write('#                '+str(ub1[8:])+'\n')
        progf.write('#DE lower bound: '+str(lb1[0:4])+'\n')
        progf.write('#                '+str(lb1[4:8])+'\n')
        progf.write('#                '+str(lb1[8:])+'\n')
        progf.write('#mixing model: '+str(mmod)+'\n')
        progf.write('#Radar grid resolution [m]: '+str(gridres)+'\n')
        progf.write('#Absorption model [liquid,gas]: ['+liq_abs+','+gas_abs+']\n')
        progf.write('#Zenith angle (degrees): '+str(theta)+'\n')
        progf.write('iter  frac.conv.  x[0]  x[1]  x[2]  x[3]  x[4]  x[5]  x[6]  x[7]  x[8]  x[9]  x[10]  x[11]  x[12]\n')
     
        c0 = clock()
        res = differential_evolution(cost1, bounds1, args=fargs,seed=seed_de, tol=tolerance, maxiter=max_it, polish=1,init='latinhypercube',callback=callback_func,mutation=mutationf,popsize=popsizef, recombination=recomb)
        c1 = clock()

        if (flag_nan == 1):   
          Nfeval = 0
          Niter = 0
          cost_value = 1.e6
          seed_de = 10
          flag_nan = 0    #reset
          print '-----------RESTART-------------'
          progf.write('iter  frac.conv.  x[0]  x[1]  x[2]  x[3]  x[4]  x[5]  x[6]  x[7]  x[8]  x[9]  x[10]  x[11]\n')
          cff.write('# Nfeval    T_B(abs.diff.)     T_B(term)      beta(abs.diff.)      beta(term)    Z(abs.diff.)    Z(term)    total abs.diff.    total_cost_value\n')
          c0 = clock()
          res = differential_evolution(cost1, bounds1, args=fargs,seed=seed_de, tol=tolerance, maxiter=max_it, polish=1,init='latinhypercube',callback=callback_func,mutation=mutationf,popsize=popsizef, recombination=recomb)
          
          c1 = clock()

          if (flag_nan==10): 
            sys.stderr.write("--------- ERROR: convergence not achieved; count "+str(count)+" ----------\n")
            cff.close()
            count+=1
            continue            

        cff.close()  
        z_res = np.copy(z_common)
        lwc_cld_res = np.copy(lwc_cld_common)
        re_cld_res = np.copy(re_cld_common)
        Nz_cld_res = np.copy(Nz_cld_common)
        ext_cld_res = np.copy(ext_cld_common)
        lwc_dzl_res = np.copy(lwc_dzl_common)
        re_dzl_res = np.copy(re_dzl_common)
        Nz_dzl_res = np.copy(Nz_dzl_common)
        ext_dzl_res = np.copy(ext_dzl_common)
        Ftb_res = np.copy(Ftb)
        Fradref_res = np.copy(Fradref)
        Fradref_dzl_res = np.copy(Fradref_dzl)
        Fbeta_res = np.copy(Fbeta)
               
        retf = open(retfile,'w')
        retf.write('#time[hrs] = '+str(along_track/3600.)+'\n')   #time in hrs
        retf.write('#-----cloud parameters-----\n')       
        retf.write('#boundaries   = '+str(cbase)+' , '+str(zcb)+' ; '+str(zct)+' [m]\n')
        retf.write('#nu           = '+str(xres[0])+'\n')
        retf.write('#hhat         = '+str(xres[1])+'\n')
        retf.write('#alpha        = '+str(xres[2])+'\n')
        retf.write('#Nad          = '+str(xres[3])+' [1/m3]\n')
        retf.write('#rcal         = '+str(xres[4])+'\n')
        retf.write('#lcal         = '+str(clid)+'  ;  ext_dbase: '+str(10.**xres[5])+'  ;  '+str(xres[5])+'\n')
        retf.write('#cldtop       = '+str(ctop+(xres[6]*gridres))+'  ;  '+str(xres[6])+'\n')
        retf.write('#cbasemod     = '+str(cbase_mod_lo+(xres[7]*(cbase_mod_hi-cbase_mod_lo)))+'  ;  '+str(xres[7])+'\n')      
        retf.write('#weight_cbase = '+str(xres[12])+'\n')
        retf.write('#-----drizzle parameters-----\n')       
        retf.write('#boundaries   = '+str(dbase)+' ; '+str(dtop)+' [m]\n')   
        retf.write('#nu           = '+str(xres[8])+'\n')
        retf.write('#ext_dzl_blw  = '+str((10.**xres[9])*(10.**xres[10]))+'  ;  '+str(xres[9])+'\n')
        retf.write('#ext_dzl_cbase= '+str(10.**xres[10])+'  ;  '+str(xres[10])+'\n')
        retf.write('#ext_dzl_peak = '+str((10.**xres[11])*np.max(ext_cld_common))+'  ;  '+str(xres[11])+'\n')
        retf.write('#===============================================================\n')
     
      print '------------cld-----------' 
      for i in range(z_res.size):
        print z_res[i], lwc_cld_res[i], re_cld_res[i], Nz_cld_res[i], ext_cld_res[i]
      print '------------dzl-----------' 
      for i in range(z_res.size):
        print z_res[i], lwc_dzl_res[i], re_dzl_res[i], Nz_dzl_res[i], ext_dzl_res[i]

      retf.write('#Col 1: Height [m]\n')
      retf.write('#Col 2: LWC [g/m3]\n')
      retf.write('#Col 3: r_e [m]\n')
      retf.write('#Col 4: extcoeff [1/m]\n')
      retf.write('#Col 5: N [1/m3]\n')
      retf.write('#========================= cloud ===============================\n')
      for i in range(z_res.size):
         retf.write(str(z_res[i])+'  '+str(lwc_cld_res[i])+'  '+str(re_cld_res[i])+'  '+str(ext_cld_res[i])+'  '+str(Nz_cld_res[i])+'\n')
      retf.write('#==================================================================\n')
      retf.write('88.  88.  88.  88.  88\n')         
      retf.write('#========================= drizzle ================================\n')
      for i in range(z_res.size):
        retf.write(str(z_res[i])+'  '+str(lwc_dzl_res[i])+'  '+str(re_dzl_res[i])+'  '+str(ext_dzl_res[i])+'  '+str(Nz_dzl_res[i])+'\n')
      retf.write('#==================================================================\n')
      retf.write('#'+res.message+'\n')
      retf.write('#Current function value: '+str(res.fun)+'\n')
      retf.write('#Iterations: '+str(res.nit)+'\n')
      retf.write('#Function evaluations: '+str(res.nfev)+'\n')
      retf.write('#Elapsed CPU time: '+str((c1-c0)/60.)+' minutes on '+system()+'\n')
      retf.close()    

      #-------------Printing out signals--------------
      sgnf = open(sgnfile,'w')
      sgnf.write ('#This logs the signal at the last iteration\n')
      sgnf.write ('#Height[m]     Z_fm_cld[mm6/m3]     Z_fm_dzl[mm6/m3]      Z_data[mm6/m3]      Z_err[mm6/m3]      Z_fm_tot_att[mm6/m3]\n')
      for i in range(nrref):
        sgnf.write(str(height_z[i])+'  '+str(Fradref_res[i])+'  '+str(Fradref_dzl_res[i])+'  '+str(Z_mean[i])+'  '+str(Z_err[i])+'  '+str(Fradref_att[i])+'\n')
      sgnf.write('#=======================================\n')
      sgnf.write ('#Frequency[GHz]   T_B_fm[K]     placeholder     T_B_data[K]     T_B_err[K]     placeholder\n')
      for i in range(ntb):
        sgnf.write(str(freq_brt[i])+'  '+str(Ftb_res[i])+'  '+str(0.0)+'  '+str(tb_mean[i])+'  '+str(tb_err[i])+'  '+str(0.0)+'\n')
      sgnf.write('#=======================================\n')
      sgnf.write ('#Height[m]     beta_fm[1/m/sr]      placeholder        beta_data[1/m/sr]       beta_err[1/m/sr]      placeholder\n')
      for i in range(height_beta.size):
        sgnf.write(str(height_beta[i])+'  '+str(Fbeta_res[i])+'  '+str(0.0)+'  '+str(beta_mean[i])+'  '+str(beta_err[i])+'  '+str(0.0)+'\n')
      sgnf.close()

      count += 1  
      
    except (EOFError): 
      print "End of File"
      infile.close()
      tpqfile.close()
  


if __name__ == "__main__":
  zeros = np.zeros
  array = np.array
  exp = np.exp
  interp = np.interp
  append = np.append
  where = np.where
  pi = np.pi
  sqrt = np.sqrt
  
  main()
