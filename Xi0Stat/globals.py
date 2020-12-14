import os
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import sys
import healpy as hp

dirName = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

miscPath = os.path.join(dirName, 'data', 'misc')

baseGWPath= os.path.join(dirName, 'data', 'GW') 

metaPath= os.path.join(baseGWPath, 'metadata') 

detectorPath = os.path.join(baseGWPath, 'detectors')



###########################
# CONSTANTS
###########################

O2BNS = ('GW170817',)
O3BNS = ('GW190425', )


d0GlobO2=123 # d_0 of eq. 2.125 for O2, in Mpc

zRglob = 0.5

nGlob = 1.91
gammaGlob = 1.6

clight = 2.99792458* 10**5

l_CMB, b_CMB = (263.99, 48.26)
v_CMB = 369

# Solar magnitude in B and K band
MBSun=5.498
MKSun=3.27

# Cosmologival parameters used in GLADE for z-dL conversion
H0GLADE=70
Om0GLADE=0.27

# Cosmologival parameters used for the analysis (RT minimal; table 2 of 2001.07619)
H0GLOB=67.9 #69
Om0GLOB=0.3
Xi0Glob =1.
cosmoglob = FlatLambdaCDM(H0=H0GLOB, Om0=Om0GLOB)

class PriorLimits:
    def __init__(self):
        self.H0min = 20
        self.H0max = 220
        self.Xi0min = 0.01
        self.Xi0max = 100

# Parameters of Schechter function in B band in units of 10^10 solar B band
# for h0=0.7
LBstar07 =2.45
phiBstar07  = 5.5 * 1e-3
alphaB07 =-1.07


# Parameters of Schechter function in K band in units of 10^10 solar K band
# for h0=0.7
LKstar07 = 10.56
phiKstar07 = 3.70 * 1e-3
alphaK07 =-1.02


###########################
###########################

from scipy.special import erfc, erfcinv

def sample_trunc_gaussian(mu = 1, sigma = 1, lower = 0, size = 1):

    sqrt2 = np.sqrt(2)
    Phialpha = 0.5*erfc(-(lower-mu)/(sqrt2*sigma))
    
    if np.isscalar(mu):
        arg = Phialpha + np.random.uniform(size=size)*(1-Phialpha)
        return np.squeeze(mu - sigma*sqrt2*erfcinv(2*arg))
    else:
        Phialpha = Phialpha[:,np.newaxis]
        arg = Phialpha + np.random.uniform(size=(mu.size, size))*(1-Phialpha)
        
        return np.squeeze(mu[:,np.newaxis] - sigma[:,np.newaxis]*sqrt2*erfcinv(2*arg))
    
def trunc_gaussian_pdf(x, mu = 1, sigma = 1, lower = 0):
#
#    if not np.isscalar(x) and not np.isscalar(mu):
#        x = x[np.newaxis, :]
#        if mu.ndim < 2:
#            mu = mu[:, np.newaxis]
#        if sigma.ndim < 2:
#            sigma = sigma[:, np.newaxis]
        
    Phialpha = 0.5*erfc(-(lower-mu)/(np.sqrt(2)*sigma))
    return 1/(np.sqrt(2*np.pi)*sigma)/(1-Phialpha) * np.exp(-(x-mu)**2/(2*sigma**2))
    
###########################
###########################

import multiprocessing

nCores = max(1,int(multiprocessing.cpu_count()/2)-1)

def fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))


def parmap(f, X):
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=fun, args=(f, q_in, q_out))
            for _ in range(nCores)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nCores)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]


###########################
###########################

    
def get_SchParams(Lstar, phiStar, h0):
        '''
        Input: Hubble parameter h0, values of Lstar, phiStar for h0=0.7
        Output: Schechter function parameters L_*, phi_* rescaled by h0
        '''
        Lstar = Lstar*(h0/0.7)**(-2)
        phiStar = phiStar*(h0/0.7)**(3)
        return Lstar, phiStar



def get_SchNorm(phistar, Lstar, alpha, Lcut):
        '''
        
        Input:  - Schechter function parameters L_*, phi_*, alpha
                - Lilit of integration L_cut in units of 10^10 solar lum.
        
        Output: integrated Schechter function up to L_cut in units of 10^10 solar lum.
        '''
        from scipy.special import gammaincc
        from scipy.special import gamma
                
        norm= phistar*Lstar*gamma(alpha+2)*gammaincc(alpha+2, Lcut)
        return norm



def ra_dec_from_th_phi(theta, phi):
        ra = np.rad2deg(phi)
        dec = np.rad2deg(0.5 * np.pi - theta)
        return ra, dec

  
def th_phi_from_ra_dec(ra, dec):
    theta = 0.5 * np.pi - np.deg2rad(dec)
    phi = np.deg2rad(ra)
    return theta, phi

cosmo70GLOB = FlatLambdaCDM(H0=70, Om0=Om0GLOB)

def dLGW(z, H0, Xi0, n):
    '''
    Modified GW luminosity distance
    '''
    cosmo=FlatLambdaCDM(H0=H0, Om0=Om0GLOB)
    return (cosmo.luminosity_distance(z).value)*Xi(z, Xi0, n=n) 
    
def Xi(z, Xi0, n):

    return Xi0+(1-Xi0)/(1+z)**n


zGridGLOB = np.logspace(start=-10, stop=5, base=10, num=1000)
dLGridGLOB = cosmo70GLOB.luminosity_distance(zGridGLOB).value
dcomGridGLOB = cosmo70GLOB.comoving_distance(zGridGLOB).value
HGridGLOB = cosmo70GLOB.H(zGridGLOB).value
from scipy import interpolate
dcom70fast = interpolate.interp1d(zGridGLOB, dcomGridGLOB, kind='cubic', bounds_error=False, fill_value=(0, np.NaN), assume_sorted=True)
dL70fast = interpolate.interp1d(zGridGLOB, dLGridGLOB, kind='cubic', bounds_error=False, fill_value=(0 ,np.NaN), assume_sorted=True)

H70fast = interpolate.interp1d(zGridGLOB, HGridGLOB, kind='cubic', bounds_error=False, fill_value=(70 ,np.NaN), assume_sorted=True)


def z_from_dLGW_fast(r, H0, Xi0, n):
    from scipy import interpolate
    z2dL = interpolate.interp1d(dLGridGLOB/H0*70*Xi(zGridGLOB, Xi0, n=n), zGridGLOB, kind='cubic', bounds_error=False, fill_value=(0,np.NaN), assume_sorted=True)
    return z2dL(r)
    

def z_from_dLGW(dL_GW_val, H0, Xi0, n):
    '''
    Returns redshift for a given luminosity distance dL_GW_val (in Mpc)
    
    Input:
        - dL_GW_val luminosity dist in Mpc
        - H0
        - Xi0: float. Value of Xi_0
        - n: float. Value of n

    '''   
    from scipy.optimize import fsolve
    #print(cosmo.H0)
    func = lambda z : dLGW(z, H0, Xi0, n=n) - dL_GW_val
    z = fsolve(func, 0.5)
    return z[0]

def dVdcom_dVdLGW(z, H0, Xi0, n):
# D_com^2 d D_com = D_com^2 (d D_com/d D_L^{gw}) d D_L^{gw}

# d D_com / d D_L^{gw} = d D_com /dz * ( d D_L^{gw} / dz ) ^(-1)
# [with D_L^{gw} = (Xi0 + (1-Xi0)(1+z)**(-n)) (1+z) Dcom ]
# = c/H(z) * (  (Xi0 + (1-n) (1-Xi0)(1+z)**(-n)  ) D_com + (Xi0 + (1-Xi0)(1+z)**(-n)) (1+z) c/H(z)  )^(-1)
# = (  (Xi0 + (1-n) (1-Xi0)(1+z)**(-n)  ) D_com H /c + (Xi0 + (1-Xi0)(1+z)**(-n)) (1+z)  )^(-1)

# D_com^2 / D_L^{gw}^2 remains

    h7 = H0 / 70
    
    #dcom = cosmo70GLOB.comoving_distance(z).value/h7

    #H = cosmo70GLOB.H(z).value*h7
    
    dcom = dcom70fast(z) / h7
    H  = H70fast(z) * h7
    
    dLGWsq_over_dcomsq = ((1+z)*Xi(z, Xi0=Xi0, n=n))**2
    
    jac = 1 / (H*(Xi0 + (1-n)*(1-Xi0)*(1+z)**(-n))*dcom/clight + (Xi0+(1-Xi0)*(1+z)**(-n))*(1+z) )
    
    jac /= dLGWsq_over_dcomsq
    
    return jac


def j(z):
    return cosmoglob.differential_comoving_volume(z).value*(cosmoglob.H0.value/clight)**3


def BB(dL, gamma=gammaGlob, d0=d0GlobO2):
    #print('gamma BB: %s' %gamma)
    #print('d0 BB: %s' %d0)
    a0, a1 = get_as(gamma)
    Bfit = np.exp(-a0*(dL-d0)/d0-a1*((dL-d0)/d0)**2)
        
    return np.where(dL<=d0, 1, Bfit)


def get_as(gamma):
    a0= (5.21+9.55*gamma+3.47*gamma**2)*1e-02
    a1=(7.37-0.72*gamma)*1e-02
    return a0, a1


class Logger(object):
    
    def __init__(self, fname):
        self.terminal = sys.__stdout__
        self.log = open(fname, "w+")
        self.log.write('--------- LOG FILE ---------\n')
        print('Logger created log file: %s' %fname)
        #self.write('Logger')
       
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

    def close(self):
        self.log.close
        sys.stdout = sys.__stdout__
        
    def isatty(self):
        return False
        
        
def hpx_downgrade_idx(hpx_array, nside_out=1024):
    #Computes the list of explored indices in a hpx array for the chosen nside_out
    arr_down = hp.ud_grade(hpx_array, nside_out)
    return np.where(arr_down>0.)[0] 
