import os
import numpy as np
from astropy.cosmology import FlatLambdaCDM

dirName = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

miscPath = os.path.join(dirName, 'data', 'misc')

metaPath= os.path.join(dirName, 'data', 'GW', 'metadata') 



###########################
# CONSTANTS
###########################

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
H0GLOB=69
Om0GLOB=0.3


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

    if not np.isscalar(x) and not np.isscalar(mu):
        x = x[:, np.newaxis]
        if mu.ndim < 2:
            mu = mu[np.newaxis, :]
        if sigma.ndim < 2:
            sigma = sigma[np.newaxis, :]
        
    Phialpha = 0.5*erfc(-(lower-mu)/(np.sqrt(2)*sigma))
    return 1/(np.sqrt(2*np.pi)*sigma)/(1-Phialpha) * np.exp(-(x-mu)**2/(2*sigma**2))
    
###########################
###########################

import multiprocessing

nCores = multiprocessing.cpu_count()

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



def dLGW(z, H0, Xi0, n=1.91):
    '''
    Modified GW luminosity distance
    '''
    cosmo=FlatLambdaCDM(H0=H0, Om0=Om0GLOB)
    return (cosmo.luminosity_distance(z).value)*Xi(z, Xi0, n=n) 
    
def Xi(z, Xi0, n=1.91):

    return Xi0+(1-Xi0)/(1+z)**n


def z_from_dLGW(dL_GW_val, H0, Xi0, n=1.91): 
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
