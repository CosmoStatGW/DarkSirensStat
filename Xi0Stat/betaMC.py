#!/usr/bin/env python3

####
# This module contains a class to compute beta from a realistic detection model from MC integration
####

#from beta import Beta
from Xi0Stat.globals import *

class BetaMC:#(Beta):
    
    def __init__(self, priorlimits, nSamples=1000000, observingRun = 'O2', SNRthresh = 8, verbose=False, **kwargs):
        self._observingRun = observingRun
        
        self.verbose = verbose
        self.freq = None
        self.integr = None
        self.nSamples = nSamples
        self.SNRthresh = SNRthresh
        self.zmax = z_from_dLGW(3000, H0=priorlimits.H0max, Xi0=priorlimits.Xi0min, n=nGlob)
        
        self.load_strain_sensitivity()

    def load_strain_sensitivity(self):
        import os
        if self._observingRun == 'O2':
            filename = '2017-08-06_DCH_C02_L1_O2_Sensitivity_strain_asd.txt'
        elif self._observingRun == 'O3':
            filename = '2018-10-20_DELTAL_FE_L1_O3_Sensitivity_strain_asd.txt'
            
       
        filepath = os.path.join(detectorPath, filename)

    #from astropy.cosmology import FlatLambdaCDM
    #cosmoGLADE = FlatLambdaCDM(H0=H0GLADE, Om0=Om0GLADE)  # the values used by GLADE

        if self.verbose:
            print('Loading strain sensitivity from %s...' %filepath)
            
        noise = np.loadtxt(filepath, usecols=range(2))
        f = noise[:,0]
        S = (noise[:,1])**2
        
        # O1 data is very weird at boundaries which extend further than for other files - cut them
        mask = (f > 10) & (f < 6000)
        S = S[mask]
        self.freq = f[mask]
        
        import scipy.integrate as igt
        
        self.integr = igt.cumtrapz(self.freq**(-7/3)/S, self.freq, initial = 0)

    def get_beta(self, H0s, Xi0s, n=nGlob, **kwargs):
      
        H0s = np.atleast_1d(H0s)
        Xi0s = np.atleast_1d(Xi0s)
        
        beta = np.ones((H0s.size, Xi0s.size))
        
        for i in np.arange(H0s.size):
            
            for j in np.arange(Xi0s.size):
                             
                beta[i,j] = self.compute_at( H0=H0s[i], Xi0=Xi0s[j])
                                
        return np.squeeze(beta)
    
    def compute_at(self, H0, Xi0, gamma=1.6, lamb=1, mBHmin=5, mBHmax=40):

#        def sample_discrete(nSamples, probabilities):
#            cdf = np.cumsum(probabilities)
#            cdf = cdf / cdf[-1]
#            return np.searchsorted(cdf, np.random.uniform(size=nSamples))
#

        def sample(nSamples, pdf, lower, upper):
            res = 100000
            x = np.linspace(lower, upper, res)
            cdf = np.cumsum(pdf(x))
            cdf = cdf / cdf[-1]
            return np.interp(np.random.uniform(size=nSamples), cdf, x)
            
        h7 = H0/70
        
#        z_table = np.linspace(0, 3*zmax, 4000)
#        dLgw_table = dLgw(z_table, Xi0, h7)
#        from scipy import interpolate
#        redshFromdLgw = interpolate.interp1d(dLgw_table, z_table, kind='cubic')

        zDist = lambda x: (1+x)**(lamb-1)*dcom70fast(x)**2/H70fast(x)

        z_true = sample(nSamples=self.nSamples, pdf=zDist, lower=0, upper=self.zmax)

        flatMasses = False
        
        if flatMasses:
            
            # distribution of heavier BH mass
            def pm1(m, alpha, mMin, mMax):
                valid = (m < mMax) & (m > mMin)
                return np.where(valid, 1, 0)
            # distributon of lighter BH mass
            def pm2(m, m1, mMin):
                valid = (m < m1) & (m > mMin)
                return np.where(valid, 1/(m1-mMin), 0)
            
            m1 = mBHmin+(mBHmax-mBHmin)*np.random.uniform(size=self.nSamples)
            m2 = mBHmin+(dgw.m1-mBHmin)*np.random.uniform(size=self.nSamples)

        else:
            
            # distribution of heavier BH mass
            def pm1(m, gamma, mMin, mMax):
                valid = (m < mMax) & (m > mMin)
                return np.where(valid, m**(-gamma), 0)
            # distributon of lighter BH mass
            def pm2(m, m1, mMin):
                valid = (m < m1) & (m > mMin)
                return np.where(valid, 1/(m1-mMin), 0)
            
            m1 = sample(self.nSamples, lambda x : pm1(x, gamma, mBHmin, mBHmax), mBHmin, mBHmax)
            m2 = mBHmin+(m1-mBHmin)*np.random.uniform(size=self.nSamples)

      
        mtot = m1 + m2
        Mc = (m1*m2)**(0.6)/mtot**(0.2)
        
        dist_true = Xi(z=z_true, Xi0=Xi0,n=nGlob)*dL70fast(z_true)/h7
        
       
        Mc *= (1+z_true)
        mtot *= (1+z_true)
                  
        ## COMPUTE SNR - all references point to M. Maggiore, Gravitational Waves (OUP) ##
        
        # (4.40), and redshift to obs frame
        fISCO = 2200/mtot
        # this is for the orbit, not the GW - see (4.1 and below). Then,
        fGW = 2*fISCO


        # these angles are related to theta and phi in principle, but we would need to transform to the time-dependent
        # detector frame. We skip this step and assume a uniformized distribution after rotation of the Earth
        phisample = 2*np.pi*np.random.uniform(size=self.nSamples)
        #pdf is sin(theta), cdf is (1-cos(theta))/2, inverse cdf is arccos(1-2*x)
        # ----no, actually only need cos of these angles which are uniform as they should
        costhetasample = 1-2*np.random.uniform(size=self.nSamples)
        cosinclsample = 1-2*np.random.uniform(size=self.nSamples)
        
        cosiota = cosinclsample
        
        # (9.136) note that the additional rotation of psi of u and v definitng the
        # polarization in plane perpendicular to propagagion
        # from matching detector polarization reference and source (aligned to major and minor axis)
        # drops out in sum of squares (7.271) (ok: is also mentioned in a side note)
        Fp = 0.5*(1+costhetasample**2)*np.cos(2*phisample)
        Qp = Fp*0.5*(1+cosinclsample**2)
        Fc = costhetasample*np.sin(2*phisample)
        Qc = Fc*cosinclsample
        # (7.178f)
        Qsq = Qp**2 + Qc**2
        
        GMsun_over_c3 = 4.927e-6# seconds
        clightMpc = clight/3.086e+19 #km/s -> Mpc/s

        
        # don't compute square - dynamic range of terms is already difficult enough
        SNR = 1/np.pi**(2/3)*(GMsun_over_c3*Mc)**(2.5/3)*(clightMpc/dist_true)*np.sqrt(5/6*Qsq*np.interp(fGW, self.freq, self.integr))
        
        ## SELECTION ###############
        mask = SNR > self.SNRthresh
        SNR = SNR[mask]
        if self.verbose:
            if np.sum(mask > 0):
                print("Largest distance detected in sample is " + str(dist_true[mask].max()))
        return SNR.size/self.nSamples

