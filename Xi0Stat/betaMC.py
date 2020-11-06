#!/usr/bin/env python3

####
# This module contains a class to compute beta from a realistic detection model from MC integration
####

#from beta import Beta
from globals import *

class BetaMC:#(Beta):
    
    def __init__(self, priorlimits, nSamples=1000000, 
                 observingRun = 'O2', SNRthresh = 8, 
                 properAnisotropy = True, selectionFunc = None, 
                 verbose=False, **kwargs):
    
        self._properAnisotropy = properAnisotropy
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
            #filename = '2016-12-13_C01_L1_O2_Sensitivity_strain_asd.txt'
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
        
        nEvals = 100
        
        if H0s.size > 1:
            grid = np.linspace(H0s[0]*0.9, H0s[-1]*1.1, nEvals)
            x = H0s
            betaarg = lambda i : grid[i], Xi0s[0]
            assert(Xi0s.size==1)
        else:
            grid = np.linspace(Xi0s[0]*0.9, Xi0s[-1]*1.1, nEvals)
            x = Xi0s
            betaarg = lambda i : H0s[0], grid[i]
        
        
        res = np.zeros(nEvals)
        for i in range(nEvals):
            res[i] = self.compute_at( *betaarg(i) )
        
        # localized cubic fit
        from scipy.signal import savgol_filter
        resfiltered = savgol_filter(res, 23, 3, deriv=0)
        
        # cubic interpolation to get requested values
        from scipy import interpolate
       
    
        interpolator = interpolate.interp1d(grid, resfiltered, kind='cubic')
        
        return interpolator(x)
        
#
#        beta = np.ones((H0s.size, Xi0s.size))
#
#        for i in np.arange(H0s.size):
#
#            for j in np.arange(Xi0s.size):
#
#                beta[i,j] = self.compute_at( H0=H0s[i], Xi0=Xi0s[j])
#
#        return np.squeeze(beta)
    
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
        #TEST costhetasample = 0.1 -0.2*np.random.uniform(size=self.nSamples)
        costhetasample = 1-2*np.random.uniform(size=self.nSamples)
        tsample = np.random.uniform(size=self.nSamples)
        
        costhetaL, phiL = self._equat2detector('livingston', costhetasample, phisample, tsample)
        costhetaH, phiH = self._equat2detector('hanford', costhetasample, phisample, tsample)
        
        
        cosinclsample = 1-2*np.random.uniform(size=self.nSamples)
        
        cosiota = cosinclsample
        
        # (9.136) note that the additional rotation of psi of u and v definitng the
        # polarization in plane perpendicular to propagagion
        # from matching detector polarization reference and source (aligned to major and minor axis)
        # drops out in sum of squares (7.271) (ok: is also mentioned in a side note)
        
        def Qsq(costh, phi, cosincl):
        
            Fp = 0.5*(1+costh**2)*np.cos(2*phi)
            Qp = Fp*0.5*(1+cosincl**2)
            Fc = costh*np.sin(2*phi)
            Qc = Fc*cosincl
            # (7.178f)
            return Qp**2 + Qc**2
            
        QsqL = Qsq(costhetaL, phiL, cosinclsample)
        QsqH = Qsq(costhetaH, phiH, cosinclsample)
        
        Qsq = np.minimum(QsqL, QsqH)
        
        GMsun_over_c3 = 4.927e-6# seconds
        clightMpc = clight/3.086e+19 #km/s -> Mpc/s

        
        # don't compute square - dynamic range of terms is already difficult enough
        SNR = 1/np.pi**(2/3)*(GMsun_over_c3*Mc)**(2.5/3)*(clightMpc/dist_true)*np.sqrt(5/6*Qsq*np.interp(fGW, self.freq, self.integr))
        
        ## SELECTION ###############
       
        mask = (SNR > self.SNRthresh) #& (costhetasample < 0.1) & (costhetasample > -0.1)
        
        ##### ADD MORE SELECTION HERE
        #mask = mask & (z_from_dLGW_fast(dist_true, H0=70, Xi0=1, n=nGlob) < 0.2)
        
        SNR = SNR[mask]
        if self.verbose:
            if np.sum(mask > 0):
                print("Largest distance detected in sample is " + str(dist_true[mask].max()))
        return SNR.size/self.nSamples



    def _equat2detector(self, detector, costheta, phi, t):
        
        if self._properAnisotropy == False:
        
            return costheta, phi
        
        else:
        
            trafos = np.zeros((t.size,3,3))
            
            if detector == 'livingston':
            
                c = np.cos(-0.0135118 + 2*np.pi*t)
                s = -np.sin(-0.0135118 + 2*np.pi*t)
                
                trafos[:,0,0] = -0.15713*c +  0.951057*s
                trafos[:,0,1] =  0.951057*c + 0.15713*s
                trafos[:,0,2] = -0.266086
                
                trafos[:,1,0] = -0.483595*c - 0.309017*s
                trafos[:,1,1] = -0.309017*c + 0.483595*s
                trafos[:,1,2] = -0.818929
                
                trafos[:,2,0] = -0.861073*c
                trafos[:,2,1] =  0.861073*s
                trafos[:,2,2] =  0.508482
                
            elif detector == 'hanford':
            
                c = np.cos(0.513263 + 2*np.pi*t)
                s = -np.sin(0.513263 + 2*np.pi*t)
                
                trafos[:,0,0] = 0.586405*c + 0.587785*s
                trafos[:,0,1] = 0.587785*c - 0.586405*s
                trafos[:,0,2] = 0.557348
                
                trafos[:,1,0] = -0.426048*c + 0.809017*s
                trafos[:,1,1] = 0.809017*c + 0.426048*s
                trafos[:,1,2] = -0.404937
                
                trafos[:,2,0] = -0.688921*c
                trafos[:,2,1] =  0.688921*s
                trafos[:,2,2] =  0.724837
     
            sintheta = np.sqrt(1-costheta**2)
            x = np.cos(phi)*sintheta
            y = np.sin(phi)*sintheta
            z = costheta
            
            dir = (np.vstack([x,y,z]).T)[:,np.newaxis,:]
            
            res = np.sum(dir*trafos, axis=2)
            
            costhetanew = np.clip(res[:, 2], a_min=0, a_max=1)
            sinthetanew = np.sqrt(1-costhetanew**2)
            xnew = res[:, 0]
            ynew = res[:, 1]
        
            cosphinew = np.clip(xnew/sinthetanew, a_min=0, a_max=1)
            phinew = np.arccos(cosphinew)

            return costhetanew, phinew
            
           
          
