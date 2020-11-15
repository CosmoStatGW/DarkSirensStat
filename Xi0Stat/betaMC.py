#!/usr/bin/env python3

####
# This module contains a class to compute beta from a realistic detection model from MC integration
####

#from beta import Beta
from globals import *
from keelin import *
from betaHom import *

class BetaMC:#(Beta):
    
    def __init__(self, priorlimits, selector, gals = None, nSamples=40000, observingRun = 'O2', massDist = 'O3', SNRthresh = 8, fluctuatingSNR = True, properAnisotropy = True, verbose=False, **kwargs):
    
        self._gals = gals
        
        self._properAnisotropy = properAnisotropy
        self._observingRun = observingRun
        self._massDist = massDist
        
        self.selector = selector
        
        self.verbose = verbose
        self.freq = None
        self.integr = None
        self.nSamples = nSamples
        self.SNRthresh = SNRthresh
        
        # for setting the scale - if this is chosen around the effective detector horizon comoving volume in Mpc^3, beta is around one
        self.VcomPivot = 4/3*np.pi*(400)**3
        
        # be generous to include extending the limits for filtering
        self.zmax = z_from_dLGW(3000, H0=priorlimits.H0max, Xi0=priorlimits.Xi0min, n=nGlob)
        
        # number of evaluations that will be interpolated
        self.nEvals = 2000
        # wrt these, window length for filtering
        self.nWindow = 101 # must be odd
               
        self.load_strain_sensitivity()
        
        if fluctuatingSNR:
            self._sigmaSNR = 1
        else:
            self._sigmaSNR = 0.001
            
        if massDist == 'O3':
            self.gamma1 = 1.58
            self.gamma2 = 5.59
            self.betaq  = 1.40
            self.mMin   = 3.96
            self.mMax   = 87.14
            self.b      = 0.43
            self.deltam = 4.83
            self.mBreak = self.mMin + self.b*(self.mMax - self.mMin)
        elif massDist == 'O2':
            self.gamma = 1.6
            self.mMin = 5
            self.mMax = 40
        else:
            raise ValueError
            

    def load_strain_sensitivity(self):
        import os
        if self._observingRun == 'O2':
            filename = '2017-08-06_DCH_C02_L1_O2_Sensitivity_strain_asd.txt'
            #filename = '2016-12-13_C01_L1_O2_Sensitivity_strain_asd.txt'
        elif self._observingRun == 'O3':
            filename = '2018-10-20_DELTAL_FE_L1_O3_Sensitivity_strain_asd.txt'
            
       
        filepath = os.path.join(detectorPath, filename)

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
      
    ### prepare grid ###
    
        H0s = np.atleast_1d(H0s)
        Xi0s = np.atleast_1d(Xi0s)
        
        isH0 = False
        if H0s.size > 1:
            outside = self.nWindow/(self.nEvals-2*self.nWindow) * (H0s[-1]-H0s[0])
            isH0 = True
            grid = np.linspace(np.max([5.0, H0s[0]-outside]), H0s[-1]+outside, self.nEvals)
            x = H0s
            betaarg = lambda i : (grid[i], Xi0s[0])
            assert(Xi0s.size==1)
        else:
            outside = self.nWindow/(self.nEvals-2*self.nWindow) * (Xi0s[-1]-Xi0s[0])
            grid = np.linspace(np.max([0.1, Xi0s[0]]), Xi0s[-1]+outside, self.nEvals)
            x = Xi0s
            betaarg = lambda i : (H0s[0], grid[i])
        
    
        res = np.zeros(self.nEvals)
        resCat = res.copy()
        resHom = res.copy()

    
    ### prepare catalog sampling ###

        if self.verbose:
            print('Computing Normalization Beta by MC integration.')
            
        from scipy.special import erf
        
        if self._gals != None:
            if len(self._gals._galcats) != 1:
                print('BetaMC currently only supports a single catalog added if a catalog is used at all, gals != None.')
                raise Error
            # ignore selection, get all
            cat = self._gals._galcats[0]
            dg  = cat.data
            wGal = dg.w.to_numpy()
            # sample true redshift from gal posteriors. This could be repeated for each H0, or we could even sample multiple times for one H0, as a MC strategy to include galaxy redshift errors. Once should be accurate enough given the large number of galaxies (especially in catalogs with large errors) and is quicker
           
            if cat._useDirac == False:
                zGal = sample_bounded_keelin_3(0.16, dg.z_lower, dg.z, dg.z_upper, dg.z_lowerbound, dg.z_upperbound, N=1)
            else: 
                zGal = dg.z
            inside = zGal < self.zmax
            zGal = zGal[inside]
            wGal = wGal[inside]
            thetaGal = dg.theta.to_numpy()[inside]
            phiGal = dg.phi.to_numpy()[inside]
            wGal /= cat._completeness._comovingDensityGoal
            
            # a big advantage of this pre-computation is also that the completnesses need to be evaluated only once for all H0, and due to the MC technique only once per galaxy
            
            if not self._gals._additive:
                
                complGal = cat.completeness(thetaGal, phiGal, zGal, oneZPerAngle = True)
                wGal /= complGal
                wGal *= self._gals.confidence(complGal)

            
            wGal /= self.VcomPivot
           
            for i in range(self.nEvals):
            
                if self.verbose:
                    if i % 30 == 0 :
                        print('#', end='', flush=True)
                        
                ###### estimate maximal range #####
                H0, Xi0 = betaarg(i)
                
                zmax = self._find_zmax_SNRthresh(H0, Xi0)
                
                # We will compute MC integrals sampling from inside this volume up to zmax instead of self.zmax
                # to not waste time with those that can anyway not be seen.
                
                #print("For H0 = " + str(H0) + ", Xi0 = " + str(Xi0) + ", maximum detector horizon of this run is located at z = " + str(zmax))
                
                ###### Distribute number of samples to catalog and to hom term #####
                
                # restrict galaxies to this volume.
                reallyInside = zGal < zmax
                zGalInside = zGal[reallyInside]
                wGalInside = wGal[reallyInside]
                thetaGalInside = thetaGal[reallyInside]
                phiGalInside = phiGal[reallyInside]
 
                # for the catalog term, we sample from the galaxies so restricted
                # compute total galaxy catalog prior mass
                inhomMass = np.sum(wGalInside)
                # after division by nbar (for h7=1), the sum of weights - the integral over the catalog term for all completion schemes - becomes an effective comoving volume (which we have normalized by the pivot volume, which is an overall normalization of beta and which we ignore in the following explanations). In the additive case, this is equal to the volume integral of Pcompl (unless the catalog is overcomplete and Pcompl is cut to 1)
                
                # therefore in the additive case the integral of the hom term 1-Pcompl, that we call homMass, is, Vcom - sum of weights,
                VcomMax = 4/3*np.pi*(dcom70fast(zmax))**3/self.VcomPivot # again H0 = 70 is fixed here (the h7^(-3) that would be here cancel with the h7^3 that is explicit in the hom. term, coming from nbar, which we will not consider either
                
                homMass  = VcomMax - inhomMass

                #print(VcomMax, inhomMass)
                
                # if largely overcomplete, this could happen:
                if homMass < 0:
                    homMass = 0

                # fractions of unity for each term
                inhomMass = inhomMass/(inhomMass+homMass)
                homMass = 1 - inhomMass

                # note that in the case of multiplicative completion it holds only approximately that
                # the the integral of the hom term, Vcom-\int confidence,
                # is Vcom - integral of first term
                # where integral of first term is sum of weights, since galaxy pdfs are weighted by space-dependent function 1/completeness as well as their weights and confidence. It only holds approximatively in as much as weightedgalaxypdfs/completeness is similar to nbar.
                # similarly, overcompleteness and clipping Pcomplete spoils this identity too
                # But: setting inhomMass and homMass *is in any case only a choice* here for importance sampling! Only reason is improved convergence of the MC when more samples are used for the term with larger contribution. We could set them both to 0.5 and would converge to same result.

                # finally choose the number of samples according to the expected relative size of the terms
                nSamplesCat = int(inhomMass*self.nSamples)
                nSamplesHom = self.nSamples - nSamplesCat
                
                # avoid dividing by zero later like so
                if nSamplesHom == 0:
                    nSamplesHom = 1
                elif nSamplesCat == 0:
                    nSamplesCat = 1

                #print(nSamplesHom, nSamplesCat)

                # sample hom term. This means sampling from pdf(x) = 1/VcomMax(zmax) and averaging the results
                costhetasample, phisample, zsample = self.sample_hom_position(nSamplesHom, zmax)
                SNR = self.sample_event(costhetasample, phisample, zsample, H0, Xi0 )
                
#               mask = (SNR > self.SNRthresh)

                # want the integral over gaussian at position SNR with sigma=1, from 8 to infinity
                # = 1 - cdf(8, snr) = 1 - 1/2 [1 + erf((8-snr)/sqrt(2))] = 1/2 [1 + erf((snr-8)/sqrt(2))] = cdf(snr, 8)
                contrib = 0.5*(1+erf((SNR-8)/np.sqrt(2)/self._sigmaSNR))
                
                mask = contrib > 0.01
                contrib = contrib[mask]
                
                # only the samples inside enough are relevant
                thetas = np.arccos( costhetasample[mask] )
                phis = phisample[mask]
                zs = zsample[mask]
                
                # evaluating completeness is probably the slow part
                contrib[ ~ self.selector.is_good(thetas, phis, zs)] = 0
                homweights = 1-self._gals.confidence(cat.completeness(thetas, phis, zs, oneZPerAngle = True))
                # inserting the 1/VcomMax to get a pdf of the sample draws generated a VcomMax here. Also let's not forget that a fraction of the evaluations
                # was zero, and it would be wrong to take np.mean(homweights) instead of np.sum(homweights) / nSamplesHom since homweights only contains the nonzero entries.
                res[i] = VcomMax * np.sum(homweights*contrib) / nSamplesHom
                resHom[i] = res[i]
                
                # sample the gal term. This means sampling from the wGal directly. The normalized pdf contains a 1/np.sum(wGal).
                # Like before, we multiply (keep) and divide (absorbed by sampling) by inhomMass
                
                galsample = self._sample_discrete(nSamplesCat, wGalInside)
                costhetasample = np.cos(thetaGalInside[galsample])
                phisample = phiGalInside[galsample]
                zsample = zGalInside[galsample]
                
                SNR = self.sample_event(costhetasample, phisample, zsample, H0, Xi0 )
               
                #mask = (SNR > self.SNRthresh)
                              
                #res[i] += (np.sum(wGalInside)/nSamplesCat)*np.sum(mask)
                #resCat[i] = (np.sum(wGalInside)/nSamplesCat)*np.sum(mask)
                              
                contrib = 0.5*(1+erf((SNR-8)/np.sqrt(2)/self._sigmaSNR))
                
                
                contrib[ ~ self.selector.is_good(thetaGalInside[galsample], phisample, zsample)] = 0
              
                    
                res[i] += (np.sum(wGalInside)/nSamplesCat)*np.sum(contrib)
                resCat[i] = (np.sum(wGalInside)/nSamplesCat)*np.sum(contrib)
                

        else:
        
            for i in range(self.nEvals):
            
                if self.verbose:
                    if i % 30 == 0 :
                        print('#', end='', flush=True)
                    
                H0, Xi0 = betaarg(i)
            
                zmax = self._find_zmax_SNRthresh(H0, Xi0)
                
                VcomMax = 4/3*np.pi*(dcom70fast(zmax))**3/self.VcomPivot
                
                costhetasample, phisample, zsample = self.sample_hom_position(self.nSamples, zmax)
                SNR = self.sample_event(costhetasample, phisample, zsample, H0, Xi0 )
                
                contrib = 0.5*(1+erf((SNR-8)/np.sqrt(2)/self._sigmaSNR))
                
                
                mask = contrib > 0.01
                contrib = contrib[mask]
                
                # only the samples inside enough are relevant
                thetas = np.arccos( costhetasample[mask] )
                phis = phisample[mask]
                zs = zsample[mask]
       
                
                contrib[ ~ self.selector.is_good(thetas, phis, zs) ] = 0
                #mask = (SNR > self.SNRthresh)
                #res[i] = np.sum(mask)
                res[i] = VcomMax*np.sum(contrib)/self.nSamples
                
        if self.verbose:
            print(' Done. ')
            
        # localized cubic fit
        from scipy.signal import savgol_filter
        resfiltered = savgol_filter(res, self.nWindow, 3, deriv=0)
        
        # estimate sigma from difference of smoothed signal and MC evals:
        
        sigma2 = np.sqrt( np.clip(savgol_filter( (res-resfiltered)**2, self.nWindow, 3, deriv=0), a_min = 0, a_max=None) )
        
        # This is still pretty noisy. Assume sigma is actually proportional to the signal itself. Just need to normalize.
        
        sigma = np.sum(sigma2)/np.sum(resfiltered)*resfiltered
        
        # Fit homogeneous beta
        bhom = BetaHom(400)
        
        if isH0:
            def betahom(H, r, A):
                bhom.dMax = r
                return A*bhom.get_beta(H, Xi0s[0])
        else:
            def betahom(Xi0, r, A):
                bhom.dMax = r
                return A*bhom.get_beta(H0s[0], Xi0)
            
        from scipy.optimize import curve_fit
  
        popt, pcov = curve_fit(betahom, grid, res, sigma=sigma)
        #popt2, pcov2 = curve_fit(betahom, grid, res, sigma=sigma2)
        
        reshomfit = betahom(grid, *popt)
        #reshomfit2 = betahom(grid, *popt2)
        self.chisq = np.sum( (res-reshomfit)**2 / (sigma**2) ) / self.nEvals
        #self.chisq2 = np.sum( (res-reshomfit2)**2 / (sigma2**2) ) / self.nEvals
        self.dMax = popt[0]
        self.dMaxErr = np.sqrt(pcov[0][0])
        #self.dMax2 = popt2[0]
        #self.dMaxErr2 = np.sqrt(pcov2[0][0])
    
        print("Fit assuming it works well: Fitted BetaMC with BetaHom with dMax = {:.2f} +- {:.2f}. Chisq = {:.2f}".format(self.dMax, self.dMaxErr, self.chisq))
        
        #print("Otherwise, if the fit were not to work well, a more general error estimate yields a more reliable Chisq: dMax = {:.2f} +- {:.2f}. Chisq = {:.2f}".format(self.dMax2, self.dMaxErr2, self.chisq2))
      
        # cubic interpolation to get requested values
        from scipy import interpolate
    
        interpolator = interpolate.interp1d(grid, resfiltered, kind='cubic')
        
        #return grid, res, resCat, resHom, sigma, resfiltered, reshomfit, interpolator(x)
        return interpolator(x)




    def _sample(self, nSamples, pdf, lower, upper):
        res = 100000
        x = np.linspace(lower, upper, res)
        cdf = np.cumsum(pdf(x))
        cdf = cdf / cdf[-1]
        return np.interp(np.random.uniform(size=nSamples), cdf, x)
               
    # we can still vectorize sampling if for every sample there is another upper bound,
    # by not inserting random variables all up to 1 into the inverse CDF but only up to CDF(upper)
    def _sample_vector_upper(self, pdf, lower, upper):
        nSamples = len(upper)
        res = 100000
        x = np.linspace(lower, upper.max(), res)
        cdf = np.cumsum(pdf(x))
        cdf = cdf / cdf[-1]
        probTilUpper = np.interp(upper, x, cdf)
        return np.interp(probTilUpper*np.random.uniform(size=nSamples), cdf, x)
        
    def _sample_discrete(self, nSamples, probabilities):
        cdf = np.cumsum(probabilities)
        cdf = cdf / cdf[-1]
        return np.searchsorted(cdf, np.random.uniform(size=nSamples))
            
    def sample_hom_position(self, nSamples, zmax):
        lamb=1
        zDist = lambda x: (1+x)**(lamb-1)*dcom70fast(x)**2/H70fast(x)
        zsample = self._sample(nSamples=nSamples, pdf=zDist, lower=0, upper=zmax)
        
        phisample = 2*np.pi*np.random.uniform(size=nSamples)
        #pdf is sin(theta), cdf is (1-cos(theta))/2, inverse cdf is arccos(1-2*x)
        # ----no, actually only need cos of these angles which are uniform as they should
        #TEST costhetasample = 0.1 -0.2*np.random.uniform(size=self.nSamples)
        costhetasample = 1-2*np.random.uniform(size=nSamples)
    
        return costhetasample, phisample, zsample
        
    # takes positions in equat. coords and adds the sampling of masses, inclination and computes SNR
    # the dependence on H0 and Xi0 is due to the dependence of detector reach dLGW expressed in z, which is passed here
    # When costhetasample==None and phisample==None, the events are located with best inclination and direction
    def sample_event(self, costhetasample, phisample, zsample, H0, Xi0):
    
        nSamples = zsample.size
        
        h7 = H0/70
               
#        flatMasses = False
#
#        if flatMasses:
#
#           # distribution of heavier BH mass
#           def pm1(m, alpha, mMin, mMax):
#               valid = (m < mMax) & (m > mMin)
#               return np.where(valid, 1, 0)
#           # distributon of lighter BH mass
#           def pm2(m, m1, mMin):
#               valid = (m < m1) & (m > mMin)
#               return np.where(valid, 1/(m1-mMin), 0)
#
#           m1 = mBHmin+(mBHmax-mBHmin)*np.random.uniform(size=nSamples)
#           m2 = mBHmin+(dgw.m1-mBHmin)*np.random.uniform(size=nSamples)

        if self._massDist=='O2':
           
            # distribution of heavier BH mass
            def pm1(m):
                valid = (m < self.mMax) & (m > self.mMin)
                return np.where(valid, m**(-self.gamma), 0)
#            # distributon of lighter BH mass
#            def pm2(m, m1):
#                valid = (m < m1) & (m > self.mMin)
#                return np.where(valid, 1/(m1-self.mMin), 0)
#
            m1 = self._sample(nSamples, lambda x : pm1(x), self.mMin, self.mMax)
            m2 = self.mMin+(m1-self.mMin)*np.random.uniform(size=nSamples)

        elif self._massDist=='O3':
            # distribution of heavier BH mass
            def S(x):
                maskL = x <= self.mMin + 1e-2
                maskU = x >= (self.mMin + self.deltam) - 1e-2
                s = np.empty_like(x)
                s[maskL] = 0
                s[maskU] = 1
                maskM = ~(maskL | maskU)
                s[maskM] = 1/(np.exp(self.deltam/(x[maskM]-self.mMin) + self.deltam/(x[maskM]-self.mMin - self.deltam))+1)
                return s
            
            def pm1(m):
                valid = (m < self.mMax) & (m > self.mMin)
                return np.where(valid, np.where(m < self.mBreak, m**(-self.gamma1)*S(m), self.mBreak**(-self.gamma1+self.gamma2)*m**(-self.gamma2)), 0)
            # distributon of lighter BH mass
            def pm2(m):
                valid = m > self.mMin
                return np.where(valid, m**(-self.betaq)*S(m), 0)
                
            m1 = self._sample(nSamples, pm1, self.mMin, self.mMax)
            m2 = self._sample_vector_upper(pm2, self.mMin, m1)
            
        # never seeems to happen, but let's be sure
        m2[m2>m1] = m1[m2>m1]
            
        mtot = m1 + m2
        Mc = (m1*m2)**(0.6)/mtot**(0.2)
       
        dist_true = Xi(z=zsample, Xi0=Xi0,n=nGlob)*dL70fast(zsample)/h7
       
      
        Mc *= (1+zsample)
        mtot *= (1+zsample)
                 
        ## COMPUTE SNR - all references point to M. Maggiore, Gravitational Waves (OUP) ##
       
        # (4.40), and redshift to obs frame
        fISCO = 2200/mtot
        # this is for the orbit, not the GW - see (4.1 and below). Then,
        fGW = 2*fISCO

        if costhetasample is not None:
            tsample = np.random.uniform(size=nSamples)
           
            costhetaL, phiL = self._equat2detector('livingston', costhetasample, phisample, tsample)
            costhetaH, phiH = self._equat2detector('hanford', costhetasample, phisample, tsample)
           
            cosinclsample = 1-2*np.random.uniform(size=nSamples)
           
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
        
        else:
            Qsq = 1
       
        GMsun_over_c3 = 4.927e-6# seconds
        clightMpc = clight/3.086e+19 #km/s -> Mpc/s

       
        # don't compute square - dynamic range of terms is already difficult enough
        SNR = 1/np.pi**(2/3)*(GMsun_over_c3*Mc)**(2.5/3)*(clightMpc/dist_true)*np.sqrt(5/6*Qsq*np.interp(fGW, self.freq, self.integr))
       
        return SNR
         


    # searches for the redshift after which no more events will be detected, depending on H0 and Xi0.
    # search is carried out until self.zmax
    def _find_zmax_SNRthresh(self, H0, Xi0):
        zgrid = np.linspace(self.zmax, 0.00001, 100000)
        # asssumes perfect orientation and inclination, randomizes masses as usual (not clear if heaviest are always the best visible since
        # their cutoff frequency is lower!)
        snrgrid = self.sample_event(None, None, zgrid, H0, Xi0)
        # find max z after which no more SNR is ever larger than threshold, by finding first True value (argmax returns first maximum) starting from large z
        idx = np.argmax(snrgrid >= self.SNRthresh)
        # fudge factor to be generous, tests show the accuracy is about 1 or 2 percent so we increase by 5%
        return 1.05*zgrid[idx]
        
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
            
            costhetanew = np.clip(res[:, 2], a_min=-0.99999, a_max=0.99999)
            sinthetanew = np.sqrt(1-costhetanew**2)
            xnew = res[:, 0]
            ynew = res[:, 1]
        
            cosphinew = np.clip(xnew/sinthetanew, a_min=-1, a_max=1)
            phinew = np.arccos(cosphinew)

            return costhetanew, phinew
            
           
          
