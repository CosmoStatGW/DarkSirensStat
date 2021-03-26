#
#    Copyright (c) 2021 Andreas Finke <andreas.finke@unige.ch>
#
#    All rights reserved. Use of this source code is governed by a modified BSD
#    license that can be found in the LICENSE file.



####
# This module contains a class to compute beta by MC sampling the population model
# and modelling the detection using the inspiral SNR and further selection.
####

#from beta import Beta
from globals import *
from keelin import *
from betaHom import *

from SNRtools import oSNR

class BetaMC:#(Beta):
    
    def __init__(self, priorlimits, selector,
                 gals = None, 
                 nSamples=40000, 
                 observingRun = 'O2', 
                 massDist = 'O3', 
                 SNRthresh = 8, 
                 fluctuatingSNR = True, 
                 properAnisotropy = True, 
                 verbose=False, 
                 lamb=1,
                 alpha1=1.6,
                 fullSNR=True,
                 approximant='IMRPhenomXHM',
                 **kwargs):
    
        
        self.fullSNR=fullSNR
        self.lamb=lamb
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
        
        # how lucky are the furthest away sources we want to cover: when SNRsigmas*_sigmaSNR is added to the theoretical SNR. Default: 2sigma. Degrades MC performance as larger volumes are needed with less chance of seeing the events
        
        self.SNRsigmas = 2
        
        print('Setting mass distribution to %s' %massDist)
        self.fit_hom=True
        if massDist == 'O3':
            self.gamma1 = alpha1 #1.58
            self.gamma2 = 5.59
            self.betaq  = 1.40
            self.mMin   = 3.96
            self.mMax   = 87.14
            self.b      = 0.43
            self.deltam = 4.83
            self.mBreak = self.mMin + self.b*(self.mMax - self.mMin)
            print('Parameters: gamma1=%s, gamma2=%s, betaq=%s, mMin=%s, mMax=%s, b=%s, deltam=%s, mBreak=%s' %(self.gamma1, self.gamma2, self.betaq, self.mMin, self.mMax, self.b, self.deltam, self.mBreak))
        elif massDist == 'O2':
            self.gamma = 1.6
            self.mMin = pow_law_Mmin
            self.mMax = pow_law_Mmax
            print('Parameters: gamma=%s, mMin=%s, mMax=%s' %(self.gamma, self.mMin, self.mMax))
        elif massDist == 'NS-gauss':
            self.mNSmean = BNS_gauss_mu
            self.mNSsigma = BNS_gauss_sigma
            self.mMin = self.mNSmean-7*self.mNSsigma
            self.mMax = self.mNSmean+7*self.mNSsigma
            #self.fit_hom=False
            print('Parameters: mu=%s, sigma=%s, sampling between mMin=%s, mMax=%s' %(self.mNSmean, self.mNSsigma, self.mMin, self.mMax))
        elif massDist == 'NS-flat':
            self.mMin = BNS_flat_Mmin
            self.mMax = BNS_flat_Mmax
            self.DeltaM = self.mMax-self.mMin
            #self.fit_hom=False
            print('Paramters: mMin=%s, mMax=%s' %(self.mMin, self.mMax))
        else:
            raise ValueError
        
        print('Redshift dependence has parameter lambda=%s' %self.lamb)    
        
        # COnstrunctor for full SNR
        if fullSNR:
            
            self.mySNRs={}
            for detectorname in ["L", "H"]:
       
                filepath = os.path.join(detectorPath, self.filenames[detectorname])
                myoSNR = oSNR(  filepath , verbose=True, approximant=approximant)
                myoSNR.make_interpolator()
            
                self.mySNRs[detectorname] =  myoSNR
             
            
        
        
        from scipy.optimize import minimize_scalar

        # m is the detector frame mass of each of the BH
        def goal(m):
            # for fixed total mass (that should be as small as possible, entering the integral cutoff in inverse)
            # one can analytically optimize the mass ratio for maximizing the chirp mass, and finds 1
            # thus the masses are equal in the optimal case, in which the angles are such that Qsq = 1
            # furthermore, for z fixed, dL(z) is fixed anyway, and the SNR expression is maximal if the mass-dependent rest is maximal.
            # the redshift enters into the rest in redshifting the chirp mass and the total mass only, and the masses enter only where the redshift enters - the rest is a  function of redshifted, detector frame, masses only. If the optimal source frame mass is known for a particular redshift, the rest is maxized by definition. For any other redshift, we only need to absorb this change into a change of (intrinsic) mass to keep the detector frame masses constant, which is possible as chirp mass and total mass scale the same when the mass is scaled.
            # Thus, we need to find the best mass for any fixed H0, Xi0 (which do not enter in the rest) and fixed z only once. For example we can consider z->0 where dL->z/H0 and maximize rest=SNR*dL \approx SNR*z/H0 (which remains well-defined) to find this optimal detector frame mass directly; numerically, we use a finite but small z but express SNR using the detector frame mass for increased precision.
            # Furthermore, the optimal SNRmax is then simply given by dividing this rest by whatever dL in question, and finding the maximum redshift is simply solving SNR=rest/dL(z)=SNRthresh for z, z=z(dL)=z(rest/SNRthresh). The quantity dL=rest/SNRthresh is clearly the maximum luminosity distance of the detector, self.dMax defined below. All what was said is summarized in saying that this distance is independent of H0, Xi0.

            # Caveat: This works only if the redshift is not so large to break through the lower mass bound. For Ligo/Virgo, this is z=O(10) though. Otherwise it maximum is probably obtained at the boundary but this would require some checks.
            zPiv = 0.001
            
            # assume perfect orientation for both but take min: this means that we look at sources perfectly oriented for the weaker detector, which is the limiting criterion in our detection decision elsewhere
            # (minimize -min is indeed maximize min, -> maximize the weaker one)
            
            return -self._SNR(m/(1+zPiv), m/(1+zPiv), zPiv, H0=70, Xi0=1, QsqL=1, QsqH=1)*dL70fast(zPiv) #Independent of zPiv, H0, Xi0
            
        res = minimize_scalar(goal, bounds = (self.mMin, self.mMax), method='bounded')
        
        if res.success == False:
            print(res.message, res.x)
            raise
        else:
            self.optimalDetectorFrameMass = res.x
            if self.verbose == True:
                print("Optimal detector frame mass for {} is {:.2f}".format(observingRun, res.x))
            # what was called "rest" in the comment above
            self.SNRmaxNumerator = -res.fun
            
            #
            self.dMax = self.SNRmaxNumerator/self.SNRthresh
            self.dMaxReal = (self.SNRmaxNumerator)/(self.SNRthresh-self.SNRsigmas*self._sigmaSNR)
            
            if self.verbose:
                print("Maximal (theoretical) detector reach for {} is {:.1f} Mpc, corresponding to z = {:.2}, ({:.2} - {:.2})".format(observingRun, self.dMax, z_from_dLGW_fast(self.dMax, H0=70, Xi0=1, n=nGlob),  z_from_dLGW_fast(self.dMax, H0=priorlimits.H0min, Xi0=priorlimits.Xi0max, n=nGlob),  z_from_dLGW_fast(self.dMax, H0=priorlimits.H0max, Xi0=priorlimits.Xi0min, n=nGlob)))
                
                if fluctuatingSNR:
            
                    print("Maximal (theoretical) detector reach for {} is {:.1f} Mpc, corresponding to z = {:.2}, ({:.2} - {:.2})".format(observingRun, self.dMaxReal, z_from_dLGW_fast(self.dMaxReal, H0=70, Xi0=1, n=nGlob),  z_from_dLGW_fast(self.dMaxReal, H0=priorlimits.H0min, Xi0=priorlimits.Xi0max, n=nGlob),  z_from_dLGW_fast(self.dMaxReal, H0=priorlimits.H0max, Xi0=priorlimits.Xi0min, n=nGlob)))
                
          

    def load_strain_sensitivity(self):
        import os
        
        self.filenames = {}
        self.integr = {}
        self.freq = {}
        
        if self._observingRun == 'O2':
            self.filenames["L"] = '2017-08-06_DCH_C02_L1_O2_Sensitivity_strain_asd.txt'
            self.filenames["H"] = '2017-06-10_DCH_C02_H1_O2_Sensitivity_strain_asd.txt'
        elif self._observingRun == 'O3':
            self.filenames["L"] = 'O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt'
            self.filenames["H"] = 'O3-H1-C01_CLEAN_SUB60HZ-1251752040.0_sensitivity_strain_asd.txt'

        for detectorname in ["L", "H"]:
       
            filepath = os.path.join(detectorPath, self.filenames[detectorname])

            if self.verbose:
                print('Loading strain sensitivity from %s...' %filepath)
                
            noise = np.loadtxt(filepath, usecols=range(2))
            f = noise[:,0]
            S = (noise[:,1])**2
            
            # O1 data is very weird at boundaries which extend further than for other files - cut them
            mask = (f > 10) & (f < 6000)
            S = S[mask]
            self.freq[detectorname] = f[mask]
            
            import scipy.integrate as igt
            
            self.integr[detectorname] = igt.cumtrapz(self.freq[detectorname]**(-7/3)/S, self.freq[detectorname], initial = 0)

    
    
    
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
            #if 'NS' in self._massDist:
            #    self.fit_hom=False
        
    
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
                print(zGal)
            else: 
                zGal = dg.z.to_numpy()
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
               
                #return
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
                
                mask = contrib > 0.001
                contrib = contrib[mask]
                
                # only continue computing for those that are not anyway nearly irrelevant. Note that nSamplesHom is staying large!
                thetas = np.arccos( costhetasample[mask] )
                phis = phisample[mask]
                zs = zsample[mask]
                
                # evaluating completeness is probably the slow part, in the following two lines
                contrib[ ~ self.selector.is_good(thetas, phis, zs)] = 0
                homweights = 1-self._gals.confidence(cat.completeness(thetas, phis, zs, oneZPerAngle = True))
                # inserting the 1/VcomMax to get a pdf of the sample draws generated a VcomMax here. Also let's not forget that a fraction of the evaluations
                # was zero, and it would be wrong to take np.mean(homweights) instead of np.sum(homweights) / nSamplesHom since homweights only contains the nonzero entries.
                res[i] = VcomMax * np.sum(homweights*contrib) / nSamplesHom
                resHom[i] = res[i]
                
                # sample the gal term. This means sampling from the wGal directly. The normalized pdf contains a 1/np.sum(wGal).
                # Like before, we multiply (keep) and divide (absorbed by sampling) by this sum
                # However, we can even include the selection from selector into the proposal distribution first, by changing the weights accordingly.
                
                wGalInside[ ~ self.selector.is_good(thetaGalInside, phiGalInside, zGalInside)] = 0
                
                
                # add the lambda dependent GW prior weighting, put on top of the galaxy distribution (also use this a few lines below in the normalization sum!)
                wGalInside *= (1+zGalInside)**(self.lamb-1)
                
                
                galsample = self._sample_discrete(nSamplesCat, wGalInside)
                
                costhetasample = np.cos(thetaGalInside[galsample])
                phisample = phiGalInside[galsample]
                zsample = zGalInside[galsample]
                
                SNR = self.sample_event(costhetasample, phisample, zsample, H0, Xi0 )
               
                #mask = (SNR > self.SNRthresh)
                              
                #res[i] += (np.sum(wGalInside)/nSamplesCat)*np.sum(mask)
                #resCat[i] = (np.sum(wGalInside)/nSamplesCat)*np.sum(mask)
                              
                contrib = 0.5*(1+erf((SNR-8)/np.sqrt(2)/self._sigmaSNR))
                
                #not necessary anymore, included in weights.
                #contrib[ ~ self.selector.is_good(thetaGalInside[galsample], phisample, zsample)] = 0
              
                    
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
                
                
                mask = contrib > 0.001
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
        
        
        
        if self.fit_hom:
            
            # estimate sigma from difference of smoothed signal and MC evals:
            sigma2 = np.sqrt( np.clip(savgol_filter( (res-resfiltered)**2, self.nWindow, 3, deriv=0), a_min = 0, a_max=None) )
        
            # This is still pretty noisy. Assume sigma is actually proportional to the signal itself. Just need to normalize.
            sigma = np.sum(sigma2)/np.sum(resfiltered)*resfiltered
           
            print('Fitting with BetaHom...')
        
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
            self.dMaxEff = popt[0]
            self.dMaxEffErr = np.sqrt(pcov[0][0])
        #self.dMax2 = popt2[0]
        #self.dMaxErr2 = np.sqrt(pcov2[0][0])
    
            print("Fit assuming it works well: Fitted BetaMC with BetaHom with dMax = {:.2f} +- {:.2f}. Chisq = {:.2f}".format(self.dMaxEff, self.dMaxEffErr, self.chisq))
        
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
            
    def sample_hom_position(self, nSamples, zmax, ):
        #lamb=1
        zDist = lambda x: (1+x)**(self.lamb-1)*dcom70fast(x)**2/H70fast(x)
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
                return np.where(valid, m**(self.betaq)*S(m), 0)
                
            m1 = self._sample(nSamples, pm1, self.mMin, self.mMax)
            m2 = self._sample_vector_upper(pm2, self.mMin, m1)
            
        if self._massDist=='NS-gauss':
       
            m1a = np.random.normal(loc=self.mNSmean, scale=self.mNSsigma, size=nSamples)
            m2a = np.random.normal(loc=self.mNSmean, scale=self.mNSsigma, size=nSamples)
            m1 = np.where(m1a>m2a, m1a, m2a)
            m2 = np.where(m1a<=m2a, m1a, m2a)
        
        if self._massDist=='NS-flat': 
            
            m1a = np.random.uniform(low=self.mMin, high=self.mMax, size=nSamples)
            m2a = np.random.uniform(low=self.mMin, high=self.mMax, size=nSamples)
            m1 = np.where(m1a>m2a, m1a, m2a)
            m2 = np.where(m1a<=m2a, m1a, m2a)
            
            
        # never seems to happen, but let's be sure (not that it would matter...)
        m2[m2>m1] = m1[m2>m1]
      
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
       
        return self._SNR(m1, m2, zsample, H0, Xi0, QsqL, QsqH)
         
    
    def _SNR(self, m1, m2, z, H0, Xi0, QsqL, QsqH):
        
        m1=np.asarray(m1)
        m2=np.asarray(m2)
        #z=np.asarray(z)
        #QsqL=np.asarray(QsqL)
        #QsqH=np.asarray(QsqH)
        
        if self.fullSNR:
            # call interpolator
                        
            h7 = H0/70
            # dist should be in Gpc for input of oSNR
            dist_true = Xi(z=z, Xi0=Xi0,n=nGlob)*dL70fast(z)/h7*1e-3
            
            m1det = m1*(1+z)
            m2det = m2*(1+z)
            
            # Throw away masses that are very large
            keep = (m1det<900) & (m2det<900)
            
            if not np.isscalar(QsqL):
                QsqL = QsqL[keep]
            if not np.isscalar(QsqH):
                QsqH = QsqH[keep]
            if not np.isscalar(z):
                z = z[keep]
                dist_true=dist_true[keep]
            
            SNR_L = np.zeros(m1.shape)
            SNR_H = np.zeros(m1.shape)
            
            SNR_L[keep] = self.mySNRs["L"].get_oSNR(m1det[keep], m2det[keep], dist_true)*np.sqrt(QsqL)
            SNR_H[keep] = self.mySNRs["H"].get_oSNR(m1det[keep], m2det[keep], dist_true)*np.sqrt(QsqH)
            
            return np.minimum(SNR_L, SNR_H)
            
        else:
            return self._SNR1stOrder(m1, m2, z, H0, Xi0, QsqL, QsqH)


    def _SNR1stOrder(self, m1, m2, z, H0, Xi0, QsqL, QsqH):
        mtot = m1 + m2
        Mc = (m1*m2)**(0.6)/mtot**(0.2)
        h7 = H0/70
        dist_true = Xi(z=z, Xi0=Xi0,n=nGlob)*dL70fast(z)/h7
       
        Mc *= (1+z)
        mtot *= (1+z)
                 
        ## COMPUTE SNR - all references point to M. Maggiore, Gravitational Waves (OUP) ##
       
        # (4.40), and redshift to obs frame
        fISCO = 2200/mtot
        # this is for the orbit, not the GW - see (4.1 and below). Then,
        fGW = 2*fISCO

        GMsun_over_c3 = 4.927e-6# seconds
        clightMpc = clight/3.086e+19 #km/s -> Mpc/s
       
        # don't compute square - dynamic range of terms is already difficult enough
        
        fac = np.sqrt(5/6)/np.pi**(2/3)*(GMsun_over_c3*Mc)**(2.5/3)*(clightMpc/dist_true)
        
        SNR_L = fac * np.sqrt(QsqL*np.interp(fGW, self.freq["L"], self.integr["L"]))
        SNR_H = fac * np.sqrt(QsqH*np.interp(fGW, self.freq["H"], self.integr["H"]))
       
        return np.minimum(SNR_L, SNR_H)
        
    # searches for the redshift after which no more events will be detected, depending on H0 and Xi0.
    # search is carried out until self.zmax
    def _find_zmax_SNRthresh(self, H0, Xi0):

        res = z_from_dLGW_fast(self.dMaxReal, H0=H0, Xi0=Xi0, n=nGlob)
        
        # this should normally be true - the optimal detector frame mass is about 40, while mMin is about 4...
        if self.optimalDetectorFrameMass/(1+res) >= self.mMin:
            return res
            
        # else, solve afresh with bounds
        from scipy.optimize import minimize_scalar

        def SNRmax(z):
            def goal(m):
                # assume perfect orientation for both but take min: this means that we look at sources perfectly oriented for the weaker detector, which is the limiting criterion in our detection decision elsewhere
                return -self._SNR(m, m, z, H0, Xi0, QsqL=1, QsqH=1)
            # (minimize -min is indeed maximize min, so maximize the weaker one)
            res = minimize_scalar(goal, bounds = (self.mMin, self.mMax), method='bounded')
            if res.success == False:
                print(res.message, res.x)
            else:
                return self._SNR(res.x, res.x, z, H0, Xi0, QsqL=1, QsqH=1) + self.SNRsigmas*self._sigmaSNR - self.SNRthresh

        from scipy.optimize import brentq
        res2 = brentq(SNRmax, a=0.001, b=self.zmax)
        
        return res2
        
 # old randomized algorithm (using that sample_event used to set Qsq = 1 for angles being None)
#        zl = 0.00001
#        zr = self.zmax
#        fudge = 0.1
#        for i in range(20):
#            zgrid = np.linspace(zr, zl, 10000)
#            # asssumes perfect orientation and inclination, randomizes masses as usual (not clear if heaviest are always the best visible since
#            # their cutoff frequency is lower!)
#            snrgrid = self.sample_event(None, None, zgrid, H0, Xi0)
#            # find max z after which no more SNR is ever larger than threshold, by finding first True value (argmax returns first maximum) starting from large z
#            idx = np.argmax(snrgrid >= self.SNRthresh)
#
#            if idx != 0:
#                print(idx)
#                # this z is for sure still visible
#                zl = zgrid[idx]
#            else:
#                pass
#
#            # close in from the right
#            zr = zl*(1+fudge)
#            fudge *= 0.8
#
#            print(i, zl, zr)
#        return zl
#
# as before, without the loop: fast but approximate
#        zgrid = np.linspace(zgrid[idx]*1.03, zgrid[idx]*0.97, 10000)
#
#        snrgrid = self.sample_event(None, None, zgrid, H0, Xi0)
#        idx = np.argmax(snrgrid >= self.SNRthresh)
#        # fudge factor to be generous, tests show the accuracy is about 1 or 2 percent so we increase by 5%
#        return 1.005*zgrid[idx]
        
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
            
           
          
