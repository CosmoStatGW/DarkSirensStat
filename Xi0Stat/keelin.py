import numpy as np

import matplotlib.pyplot as plt

# Symetric percentile triplet bounded Keelin (logit metalog) distribution pdf
# [Keelin, Thomas W. "The metalog distributions." Decision Analysis 13.4 (2016): 243-277]
# Keelin distributions can approximate any continuous analytic pdf. They are based on a truncated series expansion.
# The coefficients are fitted directly from the data from demanding quantiles to come out right.
# The more terms are used, the more quantiles need to be given.
#
# A python implementation of the general case is available here: https://github.com/tjefferies/pymetalog
# It will detect unfeasible input and find the closest feasible Keelin distribution for a given number of terms
# by solving a linear program. [Tom Keelin, private communication] A blog post is available here
# https://www.linkedin.com/pulse/traditional-continuous-probability-distributions-soon-rob-treichler/
#
# Here, we present a high performance implementation of a special 3-term case that is explicit
# (no linear system solution needed) and particularly convenient.
# It builds the pdf based on the quantile triplet (aquant_l, median, aquant_u), the positions
# where the CDF is equal to the probabilites (a, 0.5, 1-a) (l stands for lower, u for upper).
#
# Furthermore, the r.v. that is described by the pdf is bounded between lower and upper
# To approximate semibounded pdfs, choose e.g. lower = 0, upper = median + 3 sigma
#
# P is the total probability that is covered by the returned pdf (which excludes the boundary points)
# (if a tail is thin enough, no values in it will be returned for sufficiently small P < 1)
#
# P = 1 is ill-defined. So is lower = aquant_l, and other pathologies that cannot be represented.
# No checks are done
#
# Returns: function returns 2 vectors of length N.
#     The first is a generally nonuniform grid x
#     The second is pdf(x)
#     The x are spaced such that equal probability is contained in between them
#
#
# Vectorization capabilities:
# If (aquant_l, median, aquant_u) are vectors of the same length (and potentially upper and lower),
# the function returns instead two matrices with its rows being the results above for each triplet / quintet

def bounded_keelin_3(a, aquant_l, median, aquant_u, lower, upper, N=100, P=.9999, computePDF = True):

    # Treat scalar case as special case of vector input

    aquant_l = np.atleast_1d(aquant_l)
    aquant_u = np.atleast_1d(aquant_u)
    median = np.atleast_1d(median)
    lower = np.atleast_1d(lower)
    upper = np.atleast_1d(upper)

    # Some definitions

    gl = (aquant_l-lower)/(upper-aquant_l)
    gu = (aquant_u-lower)/(upper-aquant_u)
    gm = (median-lower)/(upper-median)

    # In this special case the coefficients can be computed analytically
    a1 = np.log(gm)
    foo = np.log((1-a)/a)
    a2 = 0.5/foo*np.log(gu/gl)
    a3 = np.log(gu*gl/gm**2)/((1-2*a)*foo)

    # Y is the cumulative probability. Explore a fraction P of the distribution
    Y = np.linspace(0.5*(1-P), 0.5*(P+1), N)

    # Prepare for vectorization
    Y = Y[np.newaxis, :]
    a1 = a1[:, np.newaxis]
    a2 = a2[:, np.newaxis]
    a3 = a3[:, np.newaxis]
    lower = lower[:, np.newaxis]
    upper = upper[:, np.newaxis]

    bar = np.log(Y/(1-Y))

    # The quantile function is by definition the inverse function of the CDF.
    # It maps probabilities to positions.
    # Theorem: the quantile function of F(X) = F(quantile function of X)

    # An unbounded r.v. is:
    quantileFunc = a1 + a2*bar + a3*(Y-0.5)*bar



    # Apply exp to the r.v. to get positive (semibounded) r.v.
    positive = np.exp(quantileFunc)

    # Apply a simple transformation to the r.v. that bounds the positive variable between lower and upper
    # (Note that coefficients have been derived such that the final result has the desired properties)
    boundedQuantileFunc = (lower+upper*positive)/(1+positive)

    boundedpdf = False

    if computePDF:
        # pdf of unbounded r.v. is 1/(quantile function'), which is
        pdf = 1/(a2/(Y*(1-Y)) + a3*((Y-0.5)/(Y*(1-Y)) + bar))
        # pdf of bounded quantile function is 1/(bounded quantile function') which is
        boundedpdf = pdf*(1+positive)**2/((upper-lower)*positive)

    # the quantile function is already evaluated and therefore gives a grid of values
    # the pdf corresponds to this grid (which was set by Y)

    return np.squeeze(boundedQuantileFunc), np.squeeze(boundedpdf)

# returns N random samples from the symetric percentile triplet bounded Keelin (logit metalog) distribution
def sample_bounded_keelin_3(a, aquant_l, median, aquant_u, lower, upper, N):

    # Treat scalar case as special case of vector input

    aquant_l = np.atleast_1d(aquant_l)
    aquant_u = np.atleast_1d(aquant_u)
    median = np.atleast_1d(median)
    lower = np.atleast_1d(lower)
    upper = np.atleast_1d(upper)

    # Some definitions

    gl = (aquant_l-lower)/(upper-aquant_l)
    gu = (aquant_u-lower)/(upper-aquant_u)
    gm = (median-lower)/(upper-median)

    # In this special case the coefficients can be computed analytically
    a1 = np.log(gm)
    foo = np.log((1-a)/a)
    a2 = 0.5/foo*np.log(gu/gl)
    a3 = np.log(gu*gl/gm**2)/((1-2*a)*foo)

    # Y is the cumulative probability. Explore a fraction P of the distribution
    Y = np.random.rand(a1.size, N)

    # Prepare for vectorization
    a1 = a1[:, np.newaxis]
    a2 = a2[:, np.newaxis]
    a3 = a3[:, np.newaxis]
    lower = lower[:, np.newaxis]
    upper = upper[:, np.newaxis]

    bar = np.log(Y/(1-Y))

    # The quantile function is by definition the inverse function of the CDF.
    # It maps probabilities to positions.
    # Theorem: the quantile function of F(X) = F(quantile function of X)

    # An unbounded r.v. is:
    quantileFunc = a1 + a2*bar + a3*(Y-0.5)*bar

    # Apply exp to the r.v. to get positive (semibounded) r.v.
    positive = np.exp(quantileFunc)

    # Apply a simple transformation to the r.v. that bounds the positive variable between lower and upper
    # (Note that coefficients have been derived such that the final result has the desired properties)
    boundedQuantileFunc = (lower+upper*positive)/(1+positive)

    boundedpdf = False

    return np.squeeze(boundedQuantileFunc)


# This function returns neighbor difference of the bounded Keelin 3-term CDF for a given grid x.
# These values approximate the probabilities pdf(x)dx.
# The mass is distibuted equally to left and right gridpoint.
#
# In general, a discrete CDF difference is different from a pdf in some ways:
# 1) no division by dx - not a density, but probability weights that sum to 1 always
# 2) unlike evaluating a pdf at a coarse grid x, it never misses a peak
# 3) the measure is included and the result is invariant.
# The last point means: for non-equally spaced x the result changes shape
# and is not anymore proportional to the pdf in
# the old coordinates (but to the pdf when transformed to the coordinates in which the points x are linearly spaced)
#
# This function is very suitable for numerical integrals of the (Keelin) pdf with another function f(r) discretized
# such that f is well-resolved (slowly varying on the grid r).
#
# The integral \int f(r) keelin_x(x(r)) (dx/dr) dr = \int f(r) keelin_r(r) dr \equiv \int f(r(x)) keelin_x(x) dx
# is approximated as the sum of the product of f(r) and the return value
# of this function, for x = x(r).
# Important: No Jacobian (dx/dr) is needed since we return an approximation of keelin_x(x(r)) dx (last integral)

# The returned function is piecewise constant (derivative of a linear interpolation)
# and may have some small peaks at the boundary
# since the CDF jumps there to 0 (left) and 1 (right) (visible for too small P).
# None of this is pathological in the sense that the sum is guarantied to be == 1.

def bounded_keelin_3_discrete_probabilities(x, a, aquant_l, median, aquant_u, lower, upper, N=100, P=.9999):

    cdfs, _ = bounded_keelin_3(a, aquant_l, median, aquant_u, lower, upper, N, P, computePDF = False)

    # extend grid by one point on each side
   
    if cdfs.ndim > 1:
        if x.ndim > 1:
            x = np.concatenate(((x[:,0] - (x[:,1]-x[:,0]))[:,np.newaxis], x, (x[:,-1] + (x[:,-1]-x[:,-2]))[:,np.newaxis]), axis=1)
            interpcdfs = np.zeros((cdfs.shape[0], x.shape[1]))
        else:
            x = np.concatenate(([x[0] - (x[1]-x[0])], x, [x[-1] + (x[-1]-x[-2])]))
            interpcdfs = np.zeros((cdfs.shape[0], x.size))
    else:
        x = np.concatenate(([x[0] - (x[1]-x[0])], x, [x[-1] + (x[-1]-x[-2])]))
        interpcdfs = np.zeros(x.size)

    
    Y = np.linspace(0.5*(1-P), 0.5*(P+1), N)

    # a cruical step is to set left and right for interp
    if cdfs.ndim > 1:
        for i in range(cdfs.shape[0]):
            foo = x
            if x.ndim > 1:
                foo = x[i,:]

            interpcdfs[i,:] = np.interp(foo, cdfs[i,:], Y, left = 0, right = 1)
        # diff is one pt shorter
        d = np.diff(interpcdfs)
        # return the mean of the left and right difference - same length as x.
        return 0.5*(d[:,:-1] + d[:, 1:])
    else:
        interpcdfs = np.interp(x, cdfs, Y, left = 0, right = 1)
        d = np.diff(interpcdfs)
        return 0.5*(d[:-1] + d[1:])


# This function returns neighbor difference of the bounded Keelin 3-term CDF for a given grid x.
# These values approximate the probabilities pdf(x)dx.
# The mass between two grid points is returned as an array one element shorter than the grid
def bounded_keelin_3_discrete_probabilities_between(x, a, aquant_l, median, aquant_u, lower, upper, N=100, P=.9999):

    cdfs, _ = bounded_keelin_3(a, aquant_l, median, aquant_u, lower, upper, N, P, computePDF = False)

   
   
    if cdfs.ndim > 1:
        if x.ndim > 1:
            interpcdfs = np.zeros((cdfs.shape[0], x.shape[1]))
        else:
            interpcdfs = np.zeros((cdfs.shape[0], x.size))
    else:
        interpcdfs = np.zeros(x.size)

    
    Y = np.linspace(0.5*(1-P), 0.5*(P+1), N)

    # a cruical step is to set left and right for interp
    if cdfs.ndim > 1:
        for i in range(cdfs.shape[0]):
            foo = x
            if x.ndim > 1:
                foo = x[i,:]

            interpcdfs[i,:] = np.interp(foo, cdfs[i,:], Y, left = 0, right = 1)
        # diff is one pt shorter
        d = np.diff(interpcdfs)
        # return the mean of the left and right difference - same length as x.
        return d
    else:
        interpcdfs = np.interp(x, cdfs, Y, left = 0, right = 1)
        d = np.diff(interpcdfs)
        return d
