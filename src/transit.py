import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import BoxLeastSquares
import lightkurve as lk
import ktransit

def build_ktransit_model(ticid, lc, rprs=0.02, vary_transit=True):
    from ktransit import FitTransit
    fitT = FitTransit()

    model = BoxLeastSquares(lc.time.value, lc.flux.value)
    results = model.autopower(0.16, minimum_period=2., maximum_period=21.)
    period = results.period[np.argmax(results.power)]
    t0 = results.transit_time[np.argmax(results.power)]
    if rprs is None:
        depth = results.depth[np.argmax(results.power)]
        rprs = depth ** 2

    fitT.add_guess_star(rho=0.022, zpt=0, ld1=0.6505,ld2=0.1041) #come up with better way to estimate this using AS
    fitT.add_guess_planet(T0=t0, period=period, impact=0.5, rprs=rprs)

    ferr = np.ones_like(lc.time.value) * 0.00001
    fitT.add_data(time=lc.time.value,flux=lc.flux.value,ferr=ferr)#*1e-3)

    vary_star = ['zpt']      # free stellar parameters
    if vary_transit:
        vary_planet = (['period', 'impact',       # free planetary parameters
            'T0', #'esinw', 'ecosw',
            'rprs']) #'impact',               # free planet parameters are the same for every planet you model
    else:
        vary_planet = (['rprs'])

    fitT.free_parameters(vary_star, vary_planet)
    fitT.do_fit()                   # run the fitting

    return fitT

def _individual_ktransit_dur(time, data):
    """ """
    inds = np.where(data < np.median(data))[0]
    first_transit = np.split(inds, np.where(np.diff(inds) != 1)[0] + 1)[0]

    dur = (time[first_transit[-1]] - time[first_transit[0]]) * 24.0

    return dur
